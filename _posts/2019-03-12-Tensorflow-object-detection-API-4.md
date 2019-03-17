---
layout: post
title:  Tensorflow object detection API源码分析【3】
subtitle:  Box
date: 2019-03-01 20:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-tensorflow-api3.jpg"
catalog: true
tags:
    - tensorflow
---
object detection中实现了两个边框类，一个基于tensorflow，一个基于numpy。这里只分析基于tensorflow的实现。

## core.box_list
包含**BoxList**类，定义了边框类和基本操作
### BoxList
- **__init__**：输入为一个[N,4]的tensor
- **num_boxes**：运行时计算并返回边框数量
- **num_boxes_static**：构造图时计算并返回边框数量
- **get_all_fields**：返回所有字段名
- **get_extra_fields**：返回非'boxes'字段名
- **add_field**：添加新字段和数据，主要用来添加标签等数据
- **has_field**：判断对象中是否有指定字段
- **get**：返回'boxes'字段的值
- **set**：设置'boxes'字段的值
- **get_field**：返回指定字段的值
- **set_field**：设置指定字段的值
- **get_center_coordinates_and_sizes**：计算边框宽高和中心点坐标
- **transpose_coordinates**：改变边框坐标中x，y坐标顺序
- **as_tensor_dict**：返回包含指定字段的字典，如不指定则返回全部

值得注意的是，BoxList中有num_boxes和num_boxes_static两个函数来获取box数量
```python
  def num_boxes(self):
    """Returns number of boxes held in collection.

    Returns:
      a tensor representing the number of boxes held in the collection.
    """
    return tf.shape(self.data['boxes'])[0]
```
该函数使用**tf.shape()函数，在运行时计算边框数量**。

```python
  def num_boxes_static(self):
    """Returns number of boxes held in collection.

    This number is inferred at graph construction time rather than run-time.

    Returns:
      Number of boxes held in collection (integer) or None if this is not
        inferrable at graph construction time.
    """
    return self.data['boxes'].get_shape()[0].value
```

而这个函数使用**get_shape()方法，边框数目在构造图时就已经确定**。

### core.box_list_ops
包含了边框对BoxList类的各种方法
- **aera**：返回BoxList对象每个边框的面积
- **height_width**：返回BoxList对象每个边框的高和宽
- **scale**：返回一个按指定比例缩放边框坐标的新BoxList
- **clip_to_window**：返回一个裁剪超出指定窗口的边框的新BoxList，可通过filter_nonoverlapping来设置是否过滤掉与窗口完全不重合的边框
- **prune_outside_window**：返回一个过滤掉超出指定窗口的边框的新BoxList和保留下来的边框索引
- **prune_completely_outside_window**：返回一个过滤掉完全不在指定窗口的边框的新BoxList
- **intersection**：计算BoxList1的每个边框与BoxList2中所有边框的交集面积
- **matched_intersection**：计算BoxList1的每个边框与BoxList2中对应边框的交集面积
- **iou**：计算BoxList1的每个边框与BoxList2中所有边框的IoU
- **matched_iou**：计算BoxList1的每个边框与BoxList2中对应边框的IoU
- **ioa**：计算BoxList1的每个边框与BoxList2中所有边框的IoA（两边框交集与另一个边框的面积比）
- **prune_non_overlapping_boxes**：计算BoxList1与BoxList2的IoA，过滤掉BoxList1中IoA小与某一阈值的边框，并返回一个新的BoxList和保留下来的边框索引
- **prune_small_boxes**：返回一个过滤掉长宽小与阈值的边框的新BoxList和保留下来的边框索引
- **change_coordinate_frame**：计算BoxList中的边框与指定窗口的相对坐标，并返回转化坐标后的新BoxList
- **sq_dist**：计算BoxList1的每个边框与BoxList2中相对应边框的平方距离
- **boolean_mask**：根据Boolean Mask返回只包含指定边框的新BoxList
- **gather**：根据索引返回只包含指定边框的新BoxList
- **concatenate**：对一组BoxList进行拼接
- **sort_by_field**：根据指定字段对Boxlist进行排序
- **visualize_boxes_in_image**：将边框绘制在图片上
- **filter_field_value_equals**：过滤掉boxlist中指定字段与指定值不相等的框
- **filter_greater_than**：过滤掉Boxlist中scores小与阈值的框
- **non_max_suppression**：NMS（非极大值抑制）算法
- **_copy_extra_fields**：将BoxList2中的非边框字段拷贝给BoxList1
- **to_normalized_coordinates**：将以绝对坐标表示的BoxList转化为以标准坐标表示的新BoxList
- **to_absolute_coordinates**：将以标准坐标表示的BoxList转化为以绝对坐标表示的新BoxList
- **refine_boxes_multi_class**：对每个类的目标边框分别执行refine_boxes()（NMS算法和边框投标算法）来提炼边框，并返回按scores排序后的BoxList
- **refine_boxes**：使用NMS算法过滤边框，并使用边框投标算法调整边框位置
- **box_voting**：边框投票算法
- **pad_or_clip_box_list**：调整BoxList中的边框数量
- **select_random_box**：从BoxList中随机选择一个边框，如果BoxList没有边框则返回默认边框
- **get_minimal_coverage_box**：生成一个能覆盖BoxList中所有边框的最小边框，如果BoxList没有边框则返回默认边框

### box_voting

```python
def box_voting(selected_boxes, pool_boxes, iou_thresh=0.5):
  """Performs box voting as described in S. Gidaris and N. Komodakis, ICCV 2015.

  Performs box voting as described in 'Object detection via a multi-region &
  semantic segmentation-aware CNN model', Gidaris and Komodakis, ICCV 2015. For
  each box 'B' in selected_boxes, we find the set 'S' of boxes in pool_boxes
  with iou overlap >= iou_thresh. The location of B is set to the weighted
  average location of boxes in S (scores are used for weighting). And the score
  of B is set to the average score of boxes in S.

  Args:
    selected_boxes: BoxList containing a subset of boxes in pool_boxes. These
      boxes are usually selected from pool_boxes using non max suppression.
    pool_boxes: BoxList containing a set of (possibly redundant) boxes.
    iou_thresh: (float scalar) iou threshold for matching boxes in
      selected_boxes and pool_boxes.

  Returns:
    BoxList containing averaged locations and scores for each box in
    selected_boxes.

  Raises:
    ValueError: if
      a) selected_boxes or pool_boxes is not a BoxList.
      b) if iou_thresh is not in [0, 1].
      c) pool_boxes does not have a scores field.
  """
  if not 0.0 <= iou_thresh <= 1.0:
    raise ValueError('iou_thresh must be between 0 and 1')
  if not isinstance(selected_boxes, box_list.BoxList):
    raise ValueError('selected_boxes must be a BoxList')
  if not isinstance(pool_boxes, box_list.BoxList):
    raise ValueError('pool_boxes must be a BoxList')
  if not pool_boxes.has_field('scores'):
    raise ValueError('pool_boxes must have a \'scores\' field')

  iou_ = iou(selected_boxes, pool_boxes)
  match_indicator = tf.to_float(tf.greater(iou_, iou_thresh))
  num_matches = tf.reduce_sum(match_indicator, 1)
  # TODO(kbanoop): Handle the case where some boxes in selected_boxes do not
  # match to any boxes in pool_boxes. For such boxes without any matches, we
  # should return the original boxes without voting.
  match_assert = tf.Assert(
      tf.reduce_all(tf.greater(num_matches, 0)),
      ['Each box in selected_boxes must match with at least one box '
       'in pool_boxes.'])

  scores = tf.expand_dims(pool_boxes.get_field('scores'), 1)
  scores_assert = tf.Assert(
      tf.reduce_all(tf.greater_equal(scores, 0)),
      ['Scores must be non negative.'])

  with tf.control_dependencies([scores_assert, match_assert]):
    sum_scores = tf.matmul(match_indicator, scores)
  averaged_scores = tf.reshape(sum_scores, [-1]) / num_matches
  # 加权计算得到新边框
  box_locations = tf.matmul(match_indicator,
                            pool_boxes.get() * scores) / sum_scores
  averaged_boxes = box_list.BoxList(box_locations)
  _copy_extra_fields(averaged_boxes, selected_boxes)
  averaged_boxes.add_field('scores', averaged_scores)
  return averaged_boxes
```

边框投票算法，对筛选后得到的边框，利用附近与它IoU大于阈值的边框对其进行修正：

$$
B'_{i,c}=\frac{\sum_{j:B_{j,c}\in \mathcal{N}(B_{i,c})}w_{j,c} \cdot B_{j,c}}{\sum_{j:B_{j,c}\in \mathcal{N}(B_{i,c})}w_{j,c}}
$$

## 参考
> [Object detection via a multi-region & semantic segmentation-aware CNN model](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Gidaris_Object_Detection_via_ICCV_2015_paper.pdf)


