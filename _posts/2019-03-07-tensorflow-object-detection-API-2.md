---
layout: post
title:  Tensorflow object detection API源码分析【2】
subtitle:   基础组件
date: 2019-03-01 20:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-tensorflow-api2.jpg"
catalog: true
tags:
    - tensorflow
---

## utils.category_util
category_util包含了处理目标检测时的类别标签的函数
- **load_categories_from_csv_file**：从CSV中读取类别标签
- **save_categories_to_csv_file**：将类别标签存入CSV文件

## utils.config_util
读取和更新配置

## utils.context_manager
不知道

## utils.dataset_util
dataset_util包含了生成TFRecord的工具函数
- **int64_feature**：将int64类型转化为tf.train.Feature类型
- **int64_list_feature**：将int64列表类型转化为tf.train.Feature类型
- **bytes_feature**：将bytes类型转化为tf.train.Feature类型
- **bytes_list_feature**：将bytes列表类型转化为tf.train.Feature类型
- **float_list_feature**：将float列表类型转化为tf.train.Feature类型
- **read_examples_list**：读取txt文档每行第一个字段，假设某行为xyz 3，则读取xyz，该函数主要用于读取索引，来查找图片（xyz.jpg）和标签（xyz.xml）
- **recursive_parse_xml_to_dict**：递归读取xml中内容

## utils.json_util

## utils.label_map_util
类别标签的工具函数，其中label map为

```txt
item {
  id: 1
  name: 'dog'
}

item {
  id: 2
  name: 'cat'
}
```

- **_validate_label_map**：检查label map的id是否大于等于0，id为0的是否是背景类
- **create_category_index**：生成以编号为索引的类别名列表（[{'id':1,'name':'dog'},{'id':2,'name':'cat'}]到['dog','cat']
- **get_max_label_map_index**：获取label map的id最大值
- **convert_label_map_to_categories**：根据label map或类别数生成形如[{'id':1,'name':'dog'},{'id':2,'name':'cat'}]的列表
- **load_labelmap**：读取label map文件并进行检查
- **get_label_map_dict**：读取label map文件，检查有无数值错误，生成形如{'cat': 3, 'dog': 1, 'background': 0, 'class_2': 2}的字典，可通过参数fill_in_gaps_and_background来设定是否填补背景类和编号空缺的类
- **create_categories_from_labelmap**：读取label map文件并生成形如[{'id':1,'name':'dog'},{'id':2,'name':'cat'}]的列表
- **create_category_index_from_labelmap**：读取label map文件并生成形如['dog','cat']的列表
- **create_class_agnostic_category_index**：创建一个未知类别

## utils.static_shape
static_shape包含获取tensor shape的辅助工具函数，但只可用于4维tensor的TensorShape

- **get_batch_size**：获取TensorShape中的batch size
- **get_height**：获取TensorShape中的高
- **get_width**：获取TensorShape中的宽
- **get_depth**：获取TensorShape中的depth

## utils.shape_utils
shape_utils包含对tensor的尺寸进行操作的相关工具函数

- **_is_tensor**：判断输入是不是一个有效的tensor类（tf.Tensor, tf.SparseTensor, tf.Variable）
- **_set_dim_0**：改变输入（维数至少为1）的第0维
- **pad_tensor**：对输入的第0维补充至指定长度
- **clip_tensor**：对输入的第0维裁剪至指定长度
- **pad_or_clip_tensor**：对输入的第0维补充或裁剪至指定长度
- **pad_or_clip_nd**：对输入补充或裁剪至指定尺寸
- **combined_static_and_dynamic_shape**：返回tensor的shape，有静态值则按照静态值，无静态值的维使用动态值
- **static_or_dynamic_map_fn**：重写tf.map_fn()，对输入shape为静态情况进行优化
- **check_min_image_dim**：检查图片的宽高是否大与指定值
- **assert_shape_equal**：判断两个shape是否相等
- **assert_shape_equal_along_first_dimension**：判断两个shape的第0维是否相等

### static_or_dynamic_map_fn

```python
def static_or_dynamic_map_fn(fn, elems, dtype=None,
                             parallel_iterations=32, back_prop=True):
  """Runs map_fn as a (static) for loop when possible.

  This function rewrites the map_fn as an explicit unstack input -> for loop
  over function calls -> stack result combination.  This allows our graphs to
  be acyclic when the batch size is static.
  For comparison, see https://www.tensorflow.org/api_docs/python/tf/map_fn.

  Note that `static_or_dynamic_map_fn` currently is not *fully* interchangeable
  with the default tf.map_fn function as it does not accept nested inputs (only
  Tensors or lists of Tensors).  Likewise, the output of `fn` can only be a
  Tensor or list of Tensors.

  TODO(jonathanhuang): make this function fully interchangeable with tf.map_fn.

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same structure as elems. Its output must have the
      same structure as elems.
    elems: A tensor or list of tensors, each of which will
      be unpacked along their first dimension. The sequence of the
      resulting slices will be applied to fn.
    dtype:  (optional) The output type(s) of fn. If fn returns a structure of
      Tensors differing from the structure of elems, then dtype is not optional
      and must have the same structure as the output of fn.
    parallel_iterations: (optional) number of batch items to process in
      parallel.  This flag is only used if the native tf.map_fn is used
      and defaults to 32 instead of 10 (unlike the standard tf.map_fn default).
    back_prop: (optional) True enables support for back propagation.
      This flag is only used if the native tf.map_fn is used.

  Returns:
    A tensor or sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.
  Raises:
    ValueError: if `elems` a Tensor or a list of Tensors.
    ValueError: if `fn` does not return a Tensor or list of Tensors
  """
  # elems只能是Tenor或元素为Tensor的列表
  if isinstance(elems, list):
    for elem in elems:
      if not isinstance(elem, tf.Tensor):
        raise ValueError('`elems` must be a Tensor or list of Tensors.')

    elem_shapes = [elem.shape.as_list() for elem in elems]
    # Fall back on tf.map_fn if shapes of each entry of `elems` are None or fail
    # to all be the same size along the batch dimension.
    for elem_shape in elem_shapes:
      if (not elem_shape or not elem_shape[0]
          or elem_shape[0] != elem_shapes[0][0]):
        return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    # 对elems中每个Tensor进行分解和打包
    # elems：[[[1, 2, 3], [4, 5, -1], [0, 6, 9]],[[0], [2], [1]]]
    # arg_tuples：([1, 2, 3],[0]),([4, 5, -1], 2),([0, 6, 9], 1)
    [print(tf.unstack(elem)) for elem in elems]
    arg_tuples = zip(*[tf.unstack(elem) for elem in elems])

    print(arg_tuples)
    # 对元组中的每一个Tensor执行fn()
    outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
  else:
    if not isinstance(elems, tf.Tensor):
      raise ValueError('`elems` must be a Tensor or list of Tensors.')
    elems_shape = elems.shape.as_list()
    # 同上，非静态shape则运行tf.map_fn()
    if not elems_shape or not elems_shape[0]:
      return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    # 对Tensor进行分解，并对得到的每一个Tensor执行fn()
    outputs = [fn(arg) for arg in tf.unstack(elems)]
  # Stack `outputs`, which is a list of Tensors or list of lists of Tensors
  if all([isinstance(output, tf.Tensor) for output in outputs]):
    return tf.stack(outputs)
  else:
    # 如果outputs是由Tensor列表组成的列表，则
    if all([isinstance(output, list) for output in outputs]):
      if all([all(
          [isinstance(entry, tf.Tensor) for entry in output_list])
              for output_list in outputs]):
        # 对输出进行打包
        # outputs：[[[1, 2, 3],[0]],[[4, 5, -1], 2],[[0, 6, 9], 1]]
        # return：[[[1, 2, 3], [4, 5, -1], [0, 6, 9]],[[0], [2], [1]]]
        return [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
  raise ValueError('`fn` should return a Tensor or a list of Tensors.')
```
该函数是对tf.map_fn()的封装，对静态shape的输入不使用tf.map_fn()而直接循环使用fn()。需要对tf.map_fn()的源码进行分析才能理解这个方法的目的。

### check_min_image_dim
```python
def check_min_image_dim(min_dim, image_tensor):
  """Checks that the image width/height are greater than some number.

  This function is used to check that the width and height of an image are above
  a certain value. If the image shape is static, this function will perform the
  check at graph construction time. Otherwise, if the image shape varies, an
  Assertion control dependency will be added to the graph.

  Args:
    min_dim: The minimum number of pixels along the width and height of the
             image.
    image_tensor: The image tensor to check size for.

  Returns:
    If `image_tensor` has dynamic size, return `image_tensor` with a Assert
    control dependency. Otherwise returns image_tensor.

  Raises:
    ValueError: if `image_tensor`'s' width or height is smaller than `min_dim`.
  """
  image_shape = image_tensor.get_shape()
  image_height = static_shape.get_height(image_shape)
  image_width = static_shape.get_width(image_shape)
  # 如果不能获取静态的宽高信息
  if image_height is None or image_width is None:
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(image_tensor)[1], min_dim),
                       tf.greater_equal(tf.shape(image_tensor)[2], min_dim)),
        ['image size must be >= {} in both height and width.'.format(min_dim)])
    # 确保断言执行
    with tf.control_dependencies([shape_assert]):
      # 返回一个image_tensor的值的拷贝
      return tf.identity(image_tensor)

  if image_height < min_dim or image_width < min_dim:
    raise ValueError(
        'image size must be >= %d in both height and width; image dim = %d,%d' %
        (min_dim, image_height, image_width))

  return image_tensor
```

在该函数中，使用了控制依赖器tf.control_dependencies来确保断言执行，由于tf.control_dependencies只有当域内是op是才会生效，所以必须使用tf.identity()进行赋值操作而不能直接return。

### assert_shape_equal

```python
def assert_shape_equal(shape_a, shape_b):
  """Asserts that shape_a and shape_b are equal.

  If the shapes are static, raises a ValueError when the shapes
  mismatch.

  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
  mismatch.

  Args:
    shape_a: a list containing shape of the first tensor.
    shape_b: a list containing shape of the second tensor.

  Returns:
    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op
    when the shapes are dynamic.

  Raises:
    ValueError: When shapes are both static and unequal.
  """
  # 当shapes为静态
  if (all(isinstance(dim, int) for dim in shape_a) and
      all(isinstance(dim, int) for dim in shape_b)):
    if shape_a != shape_b:
      raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
    else: return tf.no_op()
  # 当shapes为动态
  else:
    return tf.assert_equal(shape_a, shape_b)
```
此函数中使用isinstance(dim, int)判断dim中值类型，以此来判断shapes是否为静态。若不是静态则需要使用tf.assert_equal()函数动态判断两个shape是否相等，assert_shape_equal_along_first_dimension同理。

## utils.ops
- **expanded_shape**：在一个shape向量的指定位置添加指定数目的全1向量，来增加shape维数。
- **normalized_to_image_coordinates**：将边框相对坐标转化为绝对坐标
- **meshgrid**：np.meshgrid()的高维扩展版本
- **fixed_padding**：padding
- **pad_to_multiple**：将tensor的宽高维padding到multiple的整数倍
- **padded_one_hot_encoding**：将tensor转化为one hot编码
- **dense_to_sparse_boxes**：
- **indices_to_dense_vector**：将indices转化为indicator
- **reduce_sum_trailing_dimensions**：对一个tensor指定维数后的所有维降维求和
- **retain_groundtruth**：
- **retain_groundtruth_with_positive_classes**：
- **replace_nan_groundtruth_label_scores_with_ones**：
- **filter_groundtruth_with_crowd_boxes**：
- **filter_groundtruth_with_nan_box_coordinates**：
- **normalize_to_target**：
- **batch_position_sensitive_crop_regions**：
- **position_sensitive_crop_regions**：
- **reframe_box_masks_to_image_masks**：
- **merge_boxes_with_multiple_labels**：
- **nearest_neighbor_upsampling**：
- **matmul_gather_on_zeroth_axis**：
- **matmul_crop_and_resize**：
- **expected_classification_loss_under_sampling**：
- **foreground_probabilities_from_targets**：前景前景



















-




