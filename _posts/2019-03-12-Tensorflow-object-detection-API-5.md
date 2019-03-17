---
layout: post
title:  Tensorflow object detection API源码分析【4】
subtitle:  Anchor
date: 2019-03-01 20:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-tensorflow-api3.jpg"
catalog: true
tags:
    - tensorflow
---

Tensorflow object detection API中包含GridAnchorGenerator，MultipleGridAnchorGenerator，MultiscaleGridAnchorGenerator这3个Anchor类，它们继承于抽象基类AnchorGenerator。

## core.anchor_generator
包含抽象基类**AnchorGenerator**
- **name_scope**：纯虚函数，该函数用于返回对象的命名空间
- **check_num_anchors**：属性函数，控制generate函数是否动态检查生成的锚点数
- **num_anchors_per_location**：纯虚函数，该函数用于计算特征图上每个点包含多少Anchor
- **generate**：生成Anchor并检查
- **_generate**：纯虚函数，该函数用于生成Anchor
- **_assert_correct_number_of_anchors**：检查Anchor数目是否正确

## anchor_generators.grid_anchor_generator
包含继承于AnchorGenerator的子类

**GridAnchorGenerator**类为Faster R-CNN中所使用的Anchor方法。

- **__init__**：构造函数，所有参数均有默认值
- **name_scope**：返回对象的命名空间
- **num_anchors_per_location**：计算特征图上每个点包含多少Anchor
- **_generate**：生成Grid Anchor

此外，还有两个供GridAnchorGenerator内部使用的函数：
- **tile_anchors**：Grid Anchor生成核心计算部分
- **_center_size_bbox_to_corners_bbox**：将Anchor中心位置与Anchor尺寸转换为Anchor边框坐标

#### \_\_init\_\_
```python
  def __init__(self,
               scales=(0.5, 1.0, 2.0),
               aspect_ratios=(0.5, 1.0, 2.0),
               base_anchor_size=None,
               anchor_stride=None,
               anchor_offset=None):
    """Constructs a GridAnchorGenerator.

    Args:
      scales: a list of (float) scales, default=(0.5, 1.0, 2.0)
      aspect_ratios: a list of (float) aspect ratios, default=(0.5, 1.0, 2.0)
      base_anchor_size: base anchor size as height, width (
                        (length-2 float32 list or tensor, default=[256, 256])
      anchor_stride: difference in centers between base anchors for adjacent
                     grid positions (length-2 float32 list or tensor,
                     default=[16, 16])
      anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                     upper left element of the grid, this should be zero for
                     feature networks with only VALID padding and even receptive
                     field size, but may need additional calculation if other
                     padding is used (length-2 float32 list or tensor,
                     default=[0, 0])
    """
    # Handle argument defaults
    if base_anchor_size is None:
      base_anchor_size = [256, 256]
    base_anchor_size = tf.to_float(tf.convert_to_tensor(base_anchor_size))
    if anchor_stride is None:
      anchor_stride = [16, 16]
    anchor_stride = tf.to_float(tf.convert_to_tensor(anchor_stride))
    if anchor_offset is None:
      anchor_offset = [0, 0]
    anchor_offset = tf.to_float(tf.convert_to_tensor(anchor_offset))

    self._scales = scales
    self._aspect_ratios = aspect_ratios
    self._base_anchor_size = base_anchor_size
    self._anchor_stride = anchor_stride
    self._anchor_offset = anchor_offset
```

#### tile_anchors
```python
def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 base_anchor_size,
                 anchor_stride,
                 anchor_offset):
  """Create a tiled set of anchors strided along a grid in image space.

  This op creates a set of anchor boxes by placing a "basis" collection of
  boxes with user-specified scales and aspect ratios centered at evenly
  distributed points along a grid.  The basis collection is specified via the
  scale and aspect_ratios arguments.  For example, setting scales=[.1, .2, .2]
  and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
  .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
  and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before
  placing it over its respective center.

  Grid points are specified via grid_height, grid_width parameters as well as
  the anchor_stride and anchor_offset parameters.

  Args:
    grid_height: size of the grid in the y direction (int or int scalar tensor)
    grid_width: size of the grid in the x direction (int or int scalar tensor)
    scales: a 1-d  (float) tensor representing the scale of each box in the
      basis set.
    aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each
      box in the basis set.  The length of the scales and aspect_ratios tensors
      must be equal.
    base_anchor_size: base anchor size as [height, width]
      (float tensor of shape [2])
    anchor_stride: difference in centers between base anchors for adjacent grid
                   positions (float tensor of shape [2])
    anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                   upper left element of the grid, this should be zero for
                   feature networks with only VALID padding and even receptive
                   field size, but may need some additional calculation if other
                   padding is used (float tensor of shape [2])
  Returns:
    a BoxList holding a collection of N anchor boxes
  """
  # 计算可能的高和宽
  ratio_sqrts = tf.sqrt(aspect_ratios)
  heights = scales / ratio_sqrts * base_anchor_size[0]
  widths = scales * ratio_sqrts * base_anchor_size[1]
  # 生成特征图每个Anchor中心位置的网格
  # Get a grid of box centers
  y_centers = tf.to_float(tf.range(grid_height))
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = tf.to_float(tf.range(grid_width))
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  x_centers, y_centers = ops.meshgrid(x_centers, y_centers)
  # 生成每个x轴和Anchor宽的网格
  widths_grid, x_centers_grid = ops.meshgrid(widths, x_centers)
  # 生成每个y轴和Anchor高的网格
  heights_grid, y_centers_grid = ops.meshgrid(heights, y_centers)
  # 生成Anchor中心位置矩阵
  bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
  # 生成Anchor长宽矩阵
  bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
  bbox_centers = tf.reshape(bbox_centers, [-1, 2])
  bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
  # 生成边框坐标
  bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
  # 返回BoxList
  return box_list.BoxList(bbox_corners)
```

## object_detection.multiple_grid_anchor_generator


## builders.anchor_generator_builder
- **build**：根据配置信息来生成Anchor生成器对象


