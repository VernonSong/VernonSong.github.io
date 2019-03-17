---
layout: post
title:  Tensorflow object detection API源码分析【4】
subtitle:  BoxCoder
date: 2019-03-01 20:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-tensorflow-api3.jpg"
catalog: true
tags:
    - tensorflow
---

box coder主要用作边框回归中Anchor坐标到边框坐标的转换，由抽象基类BoxCoder定义接口，并派生出FasterRcnnBoxCoder，KeypointBoxCoder，MeanStddevBoxCoder，SquareBoxCoder共4种实现。

## box_coder
包含box coder的抽象基类BoxCoder

### BoxCoder
- **code_size**：抽象属性，返回code的大小
- **encode**：调用_encode()计算目标参数
- **decode**：调用_decode()计算边框坐标
- **_encode**：纯虚函数，该函数用于计算目标参数
- **_decode**：纯虚函数，该函数用于计算边框坐标

### batch_decode

## faster_rcnn_box_coder
包含基于Faster R-CNN所实现的BoxCoder类

### FasterRcnnBoxCoder

- **__init__**：构造函数
- **code_size**：在FasterRcnnBoxCoder中固定为4
- **_encode**：encode，通过边框和Anchor计算目标参数
- **_decode**：decode，通过Anchor和回归参数计算边框

#### _encode
```python
  def _encode(self, boxes, anchors):
    """Encode a box collection with respect to anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    # Avoid NaN in division and log below.
    # 防止值为0以及log值过小
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    tx = (xcenter - xcenter_a) / wa
    ty = (ycenter - ycenter_a) / ha
    tw = tf.log(w / wa)
    th = tf.log(h / ha)
    # Scales location targets as used in paper for joint training.
    #  在Faster R-CNN开源代码中使用[10.0, 10.0, 5.0, 5.0]对ty, tx, th, tw进行缩放
    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      th *= self._scale_factors[2]
      tw *= self._scale_factors[3]
    return tf.transpose(tf.stack([ty, tx, th, tw]))
```

## builders.box_coder_build
- **build**：根据配置信息来生成box coder对象





