---
layout: post
title:  Tensorflow object detection API源码分析【2】
subtitle:   util部分
date: 2019-03-01 20:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-tensorflow-api2.jpg"
catalog: true
tags:
    - tensorflow
---

## category_util
category_util包含了处理目标检测时的类别标签的函数
- **load_categories_from_csv_file**：从CSV中读取类别标签
- **save_categories_to_csv_file**：将类别标签存入CSV文件

## config_util
读取和更新配置

## context_manager
不知道

## dataset_util
dataset_util包含了生成TFRecord的工具函数
- **int64_feature**：将int64类型转化为tf.train.Feature类型
- **int64_list_feature**：将int64列表类型转化为tf.train.Feature类型
- **bytes_feature**：将bytes类型转化为tf.train.Feature类型
- **bytes_list_feature**：将bytes列表类型转化为tf.train.Feature类型
- **float_list_feature**：将float列表类型转化为tf.train.Feature类型
- **read_examples_list**：读取txt文档每行第一个字段，假设某行为xyz 3，则读取xyz，该函数主要用于读取索引，来查找图片（xyz.jpg）和标签（xyz.xml）
- **recursive_parse_xml_to_dict**：递归读取xml中内容

## json_util

## label_map_util
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

## learning_schedules
学习率自调整相关的工具函数

### exponential_decay_with_burnin
先常数学习率，一定步数后指数衰减
```python
def exponential_decay_with_burnin(global_step,
                                  learning_rate_base,
                                  learning_rate_decay_steps,
                                  learning_rate_decay_factor,
                                  burnin_learning_rate=0.0,
                                  burnin_steps=0,
                                  min_learning_rate=0.0,
                                  staircase=True):
  """Exponential decay schedule with burn-in period.

  In this schedule, learning rate is fixed at burnin_learning_rate
  for a fixed period, before transitioning to a regular exponential
  decay schedule.

  Args:
    global_step: int tensor representing global step.
    learning_rate_base: base learning rate.
    learning_rate_decay_steps: steps to take between decaying the learning rate.
      Note that this includes the number of burn-in steps.
    learning_rate_decay_factor: multiplicative factor by which to decay
      learning rate.
    burnin_learning_rate: initial learning rate during burn-in period.  If
      0.0 (which is the default), then the burn-in learning rate is simply
      set to learning_rate_base.
    burnin_steps: number of steps to use burnin learning rate.
    min_learning_rate: the minimum learning rate.
    staircase: whether use staircase decay.

  Returns:
    a (scalar) float tensor representing learning rate
  """
  if burnin_learning_rate == 0:
    burnin_learning_rate = learning_rate_base
  # 指数衰减学习率部分
  post_burnin_learning_rate = tf.train.exponential_decay(
      learning_rate_base,
      global_step - burnin_steps,
      learning_rate_decay_steps,
      learning_rate_decay_factor,
      staircase=staircase)
  # 添加常数学习率，并保证学习率大于min_learning_rate
  return tf.maximum(tf.where(
      tf.less(tf.cast(global_step, tf.int32), tf.constant(burnin_steps)),
      tf.constant(burnin_learning_rate),
      post_burnin_learning_rate), min_learning_rate, name='learning_rate')
```









-




