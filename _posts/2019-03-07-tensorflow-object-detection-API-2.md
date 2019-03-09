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
- **get_label_map_dict**：
- **create_category_index_from_labelmap**：读取label map文件
- **create_class_agnostic_category_index**：创建一个未知类别

## learning_schedules









-




