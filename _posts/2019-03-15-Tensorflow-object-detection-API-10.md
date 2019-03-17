---
layout: post
title:  Tensorflow object detection API源码分析【4】
subtitle: Sample
date: 2019-03-01 20:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-tensorflow-api3.jpg"
catalog: true
tags:
    - tensorflow
---

Sample相关的组件用来解决目标检测中，正样本的数目往往小与负样本的问题

## core.minibatch_sampler
包含抽象基类**MinibatchSampler**，用来对minibatch进行二次抽样，调整正负样本

### MinibatchSampler
- **__init__**：构造函数，无实现
- **subsample**：纯虚函数，
- **subsample_indicator**：静态函数，调整indicator中Ture元素数目，使其不大于指定数目

## core.balanced_positive_negative_sampler


