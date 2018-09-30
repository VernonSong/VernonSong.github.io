---
layout: post
title:  Batch Normalization原理与实现
subtitle: 
date: 2018-06-27 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-batchNormalization.jpg"
catalog: true
tags:
    - 深度学习
    - 论文翻译
---

## 概述
归一化数据是在训练前一个必不可少的工作，但在前向传播过程中，数据的分布又会发生变化，这种情况称为**内部协变量偏移（internal covariate shift）**，为减少此问题带来的影响，研究人员发明了Batch Normalization技术来在网络前向传播过程中动态的进行归一化数据，达到加速收敛于提高准确率的目的。

## 原理

改变参数很小，影响也会很大
饱和，收敛满