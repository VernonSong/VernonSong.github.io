---
layout: post
title: Attention原理与实现
subtitle: 
date: 2018-08-8 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-attention.jpg"
catalog: true
tags:
    - 深度学习
    - 论文翻译
---

## 概述
人们观察一幅图像时，总会聚焦于重要的部分，而不是把注意力平均分散到一整副图像中，基于此，人们引入Attention机制来使神经网络模型聚焦于重要的输入，以提升网络性能。

## 