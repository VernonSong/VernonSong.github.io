---
layout: post
title:  Local Monotonic Attention原理与实现
subtitle: 
date: 2018-11-22 00:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-lm_attention.jpg"
catalog: true
mathjax: true
tags:
    - 深度学习
    - 论文翻译
---

## 概述
基于Attention的encoder-decoder模型在序列任务中取得了很不错的成绩，但对于语音识别等任务，它们的输入与输出是单调对齐的，而Global Attention会从全部输入中寻找需要关注的信息，增加了计算量和学习难度，因此研究者提出了Local Monotonic Attention，减少Attention范围，并增加单调对齐的约束。

## 原理 
### Local Monotonic Attention
Local Monotonic Attention重点从两方面对Global Attention进行优化
-  **Local**：让每个decoder时间步只从一小部分encoder state计算Attention
- **Monotonic**：每次计算Local Attention的位置受到单调向右的约束。

因此，在每个decoder时间步计算Attention时，只关注固定大小的窗口，并通过$\Delta p_t$来决定窗口中心需从上一个窗口中心$p_{t-1}$向前移动多少个时间步。

#### 窗口移动距离
窗口移动距离$\Delta p_t$的计算有两种方案：有限制和无限制，如果限制移动距离不能大于$C_{max}$，则计算方法为：

$$
\Delta p_t = C_{max} * sigmoid(V_p ^T
$$





## 实现
### Tensorflow

## 参考
> [Local Monotonic Attention Mechanism for End-to-End Speech and Language Processing](https://arxiv.org/pdf/1705.08091.pdf)