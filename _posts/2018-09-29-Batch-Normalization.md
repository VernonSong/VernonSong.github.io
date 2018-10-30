---
layout: post
title:  Batch Normalization原理与实现
subtitle: 
date: 2018-06-27 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-batchNormalization.jpg"
catalog: true
mathjax: true
tags:
    - 深度学习
    - 论文翻译
---

## 概述
在神经网络训练过程中会出现内部协变量偏移，这种现象会影响收敛速度和模型泛化能力，为此，研究人员发明了Batch Normalization技术来在网络前向传播过程中动态的进行归一化数据，以此消除内部协变量偏移所带来的影响。

## 原理
### 内部协变量偏移
我们可以把一个神经网络拆解为$f_1$和$f_2$两个子网络：

$$
\ell =f_2(f_1(u,\theta_1),\theta_2)
$$

其中$\theta_1$与$\theta_2$分别是两个子网络中的参数，$\ell$为网络loss，对$f_2$进行梯度更新

$$
\theta_2 \leftarrow \theta_2 - \frac{\alpha}{m}\sum_{i=1}^m\frac{\partial F_2(x_i,\theta_2)}{\partial \theta_2}
$$

$x_i$为子网络$f_1$输出，可以发现无论是否在训练前对数据进行了处理，经过了$f_1$的计算后，传入到$f_2$中的输入又将是全新的分布，因此对于每一个batch的数据，$\theta$都要不断去调整以适应新的数据分布，这种现象称为**内部协变量偏移（internal covariate shift）**。

###
