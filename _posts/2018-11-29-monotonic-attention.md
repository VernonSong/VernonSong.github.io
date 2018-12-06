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
Local Monotonic Attention重点从两方面对Global Attention进行优化
-  **Local**：让每个decoder时间步只从一小部分encoder state计算Attention
- **Monotonic**：每次计算Local Attention的位置受到单调向右的约束。

因此，假设需要关注的信息分布为以$p_t$为中心的正态分布。在每个decoder时间步计算Attention时，通过$\Delta p_t$来决定关注信息的中心位置需从上一个中心位置$p_{t-1}$向前移动多少个时间步。

![](/img/in-post/post-ml-attention.png)

$\Delta p_t$的计算有两种方案：有限制和无限制，如果不限制移动距离，则计算方法为：

$$
\Delta p_t = \mathrm{exp} (V_p ^T tanh(W_ph_t^d))
$$

如果移动距离不能大于$C_{max}$，则通过sigmoid函数构成门控，控制移动距离带大小：

$$
\Delta p_t = C_{max} * sigmoid(V_p ^T tanh(W_ph_t^d))
$$

在计算高斯分布时，额外引入$\lambda$缩放比例

$$
\lambda_t = \mathrm{exp} (V_{\lambda} ^T tanh(W_{\lambda}h_t^d))
$$

$$
a_t^{\mathcal{N}}(s)=\lambda_t *\mathrm{exp}(-\frac{(s-p_t)^2}{2\sigma^2})
$$

超参数$\sigma$控制正态分布范围。

若单纯使用使用正态分布计算Attention，效果会比较差，因此再添加

$$
a_t^{\mathcal{S}}(s)=\mathrm{Score}(h_s^e,h_t^d)=V_s^T tanh(W_s[h_s^e,h_t^d])
$$

$$
\forall s \in[p_t-2\sigma,p_t+2\sigma]
$$

这样最终的Attention信息为：

$$
c_t = \sum_{s=(p_t-2\sigma)}^{(p_t+2\sigma)}(a_t^{\mathcal{N}}(s)*a_t^{\mathcal{S}}(s))*h_s^e
$$


## 实现
### Tensorflow

## 参考
> [Local Monotonic Attention Mechanism for End-to-End Speech and Language Processing](https://arxiv.org/pdf/1705.08091.pdf)