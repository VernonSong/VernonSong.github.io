---
layout: post
title: Label Smoothing原理与实现
subtitle: 一种多分类模型正则化方法
date: 2018-09-07 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-LabelSmoothing.jpg"
catalog: true
tags:
    - 深度学习
    - 论文翻译
---

## 概述
很多正则化方法都是以模型参数为出发点来减少过拟合，而Inception网络中提供了一种新的正则化方法，通过对输出
$$
Y
$$
添加噪声，来达到约束模型的效果。

## 原理
### one-hot缺陷
在多分类模型中，通常使用softmax层计算每个类的概率：

$$
\begin{align*}
p(k|x)=\frac{\mathrm{exp}(z_k)}{\sum_{i=1}^K\mathrm{exp}(z_i)}
\end{align*}
$$

设
$$
q(k|x)
$$
为
样本
$$
x
$$
的标签，有
$$
\sum_kq(k|x)=1
$$


使用one-hot方式对标签进行编码时

$$
q(k|x)=\begin{cases}
\begin{aligned}
1  \quad k=y
\newline   0  \quad k \neq y
\end{aligned}
\end{cases}
$$

忽略
$$
p
$$
与
$$
q
$$
之间的依赖关系，则交叉熵为：

$$
\begin{align*}
\ell =-\sum_{k=1}^K\mathrm{log}(p(k))q(k)
\end{align*}
$$

因为one-hot编码，
$$
q(k)
$$
只有当
$$
y=k
$$
时才为1，否则为0。所以当最小化交叉熵时就是在
$$
p=k
$$
最大化
$$
p(k)
$$
，
这将造成当
$$
k \neq q
$$
$$
z_y
$$
远远大于
$$
z_k
$$
对
$$
z_k
$$
,当样本中某一类远多于其他类时，可能会有两个问题：
1. 模型对样本预测的概率分布趋近于样本集标签分布，导致模型泛化性不强，容易过拟合
2. 鼓励模型放大正确类别与错误类别之见概率差，导致模型有时候会过于自信，但因为前一问题的存在，因此这种自信并不是一件好事

### label-smoothing正则化
基于此，我们修改one-hot编码，引入
$$
u(k)
$$
作为独立于
$$
x
$$
的标签噪声，
$$
\epsilon
$$
为噪声比例，则

$$
\begin{align*}
q'(k|x)=(1-\epsilon)\delta_{k,y}+\epsilon u(k)
\end{align*}
$$

设
$$
u(k)=1/K
$$
，即标签噪声均匀分布，则有

$$
\begin{align*}
q'(k|x)=(1-\epsilon)\delta_{k,y}+\epsilon u(k)
\end{align*}
$$

这便是**LSR（label-smoothing regularization），通过给标签加入噪声，对正确类别的概率加以约束，来阻止预测时正确标签的概率远远大于错误标签**。

### 实验结果
在包含1000类的ImageNet训练集中，引入label smooth正则，设
$$
u(k)=1/1000
$$
，
$$
\epsilon=0.1
$$
，网络准确率有0.2%的提升。

## 实现

### tensorflow

```python
def label_smoothing_regularization(self, chars_labels, weight=0.1):
    """Applies a label smoothing regularization.
    Uses the same method as in https://arxiv.org/abs/1512.00567.
    Args:
      chars_labels: ground truth ids of charactes,
        shape=[batch_size];
      weight: label-smoothing regularization weight.
    Returns:
      A sensor with the same shape as the input.
    """
    one_hot_labels = tf.one_hot(
      chars_labels, depth=self._params.num_char_classes, axis=-1)
    pos_weight = 1.0 - weight
    neg_weight = weight / self._params.num_char_classes
    return one_hot_labels * pos_weight + neg_weight
```

## 链接

### 论文
>[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)





