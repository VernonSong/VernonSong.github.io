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
### 传统神经网络缺陷
#### 内部协变量偏移
我们可以把一个神经网络拆解为$f_1$和$f_2$两个子网络：

$$
\ell =f_2(f_1(u,\theta_1),\theta_2)
$$

其中$\theta_1$与$\theta_2$分别是两个子网络中的参数，$\ell$为网络loss，对$f_2$进行梯度更新

$$
\theta_2 \leftarrow \theta_2 - \frac{\alpha}{m}\sum_{i=1}^m\frac{\partial F_2(x_i,\theta_2)}{\partial \theta_2}
$$

$x_i$为子网络$f_1$输出，因为$f_1$自身也在不断更新，在经过$f_1$的计算后，传入到$f_2$中的输入$x_i$又将是全新的分布，因此对于每一个batch的数据，$\theta$都要不断去调整以适应新的数据分布，这种现象称为**内部协变量偏移（internal covariate shift）**。

#### 梯度消失
当使用sigmoid或tanh激活函数时，若输入值过大或过小，此时会出现软饱和现象，导数接近于0，梯度消失，难以收敛。虽然使用ReLU激活函数或者慎重选择初始化值可以在一定程度上避免梯度消失。但可以发现上述情况发生的原因之一在于某些层的输入数据分布情况不理想。

### Batch Normalization
#### 训练阶段正向传播
在Batch Normalization诞生之前，有研究人员采用在网络中添加若干层，进行白化（Whitening）操作，使神经网络的每一层的输入有相对固定的分布，但却带来了新问题：

- 难以有效计算梯度
- 白化操作计算量太大
- 白化过程改变了原输入的表达信息

因此，简化白化操作，单独对每个特征进行normalizaiton，让每个特征数据分布情况都是均值为0，方差为1。

$$
\begin{align*}
& E[x^{(k)}]=\frac{1}{N} \sum_{i=1}^Nx_i^{(k)}
\newline &\hat{x}^{(k)}=\frac{x^{(k)-E[x^{(k)}]}}{\sqrt{\mathrm{Var}[x^{(k)}]}}
\end{align*}
$$

其中$N$为样本总数，$k$为该层特征维数。通过加入这样简化的白化层确实达到了加速收敛的效果，但这样的normalizaiton操作依然改变了原输入的表达信息。因此还需再添由可学习的参数控制的线性变换，让数据恢复表达能力。

$$
y^{(k)}=\gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)} 
$$

当$y^{(k)}=\sqrt{\mathrm{Var}[x^{(k)}]}$，$\beta^{(k)}=E[x^{(k)}] $时，normalizaiton后的数据完全恢复为原数据。

目前的normalizaiton操作是基于整个训练集，但由于在实际训练中通常采用SGD等方法，将训练集分成若干个mini-batch进行训练，在这种情况下该normalizaiton操作难以实现。因此，再次进行简化，在训练时只计算当前batch的均值和方差，这样的改动同时使normalizaiton操作可以进行梯度计算。由此的到Batch Normalization公式：

$$
\begin{align*}
 &\mu \leftarrow \frac{1}{m}\sum_{i=1}^m x_i
\newline &\sigma_{\mathcal{B}}^2 \leftarrow \frac{1}{m}\sum_{i=1}^m (x_i-\mu_{\mathcal{B}})^2
\newline &\hat{x}_i \leftarrow \frac{x_i-\mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2+\epsilon}}
\newline &y_i \leftarrow \gamma \hat{x_i} + \beta \equiv \mathrm{BN}_{\sigma , \beta}(x_i)
\end{align*}
$$

其中$m$为mini batch中样本数，$ \mathcal{B}=\{ x_1,\cdots,x_m\}$，$\mu$与$\sigma_{\mathcal{B}}^2$分别是$x$在mini batch上的均值和方差，算法中将方差加上常数$\epsilon$以保证数值稳定性。为表述方便，在上述公式中省略的特征维度$k$。

#### 训练阶段反向传播

$$
\begin{align*}
& \frac{\partial \ell}{\partial \hat{x_i}}=\frac{\partial \ell}{\partial y_i} \cdot \gamma
 \newline & \frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^2} =\sum_{i=1}^m \frac{\partial \ell}{\partial \hat{x_i}} \cdot (x_i - \mu_{\mathcal{B}}) \cdot \frac{-1}{2}(\sigma_{\mathcal{B}}^2+\epsilon)^{(-3/2)}
\newline & \frac{\partial \ell}{\partial\mu_{\mathcal{B}}}=	\left (  \sum_{i=1}\right ) 
\end{align*}
$$



## 实现
### Tensorflow API




