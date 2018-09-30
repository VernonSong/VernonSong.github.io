---
layout: post
title: 关于激活函数
subtitle:  神经网络学习
date: 2018-03-04 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-actfun.jpg"
catalog: true
tags:
    - 深度学习
---

在神经网络中，每个层都会出现叫激活函数（activation function）的函数，它对神经网络可以说是至关重要，没有它，深度学习便于简单的MLP毫无差别。

### 激活函数的作用
激活函数虽然名为激活，但其真正作用是为神经网络带入非线性因素，就像SVM中的核函数一样，使模型有更强的表达力，比如，如果没有激活函数，对于下图中的神经网络，

$$
\begin{align*}&
a^{[1]}=z^{[1]}=w^{[1]T}x+b^{[1]}
\newline&
a^{[2]}=z^{[1]}=w^{[2]T}a^{[1]}+b^{[2]}
\end{align*}
$$

将式子展开
可以发现，若没有激活函数，

$$
\begin{align*}
a^{[2]}_1=z^{[2]}_1=&w^{[2]T}_1(w^{[1]T}x+b^{[2]})+b{[1]}
\newline=&(w^{[2]}w^{[1]}x)+w^{[2]T}b^{[1]}b^{[2]}
\newline=&w'x+b'
\end{align*}
$$

可以看出，若没有激活函数，不论隐藏层有多少层，它的计算都可以用一个隐藏层来表达，结果都为线性，最多不过使复杂的线性组合。所以我们需要一个函数，将每次计算的结果转变为非线性，这样模型的表达力才能更强。

### 常见激活函数
加拿大蒙特利尔大学的Bengio教授在 ICML 2016 的文章[Noisy Activation Functions](https://arxiv.org/pdf/1603.00391v3.pdf)中给出了激活函数的定义：激活函数是映射 h:R→R，且几乎处处可导。之所以对可导有要求，使因为在反向传播时需要进行求导运算。

##### Sigmoid函数
Sigmoid函数被称为S函数，因为它的图像成S型，其式子为

$$
g(z)=\dfrac{1}{1+e^{-z}}
$$
其导数为

$$
g'(z)=g(z)(1-g(z))
$$

![](https://github.com/VernonSong/Storage/blob/master/image/Sigmoid.png?raw=true)
在以前，Sigmoid是非常常用的激活函数，它易于理解，输出范围在0（完全不激活）到1（完全激活）之间，可用作输出层，表示概率，并且求导简单。但现如今，Sigmoid函数已很少被使用，主要原因有两点，一是当z的值很大或者很小时，函数的导数趋近于0，我们称之为软饱和，它带来的结果就是在那些区域，梯度趋近于0，**权值更新缓慢**。因此初始化时需特别留意，防止权值过大。

还有原因是Sigmoid函数的输出不是0中心的。导致在反向传播中，输入神经元总是正数，w在梯度更新时，要么全部增大，要么全部减少，最终梯度下降权重更新时出现**z字型的下降**。不过当整个批量的数据的梯度被加起来后，对于权重的最终更新将会有不同的正负，这样就从一定程度上减轻了这个问题。

##### tanh函数
tanh函数可以说是Sigmoid函数的替代品，其公式为：

$$
g(z)=\dfrac{1-e^{-2z}}{1+e^{-2z}}
$$

它的导数为：

$$
g'(z)=1-g(z)^2
$$

![](https://github.com/VernonSong/Storage/blob/master/image/tanh.png?raw=true)
而它的图像，就是将Sigmoid函数图像下移，使之关于x轴对称。

tanh函数解决了Sigmoid函数不已0为中心的问题，加快了收敛速度，但是依然存在梯度饱和的问题。

##### ReLU函数
ReLU函数是近年来非常受欢迎的激活函数，它的中文名为线性整流函数，其公式为：

$$
g(z)=\begin{cases}
\begin{aligned}
0  \quad & z<0
\newline   z  \quad & z>=0
\end{aligned}
\end{cases}
$$


其导数非常简单

$$
g'(z)=\begin{cases}
\begin{aligned}
0  \quad & z<0
\newline   1  \quad & z>=0
\end{aligned}
\end{cases}
$$

![](https://github.com/VernonSong/Storage/blob/master/image/ReLU.png?raw=true)
线性整流函数被认为有一定的生物学原理，因为研究表明生物神经元的信息编码通常是比较分散及稀疏的，通常情况下，大脑中在同一时间大概只有1%-4%的神经元处于活跃状态。而线性整流函数，因为屏蔽了非正输入，使神经网络稀疏化，一般情况下，在一个使用修正线性单元（即线性整流）的神经网络中大概有50%的神经元处于激活态。

ReLU优点显著，它提供了神经网络的**稀疏表达能力**，在x<0时，ReLU函数硬饱和，在x>=0时，不存在饱和问题，梯度不衰减，**有效缓解了梯度消失问题**，因为ReLU函数没有诸如指数计算等复杂运算，简化了计算过程，使神经网络整体**计算成本下降**。

但ReLU的特性也带来了一些缺点，在反向传播时，若大的梯度导致对权值w的减少过于剧烈，导致所有样本在此处变为负数，梯度为0，权值不在更新。我们将这种情况称为**神经元死亡**。实际使用使需要合理设置学习率来避免神经元大范围死亡这种情况发生。

为改进ReLu函数，有人提出了带泄漏线性整流函数（LReLU）,原理就是在x<0时，不直接输出0，而是输出x与一个非常小的数的乘积，避免在x<0时无法更新梯度。相应的，还有elu，selu但在实际使用时，人们发现LReLU对准确率并没有太大的影响，因此大部分情况还是直接使用ReLU即可。

![](https://github.com/VernonSong/Storage/blob/master/image/lrelu.png?raw=true)

ELU与LReLU类似，只是小与0部分换为
$$
a(e^x-1)
$$

![](https://github.com/VernonSong/Storage/blob/master/image/elu.png?raw=true)


##### Softmax函数
Softmax函数常常作为多分类网络中的最后一层，公式为：

$$
g(z)_j=\dfrac{e^{z}_j}{\sum^N_{n=1}e^z_n}
$$

其导数为
可以看出函数将每个类的输出都映射到0-1之间，且和为1。虽然看起来，貌似
$$
e^z
$$
这个计算多此一举，因为
$$
g(z)_j=\dfrac{z_j}{\sum^N_{n=1}z_n}
$$
也能算出每个类的概率，但Softmax函数的好处在于它将大的值放大，小的值缩小，毕竟其实我们最终想要的是概率最大的那个结果。
与Softmax相对的是Hardmax，此函数将最大值映射为1，其余为0，这样虽然得到了概率最大的结果，但无法进行求导。而Softmax函数的导函数非常简单，推导后可得：

$$
g'(z)=a_i-y_i
$$

对于多分类问题，只有正确的那一类y才是1，其余都是0。
