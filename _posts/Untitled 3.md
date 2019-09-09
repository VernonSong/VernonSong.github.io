layout: post
title: Initialization总结
subtitle: 文本检测
date: 2019-06-01 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-pixel-anchor.jpg"
catalog: true
mathjax: true
tags:

    - 计算机视觉

    - 深度学习
typora-root-url: ../../blog



## 概述



初始化

在Multi-Branch Networks中，variance随branch增多而增加


$$
\begin{align*}
\operatorname{Var}\left[y^{(l)}\right]&=C_{i}^{(l)} n_{i}^{(l)} \operatorname{Var}\left[w^{(l)} x^{(l)}\right]
\newline &{=C_{i}^{(l)} n_{i}^{(l)} \operatorname{Var}\left[w^{(l)}\right] \mathrm{E}\left[\left(x^{(l)}\right)^{2}\right]}\newline &{=\alpha C_{i}^{(l)} n_{i}^{(l)} \operatorname{Var}\left[w^{(l)}\right] \operatorname{Var}\left[y^{(l-1)}\right]}
\end{align*}
$$


$C_{i}^{(l)}$为layer $l$的input branch数目，$n_{i}^{(l)}$为layer $l$每个branch的$\mathbf{x}^{(l)}$中element数目。$\alpha$取决于activation function，若为ReLu，则$\alpha$为0.5，Sigmoid这类则$\alpha$为1。为了使variance不会逐层增加，我们希望：


$$
\alpha C_{i}^{(l)} n_{i}^{(l)} \operatorname{Var}\left[w^{(l)}\right]=1
$$


因此$W^{(l)}$因初始化为$1 /\left(\alpha C_{i}^{(l)} n_{i}^{(l)}\right)$。



同理可从backward propagation角度推导出$W^{(l)}$的最佳初始化为$1 /\left(\alpha C_{o}^{(l)} n_{o}^{(l)}\right)$，$C_{o}^{(l)} n_{o}^{(l)}$为layer $l$的所有output branch中$\mathbf{X}^{(l)}$中element数目，由于大多数情况下

$C_{i}^{(l)} n_{i}^{(l)} \neq C_{o}^{(l)} n_{o}^{(l)}$，因此折中的方案为：


$$
\operatorname{Var}\left[w^{(l)}\right]=\frac{1}{\alpha^{2}\left(C_{i}^{(l)} n_{i}^{(l)}+C_{o}^{(l)} n_{o}^{(l)}\right)}, \quad \forall l
$$


