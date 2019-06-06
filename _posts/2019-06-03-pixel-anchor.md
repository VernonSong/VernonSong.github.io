---
layout: post
title: pixel-anchor原理与实现
subtitle: 文本检测
date: 2019-06-01 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-pixel-anchor.jpg"
catalog: true
mathjax: true
tags:
    - 计算机视觉
    - 深度学习
---

## 概述
在基于深度学习的OCR领域，目前的文本检测框架主要有两类，一类是基于像素级的图像语义分割（pixel -based model），一种是基于anchor的通用目标检测（anchor -based  model）。前者往往有着更好的精度，但在小尺度文本检测上召回率偏低。后者虽然对文本尺度不敏感，但是精度不如基于语义分割的文本检测，同时会存在Anchor Matching Dilemma，对临近的大旋转角度的文本行，难以区分具体哪个文本行该与当前anchor匹配。并且，两种框架在面对长的中文文本行时都表现乏力。因此，研究者将两者结合，设计出了pixel-anchor网络。

![](\img\in-post\post-text-detection\pixel-anchor-4.png)

## 原理
### 整体结构
pixel-anchor由pixel -based model和anchor -based  model两部分组成，这两部分共用一个特征提取网络，分别进行预测，以避免单一使用anchor base model的Anchor Matching Dilemma问题以及单一pixel -based model的小尺度文本recall较低的问题。同时pixel -based model的heat map传递给anchor -based  model的浅层feature map，使其获得拥有更大reception field的信息。

论文中采用resnet50作为特征提取网络，Res block4不再进行subsample，分别从Res block1，Res block2，Res block4获取1/4、1/8和1/16这三个尺度的feature map。

![](\img\in-post\post-text-detection\pixel-anchor-1.png)

### pixel -based model
#### 结构
常见的pixel -based model都是基于FPN，pixel-anchor将FPN与ASPP（atrous spacial pyramid pooling）结合起来，通过对1/16的feature map进行ASPP计算，增加其receptive field。与DeepLabv3+ 中的ASPP不同，论文中dilation rate设置为{3, 6, 9, 12, 15, 18} ，使feature map包含不同receptive field的信息，来应对各种大小的文本行。之后则是截至到1/4尺度的upsample。

![](\img\in-post\post-text-detection\pixel-anchor-2.png)

pixel -based model的输出包含两部分：RBOX和attention heat map。RBOX有score map，AABB，和rotation angle共6个channel的输出，attention heat map是提供给anchor -based model的score map。score map与attention heat map的不同在于，attention heat map中所有文本区域均为positive，而score map中的positive区域与EAST一样是缩小后的。


#### loss
在计算classification loss时，使用OHEM（ online hard example mining），对512个hard negative像素和512个随机negative像素，以及所有positive像素计算损失：

$$
L_{p_cls}=\frac{1}{|\Omega_{RBox}|}\sum_{i \in \Omega_{RBox}} H(P_i,p^*)+\frac{1}{|\Omega_{heatmap}|}\sum_{i \in \Omega_{heatmap}} H(P_i,p^*)
$$

$\|\cdot\|$表示集合中所有positive像素。$H$为cross entropy loss。

与classification loss相似，classification loss同样采用OHEM，对128个hard positive像素和128个随机positive像素计算loss：

$$
L_{p_loc}=\frac{1}{|N_{pos}|}\sum_{i \in \Omega_{loc}} IoU(R_i,R^*)+\lambda_{\theta} (1-\mathrm{cos}(\theta_i,\theta_i^*)
$$

$N_{pos}$为positive像素数，与EAST一样，$\lambda_{\theta}$设置为10。

最后将两个loss相加得到总pixel -based model的总loss：

$$
L_{p_dt}=L_{p_cls}+\alpha _p L\_{p_loc}
$$

论文中$\alpha$为1.

### anchor bases model
#### 结构
pixel-anchor的anchor bases model基于SSD，从特征提取网络中提取1/4，1/16两个feature map，然后通过全卷积操作获得1/32，1/64，1/64，1/64共4个feature map，其中最后两个feature map保持尺寸不变，使用atrous convolution（dilation rate=2）来增加reception field。

![](\img\in-post\post-text-detection\pixel-anchor-3.png)

之所以浅层的feature map不选择1/8而选择1/4尺度，是为了更好地检测小文本。同时，为了增大此feature map的recetion field，这个feature map还要与attention heat map结合，减少检测小文本时的false positive。在结合前，attention heat map要进行指数运算，使其数值范围在（1，e），保留背景位置信息的同时突出文本位置信息。

在进行预测时，为了更好的检测不同比例的文本，pixel-anchor提出Adaptive Predictor Layer（APL），用不同的卷积核去预测不同比例的anchor，以提供与其适合的reception field。

- 方形anchor：长宽比1:1，卷积核为3*3
- 普通水平anchor：长宽比 {1:2, 1:3, 1:5, 1:7}，卷积核为3*5
- 普通垂直anchor：长宽比 {2:1, 3:1, 5:1, 7:1}，卷积核为5*3
- 超长水平anchor：长宽比{1:15, 1:25, 1:35}，卷积核为1*n
- 超长垂直anchor：长宽比{15:1, 25:1, 35:1}，卷积核为n*1

其中1/4的feature map的分支不做超长文本的检测，因为它主要用来检测小文本。之后的feature map都进行超长文本检测，n分别为{33, 29, 15, 15, 15}。

pixel-anchor沿用并扩展了Textboxes++中的设计，对每个位置的anchor进行偏移，得到多个anchor，增加检测密度。在pixel-anchor中，方形anchor在水平和垂直方向偏移，水平anchor在垂直方向偏倚，垂直anchor在水平方向偏移。对于普通anchor，在六个分支上分别进行 {1, 2, 3, 4, 3, 2}次偏移，对于超长anchor，在5个分支上分别进行 {4, 4, 6, 4, 3}次偏移。

anchor -based model中每个anchor有9个channel的输出，1个用于预测score，另外8个预测文本框的4个顶点与anchor的4个顶点之间的偏移量。

#### loss
anchor -based model中与ground trues的minimum bounding rectangles (MBRs)与anchors最大IoU大于0.5的作为positives，小于0.5的作为negatives。训练时每个batch的negatives与positives的比例为3:1，classification loss为：

$$
L_{a_cls}=\frac{1}{|\Omega_{a}|}\sum_{i \in \Omega_{a}} H(P_i,p^*)
$$

regression loss使用smooth L1 loss：

$$
L_{a_loc}=\frac{1}{|\Omega_{a}|}\sum_{i \in pos(Omega_{a})} SL(l_i,l^*)
$$

anchor-based model总loss为：

$$
L_{a_dt}=L_{a_cls}+\alpha_aL_{a\_loc}
$$

文中$\alpha_a$推荐值为0.2。

### 整体loss

$$
L_{all}=(\alpha_{all}L_{p\_dt}+L_{a\_dt})
$$

文中$\alpha_{all}$推荐值为3。

### fusion NMS
pixel-anchor针对pixel-based model与anchor-based model的特点，在推理阶段分别选取它们擅长的部分作为结果。对于pixel-based model，只保留预测的文本框的MBR小于10像素以及MBR长宽比在[1:15, 15:1]之间的结果。对于anchor-based model，保留1/4 feature map分支上的所有anchor，但其它分支只保留超长anhor，同时将所有score大于阈值的文本框，score再加1，使anchor-based model拥有更高的优先级。

由于1/4的feature map的anchor较小，很少出现Anchor Matching Dilemma。而使用MBR计算IoU，因此本身大旋转角度的长文本框在anchor-based model中就不是positive，也极大的避免了Anchor Matching Dilemma现象。

对剩下的文本框，采用Textboxes++中的优化NMS算法，先用文本框的MBR计算IoU，进行高阈值（如0.5）的NMS，粗略过滤掉部分文本框，再对剩下的文本框计算准确的IoU，进行低阈值的（如0.2）的NMS。

## 缺点
1. 对大倾斜角度的长文本行，依然没有好的解决方案
2. 网络设计复杂

## 参考
[Pixel-Anchor: A Fast Oriented Scene Text Detector with Combined Networks](https://arxiv.org/abs/1811.07432)









