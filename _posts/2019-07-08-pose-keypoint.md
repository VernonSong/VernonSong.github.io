---
layout: post
title: 人体关键点检测小结
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
---

# Convolutional Pose Machines

Convolutional Pose Machines是2016年MPII竞赛中名列前茅的网络结构，其主要贡献是将CNN引入到Pose Machines中，并使用intermediate supervision来避免vanishing gradient。



### Pose Machines

Pose Machines由多个stage构成，每个stage包含一个人工设计的feature extracter $f_t$和一个multi-class predictor $g_t$。在第一个stage，$g_1$以$f_1$的结果作为输入，预测每个keypoint的heatmap。在之后的每个stage，$g_t$以$g_{t-1}$和$f_t$作为输入，根据前一个stage的heatmap和当前stage所提取的feature来进行更为精细的预测。



### Convolutional Pose Machines

CPM（Convolutional Pose Machines）使用CNN来代替Pose Machines每个stage中人工设计的feature extracter。之所以沿用此结构，是因为在keypoint detection这类的任务中，存在easy keypoint（如人体关键点中的头）和hard keypoint（如人体关键点中的肘部）。而多stage结构中，前一个stage所预测的easy keypoint的heatmap可以为下一个stage预测hard keypoint提供强有力的空间信息。



![](/img/in-post/post-pose-detection/post-pose-detection2.png)



在引入CNN这个过程中，还要解决两个问题，receptive field和vanishing gradient。为了捕捉两个距离较远的keypoint之间的关系，网络需要一个较大的receptive field。因此，CPM在extract feature时，先进行8-stride的subsampling，然后堆叠具有较大kernel size（$11 \times 11$）的conv layer。这样在第二个stage时，就有足够大的receptive field，同时尽量减少了使用较大kernel size的conv layer的计算消耗。而对于vanishing gradient问题，作者使用intermediate supervision的方式，对每个stage所输出的heatmap都计算loss，这样即使是第一个stage中浅层的layer，也能获取到正常的gradient flow。



![](/img/in-post/post-pose-detection/post-pose-detection1.png)



论文中，每个stage的target heatmap都是ground truth location的Gaussian peak，采用L2 distance来计算每个heatmap中每个位置的loss。当stage $t \ge 2$时将共享feature map $x'$和$g_t$的weight，$t >5$时网络性能几乎无提升。



# Stacked Hourglass Networks

与CPM同年的Stacked Hourglass同样使用了多stage的结构，不同的是，Stacked Hourglass每一个stage都采用FCN这样的bottom-up，top-down的结构，这样网络的总体结构就够就像是连续堆叠的沙漏，因此命名为Stacked Hourglass。



![](/img/in-post/post-pose-detection/post-pose-detection3.png)



在bottom-up阶段，FCN通过不断地subsampling带来更大的receptive field，获取距离较远的keypoint之间的信息。在top-down阶段，FCN通过upsampling还原heatmap尺寸，并使用skip connections带来的局部信息来提高heatmap精细度。



![](/img/in-post/post-pose-detection/post-pose-detection4.png)



在细节设计上，Stacked Hourglass使用了主流的residual结构，最终预测4-stride的heatmap。与CPM使用了同样的Intermediate Supervision方式。在预测阶段，作者通过同时预测原图与它的水平翻转图来增加准确度。同时把最终的keypoint位置向heatmap中值第二大的点的方向偏移四分之一个像素。



# Pyramid Residual Modules

Pyramid Residual Modules (PRMs)是对Stacked Hourglass的改进，通过加入feature pyramids 来获取多尺度特征，解决人体关键点检测中由于人物姿态不同导致的人体结构在图片中大小差异。同时讨论了multi-branch network中的variance叠加问题，并给出了自己的解决方案。



![](/img/in-post/post-pose-detection/post-pose-detection5.png)



## PRM设计

Pyramid Residual Modules的整体结构设计可以用如下公式表达


$$
\mathcal{P}\left(\mathbf{x}^{(l)} ; \mathbf{W}^{(l)}\right)=g\left(\sum_{c=1}^{C} f_{c}\left(\mathbf{x}^{(l)} ; \mathbf{w}_{f_{c}}^{(l)}\right) ; \mathbf{w}_{g}^{(l)}\right)+f_{0}\left(\mathbf{x}^{(l)} ; \mathbf{w}_{f_{0}}^{(l)}\right)
$$


其中$f_c$为获取$c$个level的feature pyramid的branch。作者对这一想法一共设计了四种实现方式。



![](/img/in-post/post-pose-detection/post-pose-detection6.png)



其中所有的$f_0$都为bottleneck结构，通过$1\times 1$的的convolution来将输入的feature dimension从256减少到128，经过$3 \times 3$的convolution计算后，再通过$1 \times 1$还原回256，以此来减少计算量。而对于$f_c$，作者认为由于small resolution的feature map中信息较少，因此，$f_c$不需要还原feature dimension，并且feature dimension可以随scale的减少而减少。PRM-A中$f_c$在$f_0$的基础上添加了subsampling和upsampling，通过addition来合并$c$个sub-branch的信息。PRM-B与PRM-A的不同是PRM-B选择公用dimensionality-reduction部分，减少参数和计算量。PRM-C只是改变了PRM-B中sub-branch的合并方式，使用concatenation而非addition。而PRM-D则没有使用bottom-up，top-down的方式，仅通过dilated convolution来简单的获取multi-scale features。



除了上述的设计，考虑到传统max-pooling reduce速度过快，作者选用fractional max-pooling进行更为smoothing的subsampling。并通过共享$f_c$的weights的方式进一步减少parameters。最终parameters和GFLOPs仅增加10%。



## Multi-Branch Networks问题

在Multi-Branch Networks中，variance的累加现象非常明显，因此作者考虑从两方面解决此问题。首先讨论了Multi-Branch结构中的weights  Initialization问题，目前常见的Xavier等初始化方法未考虑input和output数据来自多个branch，从而导致每个layer的variance随branch数目增加而累加。作者认为合适的weights  Initialization方法为：


$$
\operatorname{Var}\left[w^{(l)}\right]=\frac{1}{\alpha^{2}\left(C_{i}^{(l)} n_{i}^{(l)}+C_{o}^{(l)} n_{o}^{(l)}\right)}, \quad \forall l
$$

论文中提出residual结构中的identity mapping也是导致variances增加的一大元凶。


$$
\begin{aligned} \operatorname{Var}\left[\mathbf{x}^{(l+1)}\right] &=\operatorname{Var}\left[\mathbf{x}^{(l)}\right]+\operatorname{Var}\left[\mathcal{F}\left(\mathbf{x}^{(l+1)} ; \mathbf{W}^{(l)}\right)\right] \\ &>\operatorname{Var}\left[\mathbf{x}^{(l)}\right] \end{aligned}
$$

前人提出的解决方法是当feature map的resolution减少或者feature channel的dimension增加时，使用[$1 \times 1$ convolution，batch normalization，ReLu]结构来代替identity mapping，使variance重新回到一个较小的值。作者在stacked hourglass中进行实验，选择在两个residual unit相加时，替换其中一个branch的identity mapping。结果显示改进后的stacked hourglass的variance始终维持在一个较小的范围。



![](/img/in-post/post-pose-detection/post-pose-detection7.png)



# High-Resolution Networks

之前的大部分网络，都要先进行subsampling，得到具有全局信息的low resolution feature map，然后进行upsampling还原高分辨率的feature map。High-Resolution Networks（HRnets）采用了全新的并行结构，让高分辨率的 feature map从始至终贯穿整个网络。



网络结构图为


$$
\begin{aligned} 
\mathcal{N}_{11} \rightarrow \mathcal{N}_{21}  \rightarrow \mathcal{N}_{31} \rightarrow &\mathcal{N}_{41} \\ 
\searrow \mathcal{N}_{22}  \rightarrow \mathcal{N}_{32} \rightarrow &\mathcal{N}_{42} \\
 \searrow \mathcal{N}_{33}  \rightarrow &\mathcal{N}_{43} \\
 \searrow &\mathcal{N}_{44} \\
\end{aligned}
$$


这样设计的思想是，网络不仅需要high level的global feature，还需要hight level的local feature，每个stage中，global feature与local feature在接近的level进行fuse，补充各自所欠缺的信息。而对于关键点预测来说，最终的heatmap保持着与网络输入相同的高分辨率，并且精确度也要比通过upsampling得到的heatmap更高。



HRnet中不同分辨率的feature map进行信息交换时，从高分辨率到低分辨率的转换通过kernel size为$3\times 3$，strides为2的convolution，从低分辨率到高分辨率通过$1 \times 1$的convolutionn接nearest neighbor up-sampling进行转换。fuse时均使用addition。



HRnet中，每个block依照resnet来设计，feature map size减半后feature channels加倍，第1个satge中包含4个bottleneck residual unit，第2，3，4个stage分别包含1，4，3个exchange block，每个exchange block包含并行的residual block（4个residual unit）和一个exchange unit。论文中，HRnet-W32中4个branch的feature channels分别为32，64，128，256。HRnet-W48则为48，96，192，384。



论文中还用实验证明了low level的低分辨率feature map中的信息价值较小，而如果没有低分辨率feature map中的global feature，结果将会很差。





