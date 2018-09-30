---
layout: post
title: 【转】形态学运算
subtitle:  opencv与计算机视觉
date: 2017-03-08 09:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-morphology.jpg"
catalog: true
tags:
    - C++
    - OpenCV
---

## 一、理论与概念讲解——从现象到本质
首先呢，要知道形态学的高级形态，往往都是建立在腐蚀和膨胀这两个基本操作之上的。对膨胀和腐蚀心中有数了，接下来的高级形态学操作，应该就不难理解。
另外，为了下面对比和演示以及理解的方便，浅墨自己制作了一张毛笔字图，这里先上原图：
![](https://github.com/VernonSong/Storage/blob/master/image/qianmo1.png?raw=true)
OK，我们开始讲解。

### 1.1 开运算（Opening Operation）
 
开运算（Opening Operation），其实就是先腐蚀后膨胀的过程。其数学表达式如下：

dst = open(src, element) = dilate(erode(src, element), element)

开运算可以用来消除小物体、在纤细点处分离物体、平滑较大物体的边界的同时并不明显改变其面积。效果图是这样的：
![](https://github.com/VernonSong/Storage/blob/master/image/qianmo_opening.png?raw=true)
实际效果图：
![](https://github.com/VernonSong/Storage/blob/master/image/opening_demo.png?raw=true)

### 1.2 闭运算(Closing Operation)

先膨胀后腐蚀的过程称为闭运算(Closing Operation)，其数学表达式如下：

dst = close(src, element) = erode(dilate(src, element), element)

闭运算能够排除小型黑洞(黑色区域)。效果图如下所示：
![](https://github.com/VernonSong/Storage/blob/master/image/closing_demo.png?raw=true)

### 1.3 形态学梯度（MorphologicalGradient）

形态学梯度（Morphological Gradient）为膨胀图与腐蚀图之差，数学表达式如下：

dst = dilate(src, element) - erode(src, element)

对二值图像进行这一操作可以将团块（blob）的边缘突出出来。我们可以用形态学梯度来保留物体的边缘轮廓，如下所示:
![](https://github.com/VernonSong/Storage/blob/master/image/qianmo_m.png?raw=true)
实际素材效果图：
![](https://github.com/VernonSong/Storage/blob/master/image/m_demo.png?raw=true)

### 1.4 顶帽（Top Hat）

顶帽运算（Top Hat）又常常被译为”礼帽“运算。为原图像与上文刚刚介绍的“开运算“的结果图之差，数学表达式如下：

dst = src - open(src, element)

因为开运算带来的结果是放大了裂缝或者局部低亮度的区域，因此，从原图中减去开运算后的图，得到的效果图突出了比原图轮廓周围的区域更明亮的区域，且这一操作和选择的核的大小相关。
顶帽运算往往用来分离比邻近点亮一些的斑块。当一幅图像具有大幅的背景的时候，而微小物品比较有规律的情况下，可以使用顶帽运算进行背景提取。
如下所示:
![](https://github.com/VernonSong/Storage/blob/master/image/qianmo_top.png?raw=true)
素材效果图:
![](https://github.com/VernonSong/Storage/blob/master/image/m_demo.png?raw=true)

### 1.5 黑帽（Black Hat）

黑帽（Black Hat）运算为”闭运算“的结果图与原图像之差。数学表达式为：

dst = close(src, element) - src

黑帽运算后的效果图突出了比原图轮廓周围的区域更暗的区域，且这一操作和选择的核的大小相关。
所以，黑帽运算用来分离比邻近点暗一些的斑块。非常完美的轮廓效果图：
![](https://github.com/VernonSong/Storage/blob/master/image/qianmo_black.png?raw=true)
实际素材效果图：
![](https://github.com/VernonSong/Storage/blob/master/image/m_demo.png?raw=true)

## 二、深入——OpenCV源码分析溯源

本文的主角是OpenCV中的morphologyEx函数，它利用基本的膨胀和腐蚀技术，来执行更加高级的形态学变换，如开闭运算，形态学梯度，“顶帽”、“黑帽”等等。这一节我们来一起看一下morphologyEx函数的源代码。

```cpp
//-----------------------------------【erode（）函数中文注释版源代码】----------------------------    
//   说明：以下代码为来自于计算机开源视觉库OpenCV的官方源代码    
//   OpenCV源代码版本：2.4.8    
//   源码路径：…\opencv\sources\modules\imgproc\src\morph.cpp    
//   源文件中如下代码的起始行数：1369行    
//   中文注释by浅墨    
//--------------------------------------------------------------------------------------------------------     
void cv::morphologyEx( InputArray _src,OutputArray _dst, int op,  
                       InputArray kernel, Pointanchor, int iterations,  
                       int borderType, constScalar& borderValue )  
{  
//拷贝Mat数据到临时变量  
   Mat src = _src.getMat(), temp;  
   _dst.create(src.size(), src.type());  
   Mat dst = _dst.getMat();  
   
//一个大switch，根据不同的标识符取不同的操作  
   switch( op )  
    {  
   case MORPH_ERODE:  
       erode( src, dst, kernel, anchor, iterations, borderType, borderValue );  
       break;  
   case MORPH_DILATE:  
       dilate( src, dst, kernel, anchor, iterations, borderType, borderValue );  
       break;  
   case MORPH_OPEN:  
       erode( src, dst, kernel, anchor, iterations, borderType, borderValue );  
       dilate( dst, dst, kernel, anchor, iterations, borderType, borderValue );  
       break;  
   case CV_MOP_CLOSE:  
       dilate( src, dst, kernel, anchor, iterations, borderType, borderValue );  
       erode( dst, dst, kernel, anchor, iterations, borderType, borderValue );  
       break;  
   case CV_MOP_GRADIENT:  
       erode( src, temp, kernel, anchor, iterations, borderType, borderValue );  
       dilate( src, dst, kernel, anchor, iterations, borderType, borderValue );  
       dst -= temp;  
       break;  
   case CV_MOP_TOPHAT:  
       if( src.data != dst.data )  
           temp = dst;  
       erode( src, temp, kernel, anchor, iterations, borderType, borderValue );  
        dilate( temp, temp, kernel, anchor,iterations, borderType, borderValue );  
       dst = src - temp;  
       break;  
   case CV_MOP_BLACKHAT:  
       if( src.data != dst.data )  
           temp = dst;  
       dilate( src, temp, kernel, anchor, iterations, borderType, borderValue);  
       erode( temp, temp, kernel, anchor, iterations, borderType, borderValue);  
       dst = temp - src;  
       break;  
   default:  
       CV_Error( CV_StsBadArg, "unknown morphological operation" );  
    }  
}  
```

看上面的源码可以发现，其实morphologyEx函数其实就是内部一个大switch而已。根据不同的标识符取不同的操作。比如开运算MORPH_OPEN，按我们上文中讲解的数学表达式，就是先腐蚀后膨胀，即依次调用erode和dilate函数，为非常简明干净的代码。

## 三、浅出——API函数快速上手

### 3.1 morphologyEx函数详解

上面我们已经讲到，morphologyEx函数利用基本的膨胀和腐蚀技术，来执行更加高级形态学变换，如开闭运算，形态学梯度，“顶帽”、“黑帽”等等。这一节我们来了解它的参数意义和使用方法。

```cpp
void morphologyEx(  
InputArray src,  
OutputArray dst,  
int op,  
InputArraykernel,  
Pointanchor=Point(-1,-1),  
intiterations=1,  
intborderType=BORDER_CONSTANT,  
constScalar& borderValue=morphologyDefaultBorderValue() );  
```

- 第一个参数，InputArray类型的src，输入图像，即源图像，填Mat类的对象即可。图像位深应该为以下五种之一：CV_8U, CV_16U,CV_16S, CV_32F 或CV_64F。
- 第二个参数，OutputArray类型的dst，即目标图像，函数的输出参数，需要和源图片有一样的尺寸和类型。
- 第三个参数，int类型的op，表示形态学运算的类型，可以是如下之一的标识符：
   <br> MORPH_OPEN – 开运算（Opening operation）
    <br>MORPH_CLOSE – 闭运算（Closing operation）
<br>MORPH_GRADIENT -形态学梯度（Morphological gradient）
<br>MORPH_TOPHAT - “顶帽”（“Top hat”）
<br>MORPH_BLACKHAT - “黑帽”（“Black hat“）
另有CV版本的标识符也可选择，如CV_MOP_CLOSE，CV_MOP_GRADIENT，CV_MOP_TOPHAT，CV_MOP_BLACKHAT，这应该是OpenCV1.0系列版本遗留下来的标识符，和上面的“MORPH_OPEN”一样的效果。
- 第四个参数，InputArray类型的kernel，形态学运算的内核。若为NULL时，表示的是使用参考点位于中心3x3的核。我们一般使用函数 getStructuringElement配合这个参数的使用。getStructuringElement函数会返回指定形状和尺寸的结构元素（内核矩阵）。关于getStructuringElement我们上篇文章中讲过了，这里为了大家参阅方便，再写一遍：

其中，getStructuringElement函数的第一个参数表示内核的形状，我们可以选择如下三种形状之一:
       <br>矩形: MORPH_RECT
       <br>交叉形: MORPH_CROSS
       <br>椭圆形: MORPH_ELLIPSE
而getStructuringElement函数的第二和第三个参数分别是内核的尺寸以及锚点的位置。
我们一般在调用erode以及dilate函数之前，先定义一个Mat类型的变量来获得getStructuringElement函数的返回值。对于锚点的位置，有默认值Point(-1,-1)，表示锚点位于中心。且需要注意，十字形的element形状唯一依赖于锚点的位置。而在其他情况下，锚点只是影响了形态学运算结果的偏移。
getStructuringElement函数相关的调用示例代码如下：

```cpp
int g_nStructElementSize = 3; //结构元素(内核矩阵)的尺寸  
   
//获取自定义核  
Mat element =getStructuringElement(MORPH_RECT,  
       Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),  
       Point(g_nStructElementSize, g_nStructElementSize )); 
```

调用这样之后，我们便可以在接下来调用erode、dilate或morphologyEx函数时，kernel参数填保存getStructuringElement返回值的Mat类型变量。对应于我们上面的示例，就是填element变量。
- 第五个参数，Point类型的anchor，锚的位置，其有默认值（-1，-1），表示锚位于中心。
- 第六个参数，int类型的iterations，迭代使用函数的次数，默认值为1。
- 第七个参数，int类型的borderType，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_ CONSTANT。
- 第八个参数，const Scalar&类型的borderValue，当边界为常数时的边界值，有默认值morphologyDefaultBorderValue()，一般我们不用去管他。需要用到它时，可以看官方文档中的createMorphologyFilter()函数得到更详细的解释。

其中的这些操作都可以进行就地（in-place）操作。且对于多通道图像，每一个通道都是单独进行操作。
 
