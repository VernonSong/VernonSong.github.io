---
layout: post
title: 【转】dilate（膨胀）与erode（腐蚀）
subtitle:  opencv与计算机视觉
date: 2017-03-07 09:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-dilate.jpg"
catalog: true
tags:
    - C++
    - OpenCV
---

# 前言
在matlab中，有一个函数叫rangefilt，当时同事给我讲的时候我不明所以，后来看了matlab官方的文档以及例子，感觉，这特么不是PS的查找边缘加反相吗，于是当时就用了OpenCV的查找边缘函数，结果尝试后发现都不完全对，后来在stackoverflow上找到了答案。于是学习了一下相关图形学知识。

# 正文

## 一、理论与概念讲解——从现象到本质

### 1.1 形态学概述
 
形态学（morphology）一词通常表示生物学的一个分支，该分支主要研究动植物的形态和结构。而我们图像处理中指的形态学，往往表示的是数学形态学。下面一起来了解数学形态学的概念。
数学形态学（Mathematical morphology） 是一门建立在格论和拓扑学基础之上的图像分析学科，是数学形态学图像处理的基本理论。其基本的运算包括：二值腐蚀和膨胀、二值开闭运算、骨架抽取、极限腐蚀、击中击不中变换、形态学梯度、Top-hat变换、颗粒分析、流域变换、灰值腐蚀和膨胀、灰值开闭运算、灰值形态学梯度等。
 
简单来讲，形态学操作就是基于形状的一系列图像处理操作。OpenCV为进行图像的形态学变换提供了快捷、方便的函数。最基本的形态学操作有二种，他们是：膨胀与腐蚀(Dilation与Erosion)。
膨胀与腐蚀能实现多种多样的功能，主要如下：
- 消除噪声
- 分割(isolate)出独立的图像元素，在图像中连接(join)相邻的元素。
- 寻找图像中的明显的极大值区域或极小值区域
- 求出图像的梯度

我们在这里给出下文会用到的，用于对比膨胀与腐蚀运算的“浅墨”字样毛笔字原图：
![](https://github.com/VernonSong/Storage/blob/master/image/qianmo.png?raw=true)
在进行腐蚀和膨胀的讲解之前，首先需要注意，**腐蚀和膨胀是对白色部分（高亮部分）而言的，不是黑色部分。膨胀就是图像中的高亮部分进行膨胀，“领域扩张”，效果图拥有比原图更大的高亮区域。腐蚀就是原图中的高亮部分被腐蚀，“领域被蚕食”，效果图拥有比原图更小的高亮区域。**

### 1.2 膨胀
其实，膨胀就是求局部最大值的操作。
按数学方面来说，膨胀或者腐蚀操作就是将图像（或图像的一部分区域，我们称之为A）与核（我们称之为B）进行卷积。
核可以是任何的形状和大小，它拥有一个单独定义出来的参考点，我们称其为锚点（anchorpoint）。多数情况下，核是一个小的中间带有参考点和实心正方形或者圆盘，其实，我们可以把核视为模板或者掩码。
 
而膨胀就是求局部最大值的操作，核B与图形卷积，即计算核B覆盖的区域的像素点的最大值，并把这个最大值赋值给参考点指定的像素。这样就会使图像中的高亮区域逐渐增长。如下图所示，这就是膨胀操作的初衷。
[](https://github.com/VernonSong/Storage/blob/master/image/dilate.png?raw=true)
膨胀的数学表达式：
![](https://github.com/VernonSong/Storage/blob/master/image/dilate_formula.png?raw=true)
膨胀效果图（毛笔字）：
![](https://github.com/VernonSong/Storage/blob/master/image/qianmo_dilate.png?raw=true)
照片腐蚀效果图：
![](https://github.com/VernonSong/Storage/blob/master/image/dilate_demo.png?raw=true)

### 1.3 腐蚀

再来看一下腐蚀，大家应该知道，膨胀和腐蚀是一对好基友，是相反的一对操作，所以腐蚀就是求局部最小值的操作。
我们一般都会把腐蚀和膨胀对应起来理解和学习。下文就可以看到，两者的函数原型也是基本上一样的。
 
原理图：
![](https://github.com/VernonSong/Storage/blob/master/image/erode.png?raw=true)
腐蚀的数学表达式：
![](https://github.com/VernonSong/Storage/blob/master/image/erode_formula.png?raw=true)
腐蚀效果图（毛笔字）：
![](https://github.com/VernonSong/Storage/blob/master/image/qianmo_erode.png?raw=true)
照片腐蚀效果图：
![](https://github.com/VernonSong/Storage/blob/master/image/erode_demo.png?raw=true)


## 二、深入——OpenCV源码分析溯源

直接上源码吧，在…\opencv\sources\modules\imgproc\src\ morph.cpp路径中 的第1353行开始就为erode（腐蚀）函数的源码，1361行为dilate（膨胀）函数的源码。

```cpp
//-----------------------------------【erode（）函数中文注释版源代码】----------------------------   
//    说明：以下代码为来自于计算机开源视觉库OpenCV的官方源代码   
//    OpenCV源代码版本：2.4.8   
//    源码路径：…\opencv\sources\modules\imgproc\src\ morph.cpp   
//    源文件中如下代码的起始行数：1353行   
//    中文注释by浅墨   
//--------------------------------------------------------------------------------------------------------    
void cv::erode( InputArray src, OutputArraydst, InputArray kernel,  
                Point anchor, int iterations,  
                int borderType, constScalar& borderValue )  
{  
//调用morphOp函数，并设定标识符为MORPH_ERODE  
   morphOp( MORPH_ERODE, src, dst, kernel, anchor, iterations, borderType,borderValue );  
}  
```

```cpp
//-----------------------------------【dilate（）函数中文注释版源代码】----------------------------   
//    说明：以下代码为来自于计算机开源视觉库OpenCV的官方源代码   
//    OpenCV源代码版本：2.4.8   
//    源码路径：…\opencv\sources\modules\imgproc\src\ morph.cpp   
//    源文件中如下代码的起始行数：1361行   
//    中文注释by浅墨   
//--------------------------------------------------------------------------------------------------------   
void cv::dilate( InputArray src,OutputArray dst, InputArray kernel,  
                 Point anchor, int iterations,  
                 int borderType, constScalar& borderValue )  
{  
//调用morphOp函数，并设定标识符为MORPH_DILATE  
   morphOp( MORPH_DILATE, src, dst, kernel, anchor, iterations, borderType,borderValue );  
}  
```

可以发现erode和dilate这两个函数内部就是调用了一下morphOp，只是他们调用morphOp时，第一个参数标识符不同，一个为MORPH_ERODE（腐蚀），一个为MORPH_DILATE（膨胀）。
morphOp函数的源码在…\opencv\sources\modules\imgproc\src\morph.cpp中的第1286行，有兴趣的朋友们可以研究研究，这里就不费时费力花篇幅展开分析了。


## 三、浅出——API函数快速上手

### 3.1  形态学膨胀——dilate函数
 
erode函数，使用像素邻域内的局部极大运算符来膨胀一张图片，从src输入，由dst输出。支持就地（in-place）操作。
函数原型：

```cpp
    void dilate(  
    InputArray src,  
    OutputArray dst,  
    InputArray kernel,  
    Point anchor=Point(-1,-1),  
    int iterations=1,  
    int borderType=BORDER_CONSTANT,  
    const Scalar& borderValue=morphologyDefaultBorderValue()   
);  
```
 
参数详解：

- 第一个参数，InputArray类型的src，输入图像，即源图像，填Mat类的对象即可。图像通道的数量可以是任意的，但图像深度应为CV_8U，CV_16U，CV_16S，CV_32F或 CV_64F其中之一。
- 第二个参数，OutputArray类型的dst，即目标图像，需要和源图片有一样的尺寸和类型。
- 第三个参数，InputArray类型的kernel，膨胀操作的核。若为NULL时，表示的是使用参考点位于中心3x3的核。
     我们一般使用函数 getStructuringElement配合这个参数的使用。getStructuringElement函数会返回指定形状和尺寸的结构元素（内核矩阵）。
     其中，getStructuringElement函数的第一个参数表示内核的形状，我们可以选择如下三种形状之一:
- 矩形: MORPH_RECT
- 交叉形: MORPH_CROSS
- 椭圆形: MORPH_ELLIPSE

而getStructuringElement函数的第二和第三个参数分别是内核的尺寸以及锚点的位置。
我们一般在调用erode以及dilate函数之前，先定义一个Mat类型的变量来获得getStructuringElement函数的返回值。对于锚点的位置，有默认值Point(-1,-1)，表示锚点位于中心。且需要注意，十字形的element形状唯一依赖于锚点的位置。而在其他情况下，锚点只是影响了形态学运算结果的偏移。
getStructuringElement函数相关的调用示例代码如下：

```cpp
int g_nStructElementSize = 3; //结构元素(内核矩阵)的尺寸  
   
//获取自定义核  
Mat element = getStructuringElement(MORPH_RECT,  
    Size(2*g_nStructElementSize+1,2*g_nStructElementSize+1),  
    Point( g_nStructElementSize, g_nStructElementSize ));  
```

调用这样之后，我们便可以在接下来调用erode或dilate函数时，第三个参数填保存了getStructuringElement返回值的Mat类型变量。对应于我们上面的示例，就是填element变量。

- 第四个参数，Point类型的anchor，锚的位置，其有默认值（-1，-1），表示锚位于中心。
- 第五个参数，int类型的iterations，迭代使用erode（）函数的次数，默认值为1。
- 第六个参数，int类型的borderType，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_DEFAULT。
- 第七个参数，const Scalar&类型的borderValue，当边界为常数时的边界值，有默认值morphologyDefaultBorderValue()，一般我们不用去管他。需要用到它时，可以看官方文档中的createMorphologyFilter()函数得到更详细的解释。

使用dilate函数，一般我们只需要填前面的三个参数，后面的四个参数都有默认值。而且往往结合getStructuringElement一起使用。
调用范例：

```cpp
  //载入原图   
        Mat image = imread("1.jpg");  
//获取自定义核  
        Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));  
        Mat out;  
        //进行膨胀操作  
        dilate(image, out, element);  
```

用上面核心代码架起来的完整程序代码：

```cpp
//-----------------------------------【头文件包含部分】---------------------------------------  
//     描述：包含程序所依赖的头文件  
//----------------------------------------------------------------------------------------------  
#include <opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>  
#include <iostream>  
   
//-----------------------------------【命名空间声明部分】---------------------------------------  
//     描述：包含程序所使用的命名空间  
//-----------------------------------------------------------------------------------------------   
using namespace std;  
using namespace cv;  
   
//-----------------------------------【main( )函数】--------------------------------------------  
//     描述：控制台应用程序的入口函数，我们的程序从这里开始  
//-----------------------------------------------------------------------------------------------  
int main(  )  
{  
   
       //载入原图   
       Mat image = imread("1.jpg");  
   
       //创建窗口   
       namedWindow("【原图】膨胀操作");  
       namedWindow("【效果图】膨胀操作");  
   
       //显示原图  
       imshow("【原图】膨胀操作", image);  
   
<span style="white-space:pre">  </span>//获取自定义核  
       Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));  
       Mat out;  
<span style="white-space:pre">  </span>//进行膨胀操作  
       dilate(image,out, element);  
   
       //显示效果图  
       imshow("【效果图】膨胀操作", out);  
   
       waitKey(0);  
   
       return 0;  
}  
```

 运行截图：
![](https://github.com/VernonSong/Storage/blob/master/image/dilate_demo.png?raw=true)

### 3.2 形态学腐蚀——erode函数

erode函数，使用像素邻域内的局部极小运算符来腐蚀一张图片，从src输入，由dst输出。支持就地（in-place）操作。
 
看一下函数原型：

```cpp
void erode(  
    InputArray src,  
    OutputArray dst,  
    InputArray kernel,  
    Point anchor=Point(-1,-1),  
    int iterations=1,  
    int borderType=BORDER_CONSTANT,  
    const Scalar& borderValue=morphologyDefaultBorderValue()  
 );  
```

参数详解：

- 第一个参数，InputArray类型的src，输入图像，即源图像，填Mat类的对象即可。图像通道的数量可以是任意的，但图像深度应为CV_8U，CV_16U，CV_16S，CV_32F或 CV_64F其中之一。
- 第二个参数，OutputArray类型的dst，即目标图像，需要和源图片有一样的尺寸和类型。
- 第三个参数，InputArray类型的kernel，腐蚀操作的内核。若为NULL时，表示的是使用参考点位于中心3x3的核。我们一般使用函数 getStructuringElement配合这个参数的使用。getStructuringElement函数会返回指定形状和尺寸的结构元素（内核矩阵）。（具体看上文中浅出部分dilate函数的第三个参数讲解部分）
- 第四个参数，Point类型的anchor，锚的位置，其有默认值（-1，-1），表示锚位于单位（element）的中心，我们一般不用管它。
- 第五个参数，int类型的iterations，迭代使用erode（）函数的次数，默认值为1。
- 第六个参数，int类型的borderType，用于推断图像外部像素的某种边界模式。注意它有默认值BORDER_DEFAULT。
- 第七个参数，const Scalar&类型的borderValue，当边界为常数时的边界值，有默认值morphologyDefaultBorderValue()，一般我们不用去管他。需要用到它时，可以看官方文档中的createMorphologyFilter()函数得到更详细的解释。

同样的，使用erode函数，一般我们只需要填前面的三个参数，后面的四个参数都有默认值。而且往往结合getStructuringElement一起使用。
调用范例：

```cpp
 //载入原图   
        Mat image = imread("1.jpg");  
//获取自定义核  
        Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));  
        Mat out;  
        //进行腐蚀操作  
        erode(image,out, element);  
```

用上面核心代码架起来的完整程序代码：

```cpp
//-----------------------------------【头文件包含部分】---------------------------------------  
//     描述：包含程序所依赖的头文件  
//----------------------------------------------------------------------------------------------  
#include <opencv2/core/core.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>  
#include <iostream>  
   
//-----------------------------------【命名空间声明部分】---------------------------------------  
//     描述：包含程序所使用的命名空间  
//-----------------------------------------------------------------------------------------------   
using namespace std;  
using namespace cv;  
   
//-----------------------------------【main( )函数】--------------------------------------------  
//     描述：控制台应用程序的入口函数，我们的程序从这里开始  
//-----------------------------------------------------------------------------------------------  
int main(  )  
{  
       //载入原图   
       Matimage = imread("1.jpg");  
   
        //创建窗口   
       namedWindow("【原图】腐蚀操作");  
       namedWindow("【效果图】腐蚀操作");  
   
       //显示原图  
       imshow("【原图】腐蚀操作", image);  
   
          
//获取自定义核  
       Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));  
       Mat out;  
   
//进行腐蚀操作  
       erode(image,out, element);  
   
       //显示效果图  
       imshow("【效果图】腐蚀操作", out);  
   
       waitKey(0);  
   
       return 0;  
}  
```

运行结果：
![](https://github.com/VernonSong/Storage/blob/master/image/erode_demo.png?raw=true)