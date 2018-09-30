---
layout: post
title:  叉乘求任意多边形面积
subtitle:   图形计算方面知识
date: 2016-09-26 09:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-carea.png"
catalog: true
tags:
    - 算法
---
## 前言
一直对图形方面的算法有所欠缺，做POJ相关题目也比较迷茫，所以来恶补一下。

## 正文

### 叉乘法简介
对于凸多边形，很容易计算，如下图，以多边形的某一点为顶点，将其划分成几个三角形，计算这些三角形的面积，然后加起来即可。已知三角形顶点坐标，其三角形积可以利用向量的叉乘来计算。
![](https://github.com/VernonSong/Storage/blob/master/image/201410240014304222.png?raw=true)
对于凹多边形，如果还是按照上述方法划分成三角形，如下图，多边形的面积 = S_ABC + S_ACD + S_ADE, 这个面积明显超过多边形的面积。
![](https://github.com/VernonSong/Storage/blob/master/image/201410240014308216.png?raw=true)
我们根据二维向量叉乘求三角形ABC面积时，利用的是
![](https://github.com/VernonSong/Storage/blob/master/image/201410240014311097.png?raw=true)
因为向量叉乘是有方向的，所以得到的面积有正有负，不论是凸多边形还是凹多边形都能得出正确结果。
<br>如果我们不以多边形的某一点为顶点来划分三角形而是以任意一点，如下图，这个方法也是成立的：S = S_OAB + S_OBC + S_OCD + S_ODE + S_OEA。计算的时候，当我们取O点为原点时，可以简化计算。
![](https://github.com/VernonSong/Storage/blob/master/image/201410240014319915.png?raw=true)
当O点为原点时，根据向量的叉积计算公式，各个三角形的面积计算如下：
 
S_OAB = 0.5*(A_x\*B_y - A_y*B_x)   【(A_x，A_y)为A点的坐标】
<br>S_OBC = 0.5*(B_x\*C_y - B_y*C_x)
<br>S_OCD = 0.5*(C_x\*D_y - C_y*D_x)
<br>S_ODE = 0.5*(D_x\*E_y - D_y*E_x)
<br>S_OEA = 0.5*(E_x\*A_y - E_y*A_x)

### 代码实现


```cpp
struct point2d
{
	double x;
	double y;
	point2d(double xx, double yy) : x(xx), y(yy) {}
};
double computeAera(const vector<point2d> &points)
{
	int point_num = points.size();
	if (point_num < 3)return 0.0;//只有两个顶点则面积为0
	double s = 0;
	for (int i = 0; i < point_num; ++i)//取余是因为最后一个点要跟第一个点组成三角形
		s += points[i].x * points[(i + 1) % point_num].y - points[i].y * points[(i + 1) % point_num].x;
	return fabs(s / 2.0);
}
```
该算法还可以优化一下，对上面的式子合并一下同类项
<br>S = S_OAB + S_OBC + S_OCD + S_ODE + S_OEA =
<br>0.5*(A_y*(E_x-B_x) + B_y*(A_x-C_x) + C_y*(B_x-D_x) + D_y*(C_x-E_x) + E_y*(D_x-A_x))
 <br>
<br>这样减少了乘法的次数，代码如下：

```cpp
double computeArea(const vector<point2d> &points)
{
	int point_num = points.size();
	if (point_num < 3)return 0.0;
	double s = points[0].y * (points[point_num - 1].x - points[1].x);
	for (int i = 1; i < point_num; ++i)
		s += points[i].y * (points[i - 1].x - points[(i + 1) % point_num].x);
	return fabs(s / 2.0);
}
```