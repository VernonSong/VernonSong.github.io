---
layout: post
title: OpenCV实现matlab的conv2函数
subtitle:  opencv与计算机视觉
date: 2017-03-10 09:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-conv2.jpg"
catalog: true
tags:
    - C++
    - OpenCV
---
在按照matlab代码写软件的过程中，发现即使OpenCV中有大量的现成函数，但是有些函数计算并不与matlab一致，比如matlab中的conv2，它是一个二维卷积函数，并且具有三种形式，遗憾的是，OpenCV中的filter2D只有same这一种形式，因此，无法完全替代conv2。但机智的我在上网查询后，找到了一位大牛在的基于OpenCV的自制conv2函数

```cpp
enum ConvolutionType {   
/* Return the full convolution, including border */
  CONVOLUTION_FULL, 

/* Return only the part that corresponds to the original image */
  CONVOLUTION_SAME,

/* Return only the submatrix containing elements that were not influenced by the border       
*/
  CONVOLUTION_VALID
};

void conv2(const Mat &img, const Mat& kernel, ConvolutionType type, Mat& dest) {
  Mat source = img;
  if(CONVOLUTION_FULL == type) {
    source = Mat();
    const int additionalRows = kernel.rows-1, additionalCols = kernel.cols-1;
    copyMakeBorder(img, source, (additionalRows+1)/2, additionalRows/2,     
(additionalCols+1)/2, additionalCols/2, BORDER_CONSTANT, Scalar(0));
  }

  Point anchor(kernel.cols - kernel.cols/2 - 1, kernel.rows - kernel.rows/2 - 1);
  int borderMode = BORDER_CONSTANT;
  filter2D(source, dest, img.depth(), flip(kernel), anchor, 0, borderMode);

  if(CONVOLUTION_VALID == type) {
    dest = dest.colRange((kernel.cols-1)/2, dest.cols - kernel.cols/2)
           .rowRange((kernel.rows-1)/2, dest.rows - kernel.rows/2);
  }
}
```

结果正确，速度快，非常赞，同事还问我咋conv2都能写出来哈哈哈

程序摘自[国外大牛在stackoverflow的回答](http://blog.timmlinder.com/2011/07/opencv-equivalent-to-matlabs-conv2-function/)