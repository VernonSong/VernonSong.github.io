---
layout: post
title:  Python3下配置OpenCV
subtitle:  python与计算机视觉
date: 2017-08-6 09:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-py3OpenCV.jpg"
catalog: true
tags:
    - Python
    - OpenCV
---

Python好用的原因之一就在于它的很多库配置起来很方便，Anaconda里包含了会用到的绝大多数库，使用pip也可以简单的下载配置，网上也有整理好的各种库。但是自己在使用python3配置OpenCV时就遇到了个坑，因此记录一下。

因为自己使用了Anaconda，所以就直接下的OpenCV库，安装后发现执行import cv2会报错**dll load failed： 找不到指定的模块**，可是考虑自己所有的库都在，也装有VS，不可能是少VC的环境。根据网上的一些解决方案，推测是Anaconda里装的OpenCV的依赖库的版本与所下的OpenCV所要求的版本不一样导致。因此重新配置OpenCV：

首先卸载掉已经存在的OpenCV以及所依赖的numpy和scipy
<br>`pip uninstall numpy`
<br>`pip uninstall scipy`
<br>如果也装了OpenCV也一并卸载掉
<br>`pip uninstall opencv-python`

之后在[http://www.lfd.uci.edu/~gohlke/pythonlibs/](http://www.lfd.uci.edu/~gohlke/pythonlibs/)上下载所需版本，需注意32位还是64位以及Python版本。
在文件所在目录输入
<br>`pip install numpy‑1.13.3+mkl‑cp36‑cp36m‑win_amd64.whl`
<br>`pip install scipy‑1.0.0rc1‑cp36‑cp36m‑win_amd64.whl`
<br>`pip install opencv_python‑3.3.0‑cp36‑cp36m‑win_amd64.whl`

再次使用OpenCV就没有报错了。

如果使用Python2.7版本，只需在OpenCV官网下载exe安装包，安装后在OpenCV\build\python目录下找到所需的cv2.pyd文件，拷贝至Anaconda的Lib文件夹下即可。

