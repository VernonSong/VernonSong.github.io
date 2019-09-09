---
layout: post
title:  Tensorflow 安装问题
subtitle: 
date: 2019-05-01 20:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-tensorflow-install.jpg"
catalog: true
tags:
    - tensorflow
typora-root-url: ../../blog
---

tensorflow在安装过程中，会出现由于依赖项的缺失而造成的载入失败问题
![img](/img/in-post/post-tensorflow-wrong/post-tensorflow-wrong1.png)

在网上查阅问题，得到的答案基本都是重装CUDA和安装VC++依赖库，但盲目的安装不是一个好的方法，既然是无法加载dll，确定是哪个dll出现问题对解决问题更有帮助。因此对于出错的程序，查看其依赖的dll

```bash
"D:\Visual Studio 2017\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64\dumpbin" /dependents D:\Anaconda\Lib\site-packages\tensorflow\python\_pywrap_tensorflow_internal.pyd
```

得到如下结果
![img](/img/in-post/post-tensorflow-wrong/post-tensorflow-wrong2.png)

查看环境中是否有这些dll，就可以定位缺失的dll，常见的问题为CUDA以及cuDNN的版本与Tensorflow需求的版本不一致导致，如此tensorflow需求的就是10.0版本的CUDA。而安装VC++依赖库是为了解决MSVCP140.dll的缺失。

正确的定位问题比没有头绪，单纯根据别人的经验去做尝试有用的多。