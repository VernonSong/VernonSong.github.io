---
layout: post
title: Debug与Release
subtitle:  从项目再看C++
date: 2017-02-15 09:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-debug.png"
catalog: true
tags:
    - C++
---
对于Debug和Release，当时配置公司项目的编译环境的时候就想总结的来着，只是忙忘了。自己写程序做实验时，又遇到相关问题，因此来总结一下。

Debug和Release是VC预定义提供的两组编译选项的集合，Debug通常称为调试版本，Release通常称为发布版本。当然，我们也可以自定义自己的编译模式。

Debug与Release的主要区别如下：
<br>1.Debug版本包含调试信息，所以体积上要比Release版本大很多
<br>2.Debug版本才可以单步调试，打断点，并且代码加入大量检查，以帮助开发者迅速找到问题
<br>3.Release版本有优化，代码运行速度会显著提升（调用Debug版本C++的dll结果跑起来比同样C#程序慢多了。。）

在引用一些库的时候，编译时的一些选项对不上会导致编译失败，不仅仅是X64和win32平台要一样，有时编译模式也要一样，但是也因为这个纠结了很久。