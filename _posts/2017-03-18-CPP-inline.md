---
layout: post
title: 内联函数与宏定义
subtitle:  C++知识梳理
date: 2017-03-15 09:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-inline.jpg"
catalog: true
tags:
    - C++
---
## 内联函数（inline）
我们知道，一次函数调用包含一系列工作：调用前要先保存寄存器，并在返回时恢复；可能需要拷贝实参；程序转向一个新的位置继续执行。若想避免这些开销，我们可以使用一个特殊的函数——“内联函数”。

内联函数的工作原理看起来与宏类似，即在编译时讲内联函数展开，比如：

```cpp
inline int max(int a, int b)
{
	return (a > b ? a : b);
}
int main()
{
	int a = 1, b = 2;
	cout << max(a, b);
}
```
在编译时展开类似如下：

```cpp
cout << (a > b ? a : b);
```
因此，对于规模较小，流程直接，而又频繁调用的函数，使用内联函数可以使程序得到很好的优化。

## 内联函数与宏的区别
光知道上面的内容，一定会觉得，那内联函数不就是把把函数宏定义一下吗。虽然表面上内联函数与宏定义非常像，用宏定义也很容易理解内联函数，但他们还是由很大的区别：

- 宏定义是是简单粗暴的替换，而内联函数会像正常函数一样提供类型安全检查等操作以减少出错的可能
- 宏定义无法操作类的私有成员
- 内联函数在运行时可调试，而宏定义不可以
- inline关键字是给编译器看的，如果函数过长，编译器会拒绝内联，而宏定义则一定会强制替换

## 使用内联函数需要注意的地方
内联函数虽然看起来很美好，但它使用起来也需要有所注意：

- 关键字inline必须与函数定义体放在一起才能使函数成为内联，仅将inline放在函数声明前不起任何作用
- 内联函数应该在头文件中定义。可以确保在调用函数时所使用的定义是相同的，并保证在调用点该函数的定义对编译器是可见的
- 内联函数是以代码膨胀为代价的，如果函数代码略长，我们使用内联造成了较大的内存消耗，却紧换来很小的效率提高，那么便得不偿失了
- 递归函数无法内联，另外如果函数体内出现循环，那么执行函数体内代码的时间要比函数调用的开销大。
- 构造函数和析构函数看似符合内联函数的使用要求，但是它们有可能隐式地调用基类的构造函数和析构函数，而在内联函数中再调用函数并不是一件好事。