---
layout: post
title: sunday算法详解
subtitle:   更简单的字符串匹配
date: 2016-09-27 09:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-poj1654.png"
catalog: true
tags:
    - 算法
---
之前做一个笔试题，有一道编程题要用到字符串匹配，我正打算写KMP算法，却突然感到有点卡手，情急之下，只好用了笨办法。回到寝室，打算复习下KMP算法，结果发现了一个更好的字符串匹配算法——sunday算法。

### sunday算法原理
对于字符串匹配来说，要想获得更高的效率，显然是要使匹配的次数尽可能少，如何能在尽可能少的匹配次数中，对两个字符串进行比较呢，核心思想就是尽可能对不可能的匹配进行跳过，sunday算法在匹配字符串时的思路是这样的：
<br>首先两个字符串匹配![](https://github.com/VernonSong/Storage/blob/master/image/%E7%BB%98%E5%9B%BE4.png?raw=true)
我们把上面的字符串称为主串，下面的称为子串。在匹配到第二个字符的时候发现不相等，此时子串应该向后移动，那么移动多少呢？sunday算法的思想是：既然发现不同了，就直接看看第6个字符再定。因为子串的长度是5且此次匹配失败，那么如果能成功匹配，那么最早也是在第2个字符到第6个字符与子串一一对应相等才可以。
<br>
<br>非常巧的是，主串第6个字符根本不曾在子串中出现过，那么，我们就可以知道匹配跟前6个字符都没有关系了，直接把子串挪到第7个字符的位置看。![](https://github.com/VernonSong/Storage/blob/master/image/%E7%BB%98%E5%9B%BE5.png?raw=true)
这下比较第一次就发现不对了，我们继续按照之前的思路，直接从主串中与子串最后一个字符相对的字符的后一个开始看，主串第12个字符是c，我们发现子串从后往前数第三个字符是c，那么我们将子串中的这个c与主串中我们看的那个c对齐，记住一定要从后往前找子串的字符，不然可能会跳过能匹配的区间。![](https://github.com/VernonSong/Storage/blob/master/image/%E7%BB%98%E5%9B%BE6.png?raw=true)
这次一一比较后，我们很高兴的发现匹配成功，从开始到结束，我们只尝试匹配了3次，一共比较了8次！非常的高效！

### sunday算法实现
要实现高效的sunday算法，还有一点需要注意，那就是建表，比如上述例子中第一次匹配不成功，我们看主串中h这个字符决定下次匹配从哪开始，这时真的要从头到尾遍历一遍子串吗？以后每次都要这么遍历吗？显然不是，我们可以通过构建hash表来快速确定下次匹配位置，达到看到h就知道子串要向后移动6个单位，看到c就知道子串要向后移动3个单位。

```cpp
int *setCharStep(char *b)
{
	int i;
	int *charStep = new int[256];//256个ASCII字符转换
	int n = strlen(b);
	for (i = 0; i <256;i++)
		charStep[i] = n + 1;
		//从左向右扫描一遍 保存子串中每个字符所需移动步长 
		for (i = 0; i<n;i++){
			charStep[(unsigned char)b[i]] = n - i;
		}
	return charStep;
}
```
知道了如何移动子串，后面就很容易了

```cpp
int sunday(const char *a, const char *b,const int charStep[])
{
	int m = strlen(a);//主串长度
	int n = strlen(b);//子串长度
	int i = 0, j = 0;
	int tem;
	char pos;
	while (i < m)
	{
		tem = i;//记录此次匹配开始位置
		while (j<n)
		{
			if (a[i] == b[j])
			{
				i++;
				j++;
			}
			else
			{
				if (tem + n > m)
					return -1;//主串后面长度不够，不可能成功匹配
				pos = a[tem + n];
				i =tem+ charStep[(unsigned char)pos];
				j = 0;
				break;
			}
		}
		if (j == n)//子串全部匹配完毕
			return i-n;//匹配成功，返回此次匹配开始位置
	}
	return -1;
}
```
以上就是sunday算法的代码实现，比KMP算法要好理解很多，也更好实现，效率也是KMP算法的3到5倍，再也不怕写KMP算法卡手了。