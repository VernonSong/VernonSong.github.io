---
layout: post
title:  从PAT题目看哈希表
subtitle:   "PAT-1041，PAT1050，PAT1078"
date: 2016-08-11 18:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-hash.jpg"
catalog: true
tags:
    - PAT
    - 数据结构
---

哈希表又称散列表是通过将关键码映射到表中的一个位置，以加快查找数据的一种数据结构，他所要做的两项基本工作是**计算位置**和**解决冲突**，所谓计算位置就是构造哈希函数确定关键码存储位置，而解决冲突是应用某种策略解决多个关键码位置相同的问题。直接上题目

### 1078.Hashing(PAT-A)
The task of this problem is simple: insert a sequence of distinct positive integers into a hash table, and output the positions of the input numbers. The hash function is defined to be "H(key) = key % TSize" where TSize is the maximum size of the hash table. Quadratic probing (with positive increments only) is used to solve the collisions.
<br>Note that the table size is better to be prime. If the maximum size given by the user is not prime, you must re-define the table size to be the smallest prime number which is larger than the size given by the user.
<br>**Input Specification:**
<br>Each input file contains one test case. For each case, the first line contains two positive numbers: MSize (<=104) and N (<=MSize) which are the user-defined table size and the number of input numbers, respectively. Then N distinct positive integers are given in the next line. All the numbers in a line are separated by a space.
<br>**Output Specification:**
<br>For each test case, print the corresponding positions (index starts from 0) of the input numbers in one line. All the numbers in a line are separated by a space, and there must be no extra space at the end of the line. In case it is impossible to insert the number, print "-" instead.
<br>**Sample Input:**
<br>4 4
<br>10 6 4 15
<br>**Sample Output:**
<br>0 1 4 -
<br>
<br>这道题的题目粗暴直接，就是考哈希表。
<br>这道题目对于计算位置和解决冲突的方法都已明确给出（妈个叽居然要求正向增长试探，坑出翔！），计算位置是常见的用关键码对表长取余，当使用取余来计算位置时，为了减少冲突，通常将表长设置为素数，题目也这么要求了，除了这种散列函数构造方法，比较常见的还有数字分析法（从关键码中取出变化较随机的位重新组合作为散列地址）、折叠法（将关键码分割成位数相同的几部分，再进行叠加）等。解决冲突是要求采用二次探测法，二次探测法是当发生冲突时，以**增量序列{1，-1，4，-4...q²，-q²}(q<TableSize/2)**循环试探下一个存储位置，但题目中要求只能正向增长试探，那么增量序列即为{1，4,9...q²}(q<TableSize)。除了要求的二次探测法，还有双散列探测法（再弄几个哈希函数），分离链接法（跟桶排序原理一样，每个位置有一个链表）等。
<br>了解了这些，题目就以会做大半，接下来题目要求输入表长，需要放入的数的个数及具体值，让你输出每个数在表中的位置。仔细想想，会发现这是阉割版的哈希表，因为题目只要求存，没有删除和查询这样的操作，那么，只需要使用哈希表的原理，并不需要真正使用完全的哈希表就能完成。

```cpp
#include<iostream>
#include<cmath>
using namespace std;
bool isPrime(int n)//判断是否素数
{
	if (n == 1)
		return false;
	if (n == 2 || n == 3)
		return true;
	for (int i = 2; i <= sqrt(n); i++)
	{
		if (n%i == 0)
			return false;
	}
	return true;
}
int main()
{
	int size, m;
	cin >> size >> m;
	int position,q,flag;
	while (!isPrime(size))
		size++;//扩大表直至表长为素数
	bool *table = new bool[size];
	//因为只需存，所以只需知道位置是否为空
	for (int i = 0; i < size; i++)
		table[i] = 0;
	int *num = new int[m];
	for (int i = 0; i < m; i++)
		cin >> num[i];
	for (int i = 0; i < m; i++)
	{
		flag = 0;
		if (i != 0)
			cout << " ";
		position = num[i]%size;//映射
		q = 0;//增量序列参数
		while (table[(position + q*q)%size])
		{//循环直至找到的位置是空的
			if (q == size)//未发现可用位置
			{
				flag = 1;//当前关键码无存放位置
				break;
			}
			q++;
		}
		if (flag==1)
			cout << "-";
		else
		{//将当前位置设为已使用，并输出
			table[(position + q*q) % size] = 1;
			cout << (position + q*q) % size;
		}
	}
	return 0;
}
```

### 1050.String Subtraction(PAT-A)
Given two strings S1 and S2, S = S1 - S2 is defined to be the remaining string after taking all the characters in S2 from S1. Your task is simply to calculate S1 - S2 for any given strings. However, it might not be that simple to do it fast.
<br>**Input Specification:**
<br>Each input file contains one test case. Each case consists of two lines which gives S1 and S2, respectively. The string lengths of both strings are no more than 104. It is guaranteed that all the characters are visible ASCII codes and white space, and a new line character signals the end of a string.
<br>**Output Specification:**
<br>For each test case, print S1 - S2 in one line.
<br>**Sample Input:**
<br>They are students.
<br>aeiou
<br>**Sample Output:**
<br>Thy r stdnts.
<br>
<br>虽然这道题也是哈希，但是更简单。。。以ASCII码建哈希表，不需要映射直接按ASCII码把S2放入哈希表，不需要考虑冲突，只需要知道哈希表每个位置是否为空，也是bool类型的哈希表就可以。。。

```cpp
#include<iostream>
#include<string>
using namespace std;
int main()
{
	string s1, s2;
	bool hash[127] = { 0 };
	getline(cin, s1);
	getline(cin, s2);
	int p;
	for each(auto c in s2)
	{
		p = int(c);
		hash[p] = 1;
	}
	for each(auto c in s1)
	{
		p = int(c);
		if (hash[p] == 0)
			cout << c;
	}
	return 0;
}
```
<br>非常想吐槽PAT居然不支持C++11，改了代码重新提交才过。

### 1041.Be Unique(PAT-A)
Being unique is so important to people on Mars that even their lottery is designed in a unique way. The rule of winning is simple: one bets on a number chosen from [1, 104]. The first one who bets on a unique number wins. For example, if there are 7 people betting on 5 31 5 88 67 88 17, then the second one who bets on 31 wins.
<br>**Input Specification:**
<br>Each input file contains one test case. Each case contains a line which begins with a positive integer N (<=105) and then followed by N bets. The numbers are separated by a space.
<br>**Output Specification:**
<br>For each test case, print the winning number in a line. If there is no winner, print "None" instead.
<br>**Sample Input 1:**
<br>7 5 31 5 88 67 88 17
<br>**Sample Output 1:**
<br>31
<br>**Sample Input 2:**
<br>5 888 666 666 888 888
<br>**Sample Output 2:**
<br>None
<br>
<br>同样可以用哈希表解，同样是简单版，可直接按照数值放入表中（表大小为10000），不需要解决碰撞，但需要知道哪些元素发生了碰撞。

```cpp
#include<iostream>
#include<cmath>
using namespace std;
int main()
{
	int n;
	cin >> n;
	short  *hash = new short[10000];
	for (int i = 0; i < 10000; i++)
		hash[i] = -1;
	int *m = new int[n];
	for (int i = 0; i < n; i++)
	{
		cin >> m[i];
		if (hash[m[i]] == -1)
			hash[m[i]] = 0;
		else//有重复元素
			hash[m[i]] = 1;
	}
	int flag = 0;
	for (int i = 0; i < n; i++)
	{
		if (hash[m[i]] == 0)
		{
			cout << m[i];
			flag = 1;
			break;
		}
	}
	if (flag == 0)
		cout << "None";
	return 0;
}
```
都做完后发现最开始那道题是最难的。。。以为最开始那道题题目那么直白是相对最简单的。。。有毒。。。
<br>总的来说，当理解桶排序以后，理解哈希表感觉会简单很多。哈希表的**查找效率期望是O(1)**，实际中与哈希函数是否均匀，处理冲突的方法以及装填因子α（装入节点数和表长的比值），一般**装填因子在0.5到0.85**时效果比较好。
<br>
<br>虽然这几道题很简单，但是哈希表其实很实用，比如下载完东西经常会有的MD5校验就是运用哈希算法对文件生成一个数字摘要，以此判断文件在传输过程中是否发生错误，是否被篡改。还有在授权协议，数字签名等这些安全方面的领域哈希算法都有较为广泛的应用。