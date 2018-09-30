---
layout: post
title: python正则表达式
subtitle:  让代码变得优雅简洁
date: 2018-01-25 08:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-regex.jpg"
catalog: true
tags:
    - python
---

写字符串相关的程序，或者爬取网页信息时，经常需要检索字符串，找出与一定逻辑相匹配的字符串，纯按照匹配逻辑去遍历比对，当逻辑很复杂时，代码繁琐且不优雅。因此在编码时，通常使用正则表达式来简化这一过程。

## 正则表达式基本概念

### 元字符
正则表达式语言由两种基本字符类型组成：原义（正常）文本字符和元字符。元字符使正则表达式具有处理能力。所谓元字符就是指那些在正则表达式中具有特殊意义的专用字符，可以用来规定其前导字符（即位于元字符前面的字符）在目标对象中的出现模式。

很容易理解的元字符有

字符     |     描述
:-------------------------:|:-------------------------:
**\\** |   转移字符
**^** |匹配输入字符串的开始位置。多行模式也匹配“\n”或“\r”之后的位置。
**$**|匹配输入字符串的结束位置。多行模式也匹配“\n”或“\r”之前的位置。
**\***  | 匹配前面的子表达式零次或多次。例如，zo\*能匹配“z”、“zo”以及“zoo”。\*等价于{0,}。
**+**  | 匹配前面的子表达式一次或多次。例如，“zo+”能匹配“zo”以及“zoo”，但不能匹配“z”。+等价于{1,}。
**?**  | 匹配前面的子表达式零次或一次。例如，“do(es)?”可以匹配“do”或“does”中的“do”。?等价于{0,1}。
**{n}**  | n是一个非负整数。匹配确定的n次。例如，“o{2}”不能匹配“Bob”中的“o”，但是能匹配“food”中的两个o。
**{n,}**  | n是一个非负整数。至少匹配n次。例如，“o{2,}”不能匹配“Bob”中的“o”，但能匹配“foooood”中的所有o。“o{1,}”等价于“o+”。“o{0,}”则等价于“o*”。
**{n,m}**  | m和n均为非负整数，其中n<=m。最少匹配n次且最多匹配m次。例如，“o{1,3}”将匹配“fooooood”中的前三个o。“o{0,1}”等价于“o?”。请注意在逗号和两个数之间不能有空格。
**?**  | 非贪心量化（Non-greedy quantifiers）：当该字符紧跟在任何一个其他重复修饰符（*,+,?，{n}，{n,}，{n,m}）后面时，匹配模式是非贪婪的。非贪婪模式尽可能少的匹配所搜索的字符串，而默认的贪婪模式则尽可能多的匹配所搜索的字符串。例如，对于字符串“oooo”，“o+?”将匹配单个“o”，而“o+”将匹配所有“o”。
**.**  | 匹配除“\r”“\n”之外的任何单个字符。要匹配包括“\r”“\n”在内的任何字符，请使用像“(.\|\r\|\n)”的模式。
**\b**  | 匹配一个单词边界，也就是指单词和空格间的位置。例如，“er\b”可以匹配“never”中的“er”，但不能匹配“verb”中的“er”。
**\B**  |匹配非单词边界。“er\B”能匹配“verb”中的“er”，但不能匹配“never”中的“er”。
**\d**  | 匹配一个数字字符。等价于[0-9]。注意Unicode正则表达式会匹配全角数字字符。
**\D**  | 匹配一个非数字字符。等价于[^0-9]。
**\f**  | 匹配一个换页符。等价于\x0c和\cL。
**\n**  | 匹配一个换行符。等价于\x0a和\cJ。
**\r**  | 匹配一个回车符。等价于\x0d和\cM。
**\s**  |匹配任何空白字符，包括空格、制表符、换页符等等。等价于[ \f\n\r\t\v]。注意Unicode正则表达式会匹配全角空格符。
**\S**  |匹配任何非空白字符。等价于[^ \f\n\r\t\v]。
**\w**  |匹配包括下划线的任何单词字符。等价于“[A-Za-z0-9_]”。注意Unicode正则表达式会匹配中文字符。
**\W**  |匹配任何非单词字符。等价于“[^A-Za-z0-9_]”。
**\t**  |匹配一个制表符。等价于\x09和\cI。
**\t**  |匹配一个制表符。等价于\x09和\cI。

——*摘自WIKI百科*

### 字符集
正则表达式中，由中括号所包含的部分称为字符集，其作用是匹配字符集中的任意一个字符，例如\[xyz\]匹配字母x或y或z，在字符集中。

可以使用-代表一个区间，例如\[0-9\]表示数字0到9中的任意字符，常用的还有**\[\u4e00-\u9fa5\]**，它表示汉字区间。

在字符集中，**^**将代表取补集的意思，比如[^\u4e00-\u9fa5]将代表所有字符中除汉字以外的字符。
<br>

## 正则表达式使用
<br>

### re模块
python中内嵌的正则表达式模块为re，由C编写，其常用的函数有

**re.search(pattern, string, flags=0)**

此函数扫描整个字符串并返回第一个成功的匹配。其中pattern是正则表达式字符串，string为待匹配字符串，而flag为匹配方式，匹配方式有：

标志     |     含义
:-------------------------:|:-------------------------:
**re.S(DOTALL)** |   使.匹配包括换行在内的所有字符
**re.I（IGNORECASE）** |使匹配对大小写不敏感
**re.L（LOCALE）**|做本地化识别（locale-aware)匹配，法语等
**re.M(MULTILINE)**  | 多行匹配，影响^和$
**re.X(VERBOSE)**  | 该标志通过给予更灵活的格式以便将正则表达式写得更易于理解
**re.U**  | 根据Unicode字符集解析字符，这个标志影响\w,\W,\b,\B

```python
import re
str='1:一 2:二 3:三 4:四 5:五 6:六'
match = re.search('[\u4e00-\u9fa5]',str,re.S)
print(match.group())
```
**输出：**一


**re.match(pattern, string, flags=0)**

re.match 尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()依然返回none。

```python
import re
str='1:一 2:二 3:三 4:四 5:五 6:六'
match = re.match('[\u4e00-\u9fa5]',str,re.S)
print(match)
```
**输出：**NONE


**re.compile(pattern,flags=0)**

编译正则表达式模式，其中pattern是正则表达式字符串，而flag为匹配方式，匹配方式有：

```python
import re
str='1:一 2:二 3:三 4:四 5:五 6:六'
rr = re.compile('[3-5]:[\u4e00-\u9fa5]\s')
match = rr.search(str)
print(match.group())
```
**输出：**3:三


### 分组与后项引用
在匹配字符串时，经常希望取一段字符串中的子串，在正则表达式中，可以用分组来解决这一问题。我们把用小括号括起来的部分称为一组。

在python中，可以用group方法捕获所需的分组，

```python
import re
str='<P><strong>正则表达式：</strong>其实很简单</P>'
match = re.search("<strong>(?P<name>.*?)：</strong>(.*?)</P>", str, re.S)
print(match.group('name'))
print(match.group(2))
print(match.groups())
```

**输出：**正则表达式：
  <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其实很简单
 <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;('正则表达式：', '其实很简单')

同时，分组也可以用来对某一段正则匹配做次数限定，例如：

```python
import re
str='1:一 2:二 3:三 4:四 5:五 6:六'
match = re.search("([3-5]:[\u4e00-\u9fa5]\s){1,}", str, re.S)
print (match.group())
print (match.group(1))
```
**输出：**3:三 4:四 5:五
 <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5:五

需要注意的是虽然我们设定了匹配3次，但是最终结果只有一组。

### 检索和替换

**re.sub(pattern, repl, string, count=0, flags=0)**
将匹配的部分替换为新字符串，pattern为正则中的模式字符串，repl为替换的字符串，string为要查找的原始字符串，count表示模式匹配后替换的最大次数，默认0为所有匹配

```python
import re
def double(matched):
    value = int(matched.group('value'))
    return str(value * 2)
s = "1:一 2:二 3:三 4:四 5:五 6:六"
result= re.sub("d+", "", s)
print(result)
result= re.sub("(?P<value>\d)", double, s)
print(result)
```

**输出：**一 2:二 3:三 4:四 5:五 6:六
 <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2:一 4:二 6:三 8:四 10:五 12:六


### 零宽断言
零宽断言用于限定匹配的字符串前面或者后面是某些特定的内容。零宽断言有4种：
- **正向肯定(?=exp)** 匹配此位置之前的内容，此位置满足正则exp
- **正向否定(?!exp)** 匹配此位置之前的内容，此位置不满足正则exp
- **反向肯定(?<=exp)** 匹配此位置之后的内容，此位置满足正则exp
- **反向否定(?<!exp)** 匹配此位置之后的内容，此位置不满足正则exp

```python
import re
s = "one:一 2:二 three:三 4:四 five:五 6:六"
match = re.search('(?<=\d:)\w',s,re.S)
print(match.group())
```
**输出：**二



