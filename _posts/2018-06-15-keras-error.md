---
layout: post
title: keras跨线程调用
subtitle:  
date: 2018-04-27 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-error1.jpg"
catalog: true
tags:
    - keras
    - 深度学习
---

编写深度学习验证码识别接口，直接调用识别无问题，但使用flask作为服务器，便出现了
ValueError: Tensor Tensor("xxxxxx'", shape=(?, ?,37), dtype=float32) is not an element of this graph.的错误

github上有回答

I had this problem when doing inference in a different thread than where I loaded my model. Here's how I fixed the problem:

Right after loading or constructing your model, save the TensorFlow graph:

```python
graph = tf.get_default_graph()
```

In the other thread \(or perhaps in an asynchronous event handler\), do:

```python
global graph
with graph.as_default():
    (... do inference here ...)
```

使用keras时，在一个线程中加载模型，此时tensorflow指定一个系统默认图，但在另一线程调用模型时，操作是在一个新的图上，导致原本正确的操作报错。因此显式声明默认图，并在新线程中将global作用于全局，且以此图为系统默认图进行操作。