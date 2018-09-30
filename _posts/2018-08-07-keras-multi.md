---
layout: post
title: keras多显卡训练问题
subtitle:  解决keras读取多显卡训练模型bug
date: 2018-08-5 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-error2.jpg"
catalog: true
tags:
    - keras
    - 深度学习
---

keras作为tensorflow的高级封装，虽然好用了很多，但是也有个各种各样的问题，其中一大bug就是使用多显卡训练的模型，在单显卡环境下无法读取，报错

**ValueError: You are trying to load a weight file containing 1 layers into a model with 130 layers.**

出现此错误是因为使用multi_gpu_model函数进行多显卡数据并行时，模型结构发生了变化，逐层读取此模型发现，该模型结构为

- lambda
- lambda
- model
- dense

其中只有model层包含权重，且包含的权重数目与正常模型相同，因此猜测前两层Lambda层将数据分配至两张显卡，最后一层dense实际为concat，将并行的数据拼接至一起。所以只需把model层中的权重赋给标准模型，之后使用标准模型就可解决问题。

```python
def standardized_model(model,multi_model_path,result_model_path,gpus=None):
    #将模型转换为多显卡并行结构
    multi_model =  multi_gpu_model(model, gpus)
    #读取权重
    multi_model.load_weights(multi_model_path)
    #读取真正包含权重的层
    multi_layer = multi_model.layers[gpus+1]
    #读取该层所有权重
    multi_weights = multi_layer.get_weights()
    index=0
    #逐层遍历标准模型
    for layer in model.layers:
        #获取该层参数
        weights = layer.get_weights()
        #获取该层参数列表长度
        weights_len=len(weights)
        #将并行化模型对应此层的权重赋给标准模型
        layer.set_weights(multi_weights[index:index+weights_len])
        index+=len(weights)
    #保存模型
    model.save(result_model_path)
```


