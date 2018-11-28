---
layout: post
title:  CTC-Attention多任务学习原理与实现
subtitle: 
date: 2018-11-22 00:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-ctc_attention.jpg"
catalog: true
mathjax: true
tags:
    - 深度学习
    - 论文翻译
---

## 概述
在语音识别和手写字符的端到端识别中，通常使用Attention或CTC来省去传统的分割与对齐操作。虽然Attention在大部分场景下效果都优于CTC，但Attention过于灵活的对齐方式也导致了它在这类任务中存在着一些缺陷。因此，作者将CTC与Attention结合起来，减少对齐时的问题，提升模型准确率。

## 原理
### CTC与Attention缺陷
CTC在预测时假设句子中每一个字符出现的概率为条件独立事件，在遇到难以判别的字符时CTC会出现误识别的情况，因此有时需要额外的语言模型来辅助预测。而Attention由数据驱动，基于输入与之前的字符来预测下一个字符，会隐含语言模型。因此通常情况下Attention的表现要强于CTC。但是，当Attention在数据有噪声时表现较差，同时如果输入序列较长，在训练前期Attention会难以进行学习。这是因为Attention在对齐时是从所有输入中找与之对应的，缺乏CTC中单调对齐的约束。虽然Attention也可以通过添加窗口机制来减小注意力范围，但窗口的参数需要根据训练数据进行调整，并非是个通用的解决方案。

###  CTC-Attention多任务学习
为了能给予Attention恰当的约束，作者将CTC与Attention结合起来，以Attention Encoder-Decoder作为多任务学习框架（MTL）的核心任务，用CTC目标函数作辅助任务。

![](/img/in-post/post-ctc_attention/post-ctc_attention1.png)

CTC与Attention decoder共享encoder部分，由于CTC的前向后向算法（forward-backward algorithm）能够强制进行输入与输出的单调对齐。因此在输入数据噪声过多时也能获得稳定的对齐效果。同时，由于CTC的前向后向算法的对齐过程不是数据驱动的，而Attention需要学习如何对齐，因此CTC能帮助网络快速学习到如何粗略对齐，从而加速收敛。

由于新的网络有两个任务，因此loss变为：

$$
\mathcal{L}_{MTL}=\lambda \mathcal{L}_{CTC}+(1- \lambda ) \mathcal{L}_{Attention}
$$

$\lambda$为CTC任务所占权重。

### 实验结果

![](/img/in-post/post-ctc_attention/post-ctc_attention2.png)

可以发现，CTC loss的权重越大，网络学习速度越快，当CTC权重较小时，网络拥有最佳准确率，超过单纯的Attention模型。作者在实验时发现即使是没有噪声的数据，结合CTC的Attention模型依然有着超越单纯Attention模型的准确率，猜测是因为CTC不依赖于字符间内在联系，不拘束于数据集语言环境，但这还需要其他实验来证明。

## 实现

```python
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np


def sparse_tuple_from_label(self, sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    idx = tf.where(tf.not_equal(sequences, -1))
    self.sparse = tf.SparseTensor(idx, tf.gather_nd(sequences, idx), sequences.get_shape())


def ctc_attention_loss(attention_chars_logits, ctc_chars_logits, attention_labels, ctc_labels, lambda_value):
    # CTC loss
    batch_size = ctc_chars_logits.get_shape()[0].value
    net_length = ctc_chars_logits.get_shape()[1].value
    ctc_labels = (tf.cast(ctc_labels, tf.int32))
    sequence_length = np.ones(batch_size) * net_length
    ctc_loss = tf.nn.ctc_loss(labels=sparse_tuple_from_label(ctc_labels),
                      inputs=ctc_chars_logits,
                      sequence_length=sequence_length,
                      time_major=False)
    ctc_loss = tf.reduce_mean(ctc_loss)

    # attention loss
    attention_labels_list = tf.unstack(attention_labels, axis=1)
    batch_size, seq_length, _ = attention_chars_logits.shape.as_list()
    weights = tf.ones((batch_size, seq_length), dtype=tf.float32)
    attention_logits_list = tf.unstack(attention_chars_logits, axis=1)
    weights_list = tf.unstack(weights, axis=1)
    attention_loss = tf.contrib.legacy_seq2seq.sequence_loss(
        attention_logits_list,
        attention_labels_list,
        weights_list,
        softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=attention_chars_logits, labels=attention_labels),
        average_across_timesteps=False)
    loss = ctc_loss*lambda_value + attention_loss(1-lambda_value)
    tf.losses.add_loss(loss)
    total_loss = slim.losses.get_total_loss()
    return total_loss
```

## 参考
[Joint CTC-Attention based End-to-End Speech Recognition using Multi-task Learning](https://arxiv.org/pdf/1609.06773.pdf)
