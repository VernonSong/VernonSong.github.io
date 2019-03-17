---
layout: post
title:  Tensorflow object detection API源码分析【3】
subtitle:   util部分——metrics
date: 2019-03-01 20:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-tensorflow-api3.jpg"
catalog: true
tags:
    - tensorflow
---

## utils.learning_schedules
学习率自调整相关的工具函数

### exponential_decay_with_burnin
先常数学习率，一定步数后指数衰减
```python
def exponential_decay_with_burnin(global_step,
                                  learning_rate_base,
                                  learning_rate_decay_steps,
                                  learning_rate_decay_factor,
                                  burnin_learning_rate=0.0,
                                  burnin_steps=0,
                                  min_learning_rate=0.0,
                                  staircase=True):
  """Exponential decay schedule with burn-in period.

  In this schedule, learning rate is fixed at burnin_learning_rate
  for a fixed period, before transitioning to a regular exponential
  decay schedule.

  Args:
    global_step: int tensor representing global step.
    learning_rate_base: base learning rate.
    learning_rate_decay_steps: steps to take between decaying the learning rate.
      Note that this includes the number of burn-in steps.
    learning_rate_decay_factor: multiplicative factor by which to decay
      learning rate.
    burnin_learning_rate: initial learning rate during burn-in period.  If
      0.0 (which is the default), then the burn-in learning rate is simply
      set to learning_rate_base.
    burnin_steps: number of steps to use burnin learning rate.
    min_learning_rate: the minimum learning rate.
    staircase: whether use staircase decay.

  Returns:
    a (scalar) float tensor representing learning rate
  """
  if burnin_learning_rate == 0:
    burnin_learning_rate = learning_rate_base
  # 指数衰减学习率部分
  post_burnin_learning_rate = tf.train.exponential_decay(
      learning_rate_base,
      global_step - burnin_steps,
      learning_rate_decay_steps,
      learning_rate_decay_factor,
      staircase=staircase)
  # 添加常数学习率，并保证学习率大于min_learning_rate
  return tf.maximum(tf.where(
      tf.less(tf.cast(global_step, tf.int32), tf.constant(burnin_steps)),
      tf.constant(burnin_learning_rate),
      post_burnin_learning_rate), min_learning_rate, name='learning_rate')
```



