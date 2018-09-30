---
layout: post
title: keras多线程识别
subtitle:  
date: 2018-09-27 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-multikeras.jpg"
catalog: true
tags:
    - keras
    - 深度学习
---

以keras作为框架，tensorflow作为后端做多线程识别时，需要将tensorflow的计算图机制与keras对它的封装都考虑进去

```python
class MultiModel(object):
    def __init__(self,
                 modelPath1,
                 modelPath2):
        # 定义模型
        input = Input(shape=(64, None, 4), name='the_input')
        y_pred1 = SEnet363.SEnet(input, nclass)
        y_pred2 = SEnet484.SEnet(input, nclass)
        self.model1 = Model(inputs=input, outputs=y_pred1)
        self.model2 = Model(inputs=input, outputs=y_pred2)
        # 加载模型
        self.modelPath1 = os.path.join(os.path.dirname(os.getcwd()), modelPath1)
        self.modelPath2 = os.path.join(os.path.dirname(os.getcwd()), modelPath2)
        if os.path.exists(modelPath1):
            self.model1.load_weights(self.modelPath1)
        else:
            print('error path of mode l')
        if os.path.exists(modelPath2):
            self.model2.load_weights(self.modelPath2)
        else:
            print('error path of model 2')

        # 让模型预识别一次
        y_pred = self.model1.predict_on_batch(np.array(np.ones((1, 64, 192, 4), dtype=np.float)))
        y_pred = self.model2.predict_on_batch(np.array(np.ones((1, 64, 192, 4), dtype=np.float)))

        # 获取会话和图，并将计算图锁定
        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        self.graph.finalize()

    # 识别函数
    def predict1(self, X):
        with self.session.as_default():
            with self.graph.as_default():
                y_pred = self.model1.predict_on_batch(X)
            return y_pred

    def predict2(self, X):
        with self.session.as_default():
            with self.graph.as_default():
                y_pred = self.model2.predict_on_batch(X)
            return y_pred
```

graph.finalize()函数会将计算图设为只读模式，防止多线程访问时出现
*tensorflow.python.framework.errors_impl.NotFoundError: PruneForTargets: Some target nodes not found: group_deps *
等异常。同时因为初次识别会修改计算图，所以在graph.finalize()之前先识别一次来进行预热，否则会抛出*Graph is finalized and cannot be modified*异常。

进行多线程识别时，可采用线程池机制，预先创建线程，减少识别时创建与销毁线程的开销，之后每次识别时为线程池中线程分配任务。

```python
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(max_workers=2)  # 创建一个最大可容纳2个task的线程池
multi_model = MultiModel()
```

当同时识别的图片数过多时会爆显存，所以因根据线程数设定batch size

```python
def multi_predict(*args):
    # """
    # 处理图片并将划分为若干batch
    # """
    for i in range(batch_step):
        result1 = pool.submit(multi_model.predict1, X)
        result2 = pool.submit(multi_model.predict2, X)
        # 等待线程识别结束
        while (1):
            if result1.done() and result2.done():
                break
            else:
                time.sleep(0.01)
        y_pred1 = result1.result()
        y_pred2 = result2.result()
    # """
    # 获取结果并分析处理
    # """
    return result
```





