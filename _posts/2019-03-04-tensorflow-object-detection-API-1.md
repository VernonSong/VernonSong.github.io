---
layout: post
title:  Tensorflow object detection API源码分析【1】
subtitle:   Tensorflow object detection API windows平台配置
date: 2019-03-01 20:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-tensorflow-api1.jpg"
catalog: true
mathjax: true
tags:
    - tensorflow
---

2016年Google开源了自己的目标检测系统，这套系统包含了常用的目标检测算法以及训练好的模型，不断维护和更新，并且代码优雅，适合作为目标检测和Tensorflow的学习项目。由于官方的教程没有windows环境的配置，因此先从windows平台的环境配置写起。

下载[protobuf-win](https://github.com/protocolbuffers/protobuf/releases)，解压后将其中的bin目录添加到环境变量。然后切换到model\research目录，新建shell脚本：

```shell
protoc object_detection/protos/anchor_generator.proto --python_out=.
protoc object_detection/protos/argmax_matcher.proto --python_out=.
protoc object_detection/protos/bipartite_matcher.proto --python_out=.
protoc object_detection/protos/box_coder.proto --python_out=.
protoc object_detection/protos/box_predictor.proto --python_out=.
protoc object_detection/protos/eval.proto --python_out=.
protoc object_detection/protos/faster_rcnn.proto --python_out=.
protoc object_detection/protos/faster_rcnn_box_coder.proto --python_out=.
protoc object_detection/protos/graph_rewriter.proto --python_out=.
protoc object_detection/protos/grid_anchor_generator.proto --python_out=.
protoc object_detection/protos/hyperparams.proto --python_out=.
protoc object_detection/protos/image_resizer.proto --python_out=.
protoc object_detection/protos/input_reader.proto --python_out=.
protoc object_detection/protos/keypoint_box_coder.proto --python_out=.
protoc object_detection/protos/losses.proto --python_out=.
protoc object_detection/protos/matcher.proto --python_out=.
protoc object_detection/protos/mean_stddev_box_coder.proto --python_out=.
protoc object_detection/protos/model.proto --python_out=.
protoc object_detection/protos/multiscale_anchor_generator.proto --python_out=.
protoc object_detection/protos/optimizer.proto --python_out=.
protoc object_detection/protos/pipeline.proto --python_out=.
protoc object_detection/protos/post_processing.proto --python_out=.
protoc object_detection/protos/mean_stddev_box_coder.proto --python_out=.
protoc object_detection/protos/preprocessor.proto --python_out=.
protoc object_detection/protos/region_similarity_calculator.proto --python_out=.
protoc object_detection/protos/square_box_coder.proto --python_out=.
protoc object_detection/protos/ssd.proto --python_out=.
protoc object_detection/protos/ssd_anchor_generator.proto --python_out=.
protoc object_detection/protos/string_int_label_map.proto --python_out=.
protoc object_detection/protos/train.proto --python_out=.
```

然后运行，需注意，由于windows平台以及protos版本原因，无法直接使用

```bash
protoc object_detection/protos/*.proto --python_out=.
```

会报错误object_detection/protos/*.proto: No such file or directory，因此只能用bash脚本一个一个编译。编译完成后，Object Detection API就已配置完毕，可以运行object_detection_tutorial.ipynb进行测试。

![](/img/in-post/post-tensorflow-objectdetection-api.png)

## 参考
> [Installing TensorFlow Object Detection API on Windows 10](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7b)

