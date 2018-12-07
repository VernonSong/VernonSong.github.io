---
layout: post
title:  Local Monotonic Attention原理与实现
subtitle: 
date: 2018-11-22 00:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-lm_attention.jpg"
catalog: true
mathjax: true
tags:
    - 深度学习
    - 论文翻译
---

## 概述
基于Attention的encoder-decoder模型在序列任务中取得了很不错的成绩，但对于语音识别等任务，它们的输入与输出是单调对齐的，而Global Attention会从全部输入中寻找需要关注的信息，增加了计算量和学习难度，因此研究者提出了Local Monotonic Attention，减少Attention范围，并增加单调对齐的约束。

## 原理 
Local Monotonic Attention重点从两方面对Global Attention进行优化
-  **Local**：让每个decoder时间步只从一小部分encoder state计算Attention
- **Monotonic**：每次计算Local Attention的位置受到单调向右的约束。

因此，假设需要关注的信息分布为以$p_t$为中心的正态分布。在每个decoder时间步计算Attention时，通过$\Delta p_t$来决定关注信息的中心位置需从上一个中心位置$p_{t-1}$向前移动多少个时间步。

![](/img/in-post/post-ml-attention.png)

$\Delta p_t$的计算有两种方案：有限制和无限制，如果不限制移动距离，则计算方法为：

$$
\Delta p_t = \mathrm{exp} (V_p ^T tanh(W_ph_t^d))
$$

如果移动距离不能大于$C_{max}$，则通过sigmoid函数构成门控，控制移动距离带大小：

$$
\Delta p_t = C_{max} * sigmoid(V_p ^T tanh(W_ph_t^d))
$$

在计算高斯分布时，额外引入$\lambda$缩放比例

$$
\lambda_t = \mathrm{exp} (V_{\lambda} ^T tanh(W_{\lambda}h_t^d))
$$

$$
a_t^{\mathcal{N}}(s)=\lambda_t *\mathrm{exp}(-\frac{(s-p_t)^2}{2\sigma^2})
$$

超参数$\sigma$控制正态分布范围。

若单纯使用使用正态分布计算Attention，效果会比较差，因此再添加

$$
a_t^{\mathcal{S}}(s)=\mathrm{Score}(h_s^e,h_t^d)=V_s^T tanh(W_s[h_s^e,h_t^d])
$$

$$
\forall s \in[p_t-2\sigma,p_t+2\sigma]
$$

这样最终的Attention信息为：

$$
c_t = \sum_{s=(p_t-2\sigma)}^{(p_t+2\sigma)}(a_t^{\mathcal{N}}(s)*a_t^{\mathcal{S}}(s))*h_s^e
$$


## 实现
### Tensorflow
```python
def lm_attention_decoder(decoder_inputs,
                         initial_state,
                         attention_states,
                         cell,
                         output_size=None,
                         sigma=1,
                         projection_num=None,
                         loop_function=None,
                         dtype=None,
                         scope=None,
                         initial_state_attention=None):
    """
    Local Monotonic Attention decoder
    实现参考https://arxiv.org/pdf/1705.08091.pdf，基于tensorflow attention_decoder修改而来
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        sigma: sigma in normal distribution
        loop_function: If not None, this function will be applied to i-th output
            in order to generate i+1-th input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from a previously
            stored decoder state and attention states.
    Return:
        A tuple of the form (outputs, state, attention_loc), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
          shape [batch_size x output_size]. These represent the generated outputs.
          Output i is computed from input i (which is either the i-th element
          of decoder_inputs or loop_function(output {i-1}, i)) as follows.
          First, we run the cell on a combination of the input and previous
          attention masks:
            cell_output, new_state = cell(linear(input, prev_attn), prev_state).
          Then, we calculate new attention masks:
            new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
          and then we calculate the output:
            output = linear(cell_output, new_attn).
        state: The state of each decoder cell the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    """
    # 检查数据
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size
    with variable_scope.variable_scope(scope or "lm_attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype
        batch_size = array_ops.shape(decoder_inputs[0])[0]
        # local attention length
        attn_length = int(sigma*2)*2+1
        # attention size
        attn_size = attention_states.get_shape()[2].value
        # 计算正态分布
        s_opposite = []
        for i in range(-2 * sigma, 2 * sigma + 1):
            s_opposite.append(i)
        normal_distribution = math_ops.exp(-math_ops.square(tf.convert_to_tensor(s_opposite, dtype=tf.float32)) / (2 * sigma * sigma))
        # position
        p = 0
        h_s = array_ops.zeros((batch_size, attn_length), dtype=dtype)
        # 为防止attention越界，进行padding
        attention_states = tf.pad(attention_states, ([0, 0], [sigma * 2, sigma * 2], [0, 0]), "CONSTANT")
        # decoder hidden num
        if nest.is_sequence(initial_state):  # If the query is a tuple, flatten it.
            query_list = nest.flatten(initial_state)
            for q in query_list:  # Check that ndims == 2 if specified.
                ndims = q.get_shape().ndims
                if ndims:
                    assert ndims == 2
            query = array_ops.concat(query_list, 1)
        hidden_num = query.get_shape()[1].value
        if projection_num is None:
            projection_num = attn_size
        attention_vec_size = attn_size  # Size of query vectors for attention.
        state = initial_state

        # 参数
        # W_p, W_lambda, W_s
        w = []
        # V_p, V_lambda, V_s
        v = []
        for i in range(3):
            w.append(variable_scope.get_variable("AttnW_%i" % i, [hidden_num, projection_num], dtype=dtype))
        for i in range(2):
            v.append(variable_scope.get_variable("AttnV_%i" % i, [projection_num, 1], dtype=dtype))
        v.append(variable_scope.get_variable("AttnV_2", [projection_num], dtype=dtype))
        w_s = slim.model_variable('w_s', [1, 1, attn_size, projection_num])

        def lm_attention(query, p):
            """Put attention masks on hidden using hidden_features and query."""
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            # 计算delta p
            y_pd = math_ops.matmul(query, w[0])
            p_d = math_ops.exp(math_ops.matmul(y_pd, v[0]))
            # 计算lambda
            y_lamb = math_ops.matmul(query, w[1])
            lamb = math_ops.exp(math_ops.matmul(y_lamb, v[1]))
            a_n = lamb * normal_distribution
            # 因为前面补了sigma*2
            h_s = attention_states[:, p:p + attn_length, :]
            # 使用1 * 1卷积替代Ws * h_s
            # 为进行卷积运算，将attention_state转化为[batch_size,attn_length,1,attn_size]
            hidden = array_ops.reshape(h_s, [-1, attn_length, 1, attn_size])
            y_h = nn_ops.conv2d(hidden, w_s, [1, 1, 1, 1], "SAME")
            y_d = tf.matmul(query, w[2])
            y_d = array_ops.reshape(y_d, [-1, 1, 1, projection_num])
            y = math_ops.tanh(y_h + y_d)
            a_s = math_ops.reduce_sum(v[2] * math_ops.tanh(y_h + y_d), [2, 3])
            c = a_s * a_n
            c = array_ops.reshape(c, [-1, 1, attn_length])
            c = math_ops.matmul(c, h_s)
            c = array_ops.reshape(c, [-1,  attn_size])

            return c

        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = array_ops.zeros(batch_attn_size, dtype=dtype)
        if initial_state_attention:
            attns = lm_attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            # inputs为上一个时间步预测结果和attens
            inputs = [inp] + [attns]

            inputs = [math_ops.cast(e, dtype) for e in inputs]
            x = Linear(inputs, input_size, True)(inputs)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                        variable_scope.get_variable_scope(), reuse=True):
                    attns = lm_attention(state)
            else:
                attns = lm_attention(state, p)

            with variable_scope.variable_scope("AttnOutputProjection"):
                cell_output = math_ops.cast(cell_output, dtype)
                inputs = [cell_output] + [attns]
                output = Linear(inputs, output_size, True)(inputs)
            if loop_function is not None:
                prev = output
            outputs.append(output)
        return outputs, state
```
## 参考
> [Local Monotonic Attention Mechanism for End-to-End Speech and Language Processing](https://arxiv.org/pdf/1705.08091.pdf)