---
layout: post
title: attention原理与实现
subtitle: 
date: 2018-08-8 16:09:37 +08:00
author:     "VernonSong"
header-img: "img/post-bg-attention1.jpg"
catalog: true
tags:
    - 深度学习
    - 论文翻译
---

## 概述
Encoder-Decoder模型中，encoder部分将不定长度的输入编码为固定长度的向量，decoder部分基于此向量解码为不定长度的输出。但由于无论输入长度多长，编码后的向量都为定长，当输入过长时模型准确率会降低。因此研究人员设计了attention机制，让网络在每一个时间步额外关注于一部分向量，以此提升网络性能。

## 原理
传统的RNN Encoder-Decoder将encoder部分生成的向量$d$作为RNN解码层的初始状态，前一个时间步预测的结果作为当前时间步的输入。因此解码层对于原输入的信息全部依赖于初始状态$d$，当输出很长时，解码层可能会产生长期依赖问题。由于$d$为定长，当输入很长时，解码层也难以在$d$中提取有用的信息。因此解决问题的关键在于让每个时间步都能从编码结果中获取有效的信息。为了达到此目的，我们在每个时间步根据当前状态对向量中的信息添加权重，将加权后的信息作为当前时间步的额外信息来预测结果。

$$
\begin{align*}
&u_i^{(t)}=v^Ttanh(W_1h_i+W_2d_t)
\newline &a_i^t=softmax(u_i^t)
\newline &d_t'=\sum_{i=1}^{T_A}a_i^th_i
\end{align*}
$$

$h_i$为编码层在输入时间步$i$的结果，$d_t$为解码层在输出时间步$t$的隐藏状态，根据当前时间步的隐藏状态，使用softmax对编码信息加权，得到新的信息$d_t'$。结合此信息与当前时间步输出信息$o_t$，计算新的输出$o_t'$

$$
\begin{align*}
& o_t=f(h_{t-1},y_{t-1})
\newline &o_t'=W_3(\{o_t,d_t'\})
\end{align*}
$$

$f$通常为LSTM单元或RNN单元。

通过添加注意力机制，能显著提升encoder-decoder模型在处理长输入时的准确率，其它情况下准确率也有一定的提升。同时通过在每个输出时间步去关注不同的输入时间步，达成了输入与输出的软对齐。


## 实现
### Tensorflow
Tensorflow中有很好的soft attention decoder实现**tf.contrib.legacy_seq2seq.attention_decoder**，为方便可视化，在此基础上添加了attention位置的输出。

```python
def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.
    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.
    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
      output_size: Size of the output vectors; if None, we use cell.output_size.
      num_heads: Number of attention heads that read from attention_states.
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
    Returns:
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
        attention_loc: A list of the same length as decoder_inputs of 2D Tensors of
          shape [batch_size x attention_length]，These represent the attention location of each step.
    Raises:
      ValueError: when num_heads is not positive, there are no inputs, shapes
        of attention_states are not set, or input size cannot be inferred
        from the input.
  """

  # 检查数据
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s" %
                     attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype
    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    # 获取attention向量长度
    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = array_ops.shape(attention_states)[1]
    attn_size = attention_states.get_shape()[2].value

    # 使用1 * 1卷积替代W1 * h_t
    # 为进行卷积运算，将attention_state转化为[batch_size,attn_length,1,attn_size]
    hidden = array_ops.reshape(attention_states,
                               [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attns_locs=[]
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = variable_scope.get_variable(
          "AttnW_%d" % a, [1, 1, attn_size, attention_vec_size],
          dtype=dtype)
      # 添加hidden_features [batch_size, attn_size, 1, attention_vec_size]
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      print('hidden_features', hidden_features[0])
      v.append(
          variable_scope.get_variable(
              "AttnV_%d" % a, [attention_vec_size], dtype=dtype))

    state = initial_state

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      ats = [] # Results of attention location will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(query_list, 1)

      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          # 全连接层
          y = Linear(query, attention_vec_size, True)(query)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          y = math_ops.cast(y, dtype)
          # v^T * tanh(hidden_features + y) 并对attention_vec_size求和，降维
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                  [2, 3])
          # 使用softmax计算该时间步对attention_states各位置权重
          a = nn_ops.softmax(math_ops.cast(s, dtype=dtypes.float32))
          a = math_ops.cast(a, dtype)
          # 对不同位置加权后得到新信息 [batch_size, attn_size]
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
          print('d',d)
          ds.append(array_ops.reshape(d, [-1, attn_size]))
          ats.append(array_ops.reshape(a, [-1, attn_length]))
      return ds, ats

    outputs = []
    prev = None
    batch_attn_size = array_ops.stack([batch_size, attn_size])
    attns = [
        array_ops.zeros(
            batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
    ]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns = attention(initial_state)
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
          print('inp', inp)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      # inputs为上一个时间步预测结果和attens
      inputs = [inp] + attns
      inputs = [math_ops.cast(e, dtype) for e in inputs]
      x = Linear(inputs, input_size, True)(inputs)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attns, attns_loc = attention(state)
      else:
        attns, attns_loc = attention(state)
      attns_locs.append(attns_loc)

      with variable_scope.variable_scope("AttnOutputProjection"):
        cell_output = math_ops.cast(cell_output, dtype)
        inputs = [cell_output] + attns
        output = Linear(inputs, output_size, True)(inputs)
      if loop_function is not None:
        prev = output
      outputs.append(output)

  return outputs, state, attns_locs
```

## 参考
> [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)
> <br>
> [Grammar as a Foreign Language](https://arxiv.org/pdf/1412.7449.pdf)

