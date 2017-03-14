hackyなやり方ですが、一応メモを残します。

現状TensorFlow v0.11では直接attention matrixをとることができません。そのため、TensorFlowのソースコードをいじりました。

Macでpyenv/pipでTensorFlowをインストールしましたので、このファイルになります。

```
~/.pyenv/versions/3.5.1/lib/python3.5/site-packages/tensorflow/python/ops/seq2seq.py
```

元の論文[Grammar as a Foreign Language](https://arxiv.org/abs/1412.7449)の2.1のところに書いたように、attention matrixは

![attention matrix equation](https://github.com/vanhuyz/neural-machine-translation-demo/blob/master/docs/attn_equation.gif?raw=true)

から得られます。TensorFlowのコードを見ると、この行です↓

https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/python/ops/seq2seq.py#L620

以下のように編集します。

```diff
def attention(query):
  ...
- return ds
+ return ds, a
```

```diff
def attention_decoder(...)
  ...
  attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
             for _ in xrange(num_heads)]
+ a_list = []
  for a in attns:  # Ensure the second shape of attention vectors is set.
    a.set_shape([None, attn_size])
  if initial_state_attention:
-   attns = attention(initial_state)
+   attns, my_a = attention(initial_state)
+   a_list.append(my_a)
  ...
  if i == 0 and initial_state_attention:
    with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                       reuse=True):
-   attns = attention(state)
+   attns, my_a = attention(state)
  else:
-   attns = attention(state)
+   attns, my_a = attention(state)
  ...
  outputs.append(output)
+ a_list.append(my_a)
-return outputs, state,
+return outputs, state, a_list
```

```diff
def embedding_attention_seq2seq(...)
  ...
  def decoder(feed_previous_bool):
-   outputs, state = embedding_attention_decoder(...)
+   outputs, state, a_list = embedding_attention_decoder(...)
    ...
-   return outputs + state_list
+   return outputs + state_list + a_list

  outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
  outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
+ attns_len = len(attention_states)
- state_list = outputs_and_state[outputs_len:]
+ state_list = outputs_and_state[outputs_len:-attns_len]
  state = state_list[0]
  ...
- return outputs_and_state[:outputs_len], state
+ return outputs_and_state[:outputs_len], state, outputs_and_state[-attns_len:]
```

これで`embedding_attention_seq2seq`の戻り値は`outputs, states, attention_list`となっています。

listですがmatrixに変換するのはもちろん可能です。

```py3
attention_matrix = np.array(attention_list)
```

`attention_matrix`を利用して図化することができます。
図化方法はこのnotebookに書いてあります。

https://github.com/vanhuyz/neural-machine-translation-demo/blob/master/nmt-test.ipynb
