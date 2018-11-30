
# EasyKeras
更容易的Keras



[![License](https://img.shields.io/badge/license-Apache%202-4EB1BA.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-team/keras/blob/master/LICENSE)

本工具包是在Keras的基础上进一步的封装，对于NLP等领域添加更容易上手的操作。
特别的，我们主要针对以下内容进行了增加和改造。

1. 数据加载

2. 常用模型

3. 自定义层（注意力机制、语义相似度匹配机制、词嵌入等）

4. 模型的评估

——————————————
工具包结构如下
- easykeras
  - evaluation (性能评估)
  - load_data （数据加载）
  - model （常用模型）
  - self_layer （自定义层）
    - attention （注意力机制）
    - embedding（词嵌入）
    - matching（语义匹配）
    - gated（门控网络）
    - general(一般层)
  - tasks（常用任务）

