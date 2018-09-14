# 论文阅读笔记

## sequence to sequence learning with Neural Networks   2018.09.06


this paper present a general end-to-end approach, which uses a multilayered Long Short Term Memory to map the input  sequence to a vector of a fixed dimensionality, and another deep LSTM to decode the target sequence from the vector.

**DNNs can only be applied to problems whose inputs and targets can be sensibly encoded with vectors of fixed dimensionality.**

**theory:**

a sequence of inputs $(x_1, x_2, x_3, ..., x_T)$

a sequence of outputs $(y_1, y_2, y_3,..., y_T)$

$$h_t = sigmoid(W^{hx} x_t + W^{hh} h_{t - 1})$$

$$y_t = W^{yh} h_t$$

The goal of the LSTM is to estimate the conditional probability $$p(y_1, y_2, y_3, ......,y_{T_1} | x_1, x_2, x_3, ....., x_T)$$.
First, to obtaining the fixed-dimensional representation $v$ of the input sequence given by the last hidden state of the LSTM, and then computing the probability $p(y_1, y_2, ..., y_{T_1} | x_1, x_2, ..., x_T) = \prod_{t =1}^{T_1}{p(y_t | v, y_1, y_2, ..y_{t-1})}$, $T_1$ may differ from $T$.

**Tips:**

> to uses two different LSTMs, one for the input sequence, and another for the output sequence;
>
> deep LSTMs significantly outperformed shallow LSTMs;
>
> found that it extremely valuable to reverse the  order of words of the input sentence;

**最基础的seq2seq模型包含三个部分，即Encoder、Decoder、中间状态变量，Encoder通过学习输入，将其编码为一个固定大小的状态向量S，然后将S输出到Decoder进行解码。**

![Alt text](../md_data/seq2seq.jpg "basic seq2seq")

**不足：**

> Encoder将输入编码为固定大小的状态向量，其是一个信息有损压缩的过程，会出现信息丢失的问题；
>
> 随着sequence length的增加，时间维度上的序列很长，RNN模型也会出现梯度弥散问题；
>
> Encoder与Decoder的链接仅仅依靠一个固定大小的状态向量，使得Decoder无法关注输入信息更多的细节；

## effective approaches to attention-based neural machine translation 2018.09.08

**此篇论文提出了两个简单、有效的Attention Mechanism:**
> a global approach which always attends to all source words	
> a local one that only looks at a subset of soure words at a time

### neural machine translation
Neural machine translation directly models the conditional probability $p(y|x)$ of translation a source sentence to a target sentence.				
*source sentence:* $$x_1, x_2, x_3, ..., x_n$$
*target sentence:* $$y_1, y_2, y_3, ..., y_m$$
基本的Neural machine translation由两部分组成：			
> an encoder which computes a represention s for each source sentence				
> a decoder which generates one target word at a time

Hence decomposes the conditional probability as: 
$$\log{p(y|x)} = \sum_{j = 1}^{m}{\log{p(y_j | y_{\lt j}, s)}}$$
Parameterize the probability of decoding each word $y_j$ as:
$$p(y_j | y_{\lt j}, s) = softmax(g(h_j))$$
with g being the transformation function that outputs a vocabulary-sized vector and $h_j$ is the RNN hidden unit, computed as :$$h_j = f(h_{j-1}, s)$$
$$\begin{equation} f = 
\left\{
             \begin{array}{lr}
              vanilla \quad RNN \quad unit \\
              GRU \quad unit \\
              LSTM \quad unit & 
             \end{array}
\right.
\end{equation}$$
Training objective is formulated as follows:
$$J_t = \sum_{(x,y)\in D}{-\log{p(y|x)}}$$
with D being parallel training corpus.

### attention-based models
an attentional hidden state as follows: $$\overline{h_t} = tanh(W_c * [c_t; h_t])$$
the predictive distribution formulated as: 
$$p(y_t | y_{\lt t}, S) = Softmax(W_s * \overline{h_t})$$
### Global Attention
the idea of a global attentional model is to consider all the hidden states of the encoder when deriving the context vector $c_t$.
a variable-length alignment vector $a_t$:
$$a_t(s) = align(h_t, \overline{h_t}) = Softmax(score(h_t, \overline{h_t}))$$
score is referred as a content-based function for which we consider 3 different alternatives:
$$\begin{equation} score(h_t, \overline{h_t}) = 
\left\{
             \begin{array}{lr}
              h_{t}^T * \overline{h_t} \\
              h_{t}^T * W_a * \overline{h_t} \\
              v_a^T * tanh(W_a * [h_{t}; \overline{h_t}]) & 
             \end{array}
\right.
\end{equation}$$
### Local Attention
the local attention mechanism selectively focus on a small window fo context and is differentiable. 			
an aligned position $p_t$ for each target word at time $t$, the context vector$c_t$ is than derived as a weigthed average over the set of source hidden states within the window $[p_t - D, p_t + D]$, D is empirically selected.			
*Monotonic alignment(local-m)*: $$p_t = t$$
assuming source and target sequences are roughly monotonically aligned.			
*Predictive alignment(local-p)*: $$p_t = S * sigmoid(v_{p}^{T} * tanh(W_p * h_t))$$
$W_p$ and $v_p$ are parameters, $S$ is the source sentence length.		
the alignment weights are defined as :
 $$a_t(s) = align(h_t, \overline{h_t}) * exp(- \frac{(s - p_t)^2}{2*\sigma^2})$$
 the standard deviation is empirically set as $\sigma = \frac{D}{2}$			

## Attention is all you need 2018.09.11
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.
### model architecture
Most competitive neural sequence transduction models have an encoder-decoder structure, the encoder maps an input sequence of symbol representations 
$(x_1, ...,x_n)$ to a sequence of continuous representations $z = (z_1, ..., z_n)$.
Given z, the decoder generates an output sequence $(y_1, ..., y_m)$ of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.
Three key points for Transformer:		
> stacked self-attention  				
> point-wise  				
> fully connected layers for both the encoder and decoder  				

### encoder
The encoder is composed of a stack of N = 6 identical layers.		
each layer:			
> a multi-head self attention Mechanism 			
> a simple、position-wise fully connected feed-forward Networks 			

and employing a residual connection around each of the two sub-layers, followed by 
layer normalization.

### decoder
The decoder is also composed of a stack of N = 6 identical layers.			
each layer:			
> a multi-head self attention Mechanism 			
> a simple、position-wise fully connected feed-forward Networks 		
> multi-head attention over the output of the encoders stacke 		

and employing a residual connection around each of the two sub-layers, followed by 
layer normalization as well. The masking ensures that the predictions for position
$i$ can depend only on the known outputs at positions less than $i$.		

### attention
attention function: mapping a query and a set of key-value pairs to an outputs.  
**Scaled Dot-Product Attention:**		
$$attention(q_t, k, v) = \sum_{s = 1}^{m}{\frac{1}{Z} * exp(\frac{<q_t, k_s>}{\sqrt{d_k}}) * v_s}$$
$$Attention(Q, K, V) = softmax(\frac{Q * K^T}{\sqrt{d_k}}) * V$$
Two most commonly used attention functions:			
> additive attention 			
> dot-product attention 		

they are similar in theoretical complexity, but dot-product attention is much faster and more space-efficient.			
**multi-head attention:**
it beneficial to linearly project the queries, kyes and values h times with different, learned linear projections to $d_k$, $d_k$ and $d_v$	dimensions.	 for all the output values, thses are concatenated and once again projected, resulting in the final values.				
$$MultiHead(Q, K, V) = concat(head_1, head_2, ..., head_h) * W^o$$
$$head_i = attention(Q*W_{i}^{Q}, K*W_{i}^{K}, V*W_{i}^{V})$$
$$W_{i}^{Q} \in R^{d_{model} * d_k} \\ W_{i}^{K} \in R^{d_{model} * d_k} \\ W_{i}^{V} \in R^{d_{model} * d_v}$$	
$$W^{O} \in R^{d_{model} * d_k}$$
**position-wise feed-forward networks:**
this consists of two linear transformations with a ReLU activation in between.
$$FFN(x) = max(0, x*W_1 + b_1)*W_2 + b_2$$
**positional encoding:**
to add positional encodings to the input embeddings at the bottoms of the encoder and decoder stacks. in this work, using sine and cosine functions of different frequencies: $$PE(pos, 2i) = sin(pos/10000^{2i/d_{model}})$$  
$$PE(pos, 2i + 1) = cos(pos/10000^{2i/d_{model}})$$
this paper hypothesizes sin and cosine world allow the model to easily learn to attend by relative positions.			
**why self-attention:**	


| layer type | complexity per layer | sequential operations |		
| ------ | ------ | ------ |			
| self-attention | $O(n^2 * d)$ | $O(1)$ |			
| recurrent | $O(n * d^2)$ | $O(n)$ |			
| convolutional | $O(k * n * d)$ | $O(1)$ |			

## Convolutional Sequence to Sequence Learning 2018.09.11
Next we introduce a fully convolutional architecture for sequence to sequence modeling.       
**position embeddings:**      
input elements: $$X = (x_1, x_2, x_3, ..., x_m) \\ W = (w_1, w_2, w_3, ..., w_m)$$
其中，$w_j \in R^{f}$, embedding matrix $D \in R^{v * f}$.       
the absolute position of input elements:$$P = (p_1, p_2, p_3, ..., p_m)$$
其中，$p_j \in R^{f}$.     
Both are combined to obtain input element representations:
 $$e = (w_1 + p_1, w_2 + p_2, ..., w_m + p_m) \\ e_i = w_i + p_i$$
 **Convolutional Block Structrue:**       
Both encoder and decoder networks share a simple block structure, each block contarins a one dimensional convolution followed by a non-linearity.           
the output of encoder network:
$$Z^{l} = (z_{1}^{l}, z_{2}^{l}, ..., z_{m}^{l})$$
the output of decoder network:
$$H^{l} = (h_{1}^{l}, h_{2}^{l}, ..., h_{m}^{l})$$
each convolution kernel is parameterized as:
$$W \in R^{2d * 2d} \qquad b_2 \in R^{2d}$$
$X \in R^{k * d}$， X is a concatenation of k input elements embedded in d dimensions.  
$k$表示卷积窗口的宽度；$d$表示每个元素embedding所得到的向量大小；$2d$表示feature map的个数；           
卷积核可以将每个卷积窗口内的输入映射为一个大小为$Y = [A, B] \in R^{2d}$的向量，其中
$A, B的维度均为d$. 然后又使用了一个gate linear units进行处理，得到输入的最终表示：
$$v(A, B) = A \bigoplus \sigma(B)$$
其中，$v(A, B) \in R^{d}$, $\sigma$是门控函数.       
To enable deep convolutional networks, we add residual connections from the input of each convolution to the output of the block. 
$$h_{i}^{l} = v(W^{l}[h_{i-k/2}^{l-1}, ..., h_{i-k/2}^{l-1}] + b_{w}^{l}) + h_{i}^{l-1}$$
$$h_{i}^{l} = h_{i}^{l} + c_{i}^{l}$$
Finally, we compute a distribution over the T possible next target elements $y_{i + 1}$ by transforming the top decoder output $h_{i}^{L}$ via a linear layer with weights $W_{o}$ and bias $b_{o}$:
$$p(y_{i + 1} | y_1, ..., y_i, X) = softmax(W_{o} * h_{i}^{L} + b_{o}) \in R^{T}$$
**multi-step attention:**       
to combine the current decoder state with an embedding of the previous target element $g_i$: 
$$d_{i}^{l} = W_{d}^{l} * h_{l}^{l} + b_{d}^{l} + g_i$$
decoder layer $l$ the attention $a_{ij}^{l}$: 
$$a_{ij}^{l} = \frac{exp(d_{i}^{l}, z_{j}^{u})}{\sum_{t=1}^{m}{exp(d_{i}^{l}, z_{t}^{u})}}$$
each output $z_{j}^{u}$ of the last encoder block $u$ 表示encoder第u层（也就是最后一层）的第j个隐层状态.       
conditional input $c_{i}^{l}$ to the current decoder layer:
$$c_{i}^{l} = \sum_{j=1}^{m}{a_{ij}^{l}(z_{j}^{u} + e_{j})}$$
在解码器的每一层，都单独与编码器最后一层实现了attention机制。

## End-to-End Task-Completion Neural Dialogue Systems 2018.09.12
Traditional systems have a rather complex and modularized pipeline, consisting of a language understanding module, a dialogue manager and a natural language generation component.      
Recent advances of deep learning have inspired many applications of neural models to dialogue systems :       
> A Network-based End-to-End Trainable Task-oriented Dialogue System. 2017        
> Towards end-to-end learning for dialog state tracking and management using deep reinforcement learning. 2016        

this paper addresses all 3 issues by redefining the targeted system as a task-completion neural dialogue system, and the 3 issues as follows:
> inflexible question types       
> poor robustness       
> user requests during dialogues      

**Proposed Framework:**       
*Language Understanding(LU):*         
A major task of LU is to automatically classify the domain of a user query along with domain specific intents and fill in a set of slots to form a semantic frame. 
$$\overline{x} = w_1, w_2, ..., w_n, <EOS> \\ \overline{y} = s_1, ..., s_n, i_m$$
$\overline{x}$ is the input word sequence and $\overline{y}$ contains the associated slots and the sentence-level intent $i_m$. the LU is implemented with a 
single LSTM, which performs intent prediction and slot filling simultaneously.
$$\overline{y} = LSTM(\overline{x})$$
The LU objective is to maximize the conditional probability of the slots and the intent $\overline{y}$ given the word sequence $\overline{x}$:
$$p(\overline{y}|\overline{x}) = (\prod_{i}^{n}{p(s_i| w_1,..,w_i)})p(i_m|\overline{y})$$
*Dialog Management:*        
the symbolic LU output is passed to the DM, the classic DM includes two stages:     
> dialogue state tracking              
> policy learning         

## Using Recurrent Neural Networks for slot filling in spoken language understanding 2018.09.14

The goal of spoken language understanding is to convert the recognition of user input $S_i$ into a task-specific semantic representation of the user`s intention $U_i$ at each turn. the dialogue manager interprets $U_i$ and decides on the most appropriate system action $A_i$. </br>
the semantic parsing of input utterances in SLU consists of 3 tasks: </br>
> domain detection</br>
> intent determination</br>
> slot filling</br>
