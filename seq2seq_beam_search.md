# seq2seq beam search

在seq2seq模型中，beam search的方法只用在测试的情况下。因为在训练过程中，每个decoder的输出是有正确答案的，不需要beam search来加大输出的准确率。

解码是seq2seq模型的常见问题，在得到训练模型之后，我们希望能够得到句子序列条件概率值最大的序列：

$$argmax_{y}{P(y^{<1>}, y^{<2>}, y^{<3>}, ...,y^{T_y} | x^{<1>}, x^{<2>}, ,x^{<3>}, ..., x^{T_x})}$$

常用方法有贪心搜索（Greedy Search）、集束搜索（Beam Search）。

## greedy search

贪心搜索是一种比较直观的方法，在生成第一个词$y^{<1>}$的分布之后，根据条件语言模型挑选出最有可能的第一个词$y^{<1>}$。然后生成第二个词$y^{<2>}$的概率分布，挑选出第二个词$y^{<2>}$，依此类推。贪心算法始终是选择每一个最大概率的词，但我们真正需要的是挑选整个单词序列$y^{<1>}、y^{<2>}、y^{<3>}、......、y^{<T_y>}$使得整体的条件概率最大。

## beam search

beam search算法是一种平衡性能与消耗的搜索算法，其目的是在序列中解码出相对较优的路径，广泛应用于OCR、语音识别、翻译系统等场景中。beam search是一种启发式图搜索算法，通常用在图的解空间比较大的情况下，为了减少搜索所占用的空间、时间。在每一步深度扩展的时候，减掉一些质量比较差的结点，保留下一些质量较高的结点。这样可以减少空间消耗，并提高了时间效率，其缺点就是有可能丢弃掉潜在的最佳方案。

beam search算法作为一种折中手段，在相对受限的搜索空间中找出其最优解，得出的解接近于整个搜索空间中的最优解。beam search算法一般分为两部分：

> 路径搜索 ：在受限空间中检索出所有路径；
>
> 路径打分：对某一条路径进行评估打分；

beam search的一般步骤为：

> 初始化beam_size个序列，序列均为空，这些序列称为beam paths；
>
> 取下一个Frame的前N个候选值（N一般为beam_size或更大，Frame内部候选值已按照概率倒序排序），与beam paths组合形成N * beam_size条路径，称之为prob_paths；
>
> 对prob_paths进行打分，取前beam_size个prob_path作为新的beam_paths；
>
> 若解码结束则完成算法，否则返回第二部；

eg：

testing的时候，假设词表大小为3，内容为a、b、c，beam_size为2。

decoder解码的时候：

> 生成第1个词的时候，选择概率最大的2个词；假设为a、c，那么当前beam paths就是a、c
>
> 生成第2个词的时候，将当前序列a、c分别与词表中的所有词进行组合，得到新的6个序列aa、ab、ac、ca、cb、cc，然后从其中选择2个得分最高的作为当前的beam paths；假设当前beam paths为aa、cb
>
> 不断重复这个过程，直到遇到结束符为止；

$\frac{P(A, B|C)}{P(B|C)} = \frac{P(A, B, C)}{P(C)} * \frac{P(C)}{P(B, C)} = \frac{P(A, B, C)}{P(B, C)} = P(A|B, C)$

![Alt text](../md_data/beam_search.jpg "beam search")

![Alt text](../md_data/beam_search.png "beam search")

# seq2seq

**Formulas in seq2seq model:**

inputs sequence:  $(x_1, x_2, x_3, ..., x_m)$

outputs sequence: $(y_1, y_2, y_3, ..., y_n)$

## basic seq2seq

<center>![Alt text](../md_data/seq2seq_01.jpg "decoder 01")</center>

## 带输出反馈的解码模式

<center>![Alt text](../md_data/seq2seq_02.jpg "decoder 02")</center>

## 带向量编码的解码模式

<center>![Alt text](../md_data/seq2seq_03.jpg "decoder 03")</center>

## 带注意力机制的解码模式

<center>![Alt text](../md_data/seq2seq_04.jpg "decoder 04")</center>

# attention

Attention机制的基本思想是其打破了传统编码器-解码器结构在解码时依赖于内部一个固定长度向量的限制，其实现是通过保留LSTM编码器对输入序列的中间输出结果，然后训练一个模型对这些输入进行选择性学习，并在模型输出时将输出序列与之进行关联。

### Self Attention

Self Attention与传统的Attention机制非常的不同：

​	self Attention比传统的Attention mechanism效果要好，主要原因之一是，传统的Attention机制忽略了source端或target端句子中词与词之间的依赖关系；self Attention不仅可以得到source端与target端词与词之间的依赖关系，同时还可以有效获取source端或target端自身词与词之间的依赖关系。

eg：

由于该向量只关注于句子的某个特定元素，要得到完整的句子信息，则需要多个m表示句子：$$M = A * H$$  超参数：$r \quad d_a$	$$H -> n *2u \\ W_{s_1} -> d_a * 2u \\ W_{s_2} -> r * d_a \\ a-> n \\m -> 2u \\ A -> r*n \\ M -> r* 2u$$

r的物理意义为句子的不同层面，即矩阵A的每一行代表了句子不同层面的信息；

### 传统attention机制

#### global attention and soft attention 

传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，传统的attention是参数化、可导的，参数可以被嵌入到模型中直接参与训练，梯度可经过attention mechanism反响传播到模型其他部分。流程图如图所示：

![Alt text](../md_data/attention.png "attention")

inputs sequence:  $(x_1, x_2, x_3, ..., x_m)$

outputs sequence: $(y_1, y_2, y_3, ..., y_n)$		
*Encoder:*		
$$h_t = RNN(x_t, h_{t-1})$$
$$a_t(j) = align(h_t, \overline{h_j}) = softmax(score(h_t, \overline{h_j}))$$
$$softmax(score(h_t, \overline{h_j})) = \frac{exp(score(h_t, \overline{h_j}))}{\sum_{i}{exp(score(h_i, \overline{h_j}))}}$$
$$\begin{equation} score(h_t, \overline{h_t}) = 
\left\{
             \begin{array}{lr}
             \overline{h_t}^T * h_{t}\\
             \overline{h_t}^T * W_a * h_{t} \\
              v_a^T * tanh(W_a * [h_{t}; \overline{h_t}]) & 
             \end{array}
\right.
\end{equation}$$
其中，$h_t$是t时刻source端的隐状态，$\overline{h_j}$是j时刻target端的隐状态。			
*中间状态：*			
$$c_t = \sum_{j = 1}^{|x|}{a_t(j) * h_j}$$
其中，$|x|$是source端的sequence长度。			
*Decoder:*			
$$\overline{h_t} = RNN(\overline{h_{t-1}}, c_t, y_{t-1})$$
$$p(y_t | y_{\lt t}, x) = g(\overline{h_{t}}, c_t, y_{t-1})$$
or:
$$p(y_t | y_{\lt t}, x) = g(\overline{h_{t}})$$
缺点：

1. 若Encoder句子不太长时，相对global attention，其计算量并没有明显减少；
2. 位置向量$p_t$的预测并不非常准确，其会直接影响local attention的准确率；

#### hard attention

hard attention是一个随机过程，其会依据概率采样source端的一部分隐状态进行计算，而不是计算整个Encoder的隐状态；同时为了实现梯度的反向传播，需要采用蒙特卡洛采样的方法估计模块梯度。