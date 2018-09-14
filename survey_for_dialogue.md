<center><font size="12"><font color="gray">人机对话综述</font></font></center>			
<p align="right"><font size="5"><font color="blue">liguoliang@kingsoft.com</font></font></p>
<p align="right"><font size="5"><font color="blue">2018.09.14</font></font></p>
# 人机对话
人机对话系统可以分为四个部分：开放域聊天、任务驱动的多伦对话、问答、推荐

> 开放域聊天： 就是不局限对话的聊天，即在用户的query没有明确的信息或服务需求时系统做出的回应。		
> 问答： 更侧重与一问一答，即直接根据用户的问题给出精准的答案。虽然也会涉及上下文理解，通常是通过指代消解、query补全来完成，其更接近一个信息检索的过程。		
> 推荐： 根据当前的用户query和历史的用户画像主动推荐用户可能感兴趣的信息或服务。			

## 任务驱动的多轮对话
用户带着明确的需求而来，希望能得到满足特定限制条件的信息或服务，例如：订餐、订票、寻找音乐等等。 用户的需求可能比较复杂，需要分多轮进行陈述，同时用户也可能在对话过程中不断修改或完善自己的需求。 此外，当用户的陈述需求不够具体明确的时候，机器也可以通过询问、澄清或确认来帮助用户找到满意的结果。

任务驱动的多轮对话是一个决策过程，需要机器在对话过程中不断根据当前的状态决策下一步应该采取的最优动作(例如：提供结果、询问特定限制条件、澄清或确认需求等等)，从而能有效地辅助用户完成获取信息或服务的任务。 在学术文献中的Spoken Dialogue Systems一般特指任务驱动的多轮对话。

科普一下任务驱动的多轮对话系统的经典框图：
<center>![Alt text](../md_data/SDS.jpeg "任务驱动的多轮对话")</center>

## NLU
### Language understanding
在理解人类语言时需要重点解决好两个问题：意图识别(Intent Detection)和槽填充(Slot Filling),意图识别是个典型的多分类任务，而槽填充是个典型的序列标注任务。
自然语言中的query识别为结构化语义表示，语义表示通常被称为Dialogue Act，其有communicative function、slot-value pairs组成，前者表示query的类型，例如：陈述需求、询问等等，后者表达一个限制条件。				

[using recurrent neural networks for slot filling in spoken language understanding][1]				
[Attention-based recurrent neural network models for joint intent detection and slot filling][2]				
[Exploring the use of attention-based recurrent neural networks for spoken language understanding][3]				

### DM
### Dialogue State Tracking(DST)
DST就是根据多轮的对话来确定用户当前目标到底是什么的过程，一轮对话状态中，用户目的用一组slot-value表示。
概率分布，又称为置信状态（belief state），所以DST又可以称为Belief state tracking。			
用户目的在一个置信状态中的表示分为两部分：			
> 每个slot都可以对应多个value，同时每个value都有一个置信概率，形成了每个slot上的边缘置信状态；		
> 所有的slot-value pairs的概率分布形成了联合置信状态；

无论是ASR、SLU都是经典的分类问题，会存在误差，给任务驱动的对话系统引入了一个不确定性环境下做决策的问题。
要解决这个问题，首先我们希望ASR、SLU在输出分类结果的同时输出一个置信度的评分，最好能给出多个候选结果(N-Best List)以更好的保证召回。 然后DST在置信度评分、N-Best List的基础上估计所有可能的对话状态的 belief state，即用户完整需求的概率分布。

[Global-Locally Self-Attentive Dialogue State Tracker][4]				
[Gated End-to-End Memory Networks][5]				
[An End-to-end Approach for Handling Unknown Slot Values in Dialogue State Tracking][6]				
[N-best error simulation for training spoken dialogue systems][7]				

### Dialog policy
根据DST得到的置信状态来做决策的过程，输出一个系统动作(system action)。 系统动作也是一个由communicative
function、slot-value pairs组成的语义表示，表明系统要执行的动作类型、操作参数。

[Sub-domain Modelling for Dialogue Management with Hierarchical Reinforcement Learning][8]				
[A benchmarking Environment for Reinforcement Learning based task oriented dialogue management][9]				

## NLG
### Natural Language Generation
将Dialog policy输出的语义表示转化成自然语言的句子，反馈给用户，是从结构化数据中自动生成文本的过程。

[Sequence-to-Sequence Generation for Spoken Dialogue via Deep Syntax Trees and Strings][10]				
[Incorporating Copying Mechanism in Sequence-to-Sequence Learning][11]				
[An Ensemble of Retrieval and Generation-Based Dialog Systems][12]				
[Generating Sentences by Editing Prototypes][13]			
[Toward Controlled Generation of Text][14]				

## DST的评估方法
DST维护的是一个概率分布，那么这里就引入了两个问题：
> 怎么样衡量一个概率分布的优劣；
> 在哪一轮评估比较合适；

DSTC2013是第一届DST任务的公开评测，评测数据来自于匹斯堡公车路线电话自动查询系统3年的真实数据，包括5组训练集、4组测试集，提出了多种评测指标、评测schedule作为参考：
*评测指标*：			
> Hypothesis accuracy: 置信状态首位假设的准确率；			
> Mean reciprocal rank: 1/R的平均值，R是第一条正确假设在置信状态中的排序；			
> L2-norm: 置信状态的概率向量与真实状态向量之间的L2距离；			
> Average probability: 真实状态在置信状态中概率得分的平均值；			
> ROC performance: 一系列指标来刻画置信状态中首位假设的可区分性；		
> Equal error rate: 错误接受率(false accepts, FA)与错误拒绝率(false reject, FR)的相交点；		
> Correct accept 5/10/20:至少5%、10%、20%的FA时的正确接受率(correct accepts,CA)；			

*评测schedule*：				
> 每轮对话都做评估；				
> 对于一个slot-value pair，只有在其被提及时才评估；			
> 对话结束时评估；			

<center>![Alt text](../md_data/DSTC.png "DSTC")</center>

******

评测一个任务驱动的多轮对话系统，主要设计评测Language understanding、DST以及
Dialog policy三个部分。 自然语言理解可以通过准确率、召回率以及F-socre等指标进行评测；DST的评测，参考DSTC公开评测；对话策略的质量通常需要对话系统的整体效果来体现。

## End-to-End Dialogue Modelling


[a neural conversational model][15]				
[A Hierarchical Latent Variable Encoder-Decoder Model for Generating Dialogues][16]				
[Towards End-to-End Learning for Dialog State Tracking and Management using Deep Reinforcement Learning][17]				
[Learning end-to-end goal-oriented dialog][18]					


[1]: https://ieeexplore.ieee.org/abstract/document/6998838/					
[2]: https://arxiv.org/abs/1609.01454					
[3]: http://slunips2015.wixsite.com/slunips2015/accepted-papers
[4]: https://arxiv.org/abs/1805.09655			
[5]: https://arxiv.org/abs/1610.04211					
[6]: https://arxiv.org/abs/1805.01555					
[7]: http://svr-www.eng.cam.ac.uk/~sjy/papers/thgt12.pdf 				
[8]: https://arxiv.org/abs/1706.06210				
[9]: https://arxiv.org/abs/1711.11023					
[10]: https://arxiv.org/abs/1606.05491				
[11]: http://aclweb.org/anthology/P16-1154				
[12]: https://openreview.net/pdf?id=Sk03Yi10Z				
[13]: https://arxiv.org/abs/1709.08878					
[14]: https://arxiv.org/abs/1703.00955				
[15]: https://arxiv.org/abs/1506.05869				
[16]: https://arxiv.org/abs/1605.06069				
[17]: https://arxiv.org/abs/1606.02560					
[18]: https://arxiv.org/abs/1605.07683				

