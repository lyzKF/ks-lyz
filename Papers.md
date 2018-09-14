*我在这里写了一行话*
Questions:		
> 语义理解的问题 semantic understanding		
> 上下文理解的问题 context issue			
> 个性身份一致性问题 inconsistency in personlity		

Typical solution:		
> retrieval-based			
> generation-based			
> hybrid methods			

Challenges in Chatting Machines:			
> one-to-many： one input, many possible responses			
> knowledge & reasoning : real understanding requires various knowledge, world facts, or backgrounds			
> situational context : 

[Multiple responding mechanisms represented by real-valued vectors]
[Learning to Ask Questions in Open-domain Conversational Systems with Typed Decoders]
[Topic Aware Model by incorporating topic words]
[EnhancedMA using reinforcement learning]


## Sponken language understanding
> encodeer-decoder architectures for slot tagging		
> joint semantic frame parsing: slot filling and intent prediction jointly 			

three main components in the pipelined fashion:			
1. domain classification 				
2. intent classification : deep neural networks applied to this standard classification problem.		
3. (sequence tagging & slot tagging)slot filling : LSTM-based models,(attention-based)encoder-decoder architectures			
[using recurrent neural networks for slot filling in spoken language understanding]	

## Dialogue State Tracking
[]














Three typical approaches:			
> template based
> retrieval based
> generation based

template based: 		
1. respond users with hand-crafted features			
2. reliable and controllable		
3. hard to scalable				

retrieval based:
1. select a proper response from an index 				
2. fluent and informative responses			
3. easy to implement 				
4. heavily depend on a predefined index 				

generation based:
1. generate a response with NLG techniques			
2. flexible 			
3. suffer from safe response problem			
4. require more resource			

ensemble approaches:
[AN	ENSEMBLE OF	RETRIEVAL-BASED	AND	GENERATION-BASED HUMANCOMPUTER	CONVERSATION	SYSTEMS]
[Exemplar Encoder-Decoder for Neural Conversation Generation]
[Generating	Sentences by Prototype Editing]
[recent	work for response generation with a	conditional	variational	auto-encoder]


[A knowledge-grounded eural	conversation model]
[Flexible end-to-end dialogue system for knowledge grounded	conversatio]
[Commonsense Knowledge Aware Conversation Generation with Graph	Attention]

commonsense Knowledge 			
commonsense reasoning 		

# Deep learning for Conversational AI
## give an overview of recent research trends in deep learning for conversational AI

## provide a detailed overview of task-based dialogue systems

## analyse most promising research avenues and stress their current limitations

## discuss the importance of data requirements vs algorithm choices

## present an industry-based perspective on current deep conversational AI

## detect current "make it or break it" challenges in conversatioanl AI

task-oriented dialogue systems:
> goal-oriented				
> require precise understanding, it is hard to collect data 			
> modular, highly hand-crafted, restricted ability, but meaningful systems 		

chat-based conversational agents:			
> no goal 		
> large amounts of data 		
> End-to-end, highly data-driven, but meaningless responses			

Natural Language Understanding 				
Language Understanding 				
Spoken Language Understanding 				

Domain classification :				

intent classification :				
also known as act classification, label each dialogue utterance with intent.		
slot filling :		
sequence tagging or slot tagging	

SLU and DST(dialogue state tracking) resemble the task of semantic parsing, converting natural language into 
computer-executable formal meaning representations for domain-specific applications.


The neural belief tracker is a statistical DST framework, which aims to satisfy the following design goals:			
> end-to-end learnable 			
> generalisation to unseen slot values 				
> capability of leveraging the semantic content of pre-trained word vector spaces without human supervision 			

Representation learning : NBT-DNN or NBT-CNN 			


[an end-to-end approach for handling unknown slot values in dialogue state tracking]
[hybrid dialog state tracker with ASR Features]


# Dialogue Management Approaches
Rule-based:			
> huge hand-crafting effort 			
> non-adaptable and non-scalable 				
> but this is what works right now 				

Supervised:				
> learn to "mimic" the answers of a corpus 			
> assumes optimal human behaviour 				
> does not to long-term planning 			

Reinforcement learning:	
> learns through interaction in order to maximise a future reward 			
> learns in the actual dialogue environment 		
> adapts to new environment/users/situations 			
> requires less anaotation 			
> slow and expensive learning 			
> difficult to reuse data 			

1. policy-based RL : search directly for optimal policy				
2. value-based RL : estimate the optimal value fucntion						
3. model-based RL : build a model of the environment and plan using the model 		

## user simulation
Different simulation approaches:		
1. model: Rule-based(manually crafted by experts) and learning-based(trained from a corpus of dialogues)		
2. output-level: semantic level(dialogue act) and text level(natural language)		
3. error simulation: No error and language understanding error and ASR error 		
Training with user simulator:			
1. Rule-based		
2. agenda-based		
3. model/learning-based 			
4. seq2seq-based 		

[Dialogue Management Based on Hierarchical RL]
[Dialogue Management Based on Feudal RL]
[Revisiting Problems of RL-Based DM]





# Natural Language Generation
NLG is the process of deliberately constructing a natural language text in order to meet specified communicate goals.			

NLG Evaluation:			
human judgement :
> adequacy : correct meaning 		
> fluency : linguistic fluency/naturalness 			
> readability : fluency in the dialogue context 		
> variations : multiple realisations of the same meaning 		

automatic evaluation measures :
> word overlap: BLUE, Meteor, rouge 		
> embedding-based: greddy matching, embedding average 			
> task-oriented: item error rate 		

Template-Based NLG:		
> define a set of rules to map meaning representation to natural language 		

Pipeline approach to NLG:		
> Statistical approaches to pipelined NLG		

Sequntial approaches to NLG:	

RNN-Based LM NLG: 		


(Attentive) Seq2Seq for NLG:

**Recent work in Neural NLG:**		
controlled text generation based on VAE/GANs:		
[Latent intention dialogue models]	
[Re-using info from the input directly: CopyNet and PointerNet]
[Retrieval-based NLG: select from a set of predefined responses]
[Combining retireval-based and generative NLG]
[Generating Sentences by Editing Prototypes]
[Controlled text generation based on VAE/GANs]
[slot-value informed Seq2Seq models]

short summary on NLG: 			
> Evaluating NLG is hard, the best way is human evaluation 		
> in product, template-based NLG is still most common 		
> learning-based NLG are promising 			
> NN-based NLG is a conditional neural LM that learns realisation and semantic alignments jointly 		
> recent trends: adversarial modeling, E2E learning 			






# End-to-End Dialogue Modelling

**why end-to-end:**				
social chatbots (chit-char dialogue systems)
[a neural conversational model]		
[a hierarchical latent variable encoder-decoder model for generating dialogues]
[Towards end-to-end learning for dialog state tracking and management using deep reinforcement learning] 		
[Multi-domain dialogue success classifiers for policy training]				
[learning end-to-end goal-oriented dialog]
[Mem2seq: Effectively incorporating knowledge based into end-to-end task-oriented dialogue systems]


# Getting data for training dialogue systems
WOZ(Wizard-of-Oz data collection):		
> No explicit dialogue act annotations		
> system policy can be learned directly from the data 			
> interesting and diverse system behaviours			

wizard: A software wizard or setup assistant is a user interface type that presents a user with a sequence of dialog boxes that lead the user through a series of well-defined steps

M2M(machines talking to machines):			
> full control over the dialogue flow 			
> Paraphrase data collection UI is simpler to build 			
> easier to engineer particular behaviours 			
> crowdsourcers do not have to label data 	

[Building a conversational agent overnight with dialogue self-play]

Unstructured:
> pattern matched-based Chatbots 		
> Retrieval Based Chatbots 			
> Neural Generative chatbots 		

structured:			
> rule-based dialogue systems  			
> pomdp-based dialogue system  				
> neural generative dialogue systems 					


Tsung-Hsien Wen:
[sub-domain modelling for dialogue management with hierarchical reinforcement learnig]			
[A benchmarking environment for reinforcement learning based task oriented dialogue mannagement]		
[Dialogue manager domain adaptation using Gaussian process reinforcement learning]
[Policy committee for adaptation in multi-domain spoken dialogue systems]			
[Neural belief tracker: Data-driven dialogue state tracking]			
[Multi-domain dialog state tracking using recurrent neural networks]		
[on-line active reward learning for policy optimisation in spoken dialogue systems]
[Reward shaoing with recurrent neural networks for speeding up on-line policy learning in spoken dialogue systems]
[Multi-domain dialogue success classifiers for policy training]
[Semantically conditioned LSTM-based natural language generation for spoken dialogue systems]			
[Stochastic language generation in dialogue using recurrent neural networks with convolutioan sentence reranking]			
[Multi-domain neural network language generation for spoken dialogue systems]
[Latent intention dialogue models]
[A network-based end-to-end trainable task-oriented dialogue system]

Ryan Lowe:			
[how not to evaluate you dialogue system: An empirical study of unsupervised evaluation metrics for dialogue response generation]			
[Towards an automatic Turing test: Learning to evaluate dialogue responses]			
[A hierarchical latent variable encoder-decoder model for generating dialogues]