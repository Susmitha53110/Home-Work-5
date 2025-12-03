                       Susmitha Dodla / 700765311 
                           Homework – 5 
 

1. Positional Encoding Concepts 

a) Why do transformer models require positional encodings? 

Since transformers read all input tokens at once and rely purely on attention rather than recurrence or convolution, they do not naturally understand the order of words. Positional encodings supply the model with information about each token’s placement in the sequence, allowing it to differentiate between sentences containing the same words but arranged differently. 

 

b) Two essential properties of an effective positional encoding method 

Clear distinction between positions 
Each position must map to a unique vector so that different locations in the sequence are not confused by the model. 

Ability to convey positional relationships 
The encoding should help the model infer both absolute positions and how far apart tokens are. Approaches like sinusoidal encodings achieve this by creating predictable shifts that maintain relational information. 

 

c) Meaning of a unitary and norm-preserving positional encoding matrix 

If the positional encoding matrix is unitary, multiplying embeddings it behaves like a rotation that keeps distances and inner products intact. Norm-preserving means an embedding length stays the same after the encoding is added. Both properties ensure positional information is incorporated without altering the original scale or geometry of the token embeddings. 

 

2. Attention Mechanism 

a) What is an “attention score,” and why is it important? 

An attention score measures how closely a query vector aligns with a key vector, often using a scaled dot product. Higher scores show stronger relevance between tokens. These raw scores determine how much influence each token has when forming the output representation. 

 

b) What operation converts these scores into actual attention weights? 

The SoftMax function is applied to the scores, turning them into a normalized distribution where all weights are positive and sum to one. 

 

c) How is the final context vector produced? 

After obtaining the attention weights, each value vector is multiplied by its corresponding weight. The context vector is simply the sum of these weighted value vectors. 

 

3. Multi-Head Attention 

a) Main benefit of using several attention heads 

Multiple heads allow the model to pay attention to various types of relationships at the same time. Each head can learn distinct patterns, enabling richer and more flexible understanding of the sequence. 

 

b) Advantage of dividing Q, K, and V into multiple subspaces 

By splitting the vectors into smaller dimensions, each head focuses on a different aspect of the input. This increases the model’s expressiveness because different heads can capture different semantic or structural patterns. 

 

c) Why concatenate all heads and apply a linear layer afterward? 

Concatenating gathers the information learned by all heads. The following linear layer then transforms this combined representation back into the model’s expected embedding size, ensuring compatibility with later layers. 

 

4. Ethical Foundations 

Why ethics differ from laws or personal feelings 

Laws are created by governments and may not always reflect moral correctness, and feelings differ widely across individuals. Ethics, on the other hand, is a rational framework for determining right and wrong that does not rely solely on legal rules or emotional reactions. 

 

Two classical ethical theories and how they apply to AI decisions 

Utilitarianism: 
Focuses on outcomes and seeks the action that brings the greatest overall benefit. In an AI context—such as automatic loan approvals—a utilitarian system would choose the decision that maximizes total societal advantage, even if some individuals are negatively affected. 

Deontology: 
Emphasizes duties, rights, and moral principles regardless of consequences. Using the same AI scenario, a deontologist would require the system to treat all applicants fairly and avoid rule violations, even if this lowers efficiency. 

 

Why no single ethical theory works universally 

Each ethical framework captures different moral insights but also has limitations. Some real-world dilemmas prioritize consequences, while others require respect for rights or duties. Because no single approach handles every scenario perfectly, philosophers argue that no one theory can dominate all contexts. 

 

5. Types of AI Harms 

Definitions of allocational and representational harm 

Allocational harm: 
Occurs when an AI system unfairly influences access to resources, benefits, or opportunities—such as employment, financial services, or housing. 

Representational harm: 
Happens when AI systems reinforce harmful stereotypes or portray certain groups inaccurately or disrespectfully. 

 

Examples of each type of harm 

Allocational harm example: 
A hiring model that reduces the ranking of applicants from certain groups due to biased historical data, resulting in fewer job interviews. 

Representational harm example: 
A machine translation tool repeatedly associating specific professions with a particular gender, reinforcing cultural stereotypes. 

 

Why representational harm is more difficult to quantify 

Representational harm deals with meaning, portrayal, and social perception rather than measurable outcomes. Since it involves cultural and contextual interpretation, it is harder to detect using numerical metrics compared to allocational harms that can be measured through disparities in outcomes. 

 

6. Sources of Dataset Bias 

Three causes of bias during data collection or labeling 

Biased sampling: 
Data gathered from only a subgroup or limited region does not represent all users. 

Labeler subjectivity: 
Human annotators may unintentionally include their personal biases while labeling data. 

Unequal digital visibility: 
Some communities generate less online content, leading to skewed dataset representation. 

 

Groups or data types commonly under-represented 

Speakers of less common languages or dialects 

Communities with limited internet access 

Individuals from rural areas or the global South 

Cultural content not available digitally or behind paywalls 

 

How bias can grow even after preprocessing 

Models tend to strengthen patterns that appear frequently during training. Even if the dataset is cleaned, imbalances that remain can cause the model to further exaggerate biased associations as it generalizes. 

 

7. Safety, Security, and Privacy 

Definition of data poisoning and its effect on model behavior 

Data poisoning involves adding harmful or misleading samples to the training set so that the model learns incorrect patterns. This can cause consistent misclassification or allow attackers to trigger specific wrong outputs through hidden backdoor inputs. 

 

Ethical issues related to model memorization 

If a model reproduces private or copyrighted content, it can violate privacy, breach intellectual property laws, and disclose sensitive information, raising serious ethical concerns about data handling and trustworthiness. 

 

How model stealing impacts privacy and intellectual property 

By sending repeated queries to a model, an attacker can train a replica that mimics its outputs. This can expose proprietary techniques and potentially leak sensitive data embedded in the original model, undermining both privacy and intellectual ownership. 

 
CODING qUESTIONS

1. Scaled Dot-Product Attention (Theory + Code)
What is Attention?

Attention is a mechanism that allows a model to focus on important parts of the input. Instead of processing all tokens equally, the model learns which tokens are more relevant for generating the output.

Formula Used

The Scaled Dot-Product Attention follows this equation:

Attention
(
𝑄
,
𝐾
,
𝑉
)
=
softmax
(
𝑄
𝐾
𝑇
𝑑
𝑘
)
𝑉
Attention(Q,K,V)=softmax(
d
k
	​

	​

QK
T
	​

)V

Where:

Q → Query matrix

K → Key matrix

V → Value matrix

dₖ → Dimensionality of the key vectors

We divide by √dₖ to avoid excessively large scores

What My Code Does

Computes attention scores using Q @ Kᵀ

Scales the scores

Applies softmax to convert scores into probabilities

Uses these weights to compute the final context vector

Output Produced

The script prints:

Attention Weights → shows how much each token attends to others

Context Vector → weighted combination of values

This explains how "focus" is distributed in a sequence.

🧠 2. Transformer Encoder Block (Theory + Code)
What is a Transformer Encoder?

A Transformer encoder is a fundamental building block used in models like:

BERT

GPT

T5

Many machine translation systems

The encoder block mainly contains:

Multi-Head Self-Attention

Feed Forward Network (FFN)

Residual Connections

Layer Normalization

Why These Components Matter

Self-Attention: Learns relationships between words in a sentence

Multi-Head: Allows the model to attend from different perspectives

FFN: Helps the model transform information nonlinearly

Residuals: Improve gradient flow (training stability)

LayerNorm: Smooths training and avoids divergence

Dimensions Used (as required)

d_model = 128

h = 8 heads

Batch used for testing → (32, 10, 128)

Output Shape Explanation

Transformers maintain the same shape because:

Attention maps (batch, seq_len, d_model) → (batch, seq_len, d_model)

FFN projects back to the original dimension

So the verified output is:

torch.Size([32, 10, 128])
