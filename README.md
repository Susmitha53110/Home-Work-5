# Home-Work-5
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
