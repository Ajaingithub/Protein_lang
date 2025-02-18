import torch
import torch.nn as nn

### Single cell layer
### Input tensor
input_tensor = torch.tensor(
    [[0.345, 0.332, -0.135]]
)

linear_layer = nn.Linear(in_features=3, out_features=2)

output = linear_layer(input_tensor) ### operation it perform y = W.x + b, where W = weights, b = bias and X is the imput tensors

### Multiple layer i.e. hidden layers
input_tensor = torch.tensor(
    [[0.345, 0.332, -0.135, 0.689, -0.312, 0.973, 0.367, -0.771, 0.922, 0.221]]
)

model = nn.Sequential(
    nn.Linear(10,18),
    nn.Linear(18,29),
    nn.Linear(29,5)
)

output = model(input_tensor)
print(output)

### Adding the activation function
## Starting with the Sigmoid function
sigmoid = nn.Sigmoid() ## only use single tensor
output = sigmoid(torch.tensor([6]))

# Sigmoid is used for Binary
# Softmax for multi class classification
# Used in the last layer of the deep learning model
probabilities = nn.Softmax(dim=-1)

### Building a transformer using DataCamp
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

# Building the Transformer Model with PyTorch
# To build the Transformer model the following steps are necessary:
#     Importing the libraries and modules
#     Defining the basic building blocks - Multi-head Attention, Position-Wise Feed-Forward Networks, Positional Encoding
#     Building the Encoder block
#     Building the Decoder block
#     Combining the Encoder and Decoder layers to create the complete Transformer network

### Before running all these mutli head attention or self attention
### lets learn how it is doing
### Input tensor

import torch
import math

# 1. Input Tensor (x)
batch_size = 1
seq_len = 3
d_model = 4

# Random input tensor (e.g., representing a batch of 3 tokens, each of dimension 4)
x = torch.randn(batch_size, seq_len, d_model)  # Shape: [1, 3, 4]

# tensor([[[-0.1064, -1.0668, -0.3906,  0.3016,  1.2438],
#          [ 1.2965,  0.2296,  0.1502,  1.4333,  0.6552],
#          [ 1.1675,  1.8170, -0.4821,  0.1470,  0.3689],
#          [ 0.5642,  2.2381,  1.1289,  1.2596,  0.3684]],

#         [[-0.5382,  1.8615,  0.2813,  0.5388,  1.1295],
#          [ 0.2243, -0.3874, -0.2070,  2.0173,  0.4798],
#          [ 0.6017,  0.4941, -0.5506, -0.4039,  0.9897],
#          [-0.7141, -1.0866, -1.1268, -1.3112,  0.7889]]])

# Transformation of Input x into Query (Q), Key (K), and Value (V):
# Linear transformations: The input tensor x will pass through 3 separate linear layers (one for queries, one for keys, and one
# for values), each of which performs a learned transformation.

# Parameters for Multi-Head Attention
num_heads = 2
d_head = d_model // num_heads  # d_head = 4 // 2 = 2

# 2. Linear Transformation for Q, K, and V
query = torch.matmul(x, torch.randn(d_model, d_model))  # Linear transformation for Q
key = torch.matmul(x, torch.randn(d_model, d_model))    # Linear transformation for K
value = torch.matmul(x, torch.randn(d_model, d_model))  # Linear transformation for V

# 3. Reshape to split into attention heads
query = query.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)  # Shape: [batch_size, num_heads, seq_len, d_head]
key = key.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)      # Shape: [batch_size, num_heads, seq_len, d_head]
value = value.view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)  # Shape: [batch_size, num_heads, seq_len, d_head]

# 4. Resulting Shapes of Q, K, V
print(f"Query Shape: {query.shape}")  # Expected: [1, 2, 3, 2]
print(f"Key Shape: {key.shape}")      # Expected: [1, 2, 3, 2]
print(f"Value Shape: {value.shape}")  # Expected: [1, 2, 3, 2]

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # d_model: Dimensionality of the input.
        # num_heads: The number of attention heads to split the input into.
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        # This method reshapes the input x into the shape (batch_size, num_heads, seq_length, d_k). 
        # It enables the model to process multiple attention heads concurrently, allowing for parallel computation.
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        # After applying attention to each head separately, this method combines the results back into a 
        # single tensor of shape (batch_size, seq_length, d_model). This prepares the result for further processing.
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

# The forward method is where the actual computation happens:
# Apply Linear Transformations: The queries (Q), keys (K), and values (V) are first passed through linear transformations using the weights defined in the initialization.
# Split Heads: The transformed Q, K, V are split into multiple heads using the split_heads method.
# Apply Scaled Dot-Product Attention: The scaled_dot_product_attention method is called on the split heads.
# Combine Heads: The results from each head are combined back into a single tensor using the combine_heads method.
# Apply Output Transformation: Finally, the combined tensor is passed through an output linear transformation.

# In summary, the MultiHeadAttention class encapsulates the multi-head attention mechanism commonly used 
# in transformer models. It takes care of splitting the input into multiple attention heads, applying 
# attention to each head, and then combining the results. By doing so, the model can capture various 
# relationships in the input data at different scales, improving the expressive ability of the model.

# Position-wise Feed-Forward Networks
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# d_model: Dimensionality of the model's input and output.
# d_ff: Dimensionality of the inner layer in the feed-forward network.
# self.fc1 and self.fc2: Two fully connected (linear) layers with input and output dimensions as defined by d_model and d_ff.
# self.relu: ReLU (Rectified Linear Unit) activation function, which introduces non-linearity between the two linear layers.

# FeedForward is basically a fully connected layer, that transformer uses in both encoder and decoder. It 
# consists of two linear transformations with a ReLU activation in between. This helps in adding non-linearity
# to the model, allowing it to learn more complex patterns. 

# Linear Layers: self.linear_1 and self.linear_2 are linear transformations that project the input to a 
# higher-dimensional space ( d_ff ) and back to the original dimensionality ( d_model ).

# In summary, the PositionWiseFeedForward class defines a position-wise feed-forward neural network that 
# consists of two linear layers with a ReLU activation function in between. In the context of transformer 
# models, this feed-forward network is applied to each position separately and identically. It helps in 
# transforming the features learned by the attention mechanisms within the transformer, acting as an additional
# processing step for the attention outputs.

# Positional Encoding
# Positional Encoding is used to inject the position information of each token in the input sequence. It
# uses sine and cosine functions of different frequencies to generate the positional encoding.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        # d_model: The dimension of the model's input.
        # max_seq_length: The maximum length of the sequence for which positional encodings are pre-computed.
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# d_model: The dimension of the model's input.
# max_seq_length: The maximum length of the sequence for which positional encodings are pre-computed.
# pe: A tensor filled with zeros, which will be populated with positional encodings.
# position: A tensor containing the position indices for each position in the sequence.
# div_term: A term used to scale the position indices in a specific way.
# The sine function is applied to the even indices and the cosine function to the odd indices of pe.
# Finally, pe is registered as a buffer, which means it will be part of the module's state but will not be considered a trainable parameter.

# Summary
# The PositionalEncoding class adds information about the position of tokens within the sequence. Since
# the transformer model lacks inherent knowledge of the order of tokens (due to its self-attention 
# mechanism), this class helps the model to consider the position of tokens in the sequence. The sinusoidal
# functions used are chosen to allow the model to easily learn to attend to relative positions, as they 
# produce a unique and smooth encoding for each position in the sequence

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Parameters:
# __init__
#     d_model: The dimensionality of the input.
#     num_heads: The number of attention heads in the multi-head attention.
#     d_ff: The dimensionality of the inner layer in the position-wise feed-forward network.
#     dropout: The dropout rate used for regularization.

# Components:
#     self.self_attn: Multi-head attention mechanism.
#     self.feed_forward: Position-wise feed-forward neural network.
#     self.norm1 and self.norm2: Layer normalization, applied to smooth the layer's input.
#     self.dropout: Dropout layer, used to prevent overfitting by randomly setting some activations to zero during training.

# def forward
# Input:
#     x: The input to the encoder layer.
#     mask: Optional mask to ignore certain parts of the input.

# Processing Steps:
#     Self-Attention: The input x is passed through the multi-head self-attention mechanism.
#     Add & Normalize (after Attention): The attention output is added to the original input (residual connection), followed by dropout and normalization using norm1.
#     Feed-Forward Network: The output from the previous step is passed through the position-wise feed-forward network.
#     Add & Normalize (after Feed-Forward): Similar to step 2, the feed-forward output is added to the input of this stage (residual connection), followed by dropout and normalization using norm2.
#     Output: The processed tensor is returned as the output of the encoder layer.

# Summary:
# The EncoderLayer class defines a single layer of the transformer's encoder. It encapsulates a multi-head 
# self-attention mechanism followed by position-wise feed-forward neural network, with residual connections, 
# layer normalization, and dropout applied as appropriate. These components together allow the encoder to 
# capture complex relationships in the input data and transform them into a useful representation for downstream
# tasks. Typically, multiple such encoder layers are stacked to form the complete encoder part of a 
# transformer model

#### Making a Decoder Block for the transformers
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

# Parameters:
#     d_model: The dimensionality of the input.
#     num_heads: The number of attention heads in the multi-head attention.
#     d_ff: The dimensionality of the inner layer in the feed-forward network.
#     dropout: The dropout rate for regularization.

# Components:
#     self.self_attn: Multi-head self-attention mechanism for the target sequence.
#     self.cross_attn: Multi-head attention mechanism that attends to the encoder's output.
#     self.feed_forward: Position-wise feed-forward neural network.
#     self.norm1, self.norm2, self.norm3: Layer normalization components.
#     self.dropout: Dropout layer for regularization.

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

# Input:
#     x: The input to the decoder layer.
#     enc_output: The output from the corresponding encoder (used in the cross-attention step).
#     src_mask: Source mask to ignore certain parts of the encoder's output.
#     tgt_mask: Target mask to ignore certain parts of the decoder's input

# 1.	Self-Attention on Target Sequence: The input x is processed through a self-attention mechanism. The self-attention is performed with the current and previous token not peeking into any future tokens for the decoder. 
# 2.	Add & Normalize (after Self-Attention): The output from self-attention is added to the original x, followed by dropout and normalization using norm1.
# 3.	Cross-Attention with Encoder Output: The normalized output from the previous step is processed through a cross-attention mechanism that attends to the encoder's output enc_output. The encoder output would provide the embedding for the input (key and values) which decoder would use to based on that it will identify the next token. 
# 4.	Add & Normalize (after Cross-Attention): The output from cross-attention is added to the input of this stage, followed by dropout and normalization using norm2.
# 5.	Feed-Forward Network: The output from the previous step is passed through the feed-forward network. Linear Layer  RELU  Linear Layer to learn the non-linearity in the datasets.
# 6.	Add & Normalize (after Feed-Forward): The feed-forward output is added to the input of this stage, followed by dropout and normalization using norm3.
# 7.	Output: The processed tensor is returned as the output of the decoder layer.

# Summary:
# The DecoderLayer class defines a single layer of the transformer's decoder. It consists of a multi-head self-attention mechanism, a multi-head cross-attention mechanism (that attends to the encoder's 
# output), a position-wise feed-forward neural network, and the corresponding residual connections, layer normalization, and dropout layers. This combination enables the decoder to generate meaningful 
# outputs based on the encoder's representations, taking into account both the target sequence and the source sequence. As with the encoder, multiple decoder layers are typically stacked to form the 
# complete decoder part of a transformer model.

# 5. Combining the Encoder and Decoder layers to create the complete Transformer network

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
    # src_vocab_size: Source vocabulary size.
    # tgt_vocab_size: Target vocabulary size.
    # d_model: The dimensionality of the model's embeddings.
    # num_heads: Number of attention heads in the multi-head attention mechanism.
    # num_layers: Number of layers for both the encoder and the decoder.
    # d_ff: Dimensionality of the inner layer in the feed-forward network.
    # max_seq_length: Maximum sequence length for positional encoding.
    # dropout: Dropout rate for regularization.

        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    # And it defines the following components:
    # self.encoder_embedding: Embedding layer for the source sequence.
    # self.decoder_embedding: Embedding layer for the target sequence.
    # self.positional_encoding: Positional encoding component.
    # self.encoder_layers: A list of encoder layers.
    # self.decoder_layers: A list of decoder layers.
    # self.fc: Final fully connected (linear) layer mapping to target vocabulary size.
    # self.dropout: Dropout layer.

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    # This method is used to create masks for the source and target sequences, ensuring that padding tokens are ignored and that 
    # future tokens are not visible during training for the target sequence.

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    
    # This method defines the forward pass for the Transformer, taking source and target sequences and producing the output predictions.

    # Input Embedding and Positional Encoding: The source and target sequences are first embedded using their respective embedding layers and then added to their positional encodings.
    # Encoder Layers: The source sequence is passed through the encoder layers, with the final encoder output representing the processed source sequence.
    # Decoder Layers: The target sequence and the encoder's output are passed through the decoder layers, resulting in the decoder's output.
    # Final Linear Layer: The decoder's output is mapped to the target vocabulary size using a fully connected (linear) layer.

# Output:
# The final output is a tensor representing the model's predictions for the target sequence.

# Summary:
# The Transformer class brings together the various components of a Transformer model, including the embeddings, positional encoding, encoder layers, and decoder layers. 
# It provides a convenient interface for training and inference, encapsulating the complexities of multi-head attention, feed-forward networks, and layer normalization.
# This implementation follows the standard Transformer architecture, making it suitable for sequence-to-sequence tasks like machine translation, text summarization, etc. 
# The inclusion of masking ensures that the model adheres to the causal dependencies within sequences, ignoring padding tokens and preventing information leakage from future tokens.
# These sequential steps empower the Transformer model to efficiently process input sequences and produce corresponding output sequences.

# Training the PyTorch Transformer Model
# Sample data preparation
# For illustrative purposes, a dummy dataset will be crafted in this example. However, in a practical scenario, a more substantial dataset would be employed, and the process would involve text 
# preprocessing along with the creation of vocabulary mappings for both the source and target languages.

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

# Training the Model
# Next, the model will be trained utilizing the aforementioned sample data. However, in a real-world scenario, a significantly larger dataset would be employed, which would typically be 
# partitioned into distinct sets for training and validation purposes.

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Adam Parameters:
#     Learning Rate (lr or alpha):
#         Controls the size of the steps the optimizer takes toward minimizing the loss. A high learning rate can cause overshooting, and a low learning rate can cause slow convergence.
#         Typical default value: 0.001.

#     Beta1 (β1):
#         Controls the exponential decay rate for the first moment estimate (i.e., the mean of the gradients).
#         Typical default value: 0.9.
#         A higher value means that the moving average has a larger weight for previous gradients, which means the optimizer "remembers" the previous gradients for longer.

#     Beta2 (β2):
#         Controls the exponential decay rate for the second moment estimate (i.e., the variance of the gradients).
#         Typical default value: 0.999.
#         A value close to 1 ensures that the variance estimate is more stable over time.

#     Epsilon (ε):
#         A small value added to the denominator to prevent division by zero.
#         Typical default value: 1e-7.
#         This stabilizes the optimization by preventing any numerical issues when the second moment is very small.

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


transformer.eval()

# Generate random sample validation data
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

with torch.no_grad():
    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")