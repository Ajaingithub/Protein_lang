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