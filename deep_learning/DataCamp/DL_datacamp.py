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
