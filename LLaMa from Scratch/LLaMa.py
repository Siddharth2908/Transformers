##LLaMa is trained for next token prediction task. Since each output is dependent on the previous tokens, this is also called as Causal Models

import torch as torch
import math as math
import dataclasses
import typing
import config as config


## Paper : LLaMa - An open and efficient Foundation Language Models https://arxiv.org/abs/2302.13971

##Summary - Key differences to "Attention is all you need" paper

'''Key differences between LLaMa and Transformer_base as in Attention is all you need paper
1. LLaMa uses RoPE as against Sinusoidal positional embedding (SPE)
2. RoPE is applied to Q and K as against SPE applied to output of embedding layer & before passing it to attention block
3. Root Mean Squear Norm (RMS Norm) is applied on the embeddings before passing it to the attention blocks whereas in Transformer_base, 
layer norm is applied on the output of attention
4. Instead of Multi-head attention, LLaMa uses Grouped Multi-Query Attention with KV Cache
5. Instead of ReLU, LLaMa uses SwiGLU
6. LLaMa uses embedding size of 4096 as against 512
'''

## Step 1 : Create class to represent the parameters of the model
@dataclasses.dataclass
class model_parameters:
    model_config = config.get_config()
    dim = int(model_config['dim'])
    n_layers = int(model_config['n_layers'])
    
    #The grouped multi-query attention need not have the same number of heads for query, key and value unlike Transformer_base. Hence these
    #can be different
    n_q_heads = int(model_config['n_q_heads']) # This is only for the Query 
    n_kv_heads : typing.Optional[int] = model_config['n_kv_heads'] # This is for Keys (K) and the values (V)
    vocab_size = int(model_config['vocab_size']) #This will be used when we load the tokeniser. LLaMa uses ""Sentencepiece"
    
    # The below two parameters represents the hidden layer in FFN. In LLaMa, the num of heads are reduced in Grouped Query Attention(GQA)
    # and to compensate that, the layers are increased in FFN so as overall the number of parameters are maintained
    n_layer_ffn = int(model_config['n_layer_ffn'])
    ffn_num_layer_multiplier : typing.Optional[float] = model_config['ffn_num_layer_multiplier']
    epsilon_norm = float(model_config['epsilon_norm']) 
    
    # The below parameters are used for KV cache
    kv_max_batch_size = int(model_config['kv_max_batch_size'])
    kv_max_seq_len = int(model_config['kv_max_seq_len'])
    
    device:str = None

## Step 2 : Creating RoPE

## RoPE are applied on Q and K before applying attention. RoPE uses relative positional encoding
## Absolute postional encoding are fixed vectors that are added to the embedding of a token to represent its absolute position in the sentence
## This means, absolute positional encoding deals with one token at a time. The embedding value doesnt change for that sequence
## Retational positional encoding deals with two tokens at a time and it is involved when we calculate the attention. Since the attention mechanisms
## calculates the intensity of how much two words are related to each other, relative poistional embeddings tells the attention mechanism the 
## distance between the two words. So for the given two words i.e the pair, a vector representation is created which represents their distance. 
## This becomes very useful for attention heads while calculating the attention of a word to another word. 
# 
# Paper reference : https://arxiv.org/abs/2104.09864 - Roformer
# 
# In RoPE, the dot product used in the attention mechanism is a type of inner product
# Section 3.2 :  Rotary position embedding
# 3.2.1 A 2D case
# We begin with a simple case with a dimension d = 2. Under these settings, we make use of the geometric property
# of vectors on a 2D plane and its complex form to prove (refer Section (3.4.1) for more details) that a solution to our
# formulation Equation (11) is 
# fq(xm, m) = (Wqxm)e**imθ
# fk(xn, n) = (Wkxn)e**inθ
# g(xm, xn, m − n) = Re[(Wqxm)(Wkxn)∗e**i(m−n)θ]
# 3.4.2 Computational effiecient realization of rotary matrix multiplication
# Key properties of RoPE
# 1. Long term decay - The farther the tokens are, smaller the embedding. This is intutive to what we may want
# 2. RoPE is with linear attention

def rotary_matrix(dim_head : int, seq_len : int, device : str, theta_param : float = 10000.0): #This 10000 is from paper sec 3.3
    assert head_dim % 2 == 0, "Embedding dimension must be divisible by 2. Pls select even number"# RoPE can be applied if and only if the embedding is even number. 
    # Now create theta using the formula
    # theta_i =  10000**(-2(i-1)/d) for i = [1,2,...., dim/2]
    # the below are of shape (dim_head / 2)
    theta_arange = torch.arange(0, dim_head/2).float() #This is for numerator in the eqn (i-1). This helps to create a series
    theta_i = 1.0 / ((theta_param ** theta_arange)/dim_head).to(device) # This is the implementation of the theta_i

    #constructing the positions (the value 'm' which is the position of a token in a sequence)
    #Using this, we have to construct m*theta_i for all m = [m1, m2, m3...,m_Seq_len]
    # this is of shape (seq_len)
    m = torch.arange(seq_len, device=device)
    # multiply the list m with the list theta_i using torch.outer
    m_theta_i = torch.outer(m, theta_i).float() #This function takes each value of m and multiply with all values of theta_i
    # shape (seq_len) outer product dim_head/2 --> (seq_len, dim_head/2)
    # the below formula is the implementation of Euler formula for m_theta i.e e**ix = cosx + isinx
    # In this scenario, cos_m_theta and sin_m_theta is acheived by formula i.e R*e**imx = Rcosmx + Risinmx. 
    # In this scenario R is taken as 1 using torch.ones_like
    # This will create the rotation matrix which can then be multiplied with rms normalised input embeddings of Q and K
    m_theta_i_complex = torch.polar(torch.ones_like(m_theta_i),m_theta_i)
    return m_theta_i_complex
    
    # There are 4 transformations to be done 
    # Transformation 1 - Embedding vector to two dimensions when 
    # Transformation 2 - The output of transformation1  is then transformed to a single tensor where dim_1 is real and dim_2 is imaginary
    # and multiplied with a matrix of e**imx 
    # Transformation 3 - The resultant matrix is then squished into two dim pairs by arranging real valued in one column and imaginary in another
    # Transformation 4 - The final matrix is then flattened to the original embedding structure dimension
    # Paper reference : Figure 1: Implementation of Rotary Position Embedding(RoPE) in Roformer paper

def rotary_matrix_embedding(x_pos : torch.Tensor, m_theta_i_complex : torch.Tensor, device : str ):
    #Perform transformation 1 and 2
    # (Batch, seq_len, H, dim_head) --> (Batch, seq_len, H, dim_head/2)
    x_pos_complex = torch.view_as_complex(x_pos.float().reshape(*x_pos.shape[:-1],-1,2))
    # (seq_len, dim_head/2) ---> (1, seq_len, 1, dim_head/2)
    m_theta_i_complex = m_theta_i_complex.unsqueeze(0).unsqueeze(2)
    #Perform transformation 3
    # (Batch, seq_len, H, dim_head/2) * (1, seq_len, 1, dim_head/2) = (Batch, seq_len, H, dim_head)
    x_pos_rotated = x_pos_complex * m_theta_i_complex
    # (Batch, seq_len, H, dim_head, 2)
    x_rope_matrix = torch.view_as_real(x_pos_rotated)
    x_rope_matrix = x_rope_matrix.reshape(*x_pos.shape)
    return x_rope_matrix.type_as(x_pos).to(device)

#Step 3 - Create RMS Normalization
# Paper reference 

def 

















## Step 2 : Create the main class reprenting the LLaMa main architecture except softmax

class LLaMa(torch.nn.Module):

    def __init__(self, param : model_parameters) -> None:
        super().__init__()

        assert param.vocab_size != -1, "Vocab size cannot be -1, pls set the vocab size"

        self.param = param
        self.vocab_size = param.vocab_size
        self.n_layers = param.n_layers
        self.token_embeddings = torch.nn.Embedding(self.vocab_size, param.dim)

        #Create Nx encoder blocks 
        self.layers = torch.nn.ModuleList()
        for _ in range(param.n_layers):
            self.layers.append(LLaMa_Encoder(param))

        #Create RMS Norm layer
        self.norm = RMSNorm(param.dim, eps = param.epsilon_norm)

        #create output layer
        self.output = torch.nn.Linear(param.dim, self.vocab_size, bias = False)

        #Pre compute the frequencies of Rotary PE
        self.freqs_complex = rotary_matrix(self.param.dim // self.param.n_q_heads, 
                                                       self.param.kv_max_seq_len * 2, device = self.param.device)
        
    def forward_pass(self, tokens:torch.Tensor, start_pos:int):
        # The input is (batch , seq_len)
        # In LLaMa, the sequence length is always "1" as the model is predicting the next token and with KV Cache
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time will be processed, pls change the sequence length"

        #This model is good only for inference. During triaing, the sequence length is not 1 and its the entire sequence
        # During training, KV cache is not used
        #In this example, pre-trained LLaMa weights are loaded and hence only inference tasks are performed

        #Get the input and convert them from shape (Batch, seq_len) ---> (Batch, seq_len, emb_dim). This is 4096 in LLaMa
        embeddings = self.token_embeddings(tokens)

        #Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]

        #Applying the encoder block
        for layer in self.layers:
            hidden_layer = layer(embeddings, start_pos, freqs_complex)
        hidden_layer = self.norm(hidden_layer)
        output = self.output(hidden_layer).float()
        return output
    




























        





        


