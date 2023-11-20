import torch as torch
import torch.nn as nn
import math


## Paper : Attention is all you need - https://arxiv.org/pdf/1706.03762.pdf

## Step 1 : Create Embedding layer to take input
    # First - create input embeddings for encoders. The embedding vector can be of size 512 as in the paper or more. 
    # The paper refers the dimension of the embedding vector as d_model
    # Vocabulary size is also an input to the embedding vector
    # torch has an inbuilt function called "nn.Embedding" which can be used for the model. 
    # This will take input as vocab size and embedding vector size i.e d_model size
    # Create a forward pass to take in the input and conver them into embedding * sqrt(d_model) 

    #Paper reference :
    #3.4 Embeddings and Softmax
    #Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens 
    #to vectors of dimension dmodel. We also use the usual learned linear transformation and softmax function to convert the decoder output 
    # to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax
    #linear transformation, similar to [30]. In the embedding layers, we multiply those weights by √dmodel - 512 dimensions


class Embedding_Input(nn.Module):

    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward_pass (self, input_sequence):
        return self.embedding(input_sequence) * math.sqrt(self.d_model)
    

## Step 2 : Create Positional encoding layer and add it to the embedding layer
    # Size of the vector d_model and sequence length : This is the sequence length of input text for which positional encoding to be created
    # dropout given to ensure model doesnt memorize the position
    # Then create positional encoding. For this we need a matrix of shape seq_len X d_model as the positional encoding is required
    # for the length of the sequence and the shape should match embedding vector dim which is d_model i.e 512
    # For the even position of tokens, apply sin formula and odd positions of tokens apply cosine formula of positional encoding. 
    # The forumula is applied using log space to reduce computation and stability

    #paper reference:
    # 3.5 Positional Encoding
    #Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, 
    # we must inject some information about the relative or absolute position of the
    #tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.
    #  The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed. 
    # There are many choices of positional encodings, learned and fixed [9].
    #In this work, we use sine and cosine functions of different frequencies:
    #P E(pos,2i) = sin(pos/10000**(2i/dmodel)) ---> For even token frequency
    #P E(pos,2i+1) = cos(pos/10000**(2i/dmodel)) ---> For odd token frequency
    #where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. 
    # The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesized 
    # it would allow the model to easily learn to attend by relative positions, 
    # since for any fixed offset k, P Epos+k can be represented as a linear function of P Epos
  

class Postional_Encoding(nn.Module):

    def __init__(self, d_model:int, seq_len : int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        Positional_encoding = torch.zeros(seq_len, d_model)

        #create a tensor of shape sequence length to 1 dim This is for formula 'pos'
        input_seq_position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        #create a tensor of shape of embedding dimensions i.e d_model and in this case its 512. This is for formula '10000**(2i/dmodel)'
        d_model_div = torch.exp(torch.arrange(0,d_model,2).float() * (math.log(10000.0)/d_model))

        #applying sine function to even frequency. This is for formula sin(pos/10000**(2i/dmodel))
        Positional_encoding[:, 0::2] = torch.sin(input_seq_position * d_model_div)

        #applying cosine function to odd frequency This is for formula cos(pos/10000**(2i/dmodel))
        Positional_encoding[:, 1::2] = torch.cos(input_seq_position * d_model_div)

        # adding a batch and buffer. Positional encoding is stored as a buffer instead usual tensor so as its not considered during back prop
        Positional_encoding = Positional_encoding.unsqueeze(0) # This will change the shape to (1,seq_len, d_model)
        self.register_buffer('Postional_Encoding', Positional_encoding)


    def forward_pass(self, input_sequence):
        input_sequence = input_sequence + (self.Positional_encoding[:, :input_sequence.shape[1], :].requires_grad_(False))
        return self.dropout(input_sequence)


## Step 3 : Encoder creation

    #Paper reference : 
    # 3.1 Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. 
    # The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. 
    # We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. 
    #That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
    #itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
    #layers, produce outputs of dimension dmodel = 512
    # Here - d_model = 512

## Step 3a : Layer Normalization

    ## Layer normalization is used instead of batch norm in attention. 
    ## For example, lets take a batch of 3 items, each item here is a vector of dim nX1. For each item, mean is calcualted, then std dev
    ## The result is then used to calcualte layer norm value - (x - mean) / sqrt(std + epsilon)
    ## Then there are two parameters introduced, gamma/alpha which is a multiplicative function, and beta/bias which is an additive function. 
    ## This is done to introduce some randomness in the data

class Layer_Norm(nn.Module):

    def __init__(self, eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # This is same as gamma. Multuplicative
        self.bias = nn.Parameter(torch.ones(1)) ## This is same as beta. additive 

    def forward_pass(self,input_tensor):
        mean = input_tensor.mean(dim = -1, keepdim = True)
        std = input_tensor.std(dim = -1, keepdim = True)
        return self.alpha * (input_tensor/mean) / math.sqrt(std + self.eps) + self.bias

## Step 3b : Position-wise Feed forward neural network
   ## This is two tensors with a drop out. The dimension of first layer is 512 X 2048 and the second is 2048 X 512 i.e d_model X d_ff
   ## and d_ff X d_model. 
   # The output of this will result in the same the dimension as d_model
   # This FFN will be applied for every token attention tensor position wise 
   # i.e One FFN for each Token embedding output from Multi-head attention block


   #paper reference:
   #3.3 Position-wise Feed Forward Network
   #Each of the layers in our encoder and decoder contains a fully connected feed-forward network, 
   # which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between. 
   # FFN(x) = max(0, xW1 + b1)W2 + b2 (2)
   # While the linear transformations are the same across different positions, they use different parameters from layer to layer. 
   # Another way of describing this is as two convolutions with kernel size 1.The dimensionality of input and output is d_model = 512, 
   # and the inner-layer has dimensionality d_ff = 2048
    

class feedforwardblock(nn.Module):

    def __init__(self, d_model : int, d_ff: int, dropout : float) -> None:
        super().__init__()
        self.linear_layer1 = nn.Linear(d_model, d_ff) #This takes an input of shape 512. Weight W1 and Bias B1
        self.dropout = nn.Dropout(dropout)
        self.linear_layer2 = nn.Linear(d_ff, d_model) #This takes an input of shape 2048. Weight W2 and Bias B2

    def forward_pass(self, input_tensor):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_layer2(self.dropout(torch.relu(self.linear_layer1(input_tensor))))
    

## Step 3c : Multi-Head attention 
    # Output from step 2 i.e temporal embedding (Embedding + Positional encoding) which is of dimension Seq_len and d_model is passed to
    # 3 copies of input tensors are initialized as Query (Q), Key (K) and Value (V) all with same shape Seq_len X d_model
    # 3 weight matrices are intialized with random init one each for Q, K, V called Wq, Wk, Wv all with shape d_model X d_model 
    # Q*Wq, K*Wk, V*Wv which will result in Q', K', V' with dimension Seq_len X d_model
    # The resultant tensors are split into smaller tensors of dim dk to the size of the multi-heads across dimensions of d_model to the seq_len
    # For example : Q'is split into Q1, Q2, ..Q8, K'is split into K1, K2...K8, V'is split into V1, V2...V8 
    # since the paper mentioned 8 attention heads. 
    # For each smaller tensore, attention is computed - Attention(Q',K', V') = Softmax ((Q'*K'T)/Sqrt(d_model_dim))*V'
    # This will result in a smaller attention tensor for each embedding groups, then it concatenated
    # MultiHead(Q',K',V') = Concat(Head1, Head2..Headh)*Weight. This will result in multi-head attention matrix of same shape seq_len X d_model
    # All the above need to be executed for a batch of sequences as in production, the inputs are typically batch
    # To determine number of heads, then the d_model should be divisible by number of heads so as to have equal halves
    # dk = d_model/h i.e dimension of embedding vector split by number of heads

    #Paper reference
    # 3.2 Attention :
    # An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, 
    # keys, values, and output are all vectors. The output is computed as a weighted sum of the values, 
    # where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
    
    # 3.2.1 Scaled Dot-Product Attention
    # We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and 
    # keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, 
    # divide each by √dk, and apply a softmax function to obtain the weights on the values. 
    # In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. 
    # The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as:
    # Formula : Attention(Q',K', V') = Softmax ((Q'*K'T)/Sqrt(d_model_dim))*V'
    # The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. 
    # Dot-product attention is identical to our algorithm, except for the scaling factor of sqrt(1/dk). 
    # Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. 
    # While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, 
    # since it can be implemented using highly optimized matrix multiplication code.
    # While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling 
    # for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, 
    # pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, 
    # we scale the dot products by sqrt(1/dk).

    #3.2.2 Multi-Head Attention
    #Instead of performing a single attention function with dmodel-dimensional keys, values and queries,
    #we found it beneficial to linearly project the queries, keys and values h times with different, learned
    #linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of
    #queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. 
    # These are concatenated and once again projected, resulting in the final values
    #To illustrate why the dot products get large, assume that the components of q and k are independent random
    #variables with mean 0 and variance 1. Then their dot product, q · k = for a i to dk, sum(Qi, Ki) has mean 0 and variance dk.
    #Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. 
    # a single attention head, averaging inhibits this.
    # Formula : MultiHead(Q, K, V ) = Concat(head1, ..., headh)*WO where head1 = Attention(Q',K', V')
    #Where the projections are parameter matrices WQ ∈ R(dmodel×dk) , WK ∈ R(dmodel×dk) , WV ∈ R(dmodel×dv) and WO ∈ R(hdv×dmodel).
    #In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. 
    # Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model :int, num_heads_h:int, dropout : float ) -> None:
        self.d_model = d_model
        self.num_heads_h = num_heads_h
        assert d_model / num_heads_h == 0, "Shape Error : Dimension of embedding vector is not divisible by Number of heads"


        # Initialising weight tensors for Q, K, V and Output
        self.dk = d_model // num_heads_h # This is dimension of vector attented by each head
        self.WQ = nn.Linear(d_model, d_model, bias = False) # Weight for Q created with dim d_model X d_model
        self.WK = nn.Linear(d_model, d_model, bias = False) # Weight for K created with dim d_model X d_model
        self.WV = nn.Linear(d_model, d_model, bias = False) # Weight for V created with dim d_model X d_model
        self.Wo = nn.Linear(d_model, d_model, bias = False) # Weight for Output created with dim d_model X d_model
        self.dropout(dropout)

    @staticmethod #This helps to call single attention function to be called without initialising the class
    def single_attention(query, key, value, mask, dropout = nn.Dropout):
        dk = query.shape[-1]
        #shape is (batch, num_heads, input_seq_len, dk) ---> (batch, num_heads, seq_len, seq_len)
        attention_score = (torch.matmul(query,key).transpose(-2,-1))/math.sqrt(dk) #@ sign is matrix mul in pytorch This is step 2,3,4 in Jay alammar blog
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9) # masking the forward sequence with -infinity
        attention_score = attention_score.softmax(dim = -1) #This is step 2,3,4 in Jay alammar blog.Shape:(batch, num_heads, seq_len, seq_len)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (torch.matmul(attention_score,value)), attention_score #This is step 5, 6 in Jay alammar blog# attention score is used for visualization for example : Bertviz
    
    def forward_pass(self, query, key, value, mask):
        query = self.WQ(query)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.WK(key)      # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.WV(value)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, num_heads, dk) --> (batch, num_heads, seq_len, dk)
        # This step is used to create smaller tensors for each heads
        query = query.view(query.shape[0], query.shape[1], self.num_heads_h, self.dk).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads_h, self.dk).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads_h, self.dk).transpose(1,2)

        #calcualte attention
        output, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        #combine all heads together
        output = output.transpose(1, 2).contiguous().view(output.shape[0], -1, self.num_heads_h * self.dk)

        return self.Wo(output)

## Step 3d : Residual connections i.e skip connections
    #  Paper reference :
    #  5.4 Regularization
    # Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. 
    # In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. 
    # For the base model, we use a rate of Pdrop = 0.1. 

class Residual_connection(nn.Module):

    def __init__(self, dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = Layer_Norm()

    def forward_pass(self, ouptut_from_multi_head, skip_layer):
        return ouptut_from_multi_head + self.dropout(skip_layer(self.norm(ouptut_from_multi_head)))
    

## Step 3e : Encoder block and encoder object

    #Encoder block consists of Multi-head attention, add and layer norm, position wise FFN and Residual connection
    # In case of encoder, the input embedding is applied to three tensors - Q, K, V


class Encoder_Block(nn.Module):

    def __init__(self, self_attention_block : MultiHeadAttentionBlock, FFN:feedforwardblock, dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.FFN = FFN
        self.Residual_connection = nn.ModuleList([Residual_connection(dropout) for _ in range(2)])

    def forward_pass(self, input_embedding_vector, src_mask): #src_mask is used to apply mask to input embedding
        input_embedding_vector = self.Residual_connection[0](input_embedding_vector, lambda x : self.self_attention_block(input_embedding_vector, input_embedding_vector, input_embedding_vector, src_mask))
        encoded_vector = self.Residual_connection[1](input_embedding_vector, self.FFN)
        return encoded_vector


# Encoder block is made up of many encoders and hence defining encoder as an object in itself
class Encoder(nn.Module):

    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = Layer_Norm()
    
    def forward_pass(self, input_tensor, mask):
        for layer in self.layers:
            encoded_tensor = layer(input_tensor, mask)
        return self.norm(encoded_tensor)
    
## Step 4 : Decoder creation

    #Paper reference
    #3.1 Decoder
    #Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two
    #sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
    #attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
    #around each of the sub-layers, followed by layer normalization. We also modify the self-attention
    #sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
    #masking, combined with fact that the output embeddings are offset by one position, ensures that the
    #predictions for position i can depend only on the known outputs at positions less than i.

class Decoder_Block(nn.Module):

    #In decoder, in second Multi-head attention, Q will come from decoder input and K,V will come from encoder output

    def __init__(self, self_attention_block = MultiHeadAttentionBlock, encoder_decoder_cross_attention_block = MultiHeadAttentionBlock, 
                  FFN = feedforwardblock, dropout = float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.encoder_decoder_cross_attention_block = encoder_decoder_cross_attention_block
        self.FFN = FFN
        self.Residual_connection = nn.ModuleList([Residual_connection(dropout) for _ in range(3)])

    #Src_mask - mask applied to the encoder, tgt_mask - mask applied to the decoder. This is because the paper explains about a translation 
    #task. 

    def forward_pass(self,decoder_input, encoder_output, src_mask, tgt_mask):
        #for the first layer, Q,K,V comes from the decoder input with tgt_mask
        decoder_input = self.Residual_connection[0](decoder_input, lambda x: self.self_attention_block(decoder_input,decoder_input,decoder_input,tgt_mask))
        #for the second layer, Q comes from the decoder input, K,V comes from the encoder output with src_mask
        decoder_input = self.Residual_connection[1](decoder_input, lambda x: self.encoder_decoder_cross_attention_block(decoder_input, encoder_output, encoder_output, src_mask))
        decoded_output = self.Residual_connection[2](decoder_input, self.FFN)
        return decoded_output
    

class Decoder(nn.Module):

    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = Layer_Norm()

    def forward_pass(self, decoder_input, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            decoder_input = layer(decoder_input, encoder_output, src_mask, tgt_mask)
        return self.norm(decoder_input)
    
## Step 5 : Linear layer creation and softmax i.e Projection layer
# This is the final output layer with logit and softmax. This layer is adjusted with temperature in case of LLMs for the model to be creative
# The class of prediction will be to the tune of Vocab size.

class Linear_projection(nn.Module):

    def __init__(self, d_model :int, vocab_size : int) -> None:
        super().__init__()
        self.liner_proj = nn.Linear(d_model, vocab_size)

    def forward_pass(self, decoder_output):
        #(Batch size, seq_len, d_model) --> (Batch size, seq_len, vocab_size)
        return torch.log_softmax(self.liner_proj(decoder_output), dim = -1)
    

##Step 6 : Create Transformer Architecture
#This is the class where the model architecture is created using the classes and functions defined as above
#Since the paper is about translation task, this will require two embeddings, One for source language, one for target language
#This class is used to initialize the transformer architecture for model training and testing

class Transformer(nn.Module):

    def __init__(self, encoder : Encoder, decoder : Decoder, source_embed : Embedding_Input, target_embed : Embedding_Input, 
                 source_positional_encode : Postional_Encoding, target_positional_encode : Postional_Encoding, 
                 output_layer : Linear_projection) -> None:
        super().__init()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.source_positional_encode = source_positional_encode
        self.target_positional_encode = target_positional_encode
        self.output_layer = output_layer

    def encode(self, input_seq, src_mask):
        #(batch, seq_len, d_model)
        input_embed = self.source_embed(input_seq)
        input_embed = self.source_positional_encode(input_embed)
        return self.encoder(input_embed, src_mask)
    
    def decode(self, encoder_output : torch.tensor, src_mask : torch.tensor, tgt_embed:torch.tensor, tgt_mask : torch.tensor):
        #(batch, seq_len, d_model)
        tgt_embed = self.target_embed(tgt_embed)
        tgt_embed = self.target_positional_encode(tgt_embed)
        return self.decoder(tgt_embed, encoder_output, src_mask, tgt_mask)
    
    def transformer_setup(source_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, d_model = 512, num_blocks : int = 6, 
                          num_heads : int = 8, dropout : float = 0.1, d_ff : int = 2048):
        #Encoder Block set up
        # step 1 : create input with embedding layers
        source_embed = Embedding_Input(d_model, source_vocab_size)
        target_embed = Embedding_Input(d_model, tgt_vocab_size)

        #step 2 : create positional encodings
        source_positional_encode = Postional_Encoding(d_model, src_seq_len, dropout)
        target_positional_encode = Postional_Encoding(d_model, tgt_seq_len, dropout)

        #step 3 : create encoder blocks. This will be iterated basis the num_blocks
        encoder_blocks = []
        for _ in range(num_blocks):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
            encoder_FFN_block = feedforwardblock(d_model, d_ff, dropout)
            encoder_block = Encoder_Block(encoder_self_attention_block, encoder_FFN_block, dropout)
            encoder_blocks.append(encoder_block)

        
        #step 4 : create decoder blocks. This will be iterated basis the num_blocks
        decoder_blocks = []
        for _ in range(num_blocks):
            decoder_self_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
            encoder_decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, num_heads, dropout)
            decoder_FFN_block = feedforwardblock(d_model, d_ff, dropout)
            decoder_block = Decoder_Block(decoder_self_attention_block, encoder_decoder_cross_attention_block, decoder_FFN_block, dropout)
            decoder_blocks.append(decoder_block)

        #step 5 : create Encoder and Decoder of Transformers which consists of many encoder_blocks and decoder_blocks
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))

        #step 6 : create Final output linear FFN layer with Softmax
        Linear_layer = Linear_projection(d_model, tgt_vocab_size)

        #step 7 : Initialize transformers circuit
        Transformers_model = Transformer(encoder, decoder, source_embed, target_embed, source_positional_encode, target_positional_encode, 
                                         Linear_layer)
        
        #step 8 : Initialize the parameters
        for parameters in Transformers_model.parameters():
            if parameters.dim() > 1:
                nn.init.xavier_uniform_(parameters)
        return Transformers_model

    











    

















