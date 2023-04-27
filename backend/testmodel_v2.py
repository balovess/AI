# 定义 input_ids 和 output_ids
# 对 input_ids 进行编码器的嵌入和位置编码
# 对 input_ids 进行记忆机制和相对位置编码
# 对 output_ids 进行解码器的嵌入和位置编码
# 对 output_ids 进行相对位置编码
# 对 input_ids 和 output_ids 进行 Transformer 编码器和解码器的处理
# 对 input_ids 和 output_ids 进行编码器-解码器的注意力机制
# 对 input_ids 和 output_ids 进行线性投影
# 返回 output_logits
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerMAX(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nhid, nlayers, dropout, max_len,batch_size,seq_len,memory_size=128):
        super(TransformerMAX, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.max_len = max_len
        self.batch_size = batch_size
        self.seq_len =seq_len
        # Transformer Encoder Layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.nhid, dropout=self.dropout)
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=self.nlayers)
        # Transformer Decoder Layer
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.nhid, dropout=self.dropout)
        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.nlayers)
        # Linear Projection
        self.linear_projection = nn.Linear(self.d_model, self.vocab_size)
        # Token Embedding + Positional Encoding for encoder and decoder
        self.token_embedding_encoder = nn.Embedding(self.vocab_size, self.d_model)
        self.token_embedding_decoder = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding_encoder = nn.Embedding(self.seq_len, self.d_model)
        self.pos_embedding_decoder = nn.Embedding(self.seq_len, self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)   
        #初始化记忆储存
        self.memory_size = memory_size
        # Use torch.Tensor instead of nn.Parameter for memory
        self.memory = torch.zeros(self.batch_size, self.memory_size, self.d_model) # 初始化为零向量
        nn.init.xavier_normal_(self.memory)
        #初始化层归一
        self.layer_norm = torch.nn.LayerNorm(self.d_model)       
        # Multihead Attention for encoder and decoder
        self.multihead_attention_encoder = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.nhead, dropout=self.dropout)
        self.multihead_attention_decoder = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.nhead, dropout=self.dropout)
        # Encoder-Decoder Attention
        self.encoder_decoder_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.nhead, dropout=self.dropout)
        
        # Transformer Encoder and Decoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, self.nhead, self.nhid, self.dropout), 
            self.nlayers
            )
        
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.d_model, self.nhead, self.nhid, self.dropout), 
            self.nlayers
            )
        
 
        relative_positional_encoding_encoder_list = []
        
        for i in range(self.nlayers):
             relative_positional_encoding_encoder_list.append(nn.Parameter(torch.randn(2 * max_len - 1, d_model)))
             nn.init.xavier_normal_(relative_positional_encoding_encoder_list[i])
        
        self.relative_positional_encoding_encoder = nn.ParameterList(relative_positional_encoding_encoder_list)
        
 
        
        relative_positional_encoding_decoder_list = []
        
        for i in range(self.nlayers):
             relative_positional_encoding_decoder_list.append(nn.Parameter(torch.randn(2 * max_len - 1, d_model)))
             nn.init.xavier_normal_(relative_positional_encoding_decoder_list[i])
        
        self.relative_positional_encoding_decoder = nn.ParameterList(relative_positional_encoding_decoder_list)
        
        
        # Routing Mechanism
        # Use one routing weight for each layer instead of each head
        self.routing_weight = nn.Parameter(torch.randn(self.nlayers, 1))
        nn.init.xavier_normal_(self.routing_weight)
        
    def forward(self, input_ids, output_ids):
        
        seq_len = input_ids.size(1)
        
        # Token Embedding + Positional Encoding for encoder
        token_embeddings_encoder = self.token_embedding_encoder(input_ids)  # (batch_size, seq_len, d_model)
        
        pos_embeddings_encoder = self.pos_embedding_encoder(torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(input_ids.size(0), 1))  # (batch_size, seq_len, d_model)
        
        embeddings_encoder = token_embeddings_encoder + pos_embeddings_encoder  # (batch_size, seq_len, d_model)
        
        # Memory Attention
        memory_attention = torch.nn.functional.linear(embeddings_encoder, self.memory.transpose(1, 2))  # (batch_size, seq_len, memory_size)
        
        memory_attention_weights = F.softmax(memory_attention / math.sqrt(self.d_model), dim=-1)  # (batch_size, seq_len, memory_size)
        
        memory_attention_output = torch.nn.functional.linear(memory_attention_weights, self.memory)  # (batch_size, seq_len, d_model)
        
        embeddings_encoder = embeddings_encoder + memory_attention_output

        # 对增强的输入向量进行层归一化
        embeddings_encoder = self.layer_norm(embeddings_encoder)

        # Memory Update
        memory_update = torch.nn.functional.linear(embeddings_encoder.transpose(1, 2), self.memory) # (batch_size, d_model, memory_size)

        memory_update_weights = F.softmax(memory_update / math.sqrt(self.d_model), dim=-1) # (batch_size, d_model, memory_size)

        memory_update_output = torch.nn.functional.linear(memory_update_weights.transpose(1, 2), embeddings_encoder) # (batch_size, memory_size, d_model)

        self.memory = self.memory + memory_update_output # 更新记忆存储
        # 相对位置编码 for encoder
        
        relative_positions_encoder = torch.arange(seq_len) - torch.arange(seq_len).unsqueeze(0) + self.max_len - 1
        
        embeddings_list_encoder=[]
        for i in range(self.nlayers):
            embeddings_list_encoder.append(embeddings_encoder)

        for i in range(self.nlayers):
            relative_positional_embeddings_encoder = torch.index_select(self.relative_positional_encoding_encoder[i].unsqueeze(0).repeat(input_ids.size(0), 1), 1,
                                                            relative_positions_encoder.view(-1))  
            embeddings_list_encoder[i] += relative_positional_embeddings_encoder
            
            embeddings_list_encoder[i] += self.routing_weight[i] * embeddings_list_encoder[i-1]
            embeddings_list_encoder[i] += self.routing_weight[i] * embeddings_list_encoder[i+1]
            embeddings_list_encoder[i] += self.routing_weight[i] * embeddings_list_encoder[i]
            embeddings_list_encoder[i] /= self.routing_weight[i].sum()
            
            embeddings_list_encoder[i] += self.transformer_encoder(embeddings_list_encoder[i])
            
            output_logits += self.linear_projection(embeddings_list_encoder[i])
            
            output_logits /= math.sqrt(i+1)

        # Masking for decoder input
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # Token Embedding + Positional Encoding for decoder
        token_embeddings_decoder = self.token_embedding_decoder(output_ids)  # (batch_size, seq_len, d_model)
        
        pos_embeddings_decoder = self.pos_embedding_decoder(torch.arange(seq_len, device=output_ids.device).unsqueeze(0).repeat(output_ids.size(0), 1))  # (batch_size, seq_len, d_model)
        
        embeddings_decoder = token_embeddings_decoder + pos_embeddings_decoder  # (batch_size, seq_len, d_model)
        
        # 相对位置编码 for decoder
        
        relative_positions_decoder = torch.arange(seq_len) - torch.arange(seq_len).unsqueeze(0) + self.max_len - 1
        
        embeddings_list_decoder=[]
        for i in range(self.nlayers):
            embeddings_list_decoder.append(embeddings_decoder)

        for i in range(self.nlayers):
            relative_positional_embeddings_decoder = torch.index_select(self.relative_positional_encoding_decoder[i].unsqueeze(0).repeat(output_ids.size(0), 1), 1,relative_positions_decoder.view(-1))  
            embeddings_list_decoder[i] += relative_positional_embeddings_decoder
            
            embeddings_list_decoder[i] += self.routing_weight[i] * embeddings_list_decoder[i-1]
            embeddings_list_decoder[i] += self.routing_weight[i] * embeddings_list_decoder[i+1]
            embeddings_list_decoder[i] += self.routing_weight[i] * embeddings_list_decoder[i]
            embeddings_list_decoder[i] /= self.routing_weight[i].sum()
            embeddings_list_decoder[i] += self.transformer_decoder(embeddings_list_decoder[i], embeddings_list_encoder[i], tgt_mask=mask)

            output_logits += self.linear_projection(embeddings_list_decoder[i])
            
            output_logits /= math.sqrt(i+1)
            # Transformer Encoder
            embeddings_list_encoder[i] = self.transformer_encoder(embeddings_list_encoder[i])
            # Transformer Decoder
            embeddings_list_decoder[i] = self.transformer_decoder(embeddings_list_decoder[i], embeddings_list_encoder[i], tgt_mask=mask)
            # Encoder-Decoder Attention
            encoder_decoder_attention = self.encoder_decoder_attention(embeddings_list_decoder[-1], embeddings_list_encoder[-1], embeddings_list_encoder[-1], attn_mask=mask)
            encoder_decoder_attention_output = self.linear_projection(encoder_decoder_attention)
            output_logits += encoder_decoder_attention_output
            # Linear Projection
            output_logits += self.linear_projection(embeddings_list_encoder[i])
            output_logits += self.linear_projection(embeddings_list_decoder[i])

        return output_logits
        



       