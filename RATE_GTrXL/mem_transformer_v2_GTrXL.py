
import sys
import math
import functools
from einops import rearrange
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import efficientnet_b0, mobilenet_v2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


init_w = 1e-3
def initialize_weights(layer):
            if isinstance(layer, nn.Linear):
                torch.nn.init.uniform_(layer.weight, -init_w, init_w)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

 
class Model(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, dim_feedforward, conv_lat_dim,  norm_first, init, dropout, wo_ffn, use_gate, gate_mode, separate, critic_mode, actor_mode, state_dim, act_dim, obs_mode, algo):
        super(Model, self).__init__()
        
        self.algo = algo
        self.actor_mode = actor_mode
        self.separate = separate
        self.d_model = d_model
        self.num_layers = num_layers
        self.critic_mode = critic_mode
        self.obs_mode = obs_mode
        self.conv_lat_dim = self.d_model - self.d_model//4
        
        self.critic_act_encoder = nn.Linear(act_dim, d_model)  # он есть только у критика вне зависимости от конфигурации модели 
        self.pos_enc = PositionalEncoding(d_model)
        
        if obs_mode != 'state':
            self.convolution = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(4, 4), #if image_size[0] == 128 and image_size[1] == 128 else nn.MaxPool2d(2, 2),  # [32, 32]
                nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [16, 16]
                nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [8, 8]
                nn.Conv2d(64, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # [4, 4]
                nn.Conv2d(64, 64, 1, padding=0, bias=True), nn.ReLU(inplace=True),
                nn.Flatten(1),
                nn.Linear(1024, self.conv_lat_dim))
        
        # if obs_mode != 'state':
        #     self.convolution = mobilenet_v2(pretrained=True)
        #     self.convolution.classifier = nn.Sequential(
        #         nn.Dropout(p=0.2, inplace=True),
        #         nn.Linear(in_features=1280, out_features=self.conv_lat_dim, bias=True)
        #         )
        #     for param in self.convolution.parameters():
        #         param.requires_grad = False    
        #     for param in self.convolution.classifier.parameters():
        #         param.requires_grad = True
        
        
        if separate:
            # Universal step if Agent has separated structure
            self.state_enc = nn.Linear(state_dim, self.d_model//4)
            self.actor_state_encoder = nn.Linear(state_dim, d_model) if obs_mode == 'state' else nn.Linear(self.d_model, d_model)
            self.critic_state_encoder = nn.Linear(state_dim, d_model)

            if actor_mode == 'Trans':
                print('Trans actor!')
                self.trans_layers = nn.ModuleList()
                for layer in range(self.num_layers):
                    self.trans_layers.append(CustomTransformerEncoder(d_model, n_heads, dim_feedforward, dropout, wo_ffn, norm_first, use_gate, gate_mode, actor_mode, layer_num=None).to(device))
            elif actor_mode == 'Diff-Trans':
                print('Diff-Trans actor!')
                self.trans_layers = nn.ModuleList()
                for layer in range(self.num_layers):
                    self.trans_layers.append(CustomTransformerEncoder(d_model, n_heads, dim_feedforward, dropout, wo_ffn, norm_first, use_gate, gate_mode, actor_mode, layer_num=layer).to(device))    
            elif actor_mode == 'LSTM':
                print('LSTM actor!')
                self.actor_transformer_encoder = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
            
            
            
            # Select Critics architecture
            if critic_mode == 'FC':
                self._q1 = nn.Sequential(
                        nn.Linear(state_dim + act_dim, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, 1)
                    )
                self._q2 = nn.Sequential(
                        nn.Linear(state_dim + act_dim, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, 1)
                    )
            if critic_mode == 'Trans':
                self.critic_trans_layers = nn.ModuleList()
                for layer in range(self.num_layers):
                    self.critic_trans_layers.append(CustomTransformerEncoder(d_model, n_heads, dim_feedforward, dropout, wo_ffn, norm_first, use_gate, gate_mode, critic_mode, layer_num=None))
            elif critic_mode == 'Diff-Trans':
                self.critic_trans_layers = nn.ModuleList()
                for layer in range(self.num_layers):
                    self.critic_trans_layers.append(CustomTransformerEncoder(d_model, n_heads, dim_feedforward, dropout, wo_ffn, norm_first, use_gate, gate_mode, critic_mode, layer_num=layer))   
            elif critic_mode == 'LSTM':
                self.critic_transformer_encoder = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
            # else:
            #     raise TypeError("Invalid critic mode !!!")
        else:
            
            self.state_encoder = nn.Linear(state_dim, d_model)
            
            if actor_mode == 'Trans':
                self.transformer_encoder = CustomTransformerEncoder(d_model, n_heads, dim_feedforward, dropout, wo_ffn, norm_first, use_gate, gate_mode)
            # elif actor_mode == 'Mamba':
            #     self.transformer_encoder = Mamba_Block(d_model)
            else:
                raise TypeError("Invalid backbone mode !!!")    

        
        
        if self.algo == 'TD3' and actor_mode != 'FC':
            # self.actor_head = nn.Sequential(
            #     nn.Linear(d_model, d_model),
            #     nn.ReLU(),
            #     nn.Linear(d_model, d_model//2),
            #     nn.ReLU(),
            #     nn.Linear(d_model//2, act_dim),
            #     nn.Tanh()
            #     )
            # print('New  Head !')
            self.actor_head = nn.Sequential(
                nn.Linear(d_model, act_dim),
                nn.Tanh()
                )
        elif self.algo == 'TD3' and actor_mode == 'FC':
            self.actor_head = nn.Sequential(
                    nn.Linear(state_dim, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, act_dim),
                    nn.Tanh()
                    )  
        elif self.algo == 'SAC':
            self.mean_actor_head = nn.Sequential(
                nn.Linear(d_model, act_dim))
            self.log_std_actor_head = nn.Sequential(
                nn.Linear(d_model, act_dim))
        
        # БЛОК ОПРЕДЕЛЕНИЯ ГОЛОВЫ КРИТИКА
        if critic_mode != 'FC':
            self.critic_head1 = nn.Sequential(
                nn.Linear(d_model*2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1)
                )
            self.critic_head2 = nn.Sequential(
                nn.Linear(d_model*2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1)
                )
            
        # MODEL ARCHITECTURE SUMMARY
        print(f"Separate:  {self.separate}")
        print(f"actor_mode:  {self.actor_mode}")  
        print(f"critic_mode:  {self.critic_mode}")
        
        # if init:    ### ПЕРЕДЕЛАТЬ ВСЮ ЛОГИКУ ИНИЦИАЛИЗАЦИИ!!!!!!!!
        #     if critic_mode != 'FC':
        #         self.critic_head1.apply(initialize_weights)
        #         self.critic_head2.apply(initialize_weights)
        #     if self.algo == 'TD3':
        #         self.actor_head.apply(initialize_weights)
        #         self._q1.apply(initialize_weights)
        #         self._q2.apply(initialize_weights)
            

    
    def FC_forward1(self, s, a): #s = bs,context,11    a=bs,3
        a = a.unsqueeze(1)
        s = s[:,-1,:].unsqueeze(1)
        input = torch.cat((s,a), dim=-1)
        output = self._q1(input)
        return output.reshape(-1, 1)
    def FC_forward2(self, s, a): #s = bs,context,11    a=bs,3
        a = a.unsqueeze(1)
        s = s[:,-1,:].unsqueeze(1)
        input = torch.cat((s,a), dim=-1)
        output = self._q2(input)
        return output.reshape(-1, 1)


    def critic_forward(self, state, action):
        if self.separate:
            if self.critic_mode == 'FC':
                return self.FC_forward1(state, action), self.FC_forward2(state, action)

            state_embeddings = self.critic_state_encoder(state)  # state_embeddings n_e, b_s, cont, d_m
            state_embeddings = self.pos_enc(state_embeddings)
            if len(state_embeddings.shape) > 3:
                n_e, bs, cont, d_m = state_embeddings.shape
                state_embeddings = state_embeddings.view(-1, cont, d_m) # плющим для аттеншена n_e*bs, context, d_model
            
            if self.critic_mode == 'Trans' or self.critic_mode == 'Mamba':
                for _ in range( self.num_layers ):
                    state_embeddings = self.critic_transformer_encoder(state_embeddings) # state_embeddings n_e, b_s, cont, d_m
            elif self.critic_mode == 'LSTM':
                state_embeddings = self.critic_transformer_encoder(state_embeddings)[0] # state_embeddings n_e, b_s, cont, d_m

            state_embeddings = state_embeddings.view(n_e, bs, cont, d_m)  #восттанавливаем обратно num_envs, bs, context, d_model
            
            state_embedding = state_embeddings[:,-1,:] if len(state_embeddings.shape) <= 3 else state_embeddings[:,:,-1,:] #n_e,d_m | n_e,bs,d_m
        
            action_embedding = self.critic_act_encoder(action)                          #n_e,d_m | n_e,bs,d_m
        
            critic_input = torch.cat((state_embedding, action_embedding), dim=-1)      #n_e,2*d_m | n_e,bs,2*d_m     
        
            q_s_a_1 = self.critic_head1(critic_input) #n_e,1 | n_e,bs,1
            q_s_a_2 = self.critic_head2(critic_input)  

            return q_s_a_1, q_s_a_2


        else:
            state_embeddings = self.state_encoder(state)  
            state_embeddings = self.pos_enc(state_embeddings)  
            
            if len(state_embeddings.shape) > 3:
                n_e, bs, cont, d_m = state_embeddings.shape
                state_embeddings = state_embeddings.view(-1, cont, d_m) # плющим для аттеншена num_envs*bs, context, d_model
            
            for _ in range( self.num_layers ):
                    state_embeddings = self.transformer_encoder(state_embeddings)

            state_embeddings = state_embeddings.view(n_e, bs, cont, d_m)  #восттанавливаем обратно num_envs, bs, context, d_model
            
            state_embedding = state_embeddings[:,-1,:] if len(state_embeddings.shape) <= 3 else state_embeddings[:,:,-1,:] #n_e,d_m | n_e,bs,d_m
        
            #action_embedding = self.critic_act_encoder(action)                                        #n_e,d_m | n_e,bs,d_m
            action_embedding = action
        
            critic_input = torch.cat((state_embedding, action_embedding), dim=-1)               # n_e,2*d_m | n_e,bs,2*d_m
        
            q_s_a_1 = self.critic_head1(critic_input)     # n_e,1 | n_e,bs,1
            q_s_a_2 = self.critic_head2(critic_input)     # n_e,1 | n_e,bs,1

            return q_s_a_1, q_s_a_2
     
    def Q1(self, state, action):
        return self.critic_forward(state, action)[0]   
    
    def _log(t, eps):
        return torch.log(t.clamp(min = eps))
    
    def actor_forward(self, state, image_state=None, show_percentage=False):#, image_state , img_state=True):  # state: num_envs, bs, context, s_d
        if self.algo == 'TD3' and self.actor_mode == 'FC':
            return self.actor_head(state)[:,-1,:] if len(state.shape) <= 3 else self.actor_head(state)[:,:,-1,:]
        
        else:
            if image_state != None:
                n_e, bs, cont, h, w, c = image_state.shape
                image_state = torch.permute(image_state, (0,1,2,5,3,4))
                img_state = self.convolution(image_state.reshape(n_e*bs*cont, c, h, w)).reshape(n_e, bs, cont, self.conv_lat_dim) #n_e, bs, cont, d_m-d_m//4
                state = self.state_enc(state) #n_e, bs, cont, d_m//4
                state = torch.cat((state, img_state), dim=-1) #n_e, bs, cont, d_m

            if self.separate:
                state_embeddings = self.actor_state_encoder(state)  # num_envs, bs, context, d_model
            else:
                state_embeddings = self.state_encoder(state)

            state_embeddings = self.pos_enc(state_embeddings)
            
            if len(state_embeddings.shape) > 3:
                n_e, bs, cont, d_m = state_embeddings.shape
                state_embeddings = state_embeddings.view(-1, cont, d_m) # плющим для аттеншена num_envs*bs, context, d_model
            
            if self.separate:
                if self.actor_mode in ['Trans', 'Diff-Trans']:
                    for i in range( self.num_layers ):
                        state_embeddings = self.trans_layers[i](state_embeddings)
                elif self.actor_mode == 'LSTM':
                    state_embeddings = self.actor_transformer_encoder(state_embeddings)[0]        
            else:
                for _ in range( self.num_layers ):
                    state_embeddings = self.transformer_encoder(state_embeddings)

            
            state_embeddings = state_embeddings.view(n_e, bs, cont, d_m)  #восттанавливаем обратно num_envs, bs, context, d_model
            if self.algo == 'TD3':
                action = self.actor_head(state_embeddings)[:,-1,:] if len(state_embeddings.shape) <= 3 else self.actor_head(state_embeddings)[:,:,-1,:]
                if show_percentage:
                    return action
                else:
                    return action
            
            elif self.algo == 'SAC':
                state_embedding = state_embeddings[:,-1,:] if len(state_embeddings.shape) <= 3 else state_embeddings[:,:,-1,:]
                mu = self.mean_actor_head(state_embedding)
                sigma = self.log_std_actor_head(state_embedding)
                sigma = sigma.sigmoid().clamp(min = 1e-20)


                sampled_cont_actions = mu + sigma * torch.randn_like(sigma)
                squashed_cont_actions = sampled_cont_actions.tanh()
                
                cont_log_prob = torch.distributions.Normal(mu, sigma).log_prob(sampled_cont_actions)
                cont_log_prob = cont_log_prob.clamp(min = 1e-20)
                cont_log_prob = torch.log(cont_log_prob)
                cont_log_prob = torch.sum(cont_log_prob, -1).unsqueeze(1)
                
                return squashed_cont_actions, cont_log_prob
            


    

###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################




class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout, wo_ffn, norm_first, use_gate, gate_mode, mode, layer_num=None):
        super(CustomTransformerEncoder, self).__init__()
        
        self.norm_first = norm_first
        self.use_gate = use_gate
        self.wo_ffn = wo_ffn
        self.mode = mode
        
        if mode == 'Trans':
            self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        elif mode == 'Diff-Trans':
            self.self_attn = DifferentialAttention(dim=d_model, num_heads=n_heads, layer_num=layer_num)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        if not self.wo_ffn:
            self.layer_norm2 = nn.LayerNorm(d_model)
        
        
        self.dropout1 = nn.Dropout(dropout)
        if not self.wo_ffn:
            self.dropout2 = nn.Dropout(dropout)
            self.dropout_ffn = nn.Dropout(dropout)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.relu = torch.nn.ReLU()
        
        if self.use_gate:
            self.gate = Gate(input_dim=d_model , bg=2., mode=gate_mode)
            self.relu = torch.nn.ReLU()
        
        
        
        

    def forward(self, src):                             # src = bs, seq_len, d_model
        
        skip_connection = src
        
        if self.norm_first:
            src = self.layer_norm1(src)                 #bs, seq_len, d_model
        
        
        if self.mode == 'Trans':
            src2, _ = self.self_attn(src, src, src)
        elif self.mode == 'Diff-Trans':
            src2  = self.self_attn(src)
        
        if self.use_gate:
            connection, percentage1 = self.gate(skip_connection, self.relu(src2))  # ВОЗМОЖНО ПОСЛЕ RELU НАДО ТОЖЕ ДОБАВИТЬ ДРОПАУТ
        else: 
            connection = skip_connection + self.dropout1(src2)
        
        if not self.norm_first:
            connection = self.layer_norm1(connection)

        if self.wo_ffn:
            return connection
        ###########FFN PART##############
        skip_connection2 = connection
        if self.norm_first:
            connection = self.layer_norm2(connection)
        
        src3 = self.linear2(self.dropout_ffn(self.relu(self.linear1(connection))))  #bs, seq_len, d_model
        
        if self.use_gate:
            connection2, percentage2 = self.gate(skip_connection2, self.relu(src3))  # ВОЗМОЖНО ПОСЛЕ RELU НАДО ТОЖЕ ДОБАВИТЬ ДРОПАУТ
        else: 
            connection2 = skip_connection2 + self.dropout2(src3)
        
        if not self.norm_first:
            connection2 = self.layer_norm2(connection2)
        
        return connection2#, (percentage1, percentage2)
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################


class DifferentialAttention(nn.Module):
    def __init__(self, dim, num_heads, layer_num):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        self.scale_value = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_num)
        
        self.norm = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, x):
        queries = rearrange(self.q(x), "b n (h d q) -> b n (q h) d", h=self.num_heads, q=2, d=self.head_dim)
        queries = queries * self.scale_value

        keys = rearrange(self.k(x), "b n (h d k) -> b n (k h) d", h=self.num_heads, k=2, d=self.head_dim)
        v = rearrange(self.v(x), "b n (h d) -> b h n d", h=self.num_heads, d=2*self.head_dim)

        attention = torch.einsum("bnqd,bnkd->bnqk", queries, keys)
        attention = torch.nan_to_num(attention)
        attention = F.softmax(attention, dim=-1, dtype=torch.float32)

        lambda_1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        lambda_2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        lambda_value = torch.exp(lambda_1) - torch.exp(lambda_2) + self.lambda_init

        attention = rearrange(attention, "b n (q h) (k a) -> q k b n h a", q=2, k=2, h=self.num_heads, a=self.num_heads)
        attention = attention[0, 0, ...] - lambda_value * attention[1, 1, ...]

        out = torch.einsum("bnah,bhnd->bnad", attention, v)
        out = self.norm(out)
        out = out * (1 - self.lambda_init)
        out = rearrange(out, "b n h d -> b n (h d)")
        out = self.output_projection(out)

        return out





class Gate(torch.nn.Module):
    def __init__(self, input_dim, bg=2., mode='GRU'):
        """
        mode : GRU, Input, Output, Highway, ST
        bg : The gate bias. By setting bg > 0 we can explicitly initialize the gating mechanism to \
                be close to the identity map. This can greatly improve the learning speed and stability since it \
                initializes the agent close to a Markovian policy (ignore attention at the beginning).
        """
        super(Gate, self).__init__()
        self.mode = mode
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        #init.constant_(self.Wg.weight, 1.0)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        
    def forward_gru(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g, (1-z).mean()
    
    def forward_input(self, x, y):
        return torch.mul( self.sigmoid(self.Wg(x)), x) + y
    
    def forward_output(self, x, y):
        return x + torch.mul( self.sigmoid( self.Wg(x)-self.bg ) , y)
    
    def forward_highway(self, x, y):
        left = torch.mul( self.sigmoid( self.Wg(x) + self.bg ) , x)
        right = torch.mul( 1 - self.sigmoid( self.Wg(x) + self.bg ) , y)
        return left + right
    
    def forward_sig_tanh(self, x, y):
        return x + torch.mul( self.sigmoid( self.Wg(y) - self.bg ), self.tanh(self.Ug(y)) )
    
    def forward(self, x,y):
        if self.mode == 'GRU':
            return self.forward_gru(x,y)
        elif self.mode == 'Input':
            return self.forward_input(x,y)
        elif self.mode == 'Output':
            return self.forward_output(x,y)
        elif self.mode == 'Highway':
            return self.forward_highway(x,y)
        elif self.mode == 'ST':
            return self.forward_sig_tanh(x,y)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        self.encoding = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        
        self.encoding = self.encoding.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, x):
        
        
        num_envs, batch_size, seq_len, d_model = x.size()
        
        encoding = self.encoding[:, :seq_len, :].to(x.device)  # (1, seq_len, d_model)
        
        encoding = encoding.unsqueeze(0).repeat(num_envs, 1, 1, 1)  # (num_envs, seq_len, d_model)
        
        return x + encoding
