
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import Config

Activations={
    "ReLU":nn.ReLU(inplace=True)
}

class FC(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, activation=None):
        super(FC, self).__init__()

        self.hasactivation = activation is not None

        self.linear = nn.Linear(input_dim, output_dim)

        if activation is not None:
            self.activation = Activations[activation]

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout=None

    def forward(self, x):
        x = self.linear(x)

        if self.hasactivation:
            x = self.activation(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a * (x - mean) / (std + self.eps) + self.b


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0, activation=None):
        super(MLP, self).__init__()

        self.fc = FC(input_dim, hidden_dim, dropout=dropout, activation=activation)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear(self.fc(x))
# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, opt):
        super(FFN, self).__init__()

        self.mlp = MLP(
            input_dim=opt.hidden_size,
            hidden_dim=opt.ffn_size,
            output_dim=opt.hidden_size,
            dropout=opt.drop_ratio,
            activation="ReLU"
        )

    def forward(self, x):
        return self.mlp(x)

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, opt):
        super(MHAtt, self).__init__()
        self.opt = opt

        self.linear_v = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_k = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_q = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.linear_merge = nn.Linear(opt.hidden_size, opt.hidden_size)

        self.dropout = nn.Dropout(opt.drop_ratio)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt.multihead,
            int(self.opt.hidden_size / self.opt.multihead)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt.hidden_size
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        

        return torch.matmul(att_map, value)

class Decoder_multiSA_SA(nn.Module):
    def __init__(self, opt):
        super(Decoder_multiSA_SA, self).__init__()

        self.mhatt1 = MHAtt(opt)
        self.mhatt2 = MHAtt(opt)
        self.ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt.drop_ratio)
        self.norm1 = LayerNorm(opt.hidden_size)

        self.dropout2 = nn.Dropout(opt.drop_ratio)
        self.norm2 = LayerNorm(opt.hidden_size)

        self.dropout3 = nn.Dropout(opt.drop_ratio)
        self.norm3 = LayerNorm(opt.hidden_size)

    def forward(self, x, y, x_mask, y_mask): # x (64, 49, 512) y


        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))

        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        )) # (64, 49, 512) # (bs, 49, 768)

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x

class cross_attention_decoder(nn.Module):
    def __init__(self, opt):
        super(cross_attention_decoder, self).__init__()
        self.opt = opt
        self.tau = opt.tau_max
        self.dec_list = nn.ModuleList([Decoder_multiSA_SA(opt) for _ in range(opt.attention_layer)])

    def forward(self, y, x, y_mask, x_mask):

        # Input encoder last hidden vector
        # And obtain decoder last hidden vectors
        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask) 
        return y, x

    def set_tau(self, tau):
        self.tau = tau



class WPDM(nn.Module):
    """ 使用contrastive domain
      refined guided GCN to find attribute
    """

    def __init__(self, args, bert):
        """Initialize the model"""
        super(WPDM, self).__init__()
        self.encoder = bert
        self.dropout = nn.Dropout(args.drop_ratio)

        self.cross_attention = cross_attention_decoder(args)
        self.classifier = nn.Linear(args.hidden_size*args.cls_num, args.label_size)
        self.args = args    

        

        if 'initializer_range' in list(args.__dict__.keys()):
            self.initializer_range = args.initializer_range
        else:
            self.initializer_range = 0.02
        layers = []
        for _ in range(args.mlp_layers):
            layers += [nn.Linear(args.hidden_size*args.cls_num, args.hidden_size*args.cls_num), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)
        for i in range(len(self.out_mlp)):
            self._init_weights(self.out_mlp[i])
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids_ori,
        input_mask_ori,
        core_word_all_indexs, # [bs, max_len]
        target_words_all_indexes_list, # [bs, target_word_n, max_len]
        core_words_all_indexes_list,
        target_source_scores, # [bs, word_pair_len]
    ):
        outputs_ori = self.encoder(
            input_ids_ori,
            attention_mask=input_mask_ori
        ) # hidden [bs, max_len, hidden_size]
        cls_word = outputs_ori[0][:, 0, :] # [bs, hidden_size]
        core_word_index_sum = torch.where(torch.sum(core_word_all_indexs, dim=-1) == 0, torch.tensor(1).cuda(), torch.sum(core_word_all_indexs, dim=-1))
        core_word = (torch.sum(outputs_ori[0] * core_word_all_indexs.unsqueeze(-1), dim=1)/core_word_index_sum.unsqueeze(-1)) # [bs, hidden_size]
        
        
        target_words_index_sum = torch.where(torch.sum(target_words_all_indexes_list, dim=-1) == 0, torch.tensor(1).cuda(), torch.sum(target_words_all_indexes_list, dim=-1))
        target_words = torch.sum(outputs_ori[0].unsqueeze(1) * target_words_all_indexes_list.unsqueeze(-1), dim=2)/target_words_index_sum.unsqueeze(-1) # [bs, target_word_n, max_len, hidden_size] -> [bs, target_word_n, hidden_size]

        core_words_index_sum = torch.where(torch.sum(core_words_all_indexes_list, dim=-1) == 0, torch.tensor(1).cuda(), torch.sum(core_words_all_indexes_list, dim=-1))
        core_words = torch.sum(outputs_ori[0].unsqueeze(1) * core_words_all_indexes_list.unsqueeze(-1), dim=2)/core_words_index_sum.unsqueeze(-1) # [bs, target_word_n, max_len, hidden_size] -> [bs, target_word_n, hidden_size]
            
        
        target_words_cross_attention = []

        mask = torch.zeros(outputs_ori[0].size(0), 1, 1, 1).byte().to(self.args.device)
        for i in range(self.args.word_pair_max): 
            target_word, core_word_ = self.cross_attention(core_words[:,i,:].unsqueeze(1), target_words[:,i,:].unsqueeze(1), mask, mask)
            target_words_cross_attention.append(target_word) # [bs, 1, hidden_size]
        word_pairs = torch.cat(target_words_cross_attention, 1) # [bs, target_word_n, hidden_size]
        target_source_index_sum = torch.where(torch.sum(target_source_scores, dim=-1) == 0, torch.tensor(1).cuda(), torch.sum(target_source_scores, dim=-1))
        word_pair_emb = (torch.sum(word_pairs * target_source_scores.unsqueeze(-1), dim=1)/target_source_index_sum.unsqueeze(-1)) # [bs, hidden_size]
        # classification
        output = self.dropout(self.out_mlp(torch.cat([cls_word, core_word, word_pair_emb], dim=1)))
        logits = self.classifier(output)

        return logits


_models = {
    'none':{

        "WPDM":WPDM,
    
        
        },

    }
    