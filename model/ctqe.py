import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import Attention

class CTQE(nn.Module):
    def __init__(
        self,
        embed_size,
        dropout_rate,
        lstm_param,
        num_heads = 8,
        run_cls = True
    ):
        super().__init__()

        self.run_cls = run_cls
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.lstm_mem = nn.LSTM(
            embed_size,
            int(embed_size/2),
            lstm_param['n_layers'],
            bidirectional=lstm_param['bidirectional'],
            dropout=dropout_rate,
            batch_first=True,
        )

        self.lstm_q = nn.LSTM(
            embed_size,
            int(embed_size/2),
            lstm_param['n_layers'],
            bidirectional=lstm_param['bidirectional'],
            dropout=dropout_rate,
            batch_first=True,
        )

        self.weight_enc = nn.Linear(embed_size, 150) if lstm_param['bidirectional'] else nn.Linear(int(embed_size/2), 150)

        self.cls_enc = nn.Linear(150,1)
        self.cls_output = None
        self.mutihead_query = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout_rate, batch_first=True)

        self.att_query = Attention(embed_size, batch_first=True)
        self.att_weight_query = 0

        self.mutihead_ctx = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout_rate, batch_first=True)


    def forward(self, mean_ids, ids, mean_ctx, ctx):
        # ctx = [batch size, seq len, em_size] -> mem_enc = [batch size, seq len, bidirect*hidden_size]
        mem_enc, ( _, _) = self.lstm_mem(ctx)
        # ids = [batch size, seq len, em_size] -> q_enc = [batch size, seq len, bidirect*hidden_size]
        q_enc_, ( _ , _) = self.lstm_q(ids)
        q_enc = q_enc_.mean(dim=1,keepdim=True) #  [batch size, 1 , bidirect*hidden_size]

        p_enc = torch.einsum('bij,bkj->bki', q_enc,mem_enc).squeeze() #[batch size, seq len]
        weight_p = F.softmax(p_enc, dim=-1)
        #Knowledge Encoding Representation _ h = [batch size, bidirect*hidden_size]
        h = torch.sum(weight_p.unsqueeze(-1)*mem_enc, dim=1)
        o = self.weight_enc(h + q_enc.squeeze()) # [batch size, hidden_size]
        cls = F.sigmoid(self.cls_enc(F.relu(o))) # [batch size, 1]

        # ------------------------------------------------
        # ctx = [batch size, seq len, em_size]
        ctx_weight = torch.sum(weight_p.unsqueeze(-1)*ctx, dim=1).unsqueeze(1)  
        ctx_att, _ = self.mutihead_ctx(q_enc, ctx, ctx)                         

        src, _ = self.mutihead_query(ctx_weight, q_enc_, ids)     

        x = torch.cat([ids, self.dropout(src), self.dropout(ctx_att)], dim=1) 
        ### Start Old code
        # packed = nn.utils.rnn.pack_padded_sequence(x, [x.size(1)]*x.size(0), batch_first=True, enforce_sorted=False)
        # y, lengths = nn.utils.rnn.pad_packed_sequence(packed,batch_first=True)
        # prediction, self.att_weight_query = self.att_query(y, lengths)
        ### End Old code
        ### Start Edit for TensorTR
        lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        prediction, att_weight_query = self.att_query(x, lengths)
        # self.att_weight_query = att_weight_query
        ### End Edit for TensorTR

        # self.cls_output = cls

        prediction = cls*prediction + (1-cls)*mean_ids if self.run_cls else prediction



        # return prediction
        return prediction, att_weight_query, cls