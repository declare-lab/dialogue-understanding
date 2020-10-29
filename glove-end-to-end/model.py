import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor

def pad(tensor, length):
        if isinstance(tensor, Variable):
            var = tensor
            if length > var.size(0):
                if torch.cuda.is_available():
                    return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
                else:
                    return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
            else:
                return var
        else:
            if length > tensor.size(0):
                if torch.cuda.is_available():
                    return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
                else:
                    return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
            else:
                return tensor     
class MaskedNLLLoss(nn.Module):

    def __init__(
        self, 
        weight=None
    ):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(
        self, 
        pred, 
        target, 
        mask
    ):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss

class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector

        return attn_pool, alpha

class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
            #import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim

        return attn_pool, alpha


class DialogueRNNCell(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                            context_attention='simple', D_a=100, dropout=0.1):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell = nn.GRUCell(D_m+D_p,D_g)
        self.p_cell = nn.GRUCell(D_m+D_g,D_p)
        self.e_cell = nn.GRUCell(D_p,D_e)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p,D_p)

        self.dropout = nn.Dropout(dropout)

        if context_attention=='simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(D_g, D_m, D_a, context_attention)

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel,0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U,q0_sel], dim=1),
                torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0 else
                g_hist[-1])
        g_ = self.dropout(g_)
        if g_hist.size()[0]==0:
            c_ = torch.zeros(U.size()[0],self.D_g).type(U.type())
            alpha = None
        else:
            c_, alpha = self.attention(g_hist,U)
        # c_ = torch.zeros(U.size()[0],self.D_g).type(U.type()) if g_hist.size()[0]==0\
        #         else self.attention(g_hist,U)[0] # batch, D_g
        U_c_ = torch.cat([U,c_], dim=1).unsqueeze(1).expand(-1,qmask.size()[1],-1)
        qs_ = self.p_cell(U_c_.contiguous().view(-1,self.D_m+self.D_g),
                q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                    expand(-1,qmask.size()[1],-1).contiguous().view(-1,self.D_p)
            U_ss_ = torch.cat([U_,ss_],1)
            ql_ = self.l_cell(U_ss_,q0.view(-1, self.D_p)).view(U.size()[0],-1,self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0]==0\
                else e0
        e_ = self.e_cell(self._select_parties(q_,qm_idx), e0)
        e_ = self.dropout(e_)

        return g_,q_,e_,alpha

class DialogueRNN(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, context_attention='simple', 
                 listener_state=False, D_a=100, dropout=0.1):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e,
                             listener_state, context_attention, D_a, dropout)

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type()) # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2],
                                    self.D_p).type(U.type()) # batch, party, D_p
        e_ = torch.zeros(0).type(U.type()) # batch, D_e
        e = e_

        alpha = []
        for u_,qmask_ in zip(U, qmask):
            g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)],0)
            e = torch.cat([e, e_.unsqueeze(0)],0)
            if type(alpha_)!=type(None):
                alpha.append(alpha_[:,0,:])

        return e,alpha # seq_len, batch, D_e


class CNNFeatureExtractor(nn.Module):
    
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        output_size, 
        filters, 
        kernel_sizes, 
        dropout
    ):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size

    def init_pretrained_embeddings_from_numpy(
        self, 
        pretrained_word_vectors
    ):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        self.embedding.weight.requires_grad = False

    def forward(
        self, 
        x, 
        umask
    ):
        
        num_utt, batch, num_words = x.size()
        # x = x.long().view(-1, num_words) # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        x = x.long().reshape(-1, num_words) # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        emb = self.embedding(x) # (num_utt * batch, num_words) -> (num_utt * batch, num_words, embedding_dim) 
        emb = emb.transpose(-2, -1).contiguous() # (num_utt * batch, num_words, embedding_dim)  -> (num_utt * batch, embedding_dim, num_words) 
        
        convoluted = [F.relu(conv(emb)) for conv in self.convs] 
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted] 
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated))) # (num_utt * batch, len(kernel_sizes) * filters) -> (num_utt * batch, output_size)
        features = features.view(num_utt, batch, -1) # (num_utt * batch, output_size) -> (num_utt, batch, output_size)
        mask = umask.unsqueeze(-1).type(FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim) #  (num_utt, batch, 1) -> (num_utt, batch, output_size)
        features = (features * mask) # (num_utt, batch, output_size) -> (num_utt, batch, output_size)
        return features

class End2EndModelSpeaker(nn.Module):

    def __init__(
        self, 
        dataset,
        vocab_size, 
        embedding_dim,
        tokenizer,
        cls_model,
        cnn_output_size=100, 
        cnn_filters=50, 
        cnn_kernel_sizes=(2,3,4), 
        cnn_dropout=0.25,
        D_e=100, 
        D_h=100,
        n_classes=7, 
        dropout=0.1,
        attention=False,
        context_attention='general2',
        rec_dropout=0.1,
        residual=False
    ):  
        super(End2EndModelSpeaker, self).__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cnn_feat_extractor = CNNFeatureExtractor(vocab_size, embedding_dim, cnn_output_size, 
                                                      cnn_filters, cnn_kernel_sizes, cnn_dropout)
        
        self.dropout   = nn.Dropout(dropout)
        
        self.cls_model = cls_model
        self.residual = residual
        
        self.feature_dim = 2*D_e
        self.rec_dropout = nn.Dropout(rec_dropout)
        
        if self.cls_model == 'lstm':
            self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=D_e, num_layers=2, 
                                bidirectional=True, dropout=0.1)
            
            self.attention = attention
            if self.attention:
                self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        
            self.linear = nn.Linear(2*D_e, D_h)
            
        elif self.cls_model == 'dialogrnn':
            self.dialog_rnn_f = DialogueRNN(cnn_output_size, 100, 100, D_e, context_attention)
            self.dialog_rnn_r = DialogueRNN(cnn_output_size, 100, 100, D_e, context_attention)
            self.attention = attention
            if self.attention:
                self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        
            self.linear = nn.Linear(2*D_e, D_h)
            
        elif self.cls_model == 'logreg':
            self.linear = nn.Linear(cnn_output_size, D_h)
            
        if self.dataset != 'persuasion':
            self.smax_fc = nn.Linear(D_h, n_classes)
        else:
            self.smax_fc_er = nn.Linear(D_h, 11)
            self.smax_fc_ee = nn.Linear(D_h, 13)  
    
    def speaker_separator(self, encoded, qmask):

        _, batch_size, _ = qmask.size()

        s1_lengths = [len(qmask[:, j, 0].nonzero().flatten()) for j in range(batch_size)]
        s2_lengths = [len(qmask[:, j, 1].nonzero().flatten()) for j in range(batch_size)]

        s1_max_length = max(s1_lengths)
        s2_max_length = max(s2_lengths)

        s1_encoded, s1_label, s1_sentiment = [], [], []
        s2_encoded, s2_label, s2_sentiment = [], [], []

        for j in range(batch_size):
            s1_indices = qmask[:, j, 0].nonzero().flatten()
            s2_indices = qmask[:, j, 1].nonzero().flatten()

            s1_encoded.append(pad(encoded[s1_indices, j, :].unsqueeze(1), s1_max_length))
            s2_encoded.append(pad(encoded[s2_indices, j, :].unsqueeze(1), s2_max_length))
            
        s1_encoded = torch.cat(s1_encoded, dim=1)
        s2_encoded = torch.cat(s2_encoded, dim=1)

        return s1_encoded, s2_encoded


    def speaker_merge(self, s1_hidden, s2_hidden, qmask):

        num_utt, batch_size, _ = qmask.size()
        merged = torch.zeros(num_utt, batch_size, s1_hidden.shape[-1]).type(FloatTensor)

        for j in range(batch_size):
            s1_indices = qmask[:, j, 0].nonzero().flatten()
            s2_indices = qmask[:, j, 1].nonzero().flatten()

            merged[s1_indices, j, :] = s1_hidden[:len(s1_indices), j, :]
            merged[s2_indices, j, :] = s2_hidden[:len(s2_indices), j, :]

        return merged  
    
    def init_pretrained_embeddings(
        self, 
        pretrained_word_vectors
    ):
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(pretrained_word_vectors)

    def pad(
        self,
        tensor, 
        length
    ):
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).long()])
        else:
            return tensor
        
    def _reverse_seq(
        self, 
        X, 
        mask
    ):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(
        self, 
        conversations, 
        lengths,
        umask,
        qmask=None
    ):       
        lengths = torch.Tensor(lengths).long()
        start = torch.cumsum(torch.cat((lengths.data.new(1).zero_(), lengths[:-1])), 0)
        utterances = [sent.lower() for conv in conversations for sent in conv]
        sequence, sequence_lengths = self.tokenizer.batch_encode(utterances)
        sequence = torch.stack([self.pad(sequence.narrow(0, s, l), max(lengths))
                                for s, l in zip(start.data.tolist(), lengths.data.tolist())], 0).transpose(0, 1)
        
        sequence = sequence.cuda()
        umask = umask.cuda()
        
        mask = umask.unsqueeze(-1).type(FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim) #  (num_utt, batch, 1) -> (num_utt, batch, output_size)
        
        features = self.cnn_feat_extractor(sequence, umask)
        
        cnn_features = features
        features1, features2 = self.speaker_separator(features, qmask)

        
        if self.cls_model == 'lstm':
            features1, _ = self.lstm(features1)
            features2, _ = self.lstm(features2)
            merged_encoded = self.speaker_merge(features1, features2, qmask)
            features = merged_encoded * mask
            if self.attention:
                att_features = []
                for t in features:
                    att_ft, _ = self.matchatt(features, t, mask=umask)
                    att_features.append(att_ft.unsqueeze(0))
                att_features = torch.cat(att_features, dim=0)
                hidden = F.relu(self.linear(att_features))
            else:
                hidden = F.relu(self.linear(features))
                hidden = self.rec_dropout(hidden) 
            
            if self.residual:
                hidden = hidden + cnn_features
                
        if self.cls_model == 'dialogrnn':
            hidden_f, alpha_f = self.dialog_rnn_f(features, qmask)
            rev_features = self._reverse_seq(features, umask)
            rev_qmask = self._reverse_seq(qmask, umask)
            hidden_b, alpha_b = self.dialog_rnn_r(rev_features, rev_qmask)
            hidden_b = self._reverse_seq(hidden_b, umask)
            features = torch.cat([hidden_f, hidden_b],dim=-1)
            # features = self.rec_dropout(features)
            features = features * mask
            
            if self.attention:
                att_features = []
                for t in features:
                    att_ft, _ = self.matchatt(features, t, mask=umask)
                    att_features.append(att_ft.unsqueeze(0))
                att_features = torch.cat(att_features, dim=0)
                hidden = F.relu(self.linear(att_features))
            else:
                hidden = F.relu(self.linear(features))
                hidden = self.rec_dropout(hidden)
                
            if self.residual:
                hidden = hidden + cnn_features
        
        elif self.cls_model == 'logreg':
            hidden = F.relu(self.linear(features))
            hidden = self.rec_dropout(hidden)
        
        # hidden = self.dropout(hidden)
        
        if self.dataset != 'persuasion':
            log_prob = F.log_softmax(self.smax_fc(hidden), 2)
            return log_prob
        else:
            log_prob_er = F.log_softmax(self.smax_fc_er(hidden), 2)
            log_prob_ee = F.log_softmax(self.smax_fc_ee(hidden), 2)
            return log_prob_er, log_prob_ee
        
class End2EndModel(nn.Module):

    def __init__(
        self, 
        dataset,
        vocab_size, 
        embedding_dim,
        tokenizer,
        cls_model,
        cnn_output_size=100, 
        cnn_filters=50, 
        cnn_kernel_sizes=(2,3,4), 
        cnn_dropout=0.25,
        D_e=100, 
        D_h=100,
        n_classes=7, 
        dropout=0.1,
        attention=False,
        context_attention='general2',
        rec_dropout=0.1,
        residual=False
    ):  
        super(End2EndModel, self).__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cnn_feat_extractor = CNNFeatureExtractor(vocab_size, embedding_dim, cnn_output_size, 
                                                      cnn_filters, cnn_kernel_sizes, cnn_dropout)
        
        self.dropout   = nn.Dropout(dropout)
        
        self.cls_model = cls_model
        self.residual = residual
        
        self.feature_dim = 2*D_e
        self.rec_dropout = nn.Dropout(rec_dropout)
        
        if self.cls_model == 'lstm':
            self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=D_e, num_layers=2, 
                                bidirectional=True, dropout=0.1)
            
            self.attention = attention
            if self.attention:
                self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        
            self.linear = nn.Linear(2*D_e, D_h)
            
        elif self.cls_model == 'dialogrnn':
            self.dialog_rnn_f = DialogueRNN(cnn_output_size, 100, 100, D_e, context_attention)
            self.dialog_rnn_r = DialogueRNN(cnn_output_size, 100, 100, D_e, context_attention)
            self.attention = attention
            if self.attention:
                self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        
            self.linear = nn.Linear(2*D_e, D_h)
            
        elif self.cls_model == 'logreg':
            self.linear = nn.Linear(cnn_output_size, D_h)
            
        if self.dataset != 'persuasion':
            self.smax_fc = nn.Linear(D_h, n_classes)
        else:
            self.smax_fc_er = nn.Linear(D_h, 11)
            self.smax_fc_ee = nn.Linear(D_h, 13)
    
    def init_pretrained_embeddings(
        self, 
        pretrained_word_vectors
    ):
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(pretrained_word_vectors)

    def pad(
        self,
        tensor, 
        length
    ):
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).long()])
        else:
            return tensor
        
    def _reverse_seq(
        self, 
        X, 
        mask
    ):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(
        self, 
        conversations, 
        lengths,
        umask,
        qmask=None
    ):       
        lengths = torch.Tensor(lengths).long()
        start = torch.cumsum(torch.cat((lengths.data.new(1).zero_(), lengths[:-1])), 0)
        utterances = [sent.lower() for conv in conversations for sent in conv]
        sequence, sequence_lengths = self.tokenizer.batch_encode(utterances)
        sequence = torch.stack([self.pad(sequence.narrow(0, s, l), max(lengths))
                                for s, l in zip(start.data.tolist(), lengths.data.tolist())], 0).transpose(0, 1)
        
        sequence = sequence.cuda()
        umask = umask.cuda()
        
        mask = umask.unsqueeze(-1).type(FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim) #  (num_utt, batch, 1) -> (num_utt, batch, output_size)
        
        features = self.cnn_feat_extractor(sequence, umask)
        
        cnn_features = features
        
        if self.cls_model == 'lstm':
            features, _ = self.lstm(features)
            features = features * mask
            if self.attention:
                att_features = []
                for t in features:
                    att_ft, _ = self.matchatt(features, t, mask=umask)
                    att_features.append(att_ft.unsqueeze(0))
                att_features = torch.cat(att_features, dim=0)
                hidden = F.relu(self.linear(att_features))
            else:
                hidden = F.relu(self.linear(features))
                hidden = self.rec_dropout(hidden)
                
            if self.residual:
                hidden = hidden + cnn_features
                
        if self.cls_model == 'dialogrnn':
            hidden_f, alpha_f = self.dialog_rnn_f(features, qmask)
            rev_features = self._reverse_seq(features, umask)
            rev_qmask = self._reverse_seq(qmask, umask)
            hidden_b, alpha_b = self.dialog_rnn_r(rev_features, rev_qmask)
            hidden_b = self._reverse_seq(hidden_b, umask)
            features = torch.cat([hidden_f, hidden_b],dim=-1)
            # features = self.rec_dropout(features)
            features = features * mask
            
            if self.attention:
                att_features = []
                for t in features:
                    att_ft, _ = self.matchatt(features, t, mask=umask)
                    att_features.append(att_ft.unsqueeze(0))
                att_features = torch.cat(att_features, dim=0)
                hidden = F.relu(self.linear(att_features))
                hidden = self.rec_dropout(hidden)
            else:
                hidden = F.relu(self.linear(features))
                hidden = hidden
                
            if self.residual:
                hidden = hidden + cnn_features
        
        elif self.cls_model == 'logreg':
            hidden = F.relu(self.linear(features))
            hidden = self.rec_dropout(hidden)
        
        # hidden = self.dropout(hidden)
        
        if self.dataset != 'persuasion':
            log_prob = F.log_softmax(self.smax_fc(hidden), 2)
            return log_prob
        else:
            log_prob_er = F.log_softmax(self.smax_fc_er(hidden), 2)
            log_prob_ee = F.log_softmax(self.smax_fc_ee(hidden), 2)
            return log_prob_er, log_prob_ee


class End2EndShuffledMultitaskModel(nn.Module):

    def __init__(
        self, 
        dataset,
        vocab_size, 
        embedding_dim,
        tokenizer,
        cls_model,
        cnn_output_size=100, 
        cnn_filters=50, 
        cnn_kernel_sizes=(2,3,4), 
        cnn_dropout=0.25,
        D_e=100, 
        D_h=100,
        n_classes=7, 
        dropout=0.25,
        attention=False,
        context_attention='general',
        rec_dropout=0.1,
        residual=False,
        max_num_utt=100
    ):  
        super(End2EndShuffledMultitaskModel, self).__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cnn_feat_extractor = CNNFeatureExtractor(vocab_size, embedding_dim, cnn_output_size, 
                                                      cnn_filters, cnn_kernel_sizes, cnn_dropout)
        
        self.dropout   = nn.Dropout(dropout)
        
        self.cls_model = cls_model
        
        self.feature_dim = 2*D_e
        self.rec_dropout = nn.Dropout(rec_dropout)
        self.residual = residual
        
        if self.cls_model == 'lstm':
            self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=D_e, num_layers=2, 
                                bidirectional=True, dropout=0.1)
            
            self.attention = attention
            if self.attention:
                self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        
            self.linear = nn.Linear(2*D_e, D_h)
            
        elif self.cls_model == 'dialogrnn':
            self.dialog_rnn_f = DialogueRNN(cnn_output_size, 100, 100, D_e, context_attention)
            self.dialog_rnn_r = DialogueRNN(cnn_output_size, 100, 100, D_e, context_attention)
            self.attention = attention
            if self.attention:
                self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        
            self.linear = nn.Linear(2*D_e, D_h)
            
        self.seq_label_smax_fc = nn.Linear(D_h, max_num_utt)
            
        if self.dataset != 'persuasion':
            self.smax_fc = nn.Linear(D_h, n_classes)
        else:
            self.smax_fc_er = nn.Linear(D_h, 11)
            self.smax_fc_ee = nn.Linear(D_h, 13)
    
    def init_pretrained_embeddings(
        self, 
        pretrained_word_vectors
    ):
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(pretrained_word_vectors)

    def pad(
        self,
        tensor, 
        length
    ):
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).long()])
        else:
            return tensor
        
    def _reverse_seq(
        self, 
        X, 
        mask
    ):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)
    

    def forward(
        self, 
        conversations, 
        lengths,
        umask,
        qmask=None
    ):       
        lengths = torch.Tensor(lengths).long()
        start = torch.cumsum(torch.cat((lengths.data.new(1).zero_(), lengths[:-1])), 0)
        utterances = [sent.lower() for conv in conversations for sent in conv]
        sequence, sequence_lengths = self.tokenizer.batch_encode(utterances)
        sequence = torch.stack([self.pad(sequence.narrow(0, s, l), max(lengths))
                                for s, l in zip(start.data.tolist(), lengths.data.tolist())], 0).transpose(0, 1)
        
        sequence = sequence.cuda()
        umask = umask.cuda()
        
        mask = umask.unsqueeze(-1).type(FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim) #  (num_utt, batch, 1) -> (num_utt, batch, output_size)
        
        cnn_features = self.cnn_feat_extractor(sequence, umask)
                
        if self.cls_model == 'lstm':
            features, _ = self.lstm(cnn_features)
            features = features * mask
            if self.attention:
                att_features = []
                for t in features:
                    att_ft, _ = self.matchatt(features, t, mask=umask)
                    att_features.append(att_ft.unsqueeze(0))
                att_features = torch.cat(att_features, dim=0)
                hidden = F.relu(self.linear(att_features))
            else:
                hidden = F.relu(self.linear(features))
                hidden = self.rec_dropout(hidden)
                
            if self.residual:
                hidden = hidden + cnn_features
                
        if self.cls_model == 'dialogrnn':
            hidden_f, alpha_f = self.dialog_rnn_f(cnn_features, qmask)
            rev_features = self._reverse_seq(cnn_features, umask)
            rev_qmask = self._reverse_seq(qmask, umask)
            hidden_b, alpha_b = self.dialog_rnn_r(rev_features, rev_qmask)
            hidden_b = self._reverse_seq(hidden_b, umask)
            features = torch.cat([hidden_f, hidden_b],dim=-1)
            features = features * mask
            
            if self.attention:
                att_features = []
                for t in features:
                    att_ft, _ = self.matchatt(features, t, mask=umask)
                    att_features.append(att_ft.unsqueeze(0))
                att_features = torch.cat(att_features, dim=0)
                hidden = F.relu(self.linear(att_features))
            else:
                hidden = F.relu(self.linear(features))
                hidden = self.rec_dropout(hidden)
                
            if self.residual:
                hidden = hidden + cnn_features
                
        elif self.cls_model == 'logreg':
            hidden = F.relu(self.linear(features))
            hidden = self.rec_dropout(hidden)
        
        # hidden = self.dropout(hidden)
        
        seq_label_log_prob = F.log_softmax(self.seq_label_smax_fc(hidden), 2)
        
        if self.dataset != 'persuasion':
            log_prob = F.log_softmax(self.smax_fc(hidden), 2)
            return log_prob, seq_label_log_prob
        else:
            log_prob_er = F.log_softmax(self.smax_fc_er(hidden), 2)
            log_prob_ee = F.log_softmax(self.smax_fc_ee(hidden), 2)
            return [log_prob_er, log_prob_ee], seq_label_log_prob
            

        
