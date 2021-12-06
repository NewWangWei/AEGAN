import torch
from torch import nn
from misc.utils import expand_feats
from .CaptionModel import CaptionModel
from torch.autograd import Variable


def build_embeding_layer(vocab_size, dim, drop_prob):
    embed = nn.Sequential(nn.Embedding(vocab_size, dim),
                          nn.ReLU(),
                          nn.Dropout(drop_prob))
    return embed


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.episode = 1e-8
        self.opt = opt
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.query_dim = self.rnn_size
        self.h2att = nn.Linear(self.query_dim, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = torch.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / (weight.sum(1, keepdim=True) + self.episode)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        return att_res


class GraphAttentionRela(nn.Module):
    def __init__(self, opt):
        super(GraphAttentionRela, self).__init__()
        self.episode = 1e-8
        self.opt = opt
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.query_dim = self.rnn_size
        self.obj2att = nn.Linear(self.query_dim, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, obj_vecs_e, rela_feats, p_rela_feats, obj_rela_masks=None):
        att_size = obj_vecs_e.size(1)
        rela_size = rela_feats.size(1)

        rela_feats = rela_feats.unsqueeze(1).expand(rela_feats.size(0), att_size, rela_size, rela_feats.size(-1))
        p_rela_feats = p_rela_feats.unsqueeze(1).expand(p_rela_feats.size(0), att_size, rela_size, self.att_hid_size)

        att_obj = self.obj2att(obj_vecs_e.view(-1, obj_vecs_e.size(-1))).view(-1, att_size, self.att_hid_size)
        att_h = att_obj.unsqueeze(2).expand_as(p_rela_feats)
        dot = p_rela_feats + att_h
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, rela_size)

        weight = torch.softmax(dot, dim=1)  # (batch*att_size, rela_size)
        if obj_rela_masks is not None:
            weight = weight * obj_rela_masks.view(-1, rela_size).float()
            weight = weight / (weight.sum(1, keepdim=True) + self.episode)
        rela_feats_ = rela_feats.contiguous().view(-1, rela_size, rela_feats.size(-1))
        att_res = torch.bmm(weight.unsqueeze(1), rela_feats_)
        att_res = att_res.view(obj_vecs_e.shape)
        return att_res


class GraphAttentionAttr(nn.Module):
    def __init__(self, opt):
        super(GraphAttentionAttr, self).__init__()
        self.episode = 1e-8
        self.opt = opt
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.query_dim = self.rnn_size
        self.obj2att = nn.Linear(self.query_dim, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, obj_vecs_i, attr_feats, p_attr_feats, attr_masks=None):
        attr_size = attr_feats.size(2)

        att_obj = self.obj2att(obj_vecs_i).unsqueeze(2).expand_as(p_attr_feats)
        dot = p_attr_feats + att_obj
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        dot = dot.view(-1, attr_size)

        weight = torch.softmax(dot, dim=1)  # (batch*att_size, attr_size)
        if attr_masks is not None:
            weight = weight * attr_masks.view(-1, attr_size).float()
            weight = weight / (weight.sum(1, keepdim=True) + self.episode)

        attr_feats_ = attr_feats.contiguous().view(-1, attr_size, attr_feats.size(-1))
        att_res = torch.bmm(weight.unsqueeze(1), attr_feats_)
        att_res = att_res.view(obj_vecs_i.shape)
        return att_res


class GCN(nn.Module):
    def __init__(self, opt):
        super(GCN, self).__init__()
        self.opt = opt
        self.n_heads = opt.n_heads
        out_dim = opt.rnn_size

        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.att_hid_size = opt.att_hid_size
        self.ctx2att_rela = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att_attr = nn.Linear(self.rnn_size, self.att_hid_size)

        for i in range(self.n_heads):
            self.add_module('att_attr_{}'.format(i), GraphAttentionAttr(opt))
            self.add_module('att_rela_{}'.format(i), GraphAttentionRela(opt))

        self.gcn_attr = nn.Sequential(
            nn.Linear(self.rnn_size * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(opt.drop_prob_lm)
        )

        self.gcn_obj_i = nn.Sequential(
            nn.Linear(self.rnn_size * 2, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(opt.drop_prob_lm)
        )

        self.gcn_obj_e = nn.Sequential(
            nn.Linear(self.rnn_size * 3, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(opt.drop_prob_lm)
        )

        self.gcn_rela = nn.Sequential(
            nn.Linear(self.rnn_size * 3, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(opt.drop_prob_lm)
        )

    def forward(self, attr_vecs, obj_vecs_i, obj_vecs_e, rela_vecs, edges, rela_masks=None, obj_rela_masks=None,
                attr_masks=None):
        # for easily indexing the subject and object of each relation in the tensors
        obj_vecs_i, obj_vecs_e, rela_vecs, edges, ori_shape = self.feat_3d_to_2d(obj_vecs_i,
                                                                                 obj_vecs_e, rela_vecs,
                                                                                 edges)
        # rela
        s_idx = edges[:, 0].contiguous()  # index of subject
        o_idx = edges[:, 1].contiguous()  # index of object
        s_vecs = obj_vecs_e[s_idx]
        o_vecs = obj_vecs_e[o_idx]
        t_vecs = torch.cat([s_vecs, rela_vecs, o_vecs], dim=1)
        new_rela_vecs = self.gcn_rela(t_vecs) + rela_vecs

        obj_vecs_i, obj_vecs_e, new_rela_vecs = self.feat_2d_to_3d(obj_vecs_i,
                                                                   obj_vecs_e, new_rela_vecs,
                                                                   rela_masks,
                                                                   ori_shape)

        # attr
        tmp_obj_vecs_i = obj_vecs_i.unsqueeze(2).expand_as(attr_vecs)
        t_vecs = torch.cat([attr_vecs, tmp_obj_vecs_i], dim=-1)
        new_attr_vecs = self.gcn_attr(t_vecs) + attr_vecs
        new_attr_vecs = new_attr_vecs.view(new_attr_vecs.size(0), -1, new_attr_vecs.size(-1))

        # obj_vecs_i
        t_vecs = torch.cat([obj_vecs_i.unsqueeze(2).expand_as(attr_vecs), attr_vecs], dim=-1)
        t_vecs = self.gcn_obj_i(t_vecs)
        p_t_vecs = self.ctx2att_attr(t_vecs)
        new_obj_vecs_i = []
        for ii in range(self.n_heads):
            tmp = getattr(self, 'att_attr_{}'.format(ii))(obj_vecs_i, t_vecs, p_t_vecs, attr_masks)
            new_obj_vecs_i.append(tmp)
        new_obj_vecs_i = torch.stack(new_obj_vecs_i, dim=2)
        new_obj_vecs_i = new_obj_vecs_i.mean(2)
        new_obj_vecs_i = new_obj_vecs_i + obj_vecs_i

        # obj_vecs_e
        B, No = ori_shape
        t_vecs = torch.cat([s_vecs, rela_vecs, o_vecs], dim=1)
        t_vecs = self.gcn_obj_e(t_vecs)
        p_vecs = self.ctx2att_rela(t_vecs)
        t_vecs = t_vecs.view(B, -1, self.rnn_size)
        p_vecs = p_vecs.view(B, -1, self.att_hid_size)
        new_obj_vecs_e = []
        for ii in range(self.n_heads):
            tmp = getattr(self, 'att_rela_{}'.format(ii))(obj_vecs_e, t_vecs, p_vecs, obj_rela_masks)
            new_obj_vecs_e.append(tmp)
        new_obj_vecs_e = torch.stack(new_obj_vecs_e, dim=2)
        new_obj_vecs_e = new_obj_vecs_e.mean(2)
        new_obj_vecs_e = new_obj_vecs_e + obj_vecs_e

        return new_attr_vecs, new_obj_vecs_i, new_obj_vecs_e, new_rela_vecs

    # def feat_3d_to_2d(self, obj_vecs, attr_vecs, rela_vecs, edges):
    def feat_3d_to_2d(self, obj_vecs_i, obj_vecs_e, rela_vecs, edges):
        """
        convert 3d features of shape (B, N, d) into 2d features of shape (B*N, d)
        """
        B, No = obj_vecs_i.shape[:2]
        obj_vecs_i = obj_vecs_i.view(-1, obj_vecs_i.size(-1))
        obj_vecs_e = obj_vecs_e.view(-1, obj_vecs_e.size(-1))
        rela_vecs = rela_vecs.view(-1, rela_vecs.size(-1))

        obj_offsets = edges.new_tensor(range(0, B * No, No))
        edges = edges + obj_offsets.view(-1, 1, 1)
        edges = edges.view(-1, edges.size(-1))
        return obj_vecs_i, obj_vecs_e, rela_vecs, edges, (B, No)

    def feat_2d_to_3d(self, obj_vecs_i, obj_vecs_e, rela_vecs, rela_masks, ori_shape):
        """
        convert 2d features of shape (B*N, d) back into 3d features of shape (B, N, d)
        """
        B, No = ori_shape
        obj_vecs_i = obj_vecs_i.view(B, No, -1)
        obj_vecs_e = obj_vecs_e.view(B, No, -1)
        rela_vecs = rela_vecs.view(B, -1, rela_vecs.size(-1)) * rela_masks
        return obj_vecs_i, obj_vecs_e, rela_vecs


class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.opt = opt
        self.n_heads = opt.n_heads
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_hid_size = opt.att_hid_size
        self.seq_per_img = opt.seq_per_img
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.att_feat_size = opt.att_feat_size
        if opt.use_box:
            self.att_feat_size = self.att_feat_size + 5  # concat box position features
        # self.sg_label_embed_size = opt.sg_label_embed_size
        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = build_embeding_layer(self.vocab_size + 1, self.input_encoding_size, self.drop_prob_lm)
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        self.proj_attr = nn.Sequential(*[nn.Linear(self.rnn_size,
                                                   self.rnn_size), nn.ReLU(), nn.Dropout(0.5)])
        # self.proj_obj = nn.Sequential(*[nn.Linear(self.rnn_size + self.rnn_size * 1,
        #                                           self.rnn_size), nn.ReLU(), nn.Dropout(0.5)])
        self.proj_obj_i = nn.Sequential(*[nn.Linear(self.rnn_size + self.rnn_size * 1,
                                                    self.rnn_size), nn.ReLU(), nn.Dropout(0.5)])
        self.proj_obj_e = nn.Sequential(*[nn.Linear(self.rnn_size + self.rnn_size * 1,
                                                    self.rnn_size), nn.ReLU(), nn.Dropout(0.5)])
        self.proj_rela = nn.Sequential(*[nn.Linear(self.rnn_size,
                                                   self.rnn_size), nn.ReLU(), nn.Dropout(0.5)])
        self.gcn = GCN(opt)

        self.ctx2att_attr = nn.Linear(self.rnn_size, self.att_hid_size)

        self.ctx2att_obj = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att_rela = nn.Linear(self.rnn_size, self.att_hid_size)

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        self.logit_node = nn.Linear(self.rnn_size, 3)

        self.logit_tag = nn.Linear(self.rnn_size, 4)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed[0].weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz, weak_rela):
        assert bsz == weak_rela.size(0), 'make sure of same batch size'
        rela_mask = weak_rela > 0

        weak_rela_embedding = torch.sum(self.embed(weak_rela) * rela_mask.unsqueeze(-1).float(), dim=1) / \
                              (torch.sum(rela_mask, dim=1, keepdim=True).float() + 1e-20)
        h = torch.stack([weak_rela_embedding for _ in range(self.num_layers)], 0)
        return (h, h)

    def _embed_vrg(self, attr_labels, obj_labels, rela_labels):
        attr_embed = self.embed(attr_labels)
        obj_embed = self.embed(obj_labels)
        rela_embed = self.embed(rela_labels)

        return attr_embed, obj_embed, rela_embed

    def _proj_vrg(self, attr_embed, obj_embed, rela_embed, att_feats):
        "project node features in paper"
        attr_vecs = self.proj_attr(attr_embed)

        obj_embed = obj_embed.view(obj_embed.size(0), obj_embed.size(1), -1)
        obj_vecs_i = self.proj_obj_i(torch.cat([att_feats, obj_embed], dim=-1))
        obj_vecs_e = self.proj_obj_e(torch.cat([att_feats, obj_embed], dim=-1))

        rela_vecs = self.proj_rela(rela_embed)
        return attr_vecs, obj_vecs_i, obj_vecs_e, rela_vecs

    def _prepare_vrg_features(self, sg_data, att_feats, att_masks):
        """
        the raw data the are needed:
            - obj_labels: (B, No, ?)
            - attr_labels:(B,No,?)
            - rela_labels: (B, Nr, ?)
            - rela_triplets: (subj_index, obj_index, rela_label) of shape (B, Nr, 3)
            - rela_edges: LongTensor of shape (B, Nr, 2), where rela_edges[b, k] = [i, j] indicates the
                        presence of the relation triple: ( obj[b][i], rela[b][k], obj[b][j] ),
                        i.e. the k-th relation of the b-th sample which is between the i-th and j-th objects
        """
        attr_masks = sg_data['attr_masks']
        attr_labels = sg_data['attr']
        obj_labels = sg_data['obj_labels']
        rela_masks = sg_data['rela_masks']
        obj_rela_masks = sg_data['obj_rela_masks']
        rela_edges, rela_labels = sg_data['rela_edges'], sg_data['rela_feats']

        att_masks, rela_masks = att_masks.unsqueeze(-1), rela_masks.unsqueeze(-1)

        attr_embed, obj_embed, rela_embed = self._embed_vrg(attr_labels, obj_labels, rela_labels)
        attr_vecs, obj_vecs_i, obj_vecs_e, rela_vecs = self._proj_vrg(attr_embed, obj_embed, rela_embed, att_feats)
        # node embedding with simple gnns
        new_attr_vecs, new_obj_vecs_i, new_obj_vecs_e, new_rela_vecs = self.gcn(attr_vecs, obj_vecs_i,
                                                                                obj_vecs_e,
                                                                                rela_vecs, rela_edges, rela_masks,
                                                                                obj_rela_masks, attr_masks)

        # origin_obj_vecs = self.proj_obj(torch.cat([att_feats, obj_embed.squeeze(2)], dim=-1))
        #
        # tmp1 = torch.sum(origin_obj_vecs * new_obj_vecs_i, dim=-1, keepdim=True)
        # tmp2 = torch.sum(origin_obj_vecs * new_obj_vecs_e, dim=-1, keepdim=True)
        # weights = torch.softmax(torch.cat([tmp1, tmp2], dim=-1), dim=-1)
        # new_obj_vecs = weights[:, :, 0:1] * new_obj_vecs_i + weights[:, :, 1:2] * new_obj_vecs_e

        return new_attr_vecs, new_obj_vecs_i, new_obj_vecs_e, new_rela_vecs

    def prepare_core_args(self, sg_data, fc_feats, att_feats, att_masks):
        rela_masks = sg_data['rela_masks']
        attr_masks = sg_data['attr_masks'].view(sg_data['attr_masks'].size(0), -1)
        # embed fc and att features
        fc_feats = self.fc_embed(fc_feats)

        att_feats = self.att_embed(att_feats)

        attr_feats, obj_vecs_i, obj_vecs_e, rela_feats = self._prepare_vrg_features(sg_data, att_feats, att_masks)

        obj_feats = obj_vecs_i + obj_vecs_e

        p_obj_feats = self.ctx2att_obj(obj_feats)
        p_attr_feats = self.ctx2att_attr(attr_feats)
        p_rela_feats = self.ctx2att_rela(rela_feats)

        core_args = [fc_feats, att_feats, attr_feats, obj_feats, rela_feats, p_attr_feats, \
                     p_obj_feats, p_rela_feats, \
                     att_masks, rela_masks, attr_masks]
        return core_args

    def _forward(self, sg_data, fc_feats, att_feats, seq, weak_rela, att_masks=None):
        core_args = self.prepare_core_args(sg_data, fc_feats, att_feats, att_masks)
        # make seq_per_img copies of the encoded inputs:  shape: (B, ...) => (B*seq_per_image, ...)
        core_args = expand_feats(core_args, self.seq_per_img)
        weak_rela = expand_feats([weak_rela], self.seq_per_img)[0]

        batch_size = fc_feats.size(0) * self.seq_per_img
        state = self.init_hidden(batch_size, weak_rela)
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size + 1)
        outputs_tag = fc_feats.new_zeros(batch_size, seq.size(1) - 1, 4)

        # teacher forcing
        for i in range(seq.size(1) - 1):
            # scheduled sampling
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            # output, state = self.get_logprobs_state(it, state, core_args)
            output, output_tag, state = self.get_logprobs_state(it, state, core_args)
            outputs[:, i] = output
            outputs_tag[:, i] = output_tag

        _, _, attr_feats, obj_feats, rela_feats, _, _, _, att_masks, rela_masks, attr_masks = core_args
        node_feats = torch.cat([obj_feats, attr_feats, rela_feats], dim=1)

        outputs_node = torch.log_softmax(self.logit_node(node_feats), dim=-1)

        return outputs, outputs_tag, outputs_node

    def get_logprobs_state(self, it, state, core_args):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, state, core_args)
        logprobs = torch.log_softmax(self.logit(output), dim=1)
        logprobs_tag = torch.log_softmax(self.logit_tag(output), dim=1)

        # return logprobs, state
        return logprobs, logprobs_tag, state

    # sample sentences with greedy decoding
    def _sample(self, sg_data, fc_feats, att_feats, weak_rela, att_masks=None, opt={},
                _core_args=None):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        return_core_args = opt.get('return_core_args', False)
        expand_features = opt.get('expand_features', True)

        if beam_size > 1:
            return self._sample_beam(sg_data, fc_feats, att_feats, weak_rela, att_masks, opt)
        if _core_args is not None:
            # reuse the core_args calculated during generating sampled captions
            # when generating greedy captions for SCST,
            core_args = _core_args
        else:
            core_args = self.prepare_core_args(sg_data, fc_feats, att_feats, att_masks)

        # make seq_per_img copies of the encoded inputs:  shape: (B, ...) => (B*seq_per_image, ...)
        # should be True when training (xe or scst), False when evaluation
        if expand_features:
            if return_core_args:
                _core_args = core_args
            core_args = expand_feats(core_args, self.seq_per_img)
            weak_rela = expand_feats([weak_rela], self.seq_per_img)[0]
            batch_size = fc_feats.size(0) * self.opt.seq_per_img
        else:
            batch_size = fc_feats.size(0)

        state = self.init_hidden(batch_size, weak_rela)
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            logprobs, _, state = self.get_logprobs_state(it, state, core_args)
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp
            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        returns = [seq, seqLogprobs]
        if return_core_args:
            returns.append(_core_args)
        return returns

    # sample sentences with beam search
    def _sample_beam(self, sg_data, fc_feats, att_feats, weak_relas, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        core_args = self.prepare_core_args(sg_data, fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            weak_rela = expand_feats([weak_relas[k: k + 1]], beam_size)[0]
            state = self.init_hidden(beam_size, weak_rela)
            sample_core_args = []
            for item in core_args:
                if type(item) is list or item is None:
                    sample_core_args.append(item)
                    continue
                else:
                    sample_core_args.append(item[k:k + 1])
            sample_core_args = expand_feats(sample_core_args, beam_size)

            for t in range(1):
                if t == 0:  # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, _, state = self.get_logprobs_state(it, state, sample_core_args)

            self.done_beams[k] = self.beam_search(state, logprobs, sample_core_args, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)


class WGCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(WGCore, self).__init__()
        self.opt = opt
        self.n_heads = opt.n_heads
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)  # we, fc, h^2_t-1
        lang_lstm_in_dim = opt.rnn_size + opt.rnn_size * 3
        self.lang_lstm = nn.LSTMCell(lang_lstm_in_dim, opt.rnn_size)  # h^1_t, \hat v

        self.attention_obj = Attention(opt)
        self.attention_attr = Attention(opt)
        self.attention_rela = Attention(opt)

    def forward(self, xt, state, core_args):
        fc_feats, att_feats, attr_feats, obj_feats, rela_feats, p_attr_feats, \
        p_obj_feats, p_rela_feats, \
        att_masks, rela_masks, attr_masks = core_args
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        lang_lstm_input = h_att

        att_attr = self.attention_attr(h_att, attr_feats, p_attr_feats, attr_masks)
        lang_lstm_input = torch.cat([lang_lstm_input, att_attr], 1)

        att_obj = self.attention_obj(h_att, obj_feats, p_obj_feats, att_masks)
        lang_lstm_input = torch.cat([lang_lstm_input, att_obj], 1)

        att_rela = self.attention_rela(h_att, rela_feats, p_rela_feats, rela_masks)
        lang_lstm_input = torch.cat([lang_lstm_input, att_rela], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        output = torch.dropout(h_lang, self.drop_prob_lm, self.training)

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


class WGraphModel(AttModel):
    def __init__(self, opt):
        super(WGraphModel, self).__init__(opt)
        self.num_layers = 2
        self.core = WGCore(opt)
