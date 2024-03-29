from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import logging
import pickle
import torch
import torch.utils.data as data
from functools import reduce


class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split == 'train', self.opt.loader_num_workers,
                                                    self.opt)
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.att_feat_size = opt.att_feat_size
        self.logger = logging.getLogger('__main__')
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.is_limited = opt.is_limited

        # feature related options
        self.use_att = getattr(opt, 'use_att', True)
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # data dir
        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir

        # scene graph data
        self.attr_data_dir = opt.attr_data_dir
        self.sg_data_dir = opt.sg_data_dir
        self.sg_vocab = {v: k for k, v in json.load(open(opt.input_json))['ix_to_word'].items()}

        # load the json file which contains additional information about the dataset
        self.logger.info('DataLoader loading json file: %s' % opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        self.logger.info('vocab size is %d' % self.vocab_size)

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        self.logger.info('max sequence length in data is %d' % self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        self.logger.info('read %d image features' % (self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)
        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        for split in self.split_ix.keys():
            self.logger.info('assigned %d images to split %s' % (len(self.split_ix[split]), split))

        # load the width and height of images
        if self.use_box:
            self.logger.info('Loading sg_box_info')
            self.sg_box_info = pickle.load(open(opt.sg_box_info_path))

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train', self.opt.loader_num_workers,
                                                        self.opt)
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1  # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            tag = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
                tag[q, :] = self.h5_label_file['tags'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]
            tag = self.h5_label_file['tags'][ixl: ixl + seq_per_img, :self.seq_length]

        return seq, tag

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = []  # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = []  # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        sg_batch = []
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        tag_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='float32')
        infos = []
        gts = []
        wrapped = False

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att, tmp_sg, \
            ix, tmp_wrapped = self._prefetch_process[split].get()
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            sg_batch.append(tmp_sg)

            label_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1], \
            tag_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1] \
                = self.get_captions(ix, seq_per_img)

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        data = {}
        data['fc_feats'] = np.stack(reduce(lambda x, y: x + y, [[_] * 1 for _ in fc_batch]))
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1

        data['labels'] = label_batch  # np.vstack(label_batch)
        data['tags'] = tag_batch
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts  # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        sg_batch_data = self.batch_sg(sg_batch, max_att_len)
        data['sg_data'] = {k: v for k, v in sg_batch_data.items() if k != 'verb_labels'}

        data['verbs'] = sg_batch_data['verb_labels']

        max_attr_num = data['sg_data']['attr_masks'].shape[-1]
        max_rela_num = data['sg_data']['rela_feats'].shape[-1]

        node_obj = np.zeros((len(att_batch), max_att_len), dtype='int')
        node_attr = np.ones((len(att_batch), max_att_len * max_attr_num), dtype='int')
        node_rela = np.ones((len(att_batch), max_rela_num), dtype='int') * 2
        node_labels = np.concatenate([node_obj, node_attr, node_rela], axis=-1)
        node_masks = np.concatenate(
            [data['att_masks'], data['sg_data']['attr_masks'].reshape(len(att_batch), -1),
             data['sg_data']['rela_masks']],
            axis=-1)

        data['node_labels'] = np.zeros([len(att_batch) * seq_per_img, node_labels.shape[1]], dtype='int')
        data['node_masks'] = np.zeros([len(att_batch) * seq_per_img, node_masks.shape[1]], dtype='float')
        for i in range(len(att_batch)):
            for j in range(seq_per_img):
                data['node_labels'][i * seq_per_img + j] = node_labels[i]
                data['node_masks'][i * seq_per_img + j] = node_masks[i]
        return data

    def batch_sg(self, sg_batch, max_att_len):
        "batching object, attribute, and relationship data"
        obj_batch = [_['obj'] for _ in sg_batch]
        rela_batch = [_['rela'] for _ in sg_batch]
        verb_batch = [_['verb'] for _ in sg_batch]
        attr_batch = [_['attr'] for _ in sg_batch]
        attr_mask_batch = [_['attr_mask'] for _ in sg_batch]
        sg_data = {}

        # obj labels, shape: (B, No, 1)
        sg_data['obj_labels'] = np.zeros([len(obj_batch), max_att_len, 1], dtype='int')
        for i in range(len(obj_batch)):
            sg_data['obj_labels'][i, :obj_batch[i].shape[0]] = obj_batch[i]

        # verb labels, shape: (B, No)
        sg_data['verb_labels'] = np.zeros([len(verb_batch), max_att_len], dtype='int')
        for i in range(len(verb_batch)):
            sg_data['verb_labels'][i, :verb_batch[i].shape[0]] = verb_batch[i]

        # rela
        max_rela_len = max([_['edges'].shape[0] for _ in rela_batch])
        sg_data['rela_edges'] = np.zeros([len(rela_batch), max_rela_len, 2], dtype='int')
        sg_data['rela_feats'] = np.zeros([len(rela_batch), max_rela_len], dtype='int')
        # rela_masks, because no all items in rela_edges and rela_feats are meaningful
        sg_data['rela_masks'] = np.zeros(sg_data['rela_edges'].shape[:2], dtype='float32')
        sg_data['node_rela'] = np.full(sg_data['rela_edges'].shape[:2], 2, dtype='int')

        for i in range(len(rela_batch)):
            sg_data['rela_edges'][i, :rela_batch[i]['edges'].shape[0]] = rela_batch[i]['edges']
            sg_data['rela_feats'][i, :rela_batch[i]['edges'].shape[0]] = rela_batch[i]['feats']
            sg_data['rela_masks'][i, :rela_batch[i]['edges'].shape[0]] = 1

        # attr
        max_attr_len = max([_.shape[1] for _ in attr_batch])
        sg_data['attr'] = np.zeros([len(attr_batch), max_att_len, max_attr_len], dtype='int')
        for i in range(len(attr_batch)):
            for j in range(attr_batch[i].shape[0]):
                sg_data['attr'][i, j, :attr_batch[i].shape[1]] = attr_batch[i][j]

        sg_data['attr_masks'] = np.zeros([len(attr_batch), max_att_len, max_attr_len], dtype='int')
        for i in range(len(attr_batch)):
            for j in range(attr_batch[i].shape[0]):
                sg_data['attr_masks'][i, j, :int(attr_mask_batch[i][j].sum())] = 1
        if self.is_limited:
            if max_attr_len > 2:
                sg_data['attr'] = sg_data['attr'][:, :, :2]
                sg_data['attr_masks'] = sg_data['attr_masks'][:, :, :2]

        # obj_rela
        sg_data['obj_rela_masks'] = np.zeros([len(rela_batch), max_att_len, max_rela_len], dtype='int')
        for i in range(len(rela_batch)):
            for j in range(rela_batch[i]['edges'].shape[0]):
                start_ix, end_ix = rela_batch[i]['edges'][j][0], rela_batch[i]['edges'][j][1]
                sg_data['obj_rela_masks'][i, int(start_ix), j] = 1
                sg_data['obj_rela_masks'][i, int(end_ix), j] = 1

        return sg_data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        image_id = str(self.info['images'][ix]['id'])
        if self.use_att:
            att_feat = np.load(os.path.join(self.input_att_dir, '{}.npz'.format(image_id)))['feat']
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = self.get_box_feat(image_id)
                att_feat = np.hstack([att_feat, box_feat])
        else:
            att_feat = np.zeros((1, 1, 1))
        fc_feat = np.load(os.path.join(self.input_fc_dir, image_id + '.npy'))

        sg_data = self.get_graph_data(index)

        return (fc_feat,
                att_feat,
                sg_data,
                ix)

    def get_graph_data(self, index):
        image_id = str(self.info['images'][index]['id'])
        sg_use = np.load(os.path.join(self.sg_data_dir, '{}.npz'.format(image_id)))

        if sg_use['prela'].shape[0] == 0:
            triplet_p = np.array([[0, 0, self.sg_vocab['near']]], dtype=sg_use['prela'].dtype)
        else:
            triplet_p = sg_use['prela']

        triplet_w = sg_use['wrela']
        rela = {}
        rela['edges'] = np.vstack([triplet_p[:, :2], triplet_w[:, :2]])
        # print ('pw', triplet_p[:, 2].shape, triplet_w[:, 2].shape)
        rela['feats'] = np.squeeze(np.vstack([triplet_p[:, 2:], triplet_w[:, 2:]]), axis=1)

        obj = sg_use['obj'][:, 1:2]  # shape (No, ?)
        attr = sg_use['attr']
        attr_mask = sg_use['attr_mask']
        sg_data = {'obj': obj, 'rela': rela, 'verb': np.unique(triplet_w[:, 2]), 'attr': attr, 'attr_mask': attr_mask}
        return sg_data

    def get_box_feat(self, image_id):
        image = self.sg_box_info[int(image_id)]
        x1, y1, x2, y2 = np.hsplit(image['boxes'], 4)
        h, w = image[int(image_id)]['image_h'], image[int(image_id)]['image_w']
        iw, ih = x2 - x1 + 1, y2 - y1 + 1
        box_feat = np.hstack((0.5 * (x1 + x2) / w, 0.5 * (y1 + y2) / h, iw / w, ih / h, iw * ih / (w * h)))
        if self.norm_box_feat:
            box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
        return box_feat

    def __len__(self):
        return len(self.info['images'])


class SubsetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BlobFetcher():

    def __init__(self, split, dataloader, if_shuffle=False, num_workers=4, opt=None):
        self.opt = opt
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        self.num_workers = num_workers

    # Add more in the queue
    def reset(self):
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers,  # 4 is usually enough
                                                 worker_init_fn=None,
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[-1] == ix, "ix not equal"

        return tmp + [wrapped]
