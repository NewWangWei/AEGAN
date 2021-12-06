

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import seed
import h5py
import numpy as np
from PIL import Image
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

wnl = WordNetLemmatizer()


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')

    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']
            caption = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab


def encode_captions(imgs, params, wtoi, ttoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions

    label_arrays = []
    tag_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        Lt = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]
                    if w in ttoi:
                        Lt[j, k] = ttoi[w]
                    else:
                        Lt[j, k] = 3

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        tag_arrays.append(Lt)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    T = np.concatenate(tag_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', L.shape)
    return L, T, label_start_ix, label_end_ix, label_length


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']

    seed(123)  # make reproducible

    # create the vocab
    vocab = build_vocab(imgs, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table
    ttoi = json.load(open(params['tag_json']))
    # encode captions in large arrays, ready to ship to hdf5 file
    L, T, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi, ttoi)

    # create output h5 file
    N = len(imgs)
    f_lb = h5py.File(params['output_h5'] + '_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("tags", dtype='uint32', data=T)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):

        jimg = {}
        jimg['split'] = img['split']
        if 'filename' in img: jimg['file_path'] = os.path.join(img.get('filepath', ''),
                                                               img['filename'])  # copy it over, might need
        if 'cocoid' in img:
            jimg['id'] = img['cocoid']  # copy over & mantain an id, if present (e.g. coco ids, useful)
        elif 'imgid' in img:
            jimg['id'] = img['imgid']

        if params['images_root'] != '':
            with Image.open(os.path.join(params['images_root'], img['filepath'], img['filename'])) as _img:
                jimg['width'], jimg['height'] = _img.size

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])


def update_data(params):
    # coco_dicts
    coco_dic = json.load(open(params['ro_dict']))
    r2i = coco_dic['predicate_to_idx']  # index from 1 for these two dicts
    i2r = {v - 1: k.split()[0] for k, v in r2i.items()}  # start from zero and take the first verb of phrase
    # data / coco_pred_sg_rela.npy
    i2p = np.load(params['ori_rela_dict'], allow_pickle=True)[()]['i2w']  # index from 0
    i2p = {k: v.split()[0] for k, v in i2p.items()}
    # data/attribute.json
    a2i = json.load(open(params['attr_dict']))
    i2a = {(ix + 1): word for word, ix in a2i.items()}
    i2a = {ix + 1: i2a[ix + 1] for ix in range(params['num_of_attrs'])}

    print('Dict size of wrela, prela, attr --> {} {} {}'.format(len(r2i), len(i2p), len(i2a)))
    # big_vocab
    cocotalk = json.load(open(params['output_json'], 'r'))  # index from '1'
    i2w = cocotalk['ix_to_word']  # index from '1'
    w2i = {v: k for k, v in i2w.items()}
    w2i_lemma = {wnl.lemmatize(k): v for k, v in w2i.items()}

    key_to_append = list(i2r.values()) + list(i2p.values()) + list(i2a.values())
    for k in key_to_append:
        if k in w2i or k in w2i_lemma:
            continue

        idx = len(w2i) + 1
        assert idx > 9487, 'original idx changed'
        w2i.update({k: str(idx)})
        i2w.update({str(idx): k})
        print(k, str(idx))

    cocotalk['ix_to_word'] = i2w
    json.dump(cocotalk, open(params['output_json'].split('.')[0] + '_final.json', 'w'))

    print('updated size {}'.format(len(i2w)))

    cmb_folder = params['cmb_folder']
    sg_folder_final = params['sg_folder'] + '_final_5'
    if not os.path.exists(sg_folder_final):
        os.makedirs(sg_folder_final)
    for root, dirs, files in os.walk(cmb_folder):
        for name in tqdm(files):
            filename = os.path.join(root, name)
            filename2 = os.path.join(params['attribute_folder'], name)
            file_out = os.path.join(sg_folder_final, name)
            f = np.load(filename)
            f2 = np.load(filename2)
            wrela, prela, obj, attr_matrix, attr_mask = f['wrela'], f['prela'], f['obj'], f2[
                'attr'], f2['mask']  # obj, N * 2, rela, N * 3, wrela, N * 3
            for rela in prela:
                if i2p[rela[2]] in w2i:
                    rela[2] = int(w2i[i2p[rela[2]]])
                else:
                    rela[2] = int(w2i_lemma[i2p[rela[2]]])

            for rela in wrela:
                # print (rela, rela.shape)
                if i2r[rela[2]] in w2i:
                    rela[2] = int(w2i[i2r[rela[2]]])
                else:
                    rela[2] = int(w2i_lemma[i2r[rela[2]]])

            for ob in obj:
                if i2p[ob[1]] in w2i:
                    ob[1] = int(w2i[i2p[ob[1]]])
                else:
                    ob[1] = int(w2i_lemma[i2p[ob[1]]])

            for attr in attr_matrix:
                for ii in range(len(attr)):
                    if attr[ii] == 0:
                        continue
                    if i2a[attr[ii]] in w2i:
                        attr[ii] = w2i[i2a[attr[ii]]]
                    else:
                        attr[ii] = int(w2i_lemma[i2a[attr[ii]]])

            np.savez(file_out, wrela=wrela, prela=prela, obj=obj, attr=attr_matrix, attr_mask=attr_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='data/dataset_coco.json', help='input json file to process into hdf5')
    parser.add_argument('--tag_json', default='data/tags.json', help='input tag json file to process into hdf5')
    parser.add_argument('--output_json', default='data/cocotalk.json', help='output json file')
    parser.add_argument('--output_h5', default='data/cocotalk', help='output h5 file')
    parser.add_argument('--images_root', default='',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    parser.add_argument('--ro_dict', default='data/coco_dicts.json', help='coco_dicts for objs and predicates')
    parser.add_argument('--ori_rela_dict', default='data/coco_pred_sg_rela.npy',
                        help='original vrg dict for objs, predicates')
    parser.add_argument('--attr_dict', default='data/new_attribute.json',
                        help='original attribute dict')
    parser.add_argument('--cmb_folder', default='data/coco_cmb_vrg', help='original combined vrgs folder')
    parser.add_argument('--sg_folder', default='data/sg_data', help='original combined vrgs folder')
    parser.add_argument('--attribute_folder', default='data/attr', help='original attr folder')

    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--num_of_relas', default=1000, type=int, help='top n predicates chosen.')
    parser.add_argument('--num_of_attrs', default=200, type=int, help='top n predicates chosen.')
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    # main(params)
    update_data(params)
