from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def setup(opt):
    if opt.caption_model == 'wgmodel':
        from .WGraphModel import WGraphModel
        model = WGraphModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model
