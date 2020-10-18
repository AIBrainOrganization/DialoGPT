  
import numpy as np
import random
from scipy.io.wavfile import read
import torch
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join


def load_pretrained_model(finetune_model, pretrained_path, model_optim=False,
        resume=False, freeze_pretrained=False, except_for=['nothing']):
    '''
        load pretrained model to finetun_model.
        finetune_model (nn.Module): model will be fine tuned.
        pretrained_path (str): path to pretrained model. state_dict should be indexed.
        freeze_pretrained (bool or list): freeze all pretrained weight or given list.
    '''
    checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    #print(checkpoint['lm_head.weight']
    for key in checkpoint:
        print(key)
    feed_weight = checkpoint[key].copy()

    #if type(freeze_pretrained) == list:
    #    frozen_weights = freeze_pretrained
    if freeze_pretrained:
        frozen_weights = list(feed_weight.keys())
        if except_for[0] != 'nothing':
            for except_key in except_for:
                frozen_weights = [xx for xx in frozen_weights if except_key not in xx]
    else:
        frozen_weights = []

    finetune_state_dict = finetune_model.state_dict()

    # If pretrained weights have different shape or non-exist, then compensate it.
    for k, v in checkpoint['state_dict'].items():
        if k in finetune_state_dict.keys():
            cp_tensor = checkpoint['state_dict'][k]
            ft_tensor = finetune_state_dict[k]
            if cp_tensor.shape != ft_tensor.shape:
                feed_weight[k] = finetune_state_dict[k]
                if len(cp_tensor.shape) == len(ft_tensor.shape):
                    ft_import_dim = list()
                    for i_dim in range(len(cp_tensor.shape)):
                        if cp_tensor.shape[i_dim] <= ft_tensor.shape[i_dim]:
                            ft_import_dim.append(cp_tensor.shape[i_dim])
                    if len(ft_import_dim) == len(cp_tensor.shape):
                        d = ft_import_dim
                        if len(d) == 1:
                            feed_weight[k][:d[0]] = cp_tensor
                        elif len(d) == 2:
                            feed_weight[k][:d[0],:d[1]] = cp_tensor
                        elif len(d) == 3:
                            feed_weight[k][:d[0],:d[1],:d[2]] = cp_tensor
                        else:
                            print("Implement more dimensions")
                            exit()
                        print("{} weights are partially imported.".format(k))
                if k in frozen_weights:
                    frozen_weights.remove(k)
                resume = False
                print('[{}] Weights in model-will-be-finetuned is not in pretrained model. Resume is not available'.format(k))
            else:
                # k is in finetune network and shape is same.
                pass
        else:
            del feed_weight[k]
            if k in frozen_weights:
                frozen_weights.remove(k)

    # If new weights in finetune network is not in pretrained weight,
    for k, v in finetune_state_dict.items():
        if k not in feed_weight.keys():
            feed_weight[k] = v
    finetune_model.load_state_dict(feed_weight)

    # freeze params
    for name, param in finetune_model.named_parameters():
        if name in frozen_weights:
            param.requires_grad = False
        print('{}\t{}\t{}'.format(name, param.shape, param.requires_grad))

    print('loaded checkpoint %s' % (pretrained_path))

    if resume:
        start_epoch = checkpoint['epoch']
        model_optim.load_state_dict(checkpoint['optimizer'])
        for state in model_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        plot_losses = checkpoint['plot_losses']
    else:
        start_epoch = 0
        plot_losses = []

    return finetune_model, model_optim, start_epoch, plot_losses
