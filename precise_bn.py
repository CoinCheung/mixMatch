
import torch
import torch.nn as nn


@torch.no_grad()
def precise_bn(model, dl):
    bn_modules = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    model.cuda()
    model.train()
    bn_layers = [
        m
        for m in model.modules()
        if m.training and isinstance(m, bn_modules)
    ]
    actual_momentums = [bn.momentum for bn in bn_layers]
    for bn in bn_layers: bn.momentum = 1.
    running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]

    for idx, batch in enumerate(dl):
        model(batch[0].cuda())
        for i, bn in enumerate(bn_layers):
            running_mean[i] += (bn.running_mean - running_mean[i]) / (idx + 1)
            running_var[i] += (bn.running_var - running_var[i]) / (idx + 1)
    for i, bn in enumerate(bn_layers):
        bn.running_mean = running_mean[i]
        bn.running_var = running_var[i]
        bn.momentum = actual_momentums[i]

    return model


#  class PreciseBN(object):
#
#      def __init__(self):
#          self.bn_modules = (
#              nn.BatchNorm1d,
#              nn.BatchNorm2d,
#              nn.BatchNorm3d,
#              nn.SyncBatchNorm,
#          )
#
#      @torch.no_grad()
#      def update_bn_status(self, model, dl):
#          bn_layers = [
#              m
#              for m in model.modules()
#              if m.training and isinstance(m, self.bn_modules)
#          ]
#          actual_momentums = [bn.momentum for bn in bn_layers]
#          for bn in bn_layers: bn.momentum = 1.
#          running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
#          running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]
#
#          for idx, ims, _ in enumerate(dl):
#              model(ims)
#              for i, bn in enumerate(bn_layers):
#                  running_mean[i] += (bn.running_mean - running_mean[i]) / (idx + 1)
#                  running_var[i] += (bn.running_var - running_var[i]) / (idx + 1)
#
#          for i, bn in enumerate(bn_layers):
#              bn.running_mean = running_mean[i]
#              bn.running_var = running_var[i]
#              bn.momentum = actual_momentums[i]
#
#          return model
