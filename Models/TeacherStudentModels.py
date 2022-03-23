import torch.nn as nn
import torch.nn.functional as F
from Layers import layers
import torch
import numpy as np
import random


def init_separate_w(output_d, input_d, choices):
    existing_encoding = set()
    existing_encoding.add(tuple([0] * input_d))

    w = np.zeros((output_d, input_d))

    for i in range(output_d):
        while True:
            encoding = tuple(random.sample(choices, 1)[0] for j in range(input_d))
            if encoding not in existing_encoding:
                break
        for j in range(input_d):
            w[i, j] = encoding[j]
        existing_encoding.add(encoding)

    return w


def get_aug_w(w):
    # w: [output_d, input_d]
    # aug_w: [output_d + 1, input_d + 1]
    output_d, input_d = w.weight.size()
    aug_w = torch.zeros((output_d + 1, input_d + 1), dtype=w.weight.dtype, device=w.weight.device)
    aug_w[:output_d, :input_d] = w.weight.data
    aug_w[:output_d, input_d] = w.bias.data
    aug_w[output_d, input_d] = 1
    return aug_w


def set_add_noise(layer, teacher_layer, perturb):
    layer.weight.data[:] = teacher_layer.weight.data[:] + torch.randn(teacher_layer.weight.size()).cuda() * perturb
    layer.bias.data[:] = teacher_layer.bias.data[:] + torch.randn(teacher_layer.bias.size()).cuda() * perturb


def set_same_dir(layer, teacher_layer):
    norm = layer.weight.data.norm()
    r = norm / teacher_layer.weight.data.norm()
    layer.weight.data[:] = teacher_layer.weight.data * r
    layer.bias.data[:] = teacher_layer.bias.data * r


def set_same_sign(layer, teacher_layer):
    sel = (teacher_layer.weight.data > 0) * (layer.weight.data < 0) + (teacher_layer.weight.data < 0) * (
            layer.weight.data > 0)
    layer.weight.data[sel] *= -1.0

    sel = (teacher_layer.bias.data > 0) * (layer.bias.data < 0) + (teacher_layer.bias.data < 0) * (layer.bias.data > 0)
    layer.bias.data[sel] *= -1.0


def normalize_layer(layer):
    # [output, input]
    w = layer.weight.data
    for i in range(w.size(0)):
        norm = w[i].pow(2).sum().sqrt() + 1e-5
        w[i] /= norm
        if layer.bias is not None:
            layer.bias.data[i] /= norm


def init_w(layer, use_sep=True):
    sz = layer.weight.size()
    output_d = sz[0]
    input_d = 1
    for s in sz[1:]:
        input_d *= s

    if use_sep:
        choices = [-0.5, -0.25, 0, 0.25, 0.5]
        layer.weight.data[:] = torch.from_numpy(init_separate_w(output_d, input_d, choices)).view(*sz).cuda()
        if layer.bias is not None:
            layer.bias.data.uniform_(-.5, 0.5)


def init_w2(w, multiplier=5):
    w.weight.data *= multiplier
    w.bias.data.normal_(0, std=1)
    # w.bias.data *= 5
    for i, ww in enumerate(w.weight.data):
        pos_ratio = (ww > 0.0).sum().item() / w.weight.size(1) - 0.5
        w.bias.data[i] -= pos_ratio


class MLPTeacher(nn.Module):
    def __init__(self, hidden_units, in_units, out_units):
        super().__init__()
        self.hidden_layer1 = layers.Linear(in_units, hidden_units,
                                           layer_id=1)  # only want to prune layer with id 1 (first hl)
        self.hidden_layer2 = layers.Linear(hidden_units, out_units)
        self.activations = []

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = F.relu(x)
        self.activations.append(x.detach().clone())
        x = self.hidden_layer2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def reset_activations(self):
        self.activations = []


class MLPStudent(nn.Module):
    def __init__(self, hidden_units, in_units, out_units):
        super().__init__()
        self.hidden_layer1 = layers.Linear(in_units, hidden_units,
                                           layer_id=1)  # only want to prune layer with id 1 (first hl)
        self.hidden_layer2 = layers.Linear(hidden_units, out_units)
        self.activations = []

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = F.relu(x)
        self.activations.append(x.detach().clone())
        x = self.hidden_layer2(x)
        x = F.softmax(x, dim=1)
        return x

    def reset_activations(self):
        self.activations = []


class Model(nn.Module):
    def __init__(self, d, ks, d_output, multi=1, has_bn=True, has_bn_affine=True, has_bias=True, bn_before_relu=False):
        super(Model, self).__init__()
        self.d = d
        self.ks = ks
        self.has_bn = has_bn
        self.ws_linear = nn.ModuleList()
        self.ws_bn = nn.ModuleList()
        self.bn_before_relu = bn_before_relu
        last_k = d
        self.sizes = [d]
        self.activations = []

        for k in ks:
            if k == ks[0]:
                k *= multi
            self.ws_linear.append(layers.Linear(last_k, k, bias=has_bias))
            if has_bn:
                self.ws_bn.append(layers.BatchNorm1d(k, affine=has_bn_affine))
            self.sizes.append(k)
            last_k = k

        self.final_w = layers.Linear(last_k, d_output, bias=has_bias)
        self.relu = nn.ReLU()

        self.sizes.append(d_output)

    def set_teacher(self, teacher, perturb):
        for w_s, w_t in zip(self.ws, teacher.ws):
            set_add_noise(w_s, w_t, perturb)
        set_add_noise(self.final_w, teacher.final_w, perturb)

    def set_teacher_dir(self, teacher):
        for w_s, w_t in zip(self.ws, teacher.ws):
            set_same_dir(w_s, w_t)
        set_same_dir(self.final_w, teacher.final_w)

    def set_teacher_sign(self, teacher):
        for w_s, w_t in zip(self.ws, teacher.ws):
            set_same_sign(w_s, w_t)
        set_same_sign(self.final_w, teacher.final_w)

    def forward(self, x):
        hs = []
        pre_bns = []
        # bns = []
        h = x
        for i in range(len(self.ws_linear)):
            w = self.ws_linear[i]
            h = w(h)
            if i == 0:
                self.activations.append(h.detach().clone())
            if self.bn_before_relu:
                pre_bns.append(h)
                if len(self.ws_bn) > 0:
                    bn = self.ws_bn[i]
                    h = bn(h)
                h = self.relu(h)
            else:
                h = self.relu(h)
                pre_bns.append(h)
                if len(self.ws_bn) > 0:
                    bn = self.ws_bn[i]
                    h = bn(h)
            hs.append(h)
            # bns.append(h)
        y = self.final_w(hs[-1])
        return dict(hs=hs, pre_bns=pre_bns, y=y)

    def reset_activations(self):
        self.activations = []

    def init_w(self, use_sep=True):
        for w in self.ws_linear:
            init_w(w, use_sep=use_sep)
        init_w(self.final_w, use_sep=use_sep)

    def reset_parameters(self):
        for w in self.ws_linear:
            w.reset_parameters()
        for w in self.ws_bn:
            w.reset_parameters()
        self.final_w.reset_parameters()

    def normalize(self):
        for w in self.ws_linear:
            normalize_layer(w)
        normalize_layer(self.final_w)

    def from_bottom_linear(self, j):
        if j < len(self.ws_linear):
            return self.ws_linear[j].weight.data
        elif j == len(self.ws_linear):
            return self.final_w.weight.data
        else:
            raise RuntimeError("j[%d] is out of bound! should be [0, %d]" % (j, len(self.ws)))

    def from_bottom_aug_w(self, j):
        if j < len(self.ws_linear):
            return get_aug_w(self.ws_linear[j])
        elif j == len(self.ws_linear):
            return get_aug_w(self.final_w)
        else:
            raise RuntimeError("j[%d] is out of bound! should be [0, %d]" % (j, len(self.ws)))

    def num_layers(self):
        return len(self.ws_linear) + 1

    def from_bottom_bn(self, j):
        assert j < len(self.ws_bn)
        return self.ws_bn[j]
