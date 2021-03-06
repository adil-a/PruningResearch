# Code based on https://github.com/ganguli-lab/Synaptic-Flow/blob/378fcee0c1dafcecc7ec177e44989419809a106b/Pruners/pruners.py

import torch
from Models.TeacherStudentModels import MLPStudent


class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}
        self.intermediate_masks = None  # used for teacher student setting when doing structured magnitude pruning on
        # MLP

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity, mag_ts, specialized_ts):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        if specialized_ts:
            for i in range(len(self.scores)):
                weight_mask, bias_mask = self.scores[i]
                if i == 0:
                    weight_mask.mul_(self.intermediate_masks)
                    bias_mask.mul_(torch.flatten(self.intermediate_masks))
                else:
                    weight_mask.mul_(self.intermediate_masks.T)
            return
        # Threshold scores
        if mag_ts:
            global_scores = torch.flatten(self.scores[0])
        else:
            global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            if mag_ts:
                score = self.scores[0].to(self.intermediate_masks.device)
                zero = torch.tensor([0.]).to(self.intermediate_masks.device)
                one = torch.tensor([1.]).to(self.intermediate_masks.device)
                self.intermediate_masks.copy_(torch.where(score <= threshold, zero, one))
                for i in range(1, len(self.scores)):
                    weight_mask, bias_mask = self.scores[i]
                    if i == 1:
                        weight_mask.mul_(self.intermediate_masks)
                        bias_mask.mul_(torch.flatten(self.intermediate_masks))
                    else:
                        weight_mask.mul_(self.intermediate_masks.T)
            else:
                # print(threshold)
                for mask, param in self.masked_parameters:
                    score = self.scores[id(param)]
                    zero = torch.tensor([0.]).to(mask.device)
                    one = torch.tensor([1.]).to(mask.device)
                    mask.copy_(torch.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity, mag_ts, specialized_ts):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope, mag_ts=False, specialized_ts=False):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity, mag_ts, specialized_ts)
        if scope == 'local':
            self._local_mask(sparsity, mag_ts, specialized_ts)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, _ in self.masked_parameters:
            mask_copy = mask.data.view(-1)
            idx = torch.randperm(mask_copy.numel())
            mask.data = mask_copy[idx].view(mask.size())

    def invert(self):
        for v in self.scores.values():
            v.div_(v ** 2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params

class SpecializedTS(Pruner):
    def __init__(self, masked_parameters):
        super(SpecializedTS, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        self.scores[0] = (self.masked_parameters[0][0], self.masked_parameters[1][0])
        self.scores[1] = (self.masked_parameters[2][0], self.masked_parameters[3][0])


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        # if isinstance(model, MLPStudent):
        #     first_layer_weights_and_masks = self.masked_parameters[0]
        #     second_layer_weights_and_masks = self.masked_parameters[2]
        #     first_layer_abs_weights = torch.clone(first_layer_weights_and_masks[1]).detach().abs_()
        #     second_layer_abs_weights = torch.clone(second_layer_weights_and_masks[1].T).detach().abs_()
        #     first_layer_column_vec = torch.sum(first_layer_abs_weights, dim=1, keepdim=True)
        #     second_layer_column_vec = torch.sum(second_layer_abs_weights, dim=1, keepdim=True)
        #     # actual scores
        #     self.scores[0] = first_layer_column_vec + second_layer_column_vec
        #     # first layer masks
        #     self.scores[1] = (self.masked_parameters[0][0], self.masked_parameters[1][0])
        #     # second layer masks
        #     self.scores[2] = (self.masked_parameters[2][0], self.masked_parameters[3][0])
        #     self.intermediate_masks = torch.ones(first_layer_column_vec.shape).to(device)
        # else:
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(device)  # , dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)
