import sys

import torch
import torch.nn as nn 
import torch.nn.functional as F

def get_loss_with_name(name):
    try:
        identifier = getattr(sys.modules[__name__], name)
    except AttributeError:
        raise NameError(f"{name} is not a valid loss function.")
    if isinstance(identifier, type):
        return identifier
    raise TypeError(f"{name} is not a valid loss function.")


def build_loss_fn(loss_fn, pos_class_weight=1, neg_class_weight=1, task_type="classification"):


    if task_type == "classification":
        # if loss_fn == "FocalLoss":
        #     alpha = kwargs.pop("focal_alpha",0.25)
        #     gamma = kwargs.pop("focal_gamma",2.0)
        #     return FocalLoss(alpha=alpha,gamma=gamma)
        if loss_fn == "WeakCrossEntropyLoss":
            return WeakCrossEntropyLoss()
        else:
            neg_class_weight = neg_class_weight
            pos_class_weight = pos_class_weight
            loss_weights = torch.tensor([float(neg_class_weight),float(pos_class_weight)])
            return nn.CrossEntropyLoss(weight=loss_weights)

    elif task_type == "regression" or task_type == "autoencoder" :
        return nn.MSELoss()

    else:
        raise ValueError(f"{task_type} is not a valid task type!")

class FocalLoss(nn.Module):
    """
    binary focal loss
    """

    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = [alpha, 1-alpha]
        # self.nllLoss = nn.NLLLoss(weight=self.weight)
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight= input_tensor.new(self.weight),
            reduction = self.reduction
        )


class WeakCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = 'mean', weight=None, eps: float = 1e-8) -> None:
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        log_likelihood = F.log_softmax(input, dim=-1)
        # likelihood = F.softmax(input, dim=-1)
        loss = -(target * log_likelihood[:, 1] + (1-target) * log_likelihood[:, 0])

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        raise NotImplementedError