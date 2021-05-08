import sys

import torch
from torch._C import Value
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.utils.dummy_pt_objects import RobertaForQuestionAnswering

def get_loss_with_name(name):
    try:
        identifier = getattr(sys.modules[__name__], name)
    except AttributeError:
        raise NameError(f"{name} is not a valid loss function.")
    if isinstance(identifier, type):
        return identifier
    raise TypeError(f"{name} is not a valid loss function.")


def build_loss_fn(kwargs):
    loss_fn = kwargs.pop("loss_fn",None)
    
    if loss_fn is None:
        raise ValueError("Must pass a loss function name")
    
    if loss_fn == "CrossEntropyLoss":
        neg_class_weight = kwargs.pop("neg_class_weight")
        pos_class_weight = kwargs.pop("pos_class_weight")
        loss_weights = torch.tensor([float(neg_class_weight),float(pos_class_weight)])
        return nn.CrossEntropyLoss(weight=loss_weights)
    
    elif loss_fn == "FocalLoss":
        alpha = kwargs.pop("focal_alpha")
        gamma = kwargs.pop("focal_gamma")
        return FocalLoss(alpha=alpha,gamma=gamma)

    else:
        raise ValueError(f"{loss_fn} is not a valid loss function!")

class FocalLoss(nn.Module):
    """
    binary focal loss
    """

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.weight = torch.Tensor([alpha, 1-alpha])
        self.nllLoss = nn.NLLLoss(weight=self.weight)
        self.gamma = gamma

    def forward(self, input, target):
        softmax = F.softmax(input, dim=1)
        log_logits = torch.log(softmax)
        fix_weights = (1 - softmax) ** self.gamma
        logits = fix_weights * log_logits
        loss = self.nllLoss(logits, target)
        return loss