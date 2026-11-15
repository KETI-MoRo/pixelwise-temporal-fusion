import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

from inspect import signature

def aux_criterion(inputs, target):
    # for FCN and deeplab
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

class criterion_CEloss(nn.Module):
    def __init__(self,weight=None):
        super(criterion_CEloss, self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, output, target):
        return self.loss(F.log_softmax(output, dim=1), target)

class Bicriterion(nn.Module):
    def __init__(self, mask_loss):
        super(Bicriterion, self).__init__()
        self.mask_loss = mask_loss

    def forward(self, output, target):
        # output can be Tensor or dict with 'out'
        logits = output['out'] if isinstance(output, dict) else output
        return self.mask_loss(logits, target)

# 이진클래스 Dice
class DiceLoss_Binary(nn.Module):
    def __init__(self):
        super(DiceLoss_Binary, self).__init__()
        self.ep = 1e-8

    def forward(self, output, target):
        pred = (F.softmax(output, dim=1)[:, 0]).float()
        target = (target > 0).float()
        intersection = 2 * torch.sum(pred * target) + self.ep
        union = torch.sum(pred) + torch.sum(target) + self.ep
        loss = 1 - intersection / union
        return loss

# 멀티클래스 Dice (무게중심: IoU 직접 개선)
class DiceLoss_Multi(nn.Module):
    def __init__(self, num_classes=None, ignore_index=255, smooth=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes  # None이면 logits에서 C를 사용
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.register_buffer('class_weights', class_weights if class_weights is not None else None)

    def forward(self, logits, target):
        # logits: (N,C,H,W), target: (N,H,W)
        N, C, H, W = logits.shape
        if self.num_classes is None:
            self.num_classes = C
        probs = F.softmax(logits, dim=1)

        # ignore mask
        valid = (target != self.ignore_index)
        if not valid.any():
            return logits.new_tensor(0.0, requires_grad=True)

        # one-hot
        oh = torch.zeros((N, C, H, W), device=logits.device, dtype=probs.dtype)
        tgt = target.clone()
        tgt[~valid] = 0
        oh.scatter_(1, tgt.unsqueeze(1), 1.0)

        # mask out ignored pixels
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        oh    = oh    * mask

        dims = (0, 2, 3)
        inter = (probs * oh).sum(dims)                 # (C,)
        denom = probs.sum(dims) + oh.sum(dims)         # (C,)
        dice  = (2 * inter + self.smooth) / (denom + self.smooth)  # (C,)

        # 클래스 가중 평균(옵션)
        if self.class_weights is not None:
            w = self.class_weights
            w = w / (w.sum() + 1e-12)
            loss = 1.0 - (w * dice).sum()
        else:
            loss = 1.0 - dice.mean()
        return loss

# BCE 기반 이진 세그용
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        if isinstance(pred, OrderedDict):
            pred = pred['out']
        if true.dim() == 3:
            true.unsqueeze_(1)
            true = torch.cat([1-true, true], dim=1)
        true = true.float()
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

# 멀티클래스 Focal (CE 기반: 희소클래스 집중)
class FocalLossCE(nn.Module):
    """
    Multiclass focal loss built on top of CrossEntropy (not BCE).
    - gamma: focusing parameter
    - alpha: None(무시) | float(스칼라) | Tensor(shape=[C]) per-class weights
    - class_weights: CE의 weight(loss_weight)와 동시 사용 가능
    - label_smoothing은 focal과 상충 가능하므로 보통 0.0 권장
    """
    def __init__(self, gamma=2.0, alpha=None, class_weights=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer('alpha', alpha if isinstance(alpha, torch.Tensor) else None)
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, reduction='none')
        self.alpha_scalar = float(alpha) if (alpha is not None and not torch.is_tensor(alpha)) else None

    def forward(self, logits, target):
        # logits: (N,C,H,W), target: (N,H,W)

        # dict를 언랩
        # logits may be dict or Tensor
        if isinstance(logits, dict):
            logits = logits['out']

        N, C, H, W = logits.shape
        valid = (target != self.ignore_index)
        if not valid.any():
            return logits.new_tensor(0.0, requires_grad=True)

        # per-pixel CE
        ce = self.ce(logits, target)  # (N,H,W)
        # pt = exp(-CE)
        pt = torch.exp(-ce)

        # focal weight
        fw = (1.0 - pt) ** self.gamma

        # alpha handling
        if self.alpha is not None:                # per-class alpha
            with torch.no_grad():
                a = self.alpha[target.clamp_min(0)]  # (N,H,W)
            fw = fw * a
        elif self.alpha_scalar is not None:       # scalar alpha
            fw = fw * self.alpha_scalar

        loss = fw * ce
        loss = loss[valid].mean()
        return loss


# --- 폴백: 파이토치 구버전에서 label_smoothing을 직접 구현 ---
class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, smoothing=0.0):
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else None)
        self.ignore_index = ignore_index
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        # logits: (N, C, H, W), target: (N, H, W) with long dtype
        N, C, H, W = logits.shape
        # mask ignore_index
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            # 모든 픽셀이 ignore면 0.으로 처리
            return logits.new_tensor(0.0, requires_grad=True)

        # (N*H*W, C) / (N*H*W,)
        logits_f = logits.permute(0, 2, 3, 1)[valid_mask].reshape(-1, C)
        target_f = target[valid_mask].reshape(-1)

        log_probs = F.log_softmax(logits_f, dim=1)  # (M, C)
        # one-hot with label smoothing: y = (1-ε) * onehot + ε/C
        eps = self.smoothing
        # NLL part for correct class
        nll = -log_probs.gather(dim=1, index=target_f.unsqueeze(1)).squeeze(1)
        # mean log-prob over classes
        smooth = -log_probs.mean(dim=1)

        # class weight (optional)
        if self.weight is not None:
            w = self.weight[target_f]  # (M,)
            loss = w * ((1.0 - eps) * nll + eps * smooth)
        else:
            loss = (1.0 - eps) * nll + eps * smooth

        return loss.mean()


def _ce_with_optional_smoothing(weight=None, ignore_index=255, label_smoothing=0.0):
    # 파이토치가 CE의 label_smoothing을 지원하는지 확인
    if 'label_smoothing' in signature(nn.CrossEntropyLoss).parameters:
        return nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            label_smoothing=float(label_smoothing)
        )
    else:
        # 폴백(구버전)
        return SmoothCrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            smoothing=float(label_smoothing)
        )


def get_loss(
        name,
        loss_weight=None,
        label_smoothing: float = 0.0,
        *,
        focal_gamma: float = 2.0,
        focal_alpha=None,                 # None | float | Tensor([C])
        dice_lambda: float = 0.5,
        ignore_index: int = 255
        ):
    """
    name: 'ce' | 'focalloss_binary' | 'focalloss_multi' | 'aux' | 'bi' | 'dice_binary' | 'dice_multi' ...
    loss_weight: class weight tensor or None (이미 caller에서 device로 이동했으면 그대로 전달)
    label_smoothing: epsilon (0.0 이면 비활성)
    """
    print("loss.py$ Loss: {}".format(name))

    if name == 'focalloss_binary':
        # BCE 기반 이진 세그용
        BCEseg = nn.BCEWithLogitsLoss()
        FLseg = FocalLoss(BCEseg)
        return FLseg

    elif name == 'focalloss_multi':
        # 멀티클래스 Focal (CE 기반: 희소클래스 집중)
        # alpha를 per-class로 쓰고 싶으면, caller에서 Tensor를 loss_weight와 별도로 전달해야 함.
        return FocalLossCE(
            gamma=focal_gamma,
            alpha=focal_alpha,
            class_weights=loss_weight,
            ignore_index=ignore_index,
            )
    
    elif name == 'ce':
        # CE + (선택적) Label Smoothing
        if label_smoothing < 0.0 or label_smoothing >= 1.0:
            raise ValueError("label_smoothing must satisfy 0.0 <= epsilon < 1.0")
        if label_smoothing > 0.0:
            print(f"loss.py$ [loss] Using CrossEntropy with label_smoothing={label_smoothing}")
        return _ce_with_optional_smoothing(
            weight=loss_weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
            )
    
    elif name == 'ce_dice_multi':
        # CE + λ·Dice (멀티클래스)
        if label_smoothing < 0.0 or label_smoothing >= 1.0:
            raise ValueError("label_smoothing must satisfy 0.0 <= epsilon < 1.0")
        ce = _ce_with_optional_smoothing(
            weight=loss_weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
            )
        dice = DiceLoss_Multi(
            num_classes=None,
            ignore_index=ignore_index,
            smooth=1.0,
            class_weights=None
            )
        def _combo(output, target):
            logits = output['out'] if isinstance(output, dict) else output
            return ce(logits, target) + float(dice_lambda) * dice(logits, target)
        return _combo
    
    elif name == 'aux':
        return aux_criterion
    
    elif name == 'bi':
        mask_loss = _ce_with_optional_smoothing(
            weight=loss_weight,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
            )
        return Bicriterion(mask_loss)
    
    elif name == 'dice_binary':
        dice_loss = DiceLoss_Binary()
        return Bicriterion(dice_loss)
    
    elif name == 'dice_multi':
        # 멀티클래스 Dice
        dice_loss = DiceLoss_Multi(
            num_classes=None,
            ignore_index=ignore_index,
            smooth=1.0,
            class_weights=None
            )
        return Bicriterion(dice_loss)
    
    else:
        raise ValueError(name)
