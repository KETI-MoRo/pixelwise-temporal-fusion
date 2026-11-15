import argparse
import datetime
import os
import math
import time
import shutil
import numpy as np

import torch
import torch.nn.functional as F

from train import utils
from train import loss

from dataset import dataset_dict

# Optical Flow 관련 import
from util.flow_utils import read_flo_file, warp_using_flow_cuda

# FileOutputTerminal 클래스 import
import sys
from util.FileOutputTerminal import FileOutputTerminal

from model import model_dict
from model.fusion_module import FusionModule # ./src/model/fusion_module.py

from PIL import Image

# evaluate()에서 사용하는 모듈
from train.utils import ConfusionMatrix, MetricLogger

# seed 관련 import
import random

# EMA(Exponential Moving Average) weight tracking (학습 끝마다 EMA 가중치로 평가)
import contextlib
from util.ema_utils import EMA

# 호환형: Python 3.7+ (PEP 604 미사용)
from typing import Union


# `def save_p_or_f()`에서 사용하는, `.png`로 저장할 때 사용하는 색상 테이블
# https://github.com/DoctorKey/C-3PO/blob/main/src/dataset/idx2color.txt
'''
# ChangeSim 데이터셋에서 사용하는 idx2color
idx2color = [
    [  0,   0,   0],
    [ 81,  38,   0],
    [ 41,  36, 132],
    [ 25,  48,  16],
    [131, 192,  13]
]
'''
# C-3PO에서 사용하는 idx2color
#   파워포인트 스포이드로 class color 맞춘 결과값
idx2color = [
    [  0,   0,   0],    # S (static)
    [175, 218,   0],    # N (new)
    [215, 220, 124],    # M (missing)
    [231, 208, 240],    # Ro (rotated)
    [125,  64, 243]     # Re (Replaced)
]

# resume 시 args 변경 감지를 보강하는 헬퍼 함수
def assert_resume_args_compatible(current_args, ckpt_args):
    STRICT_RESUME_ARG_NAMES = [
        "train_dataset", "test_dataset", "input_size", "data_cv",
        "model", "mtf", "msf",
        "fusion_module_mode", "fusion_module_mode_variant",
        "fusion_module_channels", "fusion_module_kernel_size",
        "fusion_module_dropout_p", "fusion_module_dropout_cap",
        "opt", "lr", "momentum", "weight_decay",
        "lr_scheduler", "warmup", "warmup_period",
        "sgdr_t0", "sgdr_tmult", "sgdr_eta_min_mul",
        "loss", "loss_weight",
        "class_weights_gamma", "class_weights_boost_change",
        "class_weights_normalize", "class_weights_clip_min", "class_weights_clip_max",
        "loss_fusion_weight",
        "focal_gamma", "focal_alpha", "dice_weight",
        "ignore_index", "label_smoothing",
        "batch_size", "use_amp",
        "fusion_detach_epochs",
        "use_ema", "ema_decay",
        "seed", "deterministic", "disable_tf32",
        "randomflip",
    ]

    mismatches = []

    for name in STRICT_RESUME_ARG_NAMES:
        assert hasattr(current_args, name), f"Current args missing required field: {name}"
        assert hasattr(ckpt_args, name), f"Checkpoint args missing required field: {name}"

        current_val = getattr(current_args, name)
        ckpt_val = getattr(ckpt_args, name)

        if current_val != ckpt_val:
            mismatches.append((name, ckpt_val, current_val))

    assert not mismatches, (
        f"Resume aborted for checkpoint: {current_args.resume}\n"
        "CLI args do not match checkpoint args.\n"
        + "\n".join(
            [f" - {name}: checkpoint={old!r}, current={new!r}" for name, old, new in mismatches]
        )
    )

# resume 블록을 위한, optimizer state 디바이스 보정
def move_optimizer_state_to_device(optimizer, device: Union[torch.device, str]):
    """
    Some checkpoints save optimizer states (e.g., exp_avg, exp_avg_sq) on CPU.
    After optimizer.load_state_dict(...), move all state tensors to `device`.
    """
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        # Flat tensors (most common: exp_avg, exp_avg_sq, momentum_buffer ...)
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)
            # Nested containers (rare but safe to handle)
            elif isinstance(v, dict):
                for kk, vv in list(v.items()):
                    if torch.is_tensor(vv):
                        v[kk] = vv.to(device)
            elif isinstance(v, (list, tuple)):
                new_seq = []
                changed = False
                for vv in v:
                    if torch.is_tensor(vv):
                        new_seq.append(vv.to(device))
                        changed = True
                    else:
                        new_seq.append(vv)
                if changed:
                    state[k] = type(v)(new_seq)

# '--fusion-detach-epochs' 옵션이 사용하는 함수
# detach fusion-to-backbone gradients하는 스케줄을 구현
def fusion_input_with_curriculum(output, epoch, detach_epochs: int):
    """
    Returns the tensor to feed into FusionModule given the curriculum schedule.
    - First N epochs: output.detach()
    - Next N epochs:  linear ramp from detach to full (alpha = 1/N ... 1.0)
    - Afterwards:     output
    """
    if detach_epochs <= 0:
        return output  # disabled

    if epoch < detach_epochs:
        # detach phase
        #print(f"fusion-detach-epochs$ Epoch {epoch}: fusion uses DETACH (no grad to backbone).")
        return output.detach()
    
    elif epoch < 2 * detach_epochs:
        # ramp phase
        # epoch = N .. 2N-1  → steps = 1..N, alpha = steps / N
        steps = (epoch - detach_epochs) + 1
        N = float(detach_epochs)
        alpha = min(max(steps / N, 0.0), 1.0)
        #print(f"fusion-detach-epochs$ Epoch {epoch}: fusion uses RAMP (alpha={alpha:.3f}).")
        return alpha * output + (1.0 - alpha) * output.detach()
    
    else:
        # full attach
        return output

# seed 관련 유틸리티
def seed_everything(seed: int, deterministic: bool = False, disable_tf32: bool = False):
    import torch, os, random, numpy as np

    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # TF32 끄기(옵션)
    if disable_tf32 and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
    if disable_tf32 and hasattr(torch.backends, "cudnn"):
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False

    if deterministic:
        # best-effort determinism (이 버전에선 set_deterministic(True) 사용하지 않음)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 신버전이면 warn_only로 안전하게
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

        # cuBLAS 결정성
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True

# seed 관련 유틸리티
def seed_worker(worker_id):
    # 각 DataLoader 워커마다 Python/NumPy 시드를 일관되게 설정
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Cosine Warm Restarts (SGDR)
def lr_schedule_cosine_restarts_with_warmup(step, num_warmup, base_lr, eta_min,
                                            steps_per_epoch, t0_epochs, tmult):
    """
    CosineAnnealingWarmRestarts를 '스텝 단위'로 재현 (Warmup 포함).
    - step: 0.. (LambdaLR의 last_epoch과 동일, 매 batch마다 1씩 증가)
    - num_warmup: 워ーム업 스텝 수
    - base_lr: 초기 LR
    - eta_min: 각 사이클에서의 최저 LR
    - steps_per_epoch: 한 에폭의 스텝 수 (len(data_loader))
    - t0_epochs: 첫 사이클 길이(에폭 단위)
    - tmult: 다음 사이클 길이 배수
    반환: lr / base_lr  (LambdaLR용 배율)
    """
    # Warmup 구간
    if step < num_warmup:
        return step / float(max(1, num_warmup))

    # Warmup 이후, SGDR 사이클 계산 (모두 step 기준)
    t0_steps = max(int(round(t0_epochs * steps_per_epoch)), 1)
    t_cur = step - num_warmup  # warmup 이후 경과 스텝
    T_i = t0_steps             # 현재 사이클 길이(스텝)
    cycle_start = 0

    # 현재 사이클 찾기
    while t_cur >= T_i:
        t_cur -= T_i
        cycle_start += T_i
        T_i = int(round(T_i * tmult))

    # Cosine (t_cur in [0, T_i))
    cos_inner = math.pi * t_cur / float(max(1, T_i))
    lr = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(cos_inner))

    return lr / base_lr

# cosine learning rate 스케줄 함수 정의
def lr_schedule_cosine_with_warmup(step, total_iters, num_warmup, base_lr, final_lr):
    if step < num_warmup:
        return step / float(max(1, num_warmup))
    progress = (step - num_warmup) / float(max(1, total_iters - num_warmup))
    scaled = ((1 + math.cos(math.pi * progress)) / 2) * (base_lr - final_lr) + final_lr
    return scaled / base_lr

# linear learning rate 스케줄 함수 정의
def lr_schedule_linear_with_warmup(step, total_iters, num_warmup, base_lr, final_lr):
    if step < num_warmup:
        return step / float(max(1, num_warmup))
    progress = (step - num_warmup) / float(max(1, total_iters - num_warmup))
    scaled = (1 - progress) * (base_lr - final_lr) + final_lr
    return scaled / base_lr

# exp learning rate 스케줄 함수 정의
def lr_schedule_exp_with_warmup(step, total_iters, num_warmup, base_lr, final_lr):
    if step < num_warmup:
        return step / float(max(1, num_warmup))
    progress = (step - num_warmup) / float(max(1, total_iters - num_warmup))
    decay = (1 - progress) ** 0.9
    scaled = decay * (base_lr - final_lr) + final_lr
    return scaled / base_lr

# Warmup 이후 LR을 점진적으로 감소시켜 학습이 길게 이어지도록 함.
def get_scheduler_function(name, total_iters, final_lr=0.0, base_lr=1.0, num_warmup=0):
    
    # warm-up 구간 처리 포함

    # 단순 선택자 역할
    if name == 'cosine':
        return lambda step: lr_schedule_cosine_with_warmup(step, total_iters, num_warmup, base_lr, final_lr)
    elif name == 'linear':
        return lambda step: lr_schedule_linear_with_warmup(step, total_iters, num_warmup, base_lr, final_lr)
    elif name == 'exp':
        return lambda step: lr_schedule_exp_with_warmup(step, total_iters, num_warmup, base_lr, final_lr)
    elif name == 'none':
        print("Using constant learning rate schedule.")
        return lambda step: 1.0
    else:
        raise ValueError(f"Unknown scheduler name: {name}")

def create_dataloader(args):
    dataset = dataset_dict[args.train_dataset](args, train=True)
    dataset_test = dataset_dict[args.test_dataset](args, train=False)

    # DataLoader와 Sampler에 제너레이터/워커시드 연결
    # 고정 제너레이터 (데이터 순서/증강 일관화)
    g = torch.Generator()
    g.manual_seed(args.seed)

    train_sampler = torch.utils.data.RandomSampler(dataset, generator=g)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True,
        worker_init_fn=seed_worker,   # 워커별 시드
        generator=g                   # 샘플링 일관성
        )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn,
        worker_init_fn=seed_worker,   # 테스트도 안전하게 고정
        generator=g                   # 샘플링 일관성
        )

    return dataset, train_sampler, data_loader, dataset_test, data_loader_test

# class_weights 계산
def compute_class_weights_from_ratio(args, ratio):

    # [ 참고 ]
    # 클래스 별 ratio : [0.9154, 0.0245, 0.0282, 0.0144, 0.0175]

    """
    Build CE class weights from pure class ratios for a 5-class setup:
    [0:Static, 1:New, 2:Missing, 3:Rotated, 4:Replaced]

    Uses Inverse-Frequency Power:
        w_c = (p_c + eps)^(-gamma)
    then:
        - optional boost on change classes (1..4)
        - normalization (mean1/sum1/none)
        - clipping for stability

    Args (must be provided on args with 'class_weights_' prefix):
      .class_weights_gamma:         float (default 0.5)
      .class_weights_boost_change:  float (default 1.15)  # applied to classes 1..4
      .class_weights_normalize:     str   (default "mean1", choices: "mean1","sum1","none")
      .class_weights_clip_min:      float (default 0.25)
      .class_weights_clip_max:      float (default 3.0)

    ratio: iterable of 5 floats (sum≈1), aligned to label ids [0..4].

    Returns:
      np.ndarray float32, shape (5,)
    """
    eps = 1e-8
    r = np.asarray(ratio, dtype=np.float64).ravel()

    # --- 5-class only ---
    assert r.size == 5, f"compute_class_weights_from_ratio(): expected 5 classes, got {r.size}"
    assert np.all(np.isfinite(r)) and np.all(r >= 0), "ratio must be finite & non-negative"
    s = r.sum()
    assert s > 0, "ratio sum is 0"
    r = r / (s + eps)  # safety renorm

    # --- hyperparams (with sensible defaults) ---
    gamma        = float(getattr(args, "class_weights_gamma", 0.5))
    boost_change = float(getattr(args, "class_weights_boost_change", 1.15))
    normalize    = getattr(args, "class_weights_normalize", "mean1")
    clip_min     = float(getattr(args, "class_weights_clip_min", 0.25))
    clip_max     = float(getattr(args, "class_weights_clip_max", 3.0))

    # --- inverse-frequency power ---
    w = np.power(r + eps, -gamma)

    # --- boost change classes (1..4) ---
    if boost_change != 1.0:
        w[1:5] *= boost_change

    # --- normalization ---
    if normalize == "mean1":
        w = w / (w.mean() + eps)
    elif normalize == "sum1":
        w = w / (w.sum() + eps)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown class_weights_normalize: {normalize}")

    # --- clip for stability ---
    if clip_min is not None or clip_max is not None:
        w = np.clip(w, clip_min, clip_max)
    
    # 클래스 가중치 로그 정리
    print(f"[class ratio] {np.asarray(ratio).tolist()}")
    print(f"[class weights] {w.tolist()} | sum={w.sum():.3f}, mean={w.mean():.3f}")

    return w.astype(np.float32)



def compute_metrics_from_confusion_matrix(confusion_matrix, class_mask=None):
    """Compute segmentation metrics from a raw confusion matrix.

    Rows are ground-truth classes and columns are predicted classes.
    Returned values are percentages (0--100), except raw pixel counts.
    """
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    assert cm.ndim == 2 and cm.shape[0] == cm.shape[1], \
        f"Expected a square confusion matrix, got shape={cm.shape}"

    num_classes = cm.shape[0]
    tp = np.diag(cm)
    gt = cm.sum(axis=1)
    pred = cm.sum(axis=0)
    union = gt + pred - tp

    if class_mask is None:
        valid_class = gt > 0
    else:
        valid_class = np.asarray(class_mask, dtype=bool)
        assert valid_class.shape == (num_classes,), \
            f"class_mask must have shape ({num_classes},), got {valid_class.shape}"

    def safe_divide(numerator, denominator):
        out = np.full_like(numerator, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=out, where=denominator > 0)
        return out

    iou = safe_divide(tp, union)
    precision = safe_divide(tp, pred)
    recall = safe_divide(tp, gt)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)

    # Preserve the existing evaluation policy: only GT-present classes enter mIoU.
    iou_for_mean = iou.copy()
    iou_for_mean[~valid_class] = np.nan
    precision_for_mean = precision.copy()
    precision_for_mean[~valid_class] = np.nan
    recall_for_mean = recall.copy()
    recall_for_mean[~valid_class] = np.nan
    f1_for_mean = f1.copy()
    f1_for_mean[~valid_class] = np.nan

    change_mask = valid_class.copy()
    if num_classes > 0:
        change_mask[0] = False

    total = cm.sum()
    total_tp = tp.sum()

    def masked_nanmean(values, mask):
        selected = values[mask]
        return float(np.nanmean(selected) * 100.0) if selected.size and np.isfinite(selected).any() else 0.0

    metrics = {
        'confusion_matrix': cm.astype(np.int64),
        'gt_pixels': gt.astype(np.int64),
        'pred_pixels': pred.astype(np.int64),
        'class_mask': valid_class,
        'per_class_iou': iou * 100.0,
        'per_class_precision': precision * 100.0,
        'per_class_recall': recall * 100.0,
        'per_class_f1': f1 * 100.0,
        'miou_5class': masked_nanmean(iou_for_mean, valid_class),
        'change_only_miou': masked_nanmean(iou, change_mask),
        'macro_precision': masked_nanmean(precision_for_mean, valid_class),
        'macro_recall': masked_nanmean(recall_for_mean, valid_class),
        'macro_f1': masked_nanmean(f1_for_mean, valid_class),
        'change_only_macro_precision': masked_nanmean(precision, change_mask),
        'change_only_macro_recall': masked_nanmean(recall, change_mask),
        'change_only_macro_f1': masked_nanmean(f1, change_mask),
        'micro_precision': (float(total_tp / total) * 100.0) if total > 0 else 0.0,
        'micro_recall': (float(total_tp / total) * 100.0) if total > 0 else 0.0,
        'pixel_accuracy': (float(total_tp / total) * 100.0) if total > 0 else 0.0,
    }
    return metrics


def print_and_save_evaluation_metrics(args, epoch, metrics):
    """Print detailed metrics and save epoch-level CSV/NPY outputs."""
    num_classes = metrics['confusion_matrix'].shape[0]
    class_names = ['Static', 'New', 'Missing', 'Rotated', 'Replaced'] \
        if num_classes == 5 else [f'Class_{i}' for i in range(num_classes)]

    print('=' * 88)
    print(f'Detailed validation metrics (epoch {epoch})')
    print('Rows = ground truth, columns = prediction')
    print('Raw confusion matrix:')
    print(metrics['confusion_matrix'])
    print(f"Per-class IoU (%):       {[f'{v:.4f}' for v in metrics['per_class_iou']]}")
    print(f"Per-class precision (%): {[f'{v:.4f}' for v in metrics['per_class_precision']]}")
    print(f"Per-class recall (%):    {[f'{v:.4f}' for v in metrics['per_class_recall']]}")
    print(f"Per-class F1 (%):        {[f'{v:.4f}' for v in metrics['per_class_f1']]}")
    print(f"5-class mIoU (%):                  {metrics['miou_5class']:.4f}")
    print(f"Change-only mIoU (%), excluding S: {metrics['change_only_miou']:.4f}")
    print(f"Macro precision (%):               {metrics['macro_precision']:.4f}")
    print(f"Macro recall (%):                  {metrics['macro_recall']:.4f}")
    print(f"Macro F1 (%):                      {metrics['macro_f1']:.4f}")
    print(f"Change-only macro precision (%):   {metrics['change_only_macro_precision']:.4f}")
    print(f"Change-only macro recall (%):      {metrics['change_only_macro_recall']:.4f}")
    print(f"Change-only macro F1 (%):          {metrics['change_only_macro_f1']:.4f}")
    print(f"Micro precision (%):               {metrics['micro_precision']:.4f}")
    print(f"Micro recall (%):                  {metrics['micro_recall']:.4f}")
    print(f"Pixel accuracy (%):                {metrics['pixel_accuracy']:.4f}")
    print('=' * 88)

    metrics_dir = os.path.join(args.output_dir, 'validation_metrics', f'epoch_{epoch:05d}')
    os.makedirs(metrics_dir, exist_ok=True)

    np.save(os.path.join(metrics_dir, 'confusion_matrix_counts.npy'),
            metrics['confusion_matrix'], allow_pickle=False)
    np.savetxt(os.path.join(metrics_dir, 'confusion_matrix_counts.csv'),
               metrics['confusion_matrix'], fmt='%d', delimiter=',')

    with open(os.path.join(metrics_dir, 'per_class_metrics.csv'), 'w') as f:
        f.write('class_id,class_name,gt_pixels,pred_pixels,iou_percent,precision_percent,recall_percent,f1_percent\n')
        for i, name in enumerate(class_names):
            f.write(
                f"{i},{name},{metrics['gt_pixels'][i]},{metrics['pred_pixels'][i]},"
                f"{metrics['per_class_iou'][i]:.6f},{metrics['per_class_precision'][i]:.6f},"
                f"{metrics['per_class_recall'][i]:.6f},{metrics['per_class_f1'][i]:.6f}\n"
            )

    overall_keys = [
        'miou_5class', 'change_only_miou', 'macro_precision', 'macro_recall',
        'macro_f1', 'change_only_macro_precision', 'change_only_macro_recall',
        'change_only_macro_f1', 'micro_precision', 'micro_recall', 'pixel_accuracy'
    ]
    with open(os.path.join(metrics_dir, 'overall_metrics.csv'), 'w') as f:
        f.write('metric,value_percent\n')
        for key in overall_keys:
            f.write(f"{key},{metrics[key]:.6f}\n")

    print(f'Saved detailed validation metrics: {metrics_dir}')


def prepare_train(args, model_without_ddp, fusion_module, dataset, data_loader):

    # fcn 또는 deeplabv3 계열 모델일 경우:
    if "fcn" in args.model or "deeplabv3" in args.model:

        # backbone과 classifier를 분리해서 optimizer에 전달합니다.
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
            {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        ]
    else:
        # model_without_ddp.parameters() 전체를 통째로 최적화 대상으로 설정
        params_to_optimize = [
            {"params": [p for p in model_without_ddp.parameters() if p.requires_grad]}
        ]

    # FusionModule 파라미터 추가
    params_to_optimize.append({"params": fusion_module.parameters()})

    # 옵티마이저 정의
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.opt)

    # 클래스 불균형을 보정하기 위한 loss weight 적용 로직
    if args.loss_weight:
        print("computing loss weight")
        ratio = dataset.get_mask_ratio()  # [예: 클래스별 등장 비율 계산]

        # class_weights 계산
        class_weights = compute_class_weights_from_ratio(args, ratio)
        
        # 이 비율 값을 Tensor로 만들어서 GPU에 올립니다. 이후 loss 함수에 weight로 전달하게 됩니다.
        loss_weight = torch.tensor(class_weights, dtype=torch.float32).to(args.device)
        
        print("loss weight {}".format(loss_weight))
    else:
        loss_weight = None
    
    # Label Smoothing 입력값 범위 체크
    if args.label_smoothing:
        assert 0.0 <= args.label_smoothing < 1.0, \
           "--label-smoothing must satisfy 0.0 <= x < 1.0"
    
    # Loss function
    criterion = loss.get_loss(
        args.loss,
        loss_weight, # 이 부분에서 실제 loss 함수를 만드는데, loss_weight가 주어지면 클래스별 가중치를 포함해서 만듭니다.
        label_smoothing = args.label_smoothing,    # CE smoothing, Label smoothing epsilon 전달
        focal_gamma=getattr(args, "focal_gamma", 2.0),
        focal_alpha=getattr(args, "focal_alpha", None),
        dice_lambda=getattr(args, "dice_weight", 0.5),
        ignore_index=getattr(args, "ignore_index", 255),
        )
    
    # Label Smoothing (e.g., ε=0.05)
    if not args.label_smoothing == 0:
        # Label smoothing epsilon 로그
        print(f"prepare_train()$ label_smoothing = {args.label_smoothing}")

    # 전체 step 수 계산
    total_steps = args.epochs * len(data_loader)
    steps_per_epoch = len(data_loader)

    # warm-up을 몇 step 동안 적용할지를 결정
    # `args.warmup_period`는 epoch 단위이며, 전체 warm-up step 수는 다음과 같이 계산됩니다:
    #   num_warmup = warmup_epoch * steps_per_epoch
    if args.warmup:
        num_warmup = min(
            max(round(args.warmup_period * len(data_loader)), 1000), 
            max(total_steps // 3, 1)    # Warmup step 하한 보장
            )
    else:
        num_warmup = 0

    # -------- 스케줄러 생성 --------
    if args.lr_scheduler.lower() == 'sgdr':
        # Cosine Warm Restarts (SGDR)

        # Quick sanity checks (fail fast)        
        assert 0.0 <= args.sgdr_eta_min_mul < 1.0, \
            "--sgdr-eta-min-mul must satisfy 0.0 <= x < 1.0"
            # [주의] --sgdr-eta-min-mul >= 1.0을 넣으면 바닥값이 base_lr 이상으로 올라가 코사인이 “평평”해질 수 있어요.
        assert args.sgdr_t0 >= 1, \
            "--sgdr-t0 must be >= 1 (epochs)"
        assert args.sgdr_tmult >= 1.0, \
            "--sgdr-tmult must be >= 1.0 (no shrinking cycles)"
        assert steps_per_epoch > 0, \
            "steps_per_epoch must be > 0 (check DataLoader / batch_size / drop_last)"

        # eta_min = base_lr * (사용자 지정 비율)
        eta_min = args.lr * float(args.sgdr_eta_min_mul)
        lambda_lr = lambda step: lr_schedule_cosine_restarts_with_warmup(
            step=step,
            num_warmup=num_warmup,
            base_lr=args.lr,
            eta_min=eta_min,
            steps_per_epoch=steps_per_epoch,
            t0_epochs=int(args.sgdr_t0),
            tmult=float(args.sgdr_tmult),
        )
    else:
        # 기존 스케줄들 유지
        lambda_lr = get_scheduler_function(
            args.lr_scheduler,
            total_iters=total_steps,
            final_lr=0.2 * args.lr, # 학습 마지막 단계에서 도달할 최종 학습률 (예: 0.0001 * 0.2 = 0.00002)
            base_lr=args.lr,         # 학습 초기에 사용할 시작 학습률 (예: args.lr = 0.0001)
            num_warmup=num_warmup   # warm-up을 몇 step 동안 적용할지를 반영
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # FusionModule optimizer 포함 여부 확인
    # → 학습이 잘 안 될 경우, FusionModule이 너무 작아서 그런지 감을 잡을 수 있습니다.
    num_fusion_params = sum(p.numel() for p in fusion_module.parameters())
    num_params_in_optim = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    print(f"prepare_train()$ FusionModule params: {num_fusion_params}")
    print(f"prepare_train()$ Total params in optimizer: {num_params_in_optim}")
    print(f"prepare_train()$ FusionModule ratio in optimizer: {100 * num_fusion_params / num_params_in_optim:.2f}%")


    return optimizer, criterion, lr_scheduler, scaler, num_warmup


# f_{t-1} 불러오기 함수 (예: .npy)
def load_f_t_minus_1(path_img, path_f_t, epoch_minus_1, num_classes, input_size):
    """
    Load the Bayesian fusion result f_{t-1} from a .npy file.

    Parameters:
        path_img (str): path to img_t (e.g., '/workspace/C-3PO_dataset/ChangeSim/Query/Query_Seq_Train/Warehouse_3/Seq_0/rgb/1324.png')
        path_f_t: path to `f_t.npy` (e.g., 'output/train_fusion_module/fusion_tmp/20250518_061524/00000/f_t')
        epoch_minus_1 (int): Index of the previous epoch (i.e., current epoch - 1),
                             used to locate the saved fusion result f_{t-1}.

    Returns:
        np.ndarray: fusion result with shape (C, H, W)
    """

    # epoch-1이 음수일 경우에 대한 예외처리
    # 논문에서도 "precomputed temporal prior" 또는 "offline temporal memory" 같은 용어로 자연스럽게 기술 가능합니다.
    #
    # PyTorch 및 대부분의 딥러닝 학습 루프는 epoch을 0부터 시작합니다. (0-based indexing)
    #
    # The current epoch is 0. So, there is not `.npy` file for `f_t_minus_1`.
    if epoch_minus_1 < 0:
        # return the 1/C initial numpy
        return np.full((num_classes, input_size, input_size), fill_value=1.0 / num_classes, dtype=np.float32)
    
    else:
        # Parse index from img_t path
        path_parts = path_img.split('/')
        try:
            seq_type = path_parts[-5]      # 'Query_Seq_Train'
            warehouse = path_parts[-4]     # 'Warehouse_1'
            seq_num = path_parts[-3]       # 'Seq_0'
            img_filename = path_parts[-1]  # e.g., '1324.png'
            img_index = int(os.path.splitext(img_filename)[0])  # e.g., 1324
            img_index_minus_1 = img_index - 1   # t-1
        except IndexError:
            #raise ValueError("입력 경로 형식이 예상과 다릅니다: %s" % path_img)
            raise ValueError("The input path format is not as expected: %s" % path_img)

        # t-1이 음수일 경우에 대한 예외처리
        if img_index_minus_1 < 0 :
            # return the 1/C initial numpy
            return np.full((num_classes, input_size, input_size), fill_value=1.0 / num_classes, dtype=np.float32)

        # Modify path_f_t
        path_parts = path_f_t.split('/')
        path_parts[-2] = f"{epoch_minus_1:05d}"
        path_f_t_minus_1 = '/'.join(path_parts)

        # filename for f_t_minus_1
        #   eg.) Query_Seq_Train-Warehouse_5-Seq_1-99.npy
        filename = f"./{path_f_t_minus_1}/{seq_type}-{warehouse}-{seq_num}-{img_index_minus_1}.npy"

        # npy 파일을 찾을 수 없을 때, 1/C로 초기화한 uniform f_t_minus_1을 리턴
        # 이 [WARNING] 메시지가 출력 안 하도록 설계되어있기는 함
        if not os.path.exists(filename):
            #sys.exit(f"load_f_t_minus_1()$ At epoch == {epoch}, File not found: {filename}")
            print(f"[WARNING] load_f_t_minus_1()$ No f_t for epoch_minus_1 {epoch_minus_1} found at: {filename}. Using uniform prior.")
            return np.full((num_classes, input_size, input_size), fill_value=1.0 / num_classes, dtype=np.float32)

        return np.load(filename)
    

def parse_path_indices(path):
    """
    Args:
        path (str): 입력 이미지의 경로
                       (e.g., '/workspace/C-3PO_dataset/ChangeSim/Query/Query_Seq_Train/Warehouse_1/Seq_0/rgb/1377.png')
    """

    # 경로에서 필요한 요소 추출
    parts = path.split('/')
    try:
        seq_type = parts[-5]                  # 'Query_Seq_Train'
        warehouse = parts[-4]                 # 'Warehouse_1'
        seq_num = parts[-3]                   # 'Seq_0'
        filename = os.path.basename(path) # '1377.png'
    except IndexError:
        #raise ValueError("입력 경로 형식이 예상과 다릅니다: %s" % path_img)
        raise ValueError("The input path format is not as expected: %s" % path)
    
    # warehouse와 seq에서 숫자 추출
    list_number_ = int(warehouse.replace('Warehouse_', ''))
    seq_number_ = int(seq_num.replace('Seq_', ''))
    idx_ = int(os.path.splitext(filename)[0])

    return list_number_, seq_number_, idx_

def get_flow_path_t_minus_1(flow_dir, path_img):
    """
    Args:
        flow_dir (str): `.flo`가 저장된 루트 디렉토리 경로
                       (e.g., '/workspace/flownet2-pytorch/work/inference/')
        path_img (str): 입력 이미지의 경로
                       (e.g., '/workspace/C-3PO_dataset/ChangeSim/Query/Query_Seq_Train/Warehouse_1/Seq_0/rgb/1377.png')

    Returns:
        path_flow (str): 대응되는 .flo optical flow 경로
                         (e.g., '/workspace/flownet2-pytorch/work/inference/Query_Seq_Train-1-0/001376.flo')

    Notes:
        001376.flo는 1376.png를 1377.png에 맞춰서 변형하기 위한 optical flow를 포함하고 있음

    """

    # 경로에서 필요한 요소 추출
    parts = path_img.split('/')
    try:
        seq_type = parts[-5]                  # 'Query_Seq_Train'
        warehouse = parts[-4]                 # 'Warehouse_1'
        seq_num = parts[-3]                   # 'Seq_0'
        filename = os.path.basename(path_img) # '1377.png'
    except IndexError:
        #raise ValueError("입력 경로 형식이 예상과 다릅니다: %s" % path_img)
        raise ValueError("The input path format is not as expected: %s" % path_img)

    # warehouse와 seq에서 숫자 추출
    warehouse_id = warehouse.replace('Warehouse_', '')
    seq_id = seq_num.replace('Seq_', '')

    # flo 파일명 생성 (zero padding 6자리)
    img_id = os.path.splitext(filename)[0]       # '1377'
    img_id_minus_1 = str(int(img_id) - 1)         # '1376'. [주의] 여기서 img_id_minus_1가 `-1`일 수 있음.
    img_id_padded = img_id_minus_1.zfill(6)
    flo_filename = f"{img_id_padded}.flo"

    # flo 경로 구성
    subdir = f"{seq_type}-{warehouse_id}-{seq_id}"
    path_flow = os.path.join(flow_dir, subdir, flo_filename)

    return path_flow


def train_one_epoch(args, model, fusion_module, criterion, optimizer, scaler, data_loader, lr_scheduler, num_warmup, device, epoch, print_freq, ema=None):
    model.train()   # 모델(C-3PO역할)을 학습 모드로 설정
    fusion_module.train()   # 모델을 학습 모드로 설정
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # 두 번 이전 epoch에서 사용한 path_fusion_tmp 폴더 삭제
    path_fusion_tmp = os.path.join(args.fusion_tmp_dir,'{:05d}'.format(epoch - 2))
    if os.path.exists(path_fusion_tmp):
        try:
            shutil.rmtree(path_fusion_tmp) # 폴더 삭제
        except Exception as e:
            print(f"An error occurred while deleting the folder: {e}")
    #else:
    #    print(f"Folder does not exist: {path_fusion_tmp_prev}")

    # 현재 epoch에서 사용할 path_fusion_tmp 폴더 생성
    path_fusion_tmp = os.path.join(args.fusion_tmp_dir,'{:05d}'.format(epoch))
    path_p_t = os.path.join(path_fusion_tmp,'p_t')    # p_t의 numpy 파일
    path_p_t_img = os.path.join(path_fusion_tmp,'p_t_img')    # p_t의 png 이미지파일
    path_f_t = os.path.join(path_fusion_tmp,'f_t')    # f_t의 numpy 파일
    path_f_t_img = os.path.join(path_fusion_tmp,'f_t_img')    # f_t의 png 이미지파일

    for path in [path_fusion_tmp, path_p_t, path_p_t_img, path_f_t, path_f_t_img]:
        os.makedirs(path, exist_ok=True)

    # 데이터 로딩 및 학습 진행
    for image, target, paths, _ in metric_logger.log_every(data_loader, print_freq, header):

        # 텐서를 특정 디바이스(GPU 또는 CPU)로 옮기기
        image = image.to(device)
        target = target.to(device)

        # 자동 혼합 정밀도 연산(Amp) 적용
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):

            # Step 1: p_t 계산
            output = model(image)
            if isinstance(output, dict):
                output = output["out"]

            # Step 2: load f_{t-1} + apply flow warp
            warped_f_tensor_list = []
            for i in range(len(paths)): # batch 크기에 대한 idx
                f_prev = load_f_t_minus_1(paths[i], path_f_t, epoch-1, args.num_classes, args.input_size)  # 이전 epoch의 f_{t-1}. numpy.
                flow_path = get_flow_path_t_minus_1(args.flow_dir, paths[i])    # flow_{t-1}
                flow_np = read_flo_file(flow_path, f_prev.shape)

                # numpy → tensor → CUDA 이동
                f_prev_tensor = torch.from_numpy(f_prev).to(device)

                warped = warp_using_flow_cuda(f_prev_tensor, flow_np, device=args.device)
                warped_f_tensor_list.append(warped)

            # 전체 batch dimension은 torch.stack() 으로 batch 쌓아서 일괄 처리
            warped_f = torch.stack(warped_f_tensor_list)  # 최종 (B, C, H, W)

            # Step 3: f_t (Fusion)

            # '--fusion-detach-epochs' 옵션을 적용하여, detach fusion-to-backbone gradients하는 스케줄을 구현
            fusion_output_curriculum = fusion_input_with_curriculum(output, epoch, args.fusion_detach_epochs)

            # FusionModule은 입력으로 확률분포(softmax)를 받는 구조임
            #fused_output = fusion_module(
            #    torch.nn.functional.softmax(output, dim=1),
            #    torch.nn.functional.softmax(warped_f, dim=1)
            #)
            #
            # FusionModule 입력에서 softmax 제거:
            #  - Softmax는 gradient를 매우 약하게 만들 수 있어, fusion 학습이 잘 되지 않을 수 있습니다.
            #  - FusionModule 내부에서 이미 `Conv → BN → ReLU`가 들어가 있으므로 raw logits이 적절합니다.
            #fused_output = fusion_module(output, warped_f)
            fused_output = fusion_module(fusion_output_curriculum, warped_f)
            
            # Step 4: Loss 계산 - output과 fused_output 각각 손실 계산
            loss_output = criterion(output, target)  # base model supervision
            loss_fused_output = criterion(fused_output, target)  # fusion supervision
            loss_val = (1-args.loss_fusion_weight) * loss_output + args.loss_fusion_weight * loss_fused_output    # weighted sum
            #
            # [Tip] 실전 팁
            # 두 loss를 더할 때 gradient 흐름 중복에 주의해야 합니다. 하지만 output은 detach()되지 않았기 때문에 두 방향 모두 학습에 기여합니다.
            #
            # [2-branch 구조]
            # 성능을 더 높이거나 base model(C-3PO)의 안정성을 확보하려면 output에 대해서도 loss를 포함시키는 2-branch 구조

        # 그래디언트 초기화 후 역전파 수행
        optimizer.zero_grad()  # 이전 스텝에서 누적된 그래디언트를 초기화 (반드시 필요)

        if scaler.is_enabled():  # Mixed Precision Training (AMP) 적용
            # 손실을 확장(scale up)하여 역전파 수행 (FP16의 작은 값 문제 방지)
            scaler.scale(loss_val).backward()

            # FusionModule이 gradient를 받는지 확인하는 코드
            if epoch < 2:
                for name, param in fusion_module.named_parameters():
                    if param.requires_grad:

                        # FusionModule이 gradient를 받지 못한다면 정상적인 학습이 전혀 안 되는 치명적인 상태
                        assert param.grad is not None, \
                            f" train_one_epoch()$ ASSERT FAILED: FusionModule has no gradient for '{name}' (check loss connection or optimizer)"
                        
                        # FusionModule이 gradient를 받았음 확인완료 메시지
                        #print(f"train_one_epoch()$ FusionModule gradient exists for: {name}, norm = {param.grad.norm().item():.6f}")

            # 스케일링된 그래디언트로 가중치 업데이트 (AMP 적용)
            scaler.step(optimizer)
            # 스케일링 값 조정 (다음 스텝에서 적절한 스케일링 적용)
            scaler.update()
            # EMA 업데이트 (optimizer.step 직후)
            # - 가중치가 실제로 업데이트된 직후 EMA를 업데이트해야, 최신 가중치의 지수이동평균을 추적
            if ema is not None:
                ema.update()
        else:
            # 역전파 수행: loss_val를 기준으로 각 가중치의 그래디언트 계산
            loss_val.backward()

            # FusionModule이 gradient를 받는지 확인하는 코드
            if epoch < 2:
                for name, param in fusion_module.named_parameters():
                    if param.requires_grad:

                        # FusionModule이 gradient를 받지 못한다면 정상적인 학습이 전혀 안 되는 치명적인 상태
                        assert param.grad is not None, \
                            f" train_one_epoch()$ ASSERT FAILED: FusionModule has no gradient for '{name}' (check loss connection or optimizer)"
                        
                        # FusionModule이 gradient를 받았음 확인완료 메시지
                        #print(f"train_one_epoch()$ FusionModule gradient exists for: {name}, norm = {param.grad.norm().item():.6f}")

            # 가중치를 그래디언트 방향으로 업데이트하여 학습 진행
            optimizer.step()
            # EMA 업데이트 (optimizer.step 직후)
            # - 가중치가 실제로 업데이트된 직후 EMA를 업데이트해야, 최신 가중치의 지수이동평균을 추적
            if ema is not None:
                ema.update()

        # 매 배치(batch)마다 학습률 스케줄러의 step()을 호출합니다.
        lr_scheduler.step() # 학습률 스케줄러를 한 스텝 진행하여 학습률을 갱신

        # Step 5: Save fused result

        # logits으로 numpy 저장
        f_t_cpu = fused_output.detach().cpu().numpy()
        for i in range(len(paths)): # batch 크기에 대한 idx
            list_number_, seq_number_, idx_ = parse_path_indices(paths[i])
            save_p_or_f(f_t_cpu[i], list_number_, seq_number_, idx_, path_f_t, path_f_t_img, flag_save_png=False)
            
        # p_t 저장 (softmax 안 한 결과)
        p_t_cpu = output.detach().cpu().numpy()
        for i in range(len(paths)):  # batch 크기에 대한 idx
            list_number_, seq_number_, idx_ = parse_path_indices(paths[i])
            save_p_or_f(p_t_cpu[i], list_number_, seq_number_, idx_, path_p_t, path_p_t_img, flag_save_png=False)

        # loss 확인
        print('train_one_epoch()$ loss = {}, loss_fusion = {}, weighted_loss = {}, lr = {}'.format(
            loss_output.detach().item(),
            loss_fused_output.detach().item(),
            loss_val.detach().item(),
            optimizer.param_groups[0]["lr"]))
        
        metric_logger.update(loss=loss_val.item(), lr=optimizer.param_groups[0]["lr"])


def evaluate(args, model, fusion_module, data_loader, device, epoch, flow_dir, ema=None):
    """
    Evaluate model + fusion module and compute mIoU (%).

    Args:
        model: backbone model
        fusion_module: temporal fusion module
        data_loader: validation dataloader
        device: torch.device
        epoch: current epoch for fusion path
        flow_dir: directory containing optical flow files

    Returns:
        float: mean IoU
    """

    model.eval()
    fusion_module.eval()

    num_classes = data_loader.dataset.num_classes
    confmat = ConfusionMatrix(num_classes)  # uses np.nanmean & class_mask
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # [ ignore_index 마스크 + per-class 카운트 + 안전한 mIoU 계산 ]
    # (A) 루프 시작 전 누적 카운터 준비
    #IGNORE_INDEX = 255
    # per-class GT/Pred 픽셀 카운트(검증셋에 클래스 '부재' 여부 판정에 사용)
    gt_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    pred_counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    # 현재 epoch에서 사용할 path_fusion_tmp 폴더 생성
    path_fusion_tmp = os.path.join(args.fusion_tmp_dir,'{:05d}'.format(epoch))
    path_p_t = os.path.join(path_fusion_tmp,'p_t')    # p_t의 numpy 파일
    path_p_t_img = os.path.join(path_fusion_tmp,'p_t_img')    # p_t의 png 이미지파일
    path_f_t = os.path.join(path_fusion_tmp,'f_t')    # f_t의 numpy 파일
    path_f_t_img = os.path.join(path_fusion_tmp,'f_t_img')    # f_t의 png 이미지파일

    for path in [path_fusion_tmp, path_p_t, path_p_t_img, path_f_t, path_f_t_img]:
        os.makedirs(path, exist_ok=True)
    
    # EMA 스왑 컨텍스트
    # EMA(Exponential Moving Average) weight tracking (학습 끝마다 EMA 가중치로 평가)
    # EMA가 있으면 swap, 없으면 빈 컨텍스트(nullcontext)
    nullctx = contextlib.nullcontext()
    ctx = ema.swap_weights() if (ema is not None) else nullctx

    with ctx:

        with torch.no_grad():

            use_amp = bool(getattr(args, "use_amp", False))
            with torch.cuda.amp.autocast(enabled=use_amp):  # 평가에서도 AMP 사용

                for images, targets, paths, _ in metric_logger.log_every(data_loader, 50, header):
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    # backbone forward
                    output = model(images)
                    if isinstance(output, dict):
                        output = output["out"]
                    
                    # p_t 저장 (raw logits, softmax 안 한 결과)
                    p_t_cpu = output.detach().cpu().numpy()
                    for i in range(len(paths)):  # batch 크기에 대한 idx
                        list_number_, seq_number_, idx_ = parse_path_indices(paths[i])
                        save_p_or_f(p_t_cpu[i], list_number_, seq_number_, idx_,
                                    path_p_t, path_p_t_img, flag_save_png=False)

                    # CUDA 기반 flow warping 적용
                    warped_f_tensor_list = []
                    for i in range(len(paths)):  # batch 크기에 대한 idx
                        f_prev = load_f_t_minus_1(paths[i], path_f_t, epoch-1, args.num_classes, args.input_size)  # (C, H, W) numpy

                        flow_path = get_flow_path_t_minus_1(flow_dir, paths[i])
                        flow_np = read_flo_file(flow_path, f_prev.shape)

                        f_prev_tensor = torch.from_numpy(f_prev).to(device)  # numpy → tensor (CUDA 이동)
                        warped = warp_using_flow_cuda(f_prev_tensor, flow_np, device=args.device)  # CUDA 기반 warp
                        warped_f_tensor_list.append(warped)

                    warped_f = torch.stack(warped_f_tensor_list).to(device)  # (B, C, H, W)

                    # '--fusion-detach-epochs' 옵션을 적용하여, detach fusion-to-backbone gradients하는 스케줄을 구현
                    #fusion_output_curriculum = fusion_input_with_curriculum(output, epoch, args.fusion_detach_epochs)

                    # FusionModule (raw logits → fuse)
                    #  - Softmax는 gradient를 매우 약하게 만들 수 있어, fusion 학습이 잘 되지 않을 수 있습니다.
                    #  - FusionModule 내부에서 이미 `Conv → BN → ReLU`가 들어가 있으므로 raw logits이 적절합니다.
                    fused_output = fusion_module(output, warped_f)
                    #fused_output = fusion_module(fusion_output_curriculum, warped_f)

                    assert fused_output.shape[1] == num_classes, \
                        f"evaluate()$ eval logits has {fused_output.shape[1]} channels, expected {num_classes}"

                    # f_t 저장 (raw logits)
                    # logits으로 numpy 저장 (save_p_or_f 재사용)
                    f_t_cpu = fused_output.detach().cpu().numpy()
                    for i in range(len(paths)): # batch 크기에 대한 idx
                        list_number_, seq_number_, idx_ = parse_path_indices(paths[i])
                        save_p_or_f(f_t_cpu[i], list_number_, seq_number_, idx_,
                                    path_f_t, path_f_t_img, flag_save_png=False)

                    # mIoU 계산
                    pred = fused_output.argmax(1)  # (B, H, W)

                    # [ ignore_index 마스크 + per-class 카운트 + 안전한 mIoU 계산 ]
                    # (B) 배치 처리 시: 255 무시 + 유효 픽셀만 ConfusionMatrix 업데이트
                    #
                    # 255 : 정답 라벨 중 무시해야 하는 픽셀(유효하지 않은 영역)”을 표시하는 특별한 정수값

                    # 유효 라벨만 집계 (0..num_classes-1), ignore_index(255 등) 배제
                    valid = (targets >= 0) & (targets < num_classes)

                    # ConfusionMatrix 업데이트/카운트는 유효 픽셀이 있을 때만
                    if valid.any().item():

                        # 혹시 라벨 맵에 255가 섞여도 안전
                        targets_v = targets[valid]
                        pred_v = pred[valid]

                        # ConfusionMatrix 업데이트는 '유효' 픽셀만
                        confmat.update(
                            targets_v.flatten().to(torch.int64),   # 그대로 Tensor
                            pred_v.flatten().to(torch.int64),      # 그대로 Tensor
                        )

                        # --- per-class 픽셀 카운트(부재 클래스 탐지) ---
                        gt_counts += torch.bincount(targets_v.flatten().to(torch.int64), minlength=num_classes)
                        pred_counts += torch.bincount(pred_v.flatten().to(torch.int64), minlength=num_classes)

    # [ ignore_index 마스크 + per-class 카운트 + 안전한 mIoU 계산 ]
    # (C) mIoU 계산부: ‘부재 클래스’는 NaN으로 마스킹하여 평균

    acc_global, acc, iu = confmat.compute()

    # compute & class_mask 주입 → ConfusionMatrix가 nanmean으로 평균
    # - 검증셋에 등장하지 않은 클래스(denom==0 추정)는 NaN으로 두고 평균에서 제외
    # - 그 에폭 검증셋에 실제 등장한 클래스만 평균에 포함
    class_mask = (gt_counts.detach().cpu().numpy() > 0)
    confmat.class_mask = class_mask.tolist()

    # 최종 mIoU(%)는 ConfusionMatrix에 위임 (NaN 제외 평균)
    mIoU = confmat.mIoU(iu)  # percent scale (0~100)
    
    if not any(confmat.class_mask):  # extremely rare: all classes absent
        mIoU = 0.0

    # 현재 epoch값 출력
    print('Current epoch : {}'.format(epoch))

    # confmat 출력
    print("{} {} confmat: {}".format(
        header,
        data_loader.dataset.name,
        confmat
    ))

    # per-class 픽셀 요약 출력
    print(f"GT pixels per class: {gt_counts.tolist()}")
    print(f"Pred pixels per class: {pred_counts.tolist()}")
    print(f"Class mask (GT>0): {confmat.class_mask}")

    assert hasattr(confmat, 'mat'), \
        "ConfusionMatrix must expose the raw matrix through confmat.mat"
    raw_confusion = confmat.mat.detach().cpu().numpy() \
        if torch.is_tensor(confmat.mat) else np.asarray(confmat.mat)
    detailed_metrics = compute_metrics_from_confusion_matrix(
        raw_confusion, class_mask=confmat.class_mask
    )

    # Sanity check: the new detailed calculation must preserve the existing mIoU.
    assert abs(detailed_metrics['miou_5class'] - float(mIoU)) < 1e-4, (
        f"mIoU mismatch: ConfusionMatrix={float(mIoU):.6f}, "
        f"detailed_metrics={detailed_metrics['miou_5class']:.6f}"
    )
    print_and_save_evaluation_metrics(args, epoch, detailed_metrics)

    return float(mIoU)  # percent scale (0~100)

# p 또는 f 파일 출력
def save_p_or_f( data_np, list_number_, seq_number_, idx_, path_np, path_png, flag_save_png=False):

    # 파일명 생성
    if int(list_number_) < 6 :   # Train
        filename_np = os.path.join(path_np, 'Query_Seq_Train-Warehouse_{}-Seq_{}-{}.npy'.format(list_number_,seq_number_,idx_))
        filename_png = os.path.join(path_png, 'Query_Seq_Train-Warehouse_{}-Seq_{}-{}.png'.format(list_number_,seq_number_,idx_))
    else :  # Test
        filename_np = os.path.join(path_np, 'Query_Seq_Test-Warehouse_{}-Seq_{}-{}.npy'.format(list_number_,seq_number_,idx_))
        filename_png = os.path.join(path_png, 'Query_Seq_Test-Warehouse_{}-Seq_{}-{}.png'.format(list_number_,seq_number_,idx_))

    ####################
    # png 파일로 저장

    if flag_save_png:

        # 모델의 출력에서 가장 큰 값의 인덱스를 찾아 예측 결과를 구합니다.
        # https://github.com/DoctorKey/C-3PO/blob/main/src/train.py
        # (line 113) pred = output.argmax(1)
        # (C, H, W) → (H, W), pixel별 class 예측
        pred = data_np.argmax(0).astype(np.uint8)

        # 클래스, 높이, 너비 가져오기
        class_, height_, width_ = data_np.shape

        # ★ 벡터화된 컬러링 (매우 빠름)
        # - 클래스가 5인 경우: idx2color LUT로 바로 매핑 (shape: (H, W, 3))
        # - 클래스가 5가 아닌 경우: 흑백 이미지로 처리 (0=검정, 나머지=흰색)
        if class_ == 5:
            # https://github.com/DoctorKey/C-3PO/blob/main/src/dataset/changesim.py
            # (line 177) pred = self.seg.classmap2colormap(pred.to(device))
            # 클래스가 5인 경우 idx2color 사용
            lut = np.asarray(idx2color, dtype=np.uint8)  # (클래스 수, 3)

            # 클래스 개수와 LUT 길이 일치 확인
            assert lut.shape[0] == class_, f"idx2color length ({lut.shape[0]}) must equal num_classes ({class_})"
            # pred 범위 확인 (안전)
            assert pred.max() < class_, f"pred contains invalid class id (max={pred.max()}, num_classes={class_})"
            
            img = lut[pred]  # (H, W, 3)
        else:
            # 클래스가 5가 아닌 경우 흑백 이미지로 처리
            # pred==0 → 0, 그 외 → 255 (RGB 3채널로 확장)
            img = np.where(pred[..., None] == 0, 0, 255).astype(np.uint8)  # (H, W, 1)
            img = np.repeat(img, 3, axis=2)  # (H, W, 3)

        # https://github.com/DoctorKey/C-3PO/blob/main/src/train.py
        # (line 119) output_pil.save(os.path.join(save_imgs_dir, "{}_{}.png".format(utils.get_rank(), count)))
        Image.fromarray(img, mode='RGB').save(filename_png, optimize=True)

    ####################
    # numpy 파일로 저장

    # 저장
    np.save(filename_np, data_np, allow_pickle=False)

def main(args):

    # Video Change Detection, VOS 등 temporal fusion 계열 task에서 flip augmentation을 아예 제외하는 경우가 많습니다.
    # fusion 기반 모델이기 때문에, "augmentation diversity"보다 "spatial consistency"가 훨씬 더 중요함. 이에 따라, T.RandomHorizontalFlip 제거.
    assert args.randomflip == 0.0, f"Expected randomflip to be 0.0, but got {args.randomflip}"
    if args.target_miou is not None:
        assert 0.0 <= args.target_miou <= 100.0, \
            f"--target-miou must be in [0, 100], got {args.target_miou}"

    output_dir = 'output/train_fusion_module'

    # 저장될 경로 생성
    # --------------------------------------------------
    # output_dir / fusion_tmp_dir 결정
    # --------------------------------------------------
    if args.resume:
        checkpoint_for_paths = torch.load(args.resume, map_location='cpu')
        ckpt_args = checkpoint_for_paths.get('args', None)

        assert ckpt_args is not None, \
            "Resume checkpoint does not contain 'args'. Cannot recover original run directories."
        assert hasattr(ckpt_args, 'output_dir'), \
            "Resume checkpoint args do not contain 'output_dir'."
        assert hasattr(ckpt_args, 'fusion_tmp_dir'), \
            "Resume checkpoint args do not contain 'fusion_tmp_dir'."
        
        # resume 시 args 변경 감지를 보강하는 헬퍼 함수
        assert_resume_args_compatible(args, ckpt_args)

        args.output_dir = ckpt_args.output_dir
        args.fusion_tmp_dir = ckpt_args.fusion_tmp_dir
    else:
        # 학습 결과가 저장될 경로
        date_stamped = datetime.datetime.now()  # 파일출력의 기준시간
        save_path = "{}_{}_{}/{date:%Y%m%d_%H%M%S}".format(
            args.model, args.train_dataset, args.data_cv, date=date_stamped)
        args.output_dir = os.path.join(output_dir, save_path)

        # 이전 weight를 갖고 fusion한 결과를 저장할 폴더
        # `precomputed temporal prior` 또는 `offline temporal memory`
        save_path = "fusion_tmp/{date:%Y%m%d_%H%M%S}".format(date=date_stamped)
        args.fusion_tmp_dir = os.path.join(output_dir, save_path)

    utils.mkdir(args.output_dir)
    utils.mkdir(args.fusion_tmp_dir)

    # FileOutputTerminal 클래스를 위한 세팅
    log_filename = os.path.join(args.output_dir, 'terminal.txt')
    # e.g.) 'output/train_fusion/resnet18_mtf_msf_deeplabv3_ChangeSim_Multi_0/20240721_070021/terminal.txt'
    sys.stdout = FileOutputTerminal(log_filename)

    # 'terminal.txt'에 데이터 저장
    print('main$ fusion_tmp_dir : {}'.format(args.fusion_tmp_dir))
    print(args)

    # 현재 사용 중인 GPU의 이름 가져오기
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f'main$ PyTorch device_name : {device_name}')

    # 로그 세팅 전에, 시드(seed)/결정성 세팅
    seed_everything(args.seed, deterministic=args.deterministic, disable_tf32=args.disable_tf32)
    print(f"main$ [seed] seed={args.seed}, deterministic={args.deterministic}, disable_tf32={args.disable_tf32}")

    # Cosine Warm Restarts (SGDR) 사이클 확인 로그
    if args.lr_scheduler.lower() == 'sgdr':
        print(f"main$ [SGDR] T0={args.sgdr_t0} epochs, Tmult={args.sgdr_tmult}, eta_min_mul={args.sgdr_eta_min_mul}")

    # 데이터로더 생성
    dataset, train_sampler, data_loader, dataset_test, data_loader_test = create_dataloader(args)
    args.num_classes = dataset.num_classes

    # p_t를 위한 model
    model = model_dict[args.model](args).to(args.device)

    # f_t를 위한 model
    fusion_module = FusionModule(
        in_channels=args.num_classes,
        num_classes=args.num_classes,
        mode=args.fusion_module_mode,
        variant=args.fusion_module_mode_variant,
        embed_channels=args.fusion_module_channels,
        kernel_size=args.fusion_module_kernel_size,
        dropout_p=args.fusion_module_dropout_p,
        dropout_cap=args.fusion_module_dropout_cap
    ).to(args.device)

    # EMA(Exponential Moving Average) weight tracking (학습 끝마다 EMA 가중치로 평가)
    if getattr(args, "use_ema", False):
        ema = EMA([model, fusion_module], decay=args.ema_decay)  # b=16~25면 0.9995 권장
    else:
        ema = None

    # 학습 준비
    optimizer, criterion, lr_scheduler, scaler, num_warmup = prepare_train(
        args, model, fusion_module, dataset, data_loader
    )

    # 로그 파일(`terminal.txt`)을 중간에 저장하고 닫았다가 다시 열기
    sys.stdout.save_log()

    # resume 기능으로, 이어서 학습하는 경우라면,
    if args.resume:  # resume 옵션이 있다면
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        fusion_module.load_state_dict(checkpoint['fusion_module'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        move_optimizer_state_to_device(optimizer, args.device)  # optimizer state 디바이스 보정
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        best = float(checkpoint.get('best', -1.0))  # 과거 checkpoint에 best가 없을 경우 대비
        start_epoch = int(checkpoint.get('epoch', -1)) + 1  # 과거 checkpoint에 start_epoch가 없을 경우 대비
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        # EMA 상태 복원
        if ema is not None and ('ema' in checkpoint) and (checkpoint['ema'] is not None):
            ema.load_state_dict(checkpoint['ema'], map_to_device=args.device)
            print(f"main()$ [resume] EMA restored (decay={ema.decay})")
        elif 'ema' in checkpoint and checkpoint['ema'] is not None and ema is None:
            print(f"main()$ [resume] Checkpoint has EMA, but --use-ema is off; skipping EMA restore.")
        else:
            print(f"main()$ [resume] No EMA state in checkpoint; starting fresh EMA.")
        flag = checkpoint.get('is_ema_weights', False)
        print(f"main()$ [resume] Loaded weights type: {'EMA' if flag else 'RAW'}")  # best_ema / best_raw 구분 로딩 메시지
        
        print(' ')
        print('=============================================')
        print('[Loading for Resume Training is completed.]')
        print(f' - checkpoint   : {args.resume}')
        print(f' - output_dir   : {args.output_dir}')
        print(f' - fusion_tmp   : {args.fusion_tmp_dir}')
        print(f' - start_epoch  : {start_epoch}')
        print(f' - best_mIoU    : {best:.4f}')
        print('=============================================')
        print(' ')

    else:   # 처음부터 학습한다면,
        best = float(-1.0)   # 초기화
        start_epoch = 0 # 초기화
    
    start_time = time.time()
    stop_training = False
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(args, model, fusion_module, criterion, optimizer, 
                        scaler, data_loader, lr_scheduler, num_warmup, args.device, epoch, args.print_freq, 
                        ema=ema)

        # evaluate
        if epoch % args.eval_every == 0:
            miou = evaluate(args, model, fusion_module, data_loader_test, args.device, epoch, args.flow_dir, ema=ema)
            #print("Validation mIoU: {:.4f}".format(miou))

            if args.target_miou is not None and miou >= args.target_miou:
                stop_training = True

            # 'best.pth' 저장
            if miou > best:
                best = miou

                # -------- best.pth --------
                checkpoint = {
                    'model': model.state_dict(),
                    'fusion_module': fusion_module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'best': best,
                    'args': args,
                    'ema': (ema.state_dict() if ema is not None else None),  # EMA 상태 함께 저장
                    'is_ema_weights': False,
                }
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'best.pth'))
                
                # -------- best_ema.pth (EMA 가중치로 스왑해 저장) --------
                #
                # 평가(mIoU)는 EMA 기준으로 보고 있으므로,
                # 최고 성능 스냅샷도 EMA 가중치 버전을 따로 갖고 있으면 재현/배포가 쉽습니다(best_ema.pth).
                #
                # 동시에 **raw 가중치(best_raw.pth)**도 남겨 두면, 학습 재개나 분석 시 유연합니다.
                #
                # 주기 체크포인트(checkpoint.pth)에 ema 상태를 매번 저장하면
                # 훈련 중단 후 재개에도 EMA 추적이 끊기지 않습니다.

                if ema is not None:
                    with ema.swap_weights():  # 일시 스왑 → 저장 → 자동 원복
                        best_ema = {
                            'model': model.state_dict(),
                            'fusion_module': fusion_module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'epoch': epoch,
                            'best': best,
                            'args': args,
                            'ema': ema.state_dict(),   # EMA 상태 포함
                            'is_ema_weights': True,
                        }
                        utils.save_on_master(best_ema, os.path.join(args.output_dir, 'best_ema.pth'))
                
                # --------------------------------------------------
                # Copy validation metrics corresponding to the
                # checkpoint used for evaluation:
                #   EMA OFF -> best.pth
                #   EMA ON  -> best_ema.pth
                # --------------------------------------------------
                src_metrics = os.path.join(
                    args.output_dir,
                    "validation_metrics",
                    f"epoch_{epoch:05d}",
                )

                dst_root = os.path.join(
                    args.output_dir,
                    "validation_metrics_best",
                )

                dst_metrics = os.path.join(
                    dst_root,
                    f"epoch_{epoch:05d}",
                )

                # 현재 best metrics 복사
                if os.path.exists(src_metrics):

                    # 이전 best 삭제
                    if os.path.exists(dst_root):
                        shutil.rmtree(dst_root)

                    shutil.copytree(src_metrics, dst_metrics)

                    print(f"[BEST] Validation metrics copied to:")
                    print(f"       {dst_metrics}")
                else:
                    print(f"[WARNING] Validation metrics not found:")
                    print(f"          {src_metrics}")
        
        # 매 에폭 'checkpoint.pth' 저장
        checkpoint = {
            'model': model.state_dict(),
            'fusion_module': fusion_module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'best': best,
            'args': args,
            'ema': (ema.state_dict() if ema is not None else None),  # EMA 상태 포함
            'is_ema_weights': False,
        }
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'checkpoint.pth'))

        # Optional target-mIoU stopping. Only evaluated epochs can trigger it.
        if stop_training:
            print('=' * 88)
            print(
                f"Target mIoU reached: validation mIoU={miou:.4f} >= "
                f"target={args.target_miou:.4f}. Training stops after epoch {epoch}."
            )
            print("best.pth/checkpoint.pth and validation metrics have been saved.")
            print('=' * 88)
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # 로그 파일(`terminal.txt`)을 중간에 저장하고 닫았다가 다시 열기
    sys.stdout.save_log()

# `--loss_fusion_weight`의 type을 위한 custom type checker 함수
def check_fusion_weight(value):
    f = float(value)
    if f < 0.0 or f > 1.0:
        raise argparse.ArgumentTypeError("loss_fusion_weight must be in the range [0.0, 1.0]")
    return f

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='PyTorch change detection using fusion_module.py', add_help=add_help)
    parser.add_argument('--train-dataset', default='ChangeSim_Multi', help='dataset name')  # default를 changesim으로 수정
    parser.add_argument('--test-dataset', default='ChangeSim_Multi', help='dataset name')  # default를 changesim으로 수정
    parser.add_argument('--input-size', default=256, type=int, metavar='N', help='the input-size of images')  # default를 changesim으로 수정

    # fusion 기반 모델이기 때문에, "augmentation diversity"보다 "spatial consistency"가 훨씬 더 중요함. 이에 따라, T.RandomHorizontalFlip 제거.
    parser.add_argument('--randomflip', default=0.0, type=float, help='random flip input')    # 0.0으로 설정
    #parser.add_argument('--randomflip', default=0.5, type=float, help='random flip input')
    #parser.add_argument('--randomrotate', dest="randomrotate", action="store_true", help='random rotate input')
    #parser.add_argument('--randomcrop', dest="randomcrop", action="store_true", help='random crop input')

    parser.add_argument('--model', default='resnet18_mtf_msf_deeplabv3', help='model')  # default를 changesim으로 수정
    parser.add_argument('--mtf', default='iade', help='choose branches to use')  # default를 changesim으로 수정
    parser.add_argument('--msf', default=4, type=int, help='the number of MSF layers')  # default를 changesim으로 수정
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)

    # loss.py 관련 하이퍼파라미터
    parser.add_argument('--loss', default='ce', type=str, help='the training loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='FocalLossCE: focusing parameter γ (default: 2.0)')
    parser.add_argument('--focal-alpha', type=float, default=None,
                        help='FocalLossCE: α (scalar). Leave None to disable.')
    parser.add_argument('--dice-weight', type=float, default=0.5,
                        help='CE+Dice: λ for Dice term. Loss = CE + λ*Dice (default: 0.5)')
    parser.add_argument('--ignore-index', type=int, default=255,
                        help='Ignore index used in all losses (default: 255)')

    
    # `--loss-weight` 관련 compute_class_weights_from_ratio() 함수에서 사용하는 하이퍼파라미터
    parser.add_argument('--loss-weight', dest="loss_weight", action="store_true", default=False,
                        help='Add class-weight to loss function (e.g., to handle class imbalance)') # 클래스별 가중치를 포함
    parser.add_argument("--class-weights-gamma", type=float, default=0.5)
    parser.add_argument("--class-weights-boost-change", type=float, default=1.15)
    parser.add_argument("--class-weights-normalize", type=str, default="mean1", choices=["mean1","sum1","none"])
    parser.add_argument("--class-weights-clip-min", type=float, default=0.25)
    parser.add_argument("--class-weights-clip-max", type=float, default=3.0)

    parser.add_argument('--loss_fusion_weight', default=0.5, type=check_fusion_weight,
                        help='Weight for weighted sum of loss. This weight is applied to `loss_fused_output`. Must be in [0.0, 1.0].')
    parser.add_argument('--opt', default='adam', type=str, help='the optimizer')

    parser.add_argument('--lr-scheduler', default='cosine', type=str,
                    help='the lr scheduler (cosine | linear | exp | none | sgdr)')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--warmup', default=True, dest="warmup", action="store_true", help='warmup the lr')
    parser.add_argument('--warmup-period', default=5, type=int, help='Number of epochs to apply learning rate warm-up')
    # Cosine Warm Restarts (SGDR)
    # -  주기적 restart(T0=10~15, Tmult=2)로 지역 최적 탈출 유도.
    #
    # [사용 예시 (CLI)]
    #   --lr-scheduler sgdr --sgdr-t0 12 --sgdr-tmult 2 --sgdr-eta-min-mul 0.2
    #   --warmup --warmup-period 5
    #
    parser.add_argument('--sgdr-t0', default=10, type=int,
                        help='SGDR: number of epochs for the first cycle (T0)')
    parser.add_argument('--sgdr-tmult', default=2.0, type=float,
                        help='SGDR: cycle length multiplier (Tmult)')
    parser.add_argument('--sgdr-eta-min-mul', default=0.2, type=float,
                        help='SGDR: eta_min = base_lr * sgdr-eta-min-mul (per-cycle floor; require 0.0 <= x < 1.0)')
    
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, metavar='W', help='weight decay (default: 0)', dest='weight_decay')
    parser.add_argument('--print-freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--eval-every', default=1, type=int, metavar='N', help='eval the model every n epoch')
    parser.add_argument('--target-miou', default=None, type=float,
                        help='Stop training after an evaluated epoch reaches this validation mIoU percentage (e.g., 29.6). Disabled when omitted.')
    parser.add_argument('--flow-dir', default='/workspace/flownet2-pytorch/work/inference/', type=str)
    parser.add_argument('--use-amp', action='store_true', help='use Automatic Mixed Precision')
    parser.add_argument('--data-cv', default=0, type=int, metavar='N', help='the number of cross validation')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')

    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    #
    # [Resume 사용 예시 (CLI)]
    #    python src/train_fusion_module.py \
    #      [기존 args 그대로 복붙] \
    #      --resume path/to/checkpoint.pth

    parser.add_argument('--fusion-module-mode', default='simple', type=str, help='mode of fusion_module (e.g., simple, residual_block, residual_full)')
    parser.add_argument('--fusion-module-mode-variant', default='2x2_3', type=str,
                        help="Specific architecture variant of the fusion module \
                            (e.g., 2x2_3, 3x3_5, 5x5_7, 7x7_9, \
                            2x2_3r, 3x3_5r, 5x5_7r, 7x7_9r, \
                            2x2_3f, 3x3_5f, 5x5_7f, 7x7_9f). \
                            Each variant defines the number and type of convolutional blocks.")
    #                   구체적인 fusion 구조를 선택합니다 (예: 2x2_3, 3x3_5, 2x2_2r, 3x3_3r, 2x2_3f, 3x3_5f 등)
    parser.add_argument('--fusion-module-channels', default=128, type=int,
                        help='Number of embedding channels used in FusionModule (default: 128) (e.g., 32, 64, 128 — powers of 2 are preferred for GPU efficiency)')
    parser.add_argument('--fusion-module-kernel-size', default=3, type=int,
                        help="Kernel size used in all internal convolution layers of the FusionModule (default: 3). \
                            Only odd values are supported (e.g., 3, 5, 7).")
    #                   FusionModule 내부의 convolution 커널 크기 (기본값: 3).
    #                   3, 5, 7 등의 홀수 값만 지원됩니다.")
    parser.add_argument('--fusion-module-dropout-p', default=0.0, type=float,
                        help="Dropout2d probability used inside the fusion module (0 ≤ p < dropout-cap).")
    parser.add_argument('--fusion-module-dropout-cap', default=0.3, type=float,
                        help="Upper bound for the FusionModule dropout (0 < cap < 1)")
    
    #----- 시드(seed) ----
    # 시드(seed)는 이 난수들의 “시작값”이에요.
    # 시드를 고정하면 같은 코드/데이터로 재현 가능한(run-to-run 같은) 결과를 얻을 확률이 크게 올라갑니다.
    parser.add_argument('--seed', default=2025, type=int, help='random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                        help='use deterministic algorithms (slower but reproducible)')
    parser.add_argument('--disable-tf32', action='store_true',
                        help='disable TF32 (numerically stricter; may slow down on Ampere+)')
    
    #----- 초반에 fusion → backbone으로 흘러가는 gradient만 잠시 차단 ----
    # 초반에 fusion loss까지 backbone에 gradient를 밀어넣으면,
    # backbone은 두 가지 서로 다른 목표(base loss, fusion loss)에 동시에 휘둘려서 초기 진동이 커집니다.
    #
    #  - N=--fusion-detach-epochs
    #    * 0..N-1: detach
    #    * N..2N-1: ramp attach (alpha=1/N,2/N,...,1)
    #    * 2N..: full attach
    #  - N=0이면 기능 비활성 (기존과 동일)
    #
    parser.add_argument(
        '--fusion-detach-epochs', type=int, default=0,
        help='Number of initial epochs to detach fusion-to-backbone gradients; the same number of epochs is used to linearly ramp from detach to full attach. Set 0 to disable.'
        )

    # EMA(Exponential Moving Average) weight tracking (학습 끝마다 EMA 가중치로 평가)
    parser.add_argument(
        "--use-ema", action="store_true", default=False,
        help="Enable EMA (Exponential Moving Average) for evaluation-time weights."
        )
    parser.add_argument(
        '--ema-decay', type=float, default=0.9995,
        help=(
            "EMA decay for evaluation-time shadow weights.\n"
            "Recommended: 0.999~0.9999 (≥0.9995 for larger batches).\n"
            "Lower adapts faster; higher is steadier."
        )
        # 평가 시 사용하는 EMA 섀도 가중치의 감쇠 계수.
        # 권장값: 0.999–0.9999 (큰 배치는 0.9995 이상 권장).
        # 낮으면 적응 빠름, 높으면 더 안정적(반응은 느림).
    )

    # Label Smoothing (e.g., ε=0.05)
    #
    # CE 그대로 두되 overconfidence(특히 static)에 브레이크.
    # “static에 과도 적합” 패턴 완화로 minority change 클래스 개선 → mIoU 소폭↑.
    #
    # [ 사용 가이드(권장값/로그) ]
    #
    # 처음엔 --label-smoothing 0.05로 시작하고, static이 과도하게 confident하면 0.075~0.1까지 가볍게 올려보세요.
    # loss_weight(클래스가중치)와 함께 사용 가능합니다. 위 폴백 구현은 가중치도 정상 반영합니다.
    # 로그에 ε 출력이 보이도록 위에서 넣은 print가 이미 켜져 있어요.
    parser.add_argument(
        '--label-smoothing', type=float, default=0.0,
        help='Label smoothing epsilon (e.g., 0.05). Set 0.0 to disable.'
    )

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
