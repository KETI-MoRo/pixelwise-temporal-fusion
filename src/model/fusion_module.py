"""
FusionModule
============

This module performs learnable temporal fusion between:
1. warped_f_{t-1} (Bayesian fusion result from the previous time step)
2. output_t (the model output for current time step)

Various implementations are supported:
- SimpleFusionModule: basic CNN fusion (no skip)
- ResidualFusionBlockModule: fusion with residual blocks
- ResidualFusionFullModule: entire fusion path with residual connection

Main Class:
-----------
FusionModule(...) will internally select one of the various implementations based on `mode`.

Usage:
------
    fusion_module = FusionModule(in_channels=5, embed_channels=128, num_classes=5, mode='simple')
    pred_t = fusion_module(output_curr, warped_prev)

"""

import torch
import torch.nn as nn
#import torch.nn.functional as F

##########################################
# Residual block building block
##########################################
class ResidualBlock(nn.Module):
    def __init__(self, channels, zero_init_residual: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),  # <- bn_final (gamma=0으로 초기화)
        )
        self.relu = nn.ReLU(inplace=True)

        # “last BN γ=0” 트릭
        # 블록이 초기에는 거의 ReLU(x)처럼 동작해서 학습이 안정적
        if zero_init_residual:
            # gamma는 BatchNorm의 weight입니다. 마지막 BN의 weight를 0으로 초기화해 주면 돼요.
            bn_final = self.block[-1]
            if isinstance(bn_final, nn.BatchNorm2d) and bn_final.affine:
                nn.init.zeros_(bn_final.weight)  # gamma <- 0
                # bias(beta)는 기본 0이므로 건드릴 필요 없음

    def forward(self, x):
        return self.relu(x + self.block(x))  # Skip connection


##########################################
# Mode 1: SimpleFusionModule (baseline)
##########################################
class SimpleFusionModule(nn.Module):
    def __init__(self, in_channels, num_classes,
                 embed_channels=128, kernel_size=3, variant='2x2_3',
                 dropout_p: float = 0.1, dropout_cap: float = 0.3, verbose=True):
        """
        Initialize the FusionModule.

        Args:
            in_channels (int): Number of input channels (usually equal to number of classes).
            num_classes (int): Number of output classes.
            embed_channels (int): Number of channels after embedding.
        """
        super().__init__()

        assert kernel_size % 2 == 1, "Only odd kernel sizes (3, 5, 7, ...) are supported"
        self.kernel_size = kernel_size

        # Dropout 설정 & 검증
        if not (0.0 <= float(dropout_p) < 1.0):
            raise ValueError(f"dropout_p must be in [0,1), got {dropout_p}")
        if not (0.0 < float(dropout_cap) < 1.0):
            raise ValueError(f"dropout_cap must be in (0,1), got {dropout_cap}")

        # Dropout cap 적용 (초과 시 자동 클램프)
        if float(dropout_p) > float(dropout_cap) and verbose:
            print(f"FusionModule$ dropout_p={dropout_p} exceeds cap={dropout_cap}. Using {dropout_cap}.")
        self.dropout_p = min(float(dropout_p), float(dropout_cap))
        self.dropout_cap = float(dropout_cap)

        if variant == '2x2_3':
            if verbose:
                print(f"FusionModule$ Using SimpleFusionModule (variant = '{variant}')")
            self.init_2x2_3(in_channels, num_classes, embed_channels)
        elif variant == '3x3_5':
            if verbose:
                print(f"FusionModule$ Using SimpleFusionModule (variant = '{variant}')")
            self.init_3x3_5(in_channels, num_classes, embed_channels)
        elif variant == '5x5_7':
            if verbose:
                print(f"FusionModule$ Using SimpleFusionModule (variant = '{variant}')")
            self.init_5x5_7(in_channels, num_classes, embed_channels)
        elif variant == '7x7_9':
            if verbose:
                print(f"FusionModule$ Using SimpleFusionModule (variant = '{variant}')")
            self.init_7x7_9(in_channels, num_classes, embed_channels)
        else:
            raise ValueError(f"FusionModule$ Unknown variant for SimpleFusionModule: {variant}")

    # Dropout 구현
    def _drop(self, p: float):
        
        p = max(0.0, min(float(p), self.dropout_cap))

        # p==0이면 Identity, 그 외 Dropout2d(p)
        return nn.Identity() if p <= 0.0 else nn.Dropout2d(p)

    # cnn1/2 각 2개 conv block, fusion에 3개 block
    def init_2x2_3(self, in_channels, num_classes, embed_channels):

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        # Embedding layer for current model output (output_t)
        # 현재 시점의 모델 출력(output_t)을 위한 임베딩 레이어
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )
        
        # Embedding warped f_{t-1}
        # Embedding layer for warped previous fusion result (f_{t-1})
        # 이전 시점의 fusion 결과(f_{t-1})를 위한 임베딩 레이어
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion conv block
        # Fusion block to combine f1 and f2 embeddings
        # f1과 f2 임베딩을 결합하는 fusion 블록
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p),  # single Dropout2d spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Final classifier
        # 최종 예측을 위한 classifier 레이어
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1)

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    # cnn1/2 각 3개 conv block, fusion에 5개 block
    def init_3x3_5(self, in_channels, num_classes, embed_channels):

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        # Embedding layer for current model output (output_t)
        # 현재 시점의 모델 출력(output_t)을 위한 임베딩 레이어
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )
        
        # Embedding warped f_{t-1}
        # Embedding layer for warped previous fusion result (f_{t-1})
        # 이전 시점의 fusion 결과(f_{t-1})를 위한 임베딩 레이어
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion conv block
        # Fusion block to combine f1 and f2 embeddings
        # f1과 f2 임베딩을 결합하는 fusion 블록
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p),  # single Dropout2d spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Final classifier
        # 최종 예측을 위한 classifier 레이어
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1)

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    # cnn1/2 각 5개 conv block, fusion에 7개 block
    def init_5x5_7(self, in_channels, num_classes, embed_channels):

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        # Embedding layer for current model output (output_t)
        # 현재 시점의 모델 출력(output_t)을 위한 임베딩 레이어
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )
        
        # Embedding warped f_{t-1}
        # Embedding layer for warped previous fusion result (f_{t-1})
        # 이전 시점의 fusion 결과(f_{t-1})를 위한 임베딩 레이어
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion conv block
        # Fusion block to combine f1 and f2 embeddings
        # f1과 f2 임베딩을 결합하는 fusion 블록
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p),  # single Dropout2d spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Final classifier
        # 최종 예측을 위한 classifier 레이어
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1)

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    # cnn1/2 각 7개 conv block, fusion에 9개 block
    def init_7x7_9(self, in_channels, num_classes, embed_channels):

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        # Embedding layer for current model output (output_t)
        # 현재 시점의 모델 출력(output_t)을 위한 임베딩 레이어
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )
        
        # Embedding warped f_{t-1}
        # Embedding layer for warped previous fusion result (f_{t-1})
        # 이전 시점의 fusion 결과(f_{t-1})를 위한 임베딩 레이어
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion conv block
        # Fusion block to combine f1 and f2 embeddings
        # f1과 f2 임베딩을 결합하는 fusion 블록
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p),  # single Dropout2d spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Final classifier
        # 최종 예측을 위한 classifier 레이어
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1)

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    def forward(self, output_curr, warped_prev):
        """
        Forward pass for FusionModule.

        Args:
            output_curr (Tensor): Output from current model inference (output_t), shape (B, C, H, W)
            warped_prev (Tensor): Warped fusion result from previous timestep (f_{t-1}), shape (B, C, H, W)

        Returns:
            Tensor: Prediction map (pred_t), shape (B, num_classes, H, W)
        """
        # Step 1: Embed each input
        # 각 입력을 임베딩합니다.
        f1 = self.cnn1(output_curr)
        f2 = self.cnn2(warped_prev)

        # Step 2: Concatenate and fuse
        # 두 임베딩을 concat 후 fusion conv를 적용합니다.
        fused = torch.cat([f1, f2], dim=1)

        # Step 3: Classify
        # 최종 예측 결과 생성
        fused_feature = self.fusion_conv(fused)
        return self.classifier(fused_feature)
        
    """
    # SimpleFusionModule 흐름
    
    Input: output_curr      warped_prev
       [B, 5, H, W]     [B, 5, H, W]
           │                │
     cnn1 (Conv2d → 128)    cnn2 (Conv2d → 128)
           │                │
       [B,128,H,W]      [B,128,H,W]
           └───── Concatenate ─────┘
               ↓   dim=1 (channel axis)
            [B,256,H,W]   ← f1+f2
               ↓
     FusionConv (Conv2d 256→128, ×3)
               ↓
            [B,128,H,W]
               ↓
      Classifier (Conv2d 128→5)
               ↓
           Output: [B,5,H,W]
    """


##########################################
# Mode 2: ResidualFusionBlockModule
##########################################
class ResidualFusionBlockModule(nn.Module):
    def __init__(self, in_channels, num_classes, 
                 embed_channels=128, kernel_size=3, variant='2x2_3r', 
                 dropout_p: float = 0.1, dropout_cap: float = 0.3, verbose=True):
        """
        Fusion module using residual blocks for deeper and more stable learning.

        Args:
            in_channels (int): Number of input channels (usually equal to number of classes).
            num_classes (int): Number of output classes.
            embed_channels (int): Number of channels after embedding.
        """
        super().__init__()

        assert kernel_size % 2 == 1, "Only odd kernel sizes (3, 5, 7, ...) are supported"
        self.kernel_size = kernel_size

        # Dropout 설정 & 검증
        if not (0.0 <= float(dropout_p) < 1.0):
            raise ValueError(f"dropout_p must be in [0,1), got {dropout_p}")
        if not (0.0 < float(dropout_cap) < 1.0):
            raise ValueError(f"dropout_cap must be in (0,1), got {dropout_cap}")

        # Dropout cap 적용 (초과 시 자동 클램프)
        if float(dropout_p) > float(dropout_cap) and verbose:
            print(f"FusionModule$ dropout_p={dropout_p} exceeds cap={dropout_cap}. Using {dropout_cap}.")
        self.dropout_p = min(float(dropout_p), float(dropout_cap))
        self.dropout_cap = float(dropout_cap)

        if variant == '2x2_3r':
            if verbose:
                print(f"FusionModule$ Using ResidualFusionBlockModule (variant = '{variant}')")
            self.init_2x2_3r(in_channels, num_classes, embed_channels)
        elif variant == '3x3_5r':
            if verbose:
                print(f"FusionModule$ Using ResidualFusionBlockModule (variant = '{variant}')")
            self.init_3x3_5r(in_channels, num_classes, embed_channels)
        elif variant == '5x5_7r':
            if verbose:
                print(f"FusionModule$ Using ResidualFusionBlockModule (variant = '{variant}')")
            self.init_5x5_7r(in_channels, num_classes, embed_channels)
        elif variant == '7x7_9r':
            if verbose:
                print(f"FusionModule$ Using ResidualFusionBlockModule (variant = '{variant}')")
            self.init_7x7_9r(in_channels, num_classes, embed_channels)
        else:
            raise ValueError(f"FusionModule$ Unknown variant for ResidualFusionBlockModule: {variant}")

    # Dropout 구현
    def _drop(self, p: float):

        p = max(0.0, min(float(p), self.dropout_cap))

        # p==0이면 Identity, 그 외 Dropout2d(p)
        return nn.Identity() if p <= 0.0 else nn.Dropout2d(p)


    # cnn1/2 각 2개 conv block, fusion에 3개 residual block
    def init_2x2_3r(self, in_channels, num_classes, embed_channels):
        """
        cnn1:        Conv → BN → ReLU → Conv → BN → ReLU        → 2개
        cnn2:        Conv → BN → ReLU → Conv → BN → ReLU        → 2개
        reduce:      Conv → BN → ReLU                           → 1개
        resblock:    [Conv → BN → ReLU → Conv → BN] × 3         → 6개
        classifier:  Conv (1x1)                                 → 1개
        -------------------------------------------------------------
        총합:                                                  → 12개
        """

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        # Embedding layer for current model output (output_t)
        # 현재 시점의 모델 출력(output_t)을 위한 임베딩 레이어
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Embedding warped f_{t-1}
        # Embedding layer for warped previous fusion result (f_{t-1})
        # 이전 시점의 fusion 결과(f_{t-1})를 위한 임베딩 레이어
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Reduce channel after concat (from 2C to C)
        # Channel reducer after concatenation of f1 and f2
        # f1과 f2를 concat하면 채널 수가 2배로 늘어나므로 이를 줄여주는 역할
        # - ResidualBlock을 효과적으로 사용하기 위한 구조적 필요성
        # - (Residual block 전제조건) 입력과 출력의 channel 수가 동일해야 함
        # - (Residual block 내부) `return self.relu(x + self.block(x))`
        self.reduce = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p),   # Dropout2d
        )

        # Fusion block using two residual blocks
        # residual block을 3개 쌓아서 깊이 있는 feature 통합 수행
        # Skip connection을 통해 gradient 흐름 안정화
        # - 내부의 특정 블록(self.resblock)이 residual 구조를 갖고 있기 때문에,
        #   전체 모듈 이름이 ResidualFusionModule인 것
        self.resblock = nn.Sequential(
            ResidualBlock(embed_channels),
            self._drop(self.dropout_p * 0.5),   # Dropout2d
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
        )

        # Final classifier
        # 최종 예측을 위한 classifier 레이어
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1)

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    # cnn1/2 각 3개 conv block, fusion에 5개 residual block
    def init_3x3_5r(self, in_channels, num_classes, embed_channels):

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        # Embedding layer for current model output (output_t)
        # 현재 시점의 모델 출력(output_t)을 위한 임베딩 레이어
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Embedding warped f_{t-1}
        # Embedding layer for warped previous fusion result (f_{t-1})
        # 이전 시점의 fusion 결과(f_{t-1})를 위한 임베딩 레이어
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Reduce channel after concat (from 2C to C)
        # Channel reducer after concatenation of f1 and f2
        # f1과 f2를 concat하면 채널 수가 2배로 늘어나므로 이를 줄여주는 역할
        # - ResidualBlock을 효과적으로 사용하기 위한 구조적 필요성
        # - (Residual block 전제조건) 입력과 출력의 channel 수가 동일해야 함
        # - (Residual block 내부) `return self.relu(x + self.block(x))`
        self.reduce = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p),   # Dropout2d
        )

        # Fusion block using two residual blocks
        # residual block을 쌓아서 깊이 있는 feature 통합 수행
        # Skip connection을 통해 gradient 흐름 안정화
        # - 내부의 특정 블록(self.resblock)이 residual 구조를 갖고 있기 때문에,
        #   전체 모듈 이름이 ResidualFusionModule인 것
        self.resblock = nn.Sequential(
            ResidualBlock(embed_channels),
            self._drop(self.dropout_p * 0.5),   # Dropout2d
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
        )

        # Final classifier
        # 최종 예측을 위한 classifier 레이어
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1)

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    # cnn1/2 각 5개 conv block, fusion에 7개 residual block
    def init_5x5_7r(self, in_channels, num_classes, embed_channels):

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        # Embedding layer for current model output (output_t)
        # 현재 시점의 모델 출력(output_t)을 위한 임베딩 레이어
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Embedding warped f_{t-1}
        # Embedding layer for warped previous fusion result (f_{t-1})
        # 이전 시점의 fusion 결과(f_{t-1})를 위한 임베딩 레이어
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Reduce channel after concat (from 2C to C)
        # Channel reducer after concatenation of f1 and f2
        # f1과 f2를 concat하면 채널 수가 2배로 늘어나므로 이를 줄여주는 역할
        # - ResidualBlock을 효과적으로 사용하기 위한 구조적 필요성
        # - (Residual block 전제조건) 입력과 출력의 channel 수가 동일해야 함
        # - (Residual block 내부) `return self.relu(x + self.block(x))`
        self.reduce = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p),   # Dropout2d
        )

        # Fusion block using two residual blocks
        # residual block을 쌓아서 깊이 있는 feature 통합 수행
        # Skip connection을 통해 gradient 흐름 안정화
        # - 내부의 특정 블록(self.resblock)이 residual 구조를 갖고 있기 때문에,
        #   전체 모듈 이름이 ResidualFusionModule인 것
        self.resblock = nn.Sequential(
            ResidualBlock(embed_channels),
            self._drop(self.dropout_p * 0.5),   # Dropout2d
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
        )

        # Final classifier
        # 최종 예측을 위한 classifier 레이어
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1)

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    # cnn1/2 각 7개 conv block, fusion에 9개 residual block
    def init_7x7_9r(self, in_channels, num_classes, embed_channels):

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        # Embedding layer for current model output (output_t)
        # 현재 시점의 모델 출력(output_t)을 위한 임베딩 레이어
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Embedding warped f_{t-1}
        # Embedding layer for warped previous fusion result (f_{t-1})
        # 이전 시점의 fusion 결과(f_{t-1})를 위한 임베딩 레이어
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Reduce channel after concat (from 2C to C)
        # Channel reducer after concatenation of f1 and f2
        # f1과 f2를 concat하면 채널 수가 2배로 늘어나므로 이를 줄여주는 역할
        # - ResidualBlock을 효과적으로 사용하기 위한 구조적 필요성
        # - (Residual block 전제조건) 입력과 출력의 channel 수가 동일해야 함
        # - (Residual block 내부) `return self.relu(x + self.block(x))`
        self.reduce = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p),   # Dropout2d
        )

        # Fusion block using two residual blocks
        # residual block을 쌓아서 깊이 있는 feature 통합 수행
        # Skip connection을 통해 gradient 흐름 안정화
        # - 내부의 특정 블록(self.resblock)이 residual 구조를 갖고 있기 때문에,
        #   전체 모듈 이름이 ResidualFusionModule인 것
        self.resblock = nn.Sequential(
            ResidualBlock(embed_channels),
            self._drop(self.dropout_p * 0.5),   # Dropout2d
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
            ResidualBlock(embed_channels),
        )

        # Final classifier
        # 최종 예측을 위한 classifier 레이어
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1)

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    def forward(self, output_curr, warped_prev):
        """
        Forward pass for ResidualFusionModule.

        Args:
            output_curr (Tensor): Output from current model inference (output_t), shape (B, C, H, W)
            warped_prev (Tensor): Warped fusion result from previous timestep (f_{t-1}), shape (B, C, H, W)

        Returns:
            Tensor: Prediction map (pred_t), shape (B, num_classes, H, W)
        """
        # Step 1: Embed each input
        # f1: 현재 timestep 결과의 embedding
        # f2: 이전 timestep 결과의 embedding
        f1 = self.cnn1(output_curr)
        f2 = self.cnn2(warped_prev)

        # Step 2: Concatenate and reduce
        # 두 feature를 채널 차원에서 concat한 뒤, 채널 수를 줄여줍니다
        fused = torch.cat([f1, f2], dim=1)
        reduced = self.reduce(fused)

        # 채널 수 확인
        assert reduced.shape[1] == self.classifier.in_channels, \
            f"Expected input to resblock to have {self.classifier.in_channels} channels, but got {reduced.shape[1]}."
            #f"resblock 입력 채널 수가 예상과 다릅니다. 현재: {reduced.shape[1]}, 예상: {self.classifier.in_channels}"

        # Step 3: Residual fusion
        # residual block을 통해 더욱 정교하게 feature를 융합합니다
        fused_feature = self.resblock(reduced)
        
        # Step 4: Classify
        # 최종 예측 결과 생성
        return self.classifier(fused_feature)
        
    """
    # ResidualFusionBlockModule 흐름
    
    Input: output_curr      warped_prev
       [B, 5, H, W]            [B, 5, H, W]
           │                       │
     cnn1 (Conv2d → 128)     cnn2 (Conv2d → 128)
           │                       │
       [B,128,H,W]            [B,128,H,W]
           └───── Concatenate ─────┘
               ↓
            [B,256,H,W]   ← f1+f2
               ↓
     Reduce Conv (Conv2d 256→128)
               ↓
            [B,128,H,W]
               ↓
     ResidualBlock (×3, skip connection)
               ↓
            [B,128,H,W]
               ↓
      Classifier (Conv2d 128→5)
               ↓
           Output: [B,5,H,W]
    """


##########################################
# Mode 3: ResidualFusionFullModule
##########################################
class ResidualFusionFullModule(nn.Module):
    def __init__(self, in_channels, num_classes, 
                 embed_channels=128, kernel_size=3, variant='2x2_3f', 
                 dropout_p: float = 0.1, dropout_cap: float = 0.3, verbose=True):
        """
        Fusion module with global residual structure:
        The entire fusion path is wrapped with a residual connection.

        Args:
            in_channels (int): Number of input channels (usually equal to number of classes).
            num_classes (int): Number of output classes.
            embed_channels (int): Number of channels after embedding.
        """
        super().__init__()

        assert kernel_size % 2 == 1, "Only odd kernel sizes (3, 5, 7, ...) are supported"
        self.kernel_size = kernel_size

        # Dropout 설정 & 검증
        if not (0.0 <= float(dropout_p) < 1.0):
            raise ValueError(f"dropout_p must be in [0,1), got {dropout_p}")
        if not (0.0 < float(dropout_cap) < 1.0):
            raise ValueError(f"dropout_cap must be in (0,1), got {dropout_cap}")

        # Dropout cap 적용 (초과 시 자동 클램프)
        if float(dropout_p) > float(dropout_cap) and verbose:
            print(f"FusionModule$ dropout_p={dropout_p} exceeds cap={dropout_cap}. Using {dropout_cap}.")
        self.dropout_p = min(float(dropout_p), float(dropout_cap))
        self.dropout_cap = float(dropout_cap)

        if variant == '2x2_3f':
            if verbose:
                print(f"FusionModule$ Using ResidualFusionFullModule (variant = '{variant}')")
            self.init_2x2_3f(in_channels, num_classes, embed_channels)
        elif variant == '3x3_5f':
            if verbose:
                print(f"FusionModule$ Using ResidualFusionFullModule (variant = '{variant}')")
            self.init_3x3_5f(in_channels, num_classes, embed_channels)
        elif variant == '5x5_7f':
            if verbose:
                print(f"FusionModule$ Using ResidualFusionFullModule (variant = '{variant}')")
            self.init_5x5_7f(in_channels, num_classes, embed_channels)
        elif variant == '7x7_9f':
            if verbose:
                print(f"FusionModule$ Using ResidualFusionFullModule (variant = '{variant}')")
            self.init_7x7_9f(in_channels, num_classes, embed_channels)
        else:
            raise ValueError(f"FusionModule$ Unknown variant for ResidualFusionFullModule: {variant}")

    # Dropout 구현
    def _drop(self, p: float):
        
        p = max(0.0, min(float(p), self.dropout_cap))

        # p==0이면 Identity, 그 외 Dropout2d(p)
        return nn.Identity() if p <= 0.0 else nn.Dropout2d(p)

    # cnn1/2 각 2개 conv block, fusion에 3개 fusion conv
    def init_2x2_3f(self, in_channels, num_classes, embed_channels):
        """
        cnn1:        Conv → BN → ReLU → Conv → BN → ReLU        → 2개
        cnn2:        Conv → BN → ReLU → Conv → BN → ReLU        → 2개
        fusion_core: Conv → BN → ReLU                           → 1개
                     Conv → BN → ReLU                           → 1개
                     Conv → BN                                  → 1개
                     (→ 총 3개의 Conv2d 블록)
        shortcut:    f1 + f2 (Global residual connection)       → 0개 (연산 only)
        relu:        ReLU(core + shortcut)                      → 0개 (연산 only)
        classifier:  Conv (1x1)                                 → 1개
        -------------------------------------------------------------
        총합:                                                  → 9개
        """

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Embedding warped f_{t-1}
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion block (Conv + ReLU × 3)
        self.fusion_core = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p), # Dropout 필수 spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p * 0.5), # Dropout 옵션 spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            # forward()에서 fused_feature = self.relu(core + shortcut)로 nn.ReLU(inplace=True) 수행
        )

        self.relu = nn.ReLU(inplace=True)

        # Final classifier
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1, bias=True)  # 1x1은 BN이 없으니 bias OK

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    # cnn1/2 각 3개 conv block, fusion에 5개 fusion conv
    def init_3x3_5f(self, in_channels, num_classes, embed_channels):

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Embedding warped f_{t-1}
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion block (Conv + ReLU × 5)
        self.fusion_core = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p), # Dropout 필수 spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p * 0.5), # Dropout 옵션 spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            # forward()에서 fused_feature = self.relu(core + shortcut)로 nn.ReLU(inplace=True) 수행
        )

        self.relu = nn.ReLU(inplace=True)

        # Final classifier
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1, bias=True)  # 1x1은 BN이 없으니 bias OK

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    # cnn1/2 각 5개 conv block, fusion에 7개 fusion conv
    def init_5x5_7f(self, in_channels, num_classes, embed_channels):

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Embedding warped f_{t-1}
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion block (Conv + ReLU × 7)
        self.fusion_core = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p), # Dropout 필수 spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p * 0.5), # Dropout 옵션 spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            # forward()에서 fused_feature = self.relu(core + shortcut)로 nn.ReLU(inplace=True) 수행
        )

        self.relu = nn.ReLU(inplace=True)

        # Final classifier
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1, bias=True)  # 1x1은 BN이 없으니 bias OK

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    # cnn1/2 각 7개 conv block, fusion에 9개 fusion conv
    def init_7x7_9f(self, in_channels, num_classes, embed_channels):

        k = self.kernel_size
        p = k // 2

        # Embedding output_t
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Embedding warped f_{t-1}
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
        )

        # Fusion block (Conv + ReLU × 9)
        self.fusion_core = nn.Sequential(
            nn.Conv2d(embed_channels * 2, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p), # Dropout 필수 spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            self._drop(self.dropout_p * 0.5), # Dropout 옵션 spot
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(embed_channels),
            # forward()에서 fused_feature = self.relu(core + shortcut)로 nn.ReLU(inplace=True) 수행
        )

        self.relu = nn.ReLU(inplace=True)

        # Final classifier
        self.classifier = nn.Conv2d(embed_channels, num_classes, kernel_size=1, bias=True)  # 1x1은 BN이 없으니 bias OK

        # Sanity check
        assert self.classifier.in_channels == embed_channels, \
            f"Classifier input channels must match embed_channels ({embed_channels}), got {self.classifier.in_channels}"

    def forward(self, output_curr, warped_prev):
        """
        Forward pass with global residual skip connection.

        Args:
            output_curr (Tensor): Output from current model inference (output_t), shape (B, C, H, W)
            warped_prev (Tensor): Warped fusion result from previous timestep (f_{t-1}), shape (B, C, H, W)

        Returns:
            Tensor: Prediction map (pred_t), shape (B, num_classes, H, W)
        """
        f1 = self.cnn1(output_curr)
        f2 = self.cnn2(warped_prev)
        fused = torch.cat([f1, f2], dim=1)

        # Global residual: skip connection over entire fusion block
        core = self.fusion_core(fused)
        shortcut = f1 + f2
        fused_feature = self.relu(core + shortcut)  # f1+f2 used as shortcut

        return self.classifier(fused_feature)

    """
    # ResidualFusionFullModule 흐름

    Input: output_curr      warped_prev
        [B, 5, H, W]            [B, 5, H, W]
            │                       │
        cnn1 (Conv2d → 128)     cnn2 (Conv2d → 128)
            │                       │
        [B,128,H,W]             [B,128,H,W]
            └───── Concatenate ─────┘
                ↓
            [B,256,H,W]   ← f1+f2
                ↓
    FusionCore Conv (×3) → [B,128,H,W]
                ↓
        Add residual (f1+f2) → [B,128,H,W]
                ↓
        Classifier (Conv2d 128→5)
                ↓
            Output: [B,5,H,W]
    """
    """
    output_curr ───► cnn1 ─┐
                           │               ┌─────► fusion_core ─┐
    warped_prev ───► cnn2 ─┘── concat ───►─┘                    │
                                           └──────► core        │
                    cnn1 ─► f1                                f1 + f2 (shortcut)
                    cnn2 ─► f2                                ↓
                                                            add → ReLU → Classifier
    """
        

##########################################
# Interface Class: FusionModule
# - 실제 구현은, Delegation 방식으로 설계됨
##########################################
class FusionModule(nn.Module):
    def __init__(
            self, in_channels, num_classes,
            mode='simple', variant='2x2_3', 
            embed_channels=128, kernel_size=3,
            dropout_p: float = 0.0, dropout_cap: float = 0.3,
            verbose=True):
        """
        Interface class that delegates to one of various internal fusion implementations.

        Args:
            in_channels (int): Number of input channels (e.g., 5 for 5-class segmentation).
            num_classes (int): Number of output classes.
            embed_channels (int): Embedding dimension used for internal CNNs. (c.f., )
                               (e.g., 32, 64, 128 — powers of 2 are preferred for GPU efficiency)
            mode (str): Mode of FusionModule (e.g. 'simple', 'residual_block', 'residual_full').
        """
        super().__init__()

        if mode == 'simple':
            # skip connection 없는 기본 CNN fusion
            self.impl = SimpleFusionModule(in_channels, num_classes, 
                                           embed_channels, kernel_size, variant, 
                                           dropout_p=dropout_p, dropout_cap=dropout_cap, verbose=verbose)
            if verbose:
                print(f"FusionModule$ Using mode: '{mode}': SimpleFusionModule.")
                print(f"FusionModule$ Embedding dimension used for internal CNNs: '{embed_channels}'")
        elif mode == 'residual_block':
            # Residual block만 중간에 사용하는 구조
            self.impl = ResidualFusionBlockModule(in_channels, num_classes, 
                                                  embed_channels, kernel_size, variant, 
                                                  dropout_p=dropout_p, dropout_cap=dropout_cap, verbose=verbose)
            if verbose:
                print(f"FusionModule$ Using mode: '{mode}': ResidualFusionBlockModule.")
                print(f"FusionModule$ Embedding dimension used for internal CNNs: '{embed_channels}'")
        elif mode == 'residual_full':
            # 전체 fusion path를 residual로 감싼 구조
            self.impl = ResidualFusionFullModule(in_channels, num_classes, 
                                                 embed_channels, kernel_size, variant, 
                                                 dropout_p=dropout_p, dropout_cap=dropout_cap, verbose=verbose)
            if verbose:
                print(f"FusionModule$ Using mode: '{mode}': ResidualFusionFullModule.")
                print(f"FusionModule$ Embedding dimension used for internal CNNs: '{embed_channels}'")
        else:
            raise ValueError(f"FusionModule$ Unknown mode: {mode}. Choose from ['simple', 'residual_block', 'residual_full']")

    def forward(self, output_curr, warped_prev):
        """
        Forward pass of FusionModule.

        Note:
            - No softmax is applied inside this module.
            - Input tensors (output_curr and warped_prev) are expected to be raw logits.
            - Softmax, if needed (e.g., for visualization), should be applied outside.

        Args:
            output_curr (Tensor): Raw logits from current model inference, shape (B, C, H, W)
            warped_prev (Tensor): Warped raw logits from previous fusion result, shape (B, C, H, W)

        Returns:
            Tensor: Raw logits (before softmax), shape (B, num_classes, H, W)
        """
        return self.impl(output_curr, warped_prev)
