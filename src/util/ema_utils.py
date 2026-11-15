# ========= ema_utils.py (revised) =========
import contextlib
import torch

class EMA:
    def __init__(self, modules, decay=0.999, device=None):
        """
        modules: [model, fusion_module, ...]
        decay:   0.999 ~ 0.9999 권장
        """
        self.decay = float(decay)
        self.modules = list(modules)
        self.shadow = {}         # key: (mod_idx, name) -> tensor
        self._keys = []          # (mod_idx, name) in a stable order

        for mi, m in enumerate(self.modules):
            for name, p in m.named_parameters():
                if p.requires_grad:
                    k = (mi, name)
                    self._keys.append(k)
                    self.shadow[k] = p.detach().clone().to(device if device is not None else p.device)

    @torch.no_grad()
    def update(self):
        d = self.decay
        for mi, m in enumerate(self.modules):
            for name, p in m.named_parameters():
                if not p.requires_grad:
                    continue
                k = (mi, name)
                s = self.shadow.get(k, None)
                if s is None:
                    # 새 파라미터가 추가된 드문 경우: shadow에 등록
                    self.shadow[k] = p.detach().clone()
                else:
                    s.mul_(d).add_(p.detach(), alpha=1.0 - d)  # s = d*s + (1-d)*p

    @contextlib.contextmanager
    def swap_weights(self):
        """
        평가시에만 파라미터를 EMA 버전으로 스왑. 종료 시 원복.
        """
        backup = {}
        try:
            for mi, m in enumerate(self.modules):
                for name, p in m.named_parameters():
                    if not p.requires_grad:
                        continue
                    k = (mi, name)
                    backup[k] = p.detach().clone()
                    p.data.copy_(self.shadow[k])
            yield
        finally:
            for mi, m in enumerate(self.modules):
                for name, p in m.named_parameters():
                    if not p.requires_grad:
                        continue
                    k = (mi, name)
                    p.data.copy_(backup[k])

    def state_dict(self):
        """
        체크포인트 저장용. 키 이름을 같이 저장(디버깅/호환)하지만,
        로딩 시에는 '현재 모듈 순서'에 맞춰 안전하게 재매핑합니다.
        """
        tensors = []
        for k in self._keys:
            tensors.append(self.shadow[k].cpu())
        return {
            "decay": self.decay,
            "num_modules": len(self.modules),
            "keys": self._keys,      # (mod_idx, name) 목록 (레거시/디버깅용)
            "tensors": tensors,      # shadow 파라미터 텐서 리스트
        }

    def load_state_dict(self, sd, map_to_device=None, strict_shape=True):
        """
        sd: state_dict()
        strict_shape: True면 shape 불일치 항목은 에러 대신 스킵(경고 출력).
        """
        self.decay = float(sd.get("decay", self.decay))
        saved_tensors = sd["tensors"]

        # 현재 모듈의 등록 순서대로 파라미터 목록 만들기
        current_keys = []
        for mi, m in enumerate(self.modules):
            for name, p in m.named_parameters():
                if p.requires_grad:
                    current_keys.append((mi, name))

        # 길이 맞추기: 최소 길이까지 zip
        n = min(len(saved_tensors), len(current_keys))
        for i in range(n):
            k = current_keys[i]
            t = saved_tensors[i].to(map_to_device) if map_to_device is not None else saved_tensors[i]
            # shape 체크
            if strict_shape and k in self.shadow and self.shadow[k].shape != t.shape:
                print(f"[EMA] Warning: shape mismatch for {k}: {self.shadow[k].shape} vs {t.shape} (skipped)")
                continue
            self.shadow[k] = t.detach().clone()

        # 키 목록 갱신
        self._keys = current_keys[:]
