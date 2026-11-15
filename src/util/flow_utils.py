import numpy as np
import os
import torch
import torch.nn.functional as F

from PIL import Image


# .flo 파일을 읽어서 numpy 배열로 변환하는 함수
def read_flo_file(flow_path, npy_shape, reverse_x = True, reverse_y = True):
    """
    Read .flo optical flow file in Middlebury format
    Returns a numpy array of shape (H, W, 2)

    Args:
        flow_path (str): path to .flo file
        npy_shape (tuple): shape of corresponding f_prev numpy array (C, H, W) (e.g., (5, 256, 256))
    """

    # "000-1.flo"라는 이상한 파일명이 들어올 때에는 zero flow로 return
    # Special handling for first frame (t=0 case)
    fname = os.path.basename(flow_path) # 파일명만 분리해서 확인
    if any(s in fname for s in ("-1.flo", "000-1.flo", "-00001.flo")):
        #print(f"[INFO] Flow path {flow_path} indicates t=0 → using zero flow")
        _, H, W = npy_shape  # extract height and width from (C, H, W)
        return np.zeros((H, W, 2), dtype=np.float32)
    
    with open(flow_path, 'rb') as f:

        assert flow_path[-4:] == '.flo', "The file extension is not .flo : %r" % flow_path[-4:]

        #  .flo 파일의 맨 처음 데이터는 매직 넘버(magic number)로, 특정한 값이어야 합니다.
        magic = np.fromfile(f, np.float32, count=1)[0]
        assert magic == 202021.25, "Magic number %r incorrect. Invalid .flo file" % magic

        # https://github.com/Johswald/flow-code-python/blob/master/readFlowFile.py
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])

        # npy_shape는 (C, H, W)
        _, H, W = npy_shape
        assert (h, w) == (H, W), f".flo header size {(h, w)} != expected {(H, W)}"
        
        #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
        data = np.fromfile(f, np.float32, count=2*w*h)
        assert data.size == 2*w*h, f"Unexpected .flo payload size: {data.size} vs {2*w*h}"

    # Reshape data into 3D array (columns, rows, bands)
    #flow = np.resize(data, (h, w, 2))  # resize는 길이가 맞지 않으면 데이터를 반복/잘라서 “맞춰버리기” 때문에, 파일 손상도 조용히 통과할 수 있어요.
    flow = data.reshape(h, w, 2)

    # 연속 메모리 강제 확보 → 이후 torch.from_numpy() 속도 & 호환성 ↑
    flow = np.ascontiguousarray(flow)  # ADD: ensure C-contiguous layout

    # optical flow 반전 설정 추가
    #reverse_x = True    # 예: 좌우 반전
    #reverse_y = True    # 예: 상하 반전

    if reverse_x:
        flow[..., 0] *= -1
    if reverse_y:
        flow[..., 1] *= -1
        
    return flow


def warp_using_flow_cuda(input_logits_tensor, flow, device, flipped=False):
    """
    CUDA-accelerated warping of (C, H, W) prob_map using (H, W, 2) flow.

    Args:
        input_logits_tensor (CUDA tensor): shape (C, H, W), logits tensor
        flow (np.ndarray): shape (H, W, 2), optical flow (flo 파일에서 읽은 결과)
        flipped (bool): 학습 시 horizontal flip 여부 (Optional)

    Returns:
        tensor (CUDA tensor): warped logits tensor (C, H, W)
    """

    # device 인자를 안전하게 torch.device로 변환
    device = torch.device(device) if not isinstance(device, torch.device) else device

    with torch.no_grad():  # 학습과 무관한 연산임을 명시

        # --------------------
        # Step 1. Flow 처리
        # --------------------
        C, H, W = input_logits_tensor.shape

        # (안전) 해상도 검증 (flow vs 입력(input_logits_tensor))
        assert flow.shape[:2] == (H, W), f"Flow size {flow.shape[:2]} != {(H, W)}"

        # (안전) 연속 메모리 보장 (from_numpy 호환/성능)
        flow = np.ascontiguousarray(flow)

        flow_tensor = torch.from_numpy(flow).float()  # (H, W, 2), CPU
        if flipped:
            flow_tensor = torch.flip(flow_tensor, dims=[1])  # flip width
            flow_tensor[..., 0] *= -1  # horizontal component flip

        flow_tensor = flow_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 2, H, W)

        # --------------------
        # Step 2. Grid 생성
        # --------------------
        # Create base grid: shape = (1, H, W, 2)

        # grid 생성 (입력과 동일 dtype)
        try:
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device),
                                            torch.arange(W, device=device),
                                            indexing="ij")
        except TypeError:
            # PyTorch 1.10 이전 버전
            grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device),
                                            torch.arange(W, device=device))

        base_grid = torch.stack((grid_x, grid_y), dim=0).float().to(device)  # (2, H, W)
        grid = base_grid + flow_tensor[0]  # (2, H, W)

        # Normalize to [-1, 1] for grid_sample
        grid[0] = 2.0 * grid[0] / (W - 1) - 1.0  # x
        grid[1] = 2.0 * grid[1] / (H - 1) - 1.0  # y
        grid = grid.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 2)

        # --------------------
        # Step 3. Interpolation
        # --------------------

        # (중요) dtype 정렬: grid_sample은 입력과 grid가 같은 float dtype이어야 안정적
        input_logits_tensor = input_logits_tensor.unsqueeze(0)  # (1, C, H, W)
        input_logits_tensor = input_logits_tensor.to(device=device, dtype=torch.float32).contiguous()
        grid = grid.to(dtype=input_logits_tensor.dtype).contiguous()

        # 1) 일단 zeros 패딩으로 샘플링 (연산이 가장 빠르고, 이후 OOB(Out Of Bounds)를 덮어씌움)
        warped = F.grid_sample(
            input_logits_tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )  # (1, C, H, W)

        # 2) OOB(프레임 밖) 마스크를 계산해 그 위치는 '균일 로짓'으로 채우기
        #    grid의 범위가 [-1,1]을 벗어나면 OOB

        gx = grid[..., 0]   # grid shape: (1, H, W, 2)
        gy = grid[..., 1]
        oob = (gx < -1) | (gx > 1) | (gy < -1) | (gy > 1)  # (1, H, W), bool

        if oob.any():

            # 채널 차원으로 브로드캐스트될 마스크 만들기
            #C = input_logits_tensor.shape[1]   # line 88에서 이미 C가 초기화되어있음.
            mask = oob.unsqueeze(1).expand(-1, C, -1, -1)  # (1, C, H, W)

            # NOTE: 모든 채널에 같은 로짓값을 넣으면 softmax 후 확률은 1/C가 됩니다.
            # 0.0로 채워도 동일하지만, 여기서는 명시적으로 1.0/C를 사용합니다.
            warped = warped.masked_fill(mask, 1.0 / C)

    return warped.squeeze(0)  # (C, H, W)


if __name__ == "__main__":
    
    #####################################
    #
    # flow_utils.py에 대한 테스트코드
    #
    #####################################

    # [NOTE] `00000.flo`는 `0.png`와 `1.png`의 FlowNet2에 대한 결과임.

    path_input_img = '/workspace/C-3PO_dataset/ChangeSim256x256/Query/Query_Seq_Train/Warehouse_0/Seq_1/rgb/50.png'
    path_flo_file = '/workspace/flownet2-pytorch/work/inference/Query_Seq_Train-0-1/000050.flo'
    #path_input_img = '/workspace/C-3PO_dataset/ChangeSim256x256/Query/Query_Seq_Train/Warehouse_0/Seq_1/rgb/113.png'
    #path_flo_file = '/workspace/flownet2-pytorch/work/inference/Query_Seq_Train-0-1/000113.flo'
    path_output_npy = './output/flow_utils/test_flow_utils.npy'
    path_output_img = './output/flow_utils/test_flow_utils.png'

    # 출력폴더가 없으면 출력폴더 생성
    os.makedirs('./output/flow_utils', exist_ok=True)


    # ① PNG 이미지를 numpy로 로드
    img_np = np.array(Image.open(path_input_img))  # (H, W, 3), dtype=uint8
    img_np = img_np.astype(np.float32) / 255.0     # [0, 1] 범위 정규화

    # ② (C, H, W)로 변환
    img_np_chw = np.transpose(img_np, (2, 0, 1))  # (3, H, W)

    # ③ Torch tensor로 변환 (CUDA로 이동)
    img_tensor = torch.from_numpy(img_np_chw).cuda()

    # ④ Optical flow 불러오기 (.flo → numpy)
    flow_np = read_flo_file(path_flo_file, npy_shape=img_np_chw.shape)  # (H, W, 2)

    # ⑤ CUDA 기반 warping
    warped_tensor = warp_using_flow_cuda(img_tensor, flow_np, device=torch.device('cuda'), flipped=False)  # (3, H, W)

    # ⑥ 저장 (Numpy, PNG)
    warped_np = warped_tensor.detach().cpu().numpy()  # (3, H, W)
    np.save(path_output_npy, warped_np)

    # ⑦ (H, W, 3)로 바꾸고 clamp 후 저장
    warped_img = np.transpose(warped_np, (1, 2, 0))  # (H, W, 3)
    warped_img = np.clip(warped_img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(warped_img).save(path_output_img)

    print(f"Warped tensor saved: {path_output_npy}")
    print(f"Warped image saved:  {path_output_img}")
