# pixelwise-temporal-fusion


## Installation

- An NVIDIA GPU is required.
- This repository uses Docker with a bind mount.
    - [Reference] [Installing the NVIDIA Docker Toolkit](docs/install_docker.md)

### Prepare the bind mount

- The instructions below assume that `~/workspace` is used as the bind mount source directory.

### Clone the repository

```
cd ~/workspace
git clone https://github.com/KETI-MoRo/pixelwise-temporal-fusion.git
```

### Build the Docker image
```
cd pixelwise-temporal-fusion/
sudo docker build -t pixelwise-temporal-fusion:1.0 .
```

### Create the Docker container
- Create the container using the bind mount prepared above.
```
sudo docker run -d \
  --name pixelwise-temporal-fusion \
  --gpus all \
  --shm-size=20g \
  -v ~/workspace:/workspace \
  pixelwise-temporal-fusion:1.0 \
  bash -c "sleep infinity"
```

### Enter the Docker container
```
sudo docker exec -it pixelwise-temporal-fusion /bin/bash
```

## Dataset Preparation

### ChangeSim
- Dataset usage and data details: [ChangeSim data documentation](https://github.com/SAMMiCA/ChangeSim/blob/main/data.md)
- If you use the ChangeSim dataset, please cite:

  ```bibtex
  @inproceedings{park2021changesim,
    author = {Park, Jin-Man and Jang, Jae-hyuk and Yoo, Sahng-Min and Lee, Sun-Kyung and Kim, Ue-hwan and Kim, Jong-Hwan},
    title = {ChangeSim: Towards End-to-End Online Scene Change Detection in Industrial Indoor Environments},
    booktitle = {2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year = {2021},
    organization = {IEEE},
    url = {https://arxiv.org/abs/2103.05368}
  }
  ```
- ChangeSim GitHub Pages: [Link](https://sammica.github.io/ChangeSim/)
- Dataset download
    - [GitHub Repository README.md](https://github.com/SAMMiCA/ChangeSim?tab=readme-ov-file#dataset-download)
- Download the ChangeSim zip files.
    | File name           | Size     |
    |---------------------|----------|
    | Ref_Seq_Train.zip   | 52.8GB   |
    | Ref_Seq_Test.zip    | 30.2GB   |
    | Query_Seq_Train.zip | 42.8GB   |
    | Query_Seq_Test.zip  | 30.3GB   |
- Unzip `Query_Seq_Train.zip` and `Query_Seq_Test.zip`.
- Example directory structure after extraction:
    ```
    root@73645f46c205:/workspace/C-3PO_dataset/ChangeSim$ tree -d -L 3
    .
    └── Query
        ├── Query_Seq_Test
        │   ├── Warehouse_6
        │   ├── Warehouse_7
        │   ├── Warehouse_8
        │   └── Warehouse_9
        └── Query_Seq_Train
            ├── Warehouse_0
            ├── Warehouse_1
            ├── Warehouse_2
            ├── Warehouse_3
            ├── Warehouse_4
            └── Warehouse_5
    ```

### Resize the ChangeSim dataset images
- Optical flow must be applied iteratively to the temporal fusion outputs.
- To reduce the computational load, optical flow is precomputed using [FlowNet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch/tree/71034046166735a79a5b82df78de72d806e82842), and the saved results are used during temporal fusion.
- The optical flow fields produced by FlowNet2 must have the same spatial dimensions as the images used for temporal fusion.
- The image size used for temporal fusion is determined by the change detection network, [C-3PO](https://github.com/DoctorKey/C-3PO/tree/7aadc692c3313f7212b1c0f8114f7a2005419617).
    - [NOTE] The [default](https://github.com/DoctorKey/C-3PO/blob/7aadc692c3313f7212b1c0f8114f7a2005419617/exp/changesim/imagenet_resnet18_mtf_iade_msf4_deeplabv3_changesim_multi.sh) input size for `C-3PO` is 256 × 256.
        ```bash
        python3 -m torch.distributed.launch --nproc_per_node=4 --use_env src/train.py --train-dataset ChangeSim_Multi --test-dataset ChangeSim_Multi --input-size 256 --model resnet18_mtf_msf_deeplabv3 --mtf iade --msf 4  --warmup --loss ce
        ```
- [`dataset_image_resize.py`](src/dataset_image_resize.py) resizes the dataset images to match the default input size above.
- Usage:
    ```bash
    $ cd /workspace/pixelwise-temporal-fusion/
    $ python /workspace/pixelwise-temporal-fusion/src/dataset_image_resize.py --path_sequence /Query/Query_Seq_Train/Warehouse_5/Seq_1/rgb
    ```
- `dataset_image_resize.py` processes one ChangeSim sequence at a time. To process all sequences, run the script for each `rgb` directory.
- Example output directory structure:
    - [NOTE] We use only the `Query` data for this repository. Therefore, we resize only the images in the `Query` folder.
    ```
    root@73645f46c205:/workspace/C-3PO_dataset/ChangeSim256x256$ tree -d -L 2
    .
    └── Query
        ├── Query_Seq_Test
        └── Query_Seq_Train
    ```

### FlowNet2
- Original repository: [FlowNet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch/tree/71034046166735a79a5b82df78de72d806e82842)
- Git commit:
    ```
    git checkout 71034046166735a79a5b82df78de72d806e82842
    ```
- Refer to the official FlowNet2 documentation for detailed installation and usage instructions.
- Example directory structure for FlowNet2 outputs computed from the resized images in the `ChangeSim256x256` folder:
    ```
    root@73645f46c205:/workspace/flownet2-pytorch$ tree ./ -L 3
    ./
    └── work
        └── inference
            ├── Query_Seq_Test-6-0
            ├── Query_Seq_Test-6-1
            ├── Query_Seq_Test-7-0
            ├── Query_Seq_Test-7-1
            ├── Query_Seq_Test-8-0
            ├── Query_Seq_Test-8-1
            ├── Query_Seq_Test-9-0
            ├── Query_Seq_Test-9-1
            ├── Query_Seq_Train-0-0
            ├── Query_Seq_Train-0-1
            ├── Query_Seq_Train-1-0
            ├── Query_Seq_Train-1-1
            ├── Query_Seq_Train-2-0
            ├── Query_Seq_Train-2-1
            ├── Query_Seq_Train-3-0
            ├── Query_Seq_Train-3-1
            ├── Query_Seq_Train-4-0
            ├── Query_Seq_Train-4-1
            ├── Query_Seq_Train-5-0
            └── Query_Seq_Train-5-1
    ```

### Directory Structure of the ChangeSim Dataset and FlowNet2 Outputs

The following directory listings were generated using the `tree` command after dataset preparation. Intermediate entries have been omitted to reduce file size. The total file count is provided in the summary line of each listing.

- [ChangeSim dataset](exp/tree_dataset/tree_ChangeSim.txt)
- [FlowNet2 outputs](exp/tree_dataset/tree_flownet2.txt)


## Running the Recipes from Table 1

Inside the container, navigate to the repository root:

```bash
cd /workspace/pixelwise-temporal-bayesian-fusion
```

Choose a recipe below and run it individually.

- C-3PO (Reproduced)
    - C-3PO serves as the baseline in this paper. Our implementation builds on the [official C-3PO codebase](https://github.com/DoctorKey/C-3PO), which we extend with the proposed pixel-wise temporal fusion module.
    ```bash
    bash exp/table1+2/C-3PO.sh
    ```
- Q1
    ```bash
    bash exp/table1+2/Q1.sh
    ```
- A3
    ```bash
    bash exp/table1+2/A3.sh
    ```
- B2
    ```bash
    bash exp/table1+2/B2.sh
    ```
- T2
    ```bash
    bash exp/table1+2/T2.sh
    ```


## Citation

If you use this repository in your research, please cite:

```bibtex
@Article{s26227168,
    AUTHOR = {Kim, Dong Yeop and Kim, Keunhwan and Hwang, Jung-Hoon and Kim, Euntai},
    TITLE = {Revisiting Scene Change Detection with Pixel-Wise Temporal Fusion},
    JOURNAL = {Sensors},
    VOLUME = {26},
    YEAR = {2026},
    NUMBER = {22},
    ARTICLE-NUMBER = {7168},
    URL = {https://www.mdpi.com/1424-8220/26/22/7168},
    ISSN = {1424-8220},
    DOI = {10.3390/s26227168}
}
```
