import os
import socket   # hostname에 따라서 path를 다르게 하기 위한 모듈

def get_dataset_path(name):

    # 현재 호스트 이름 가져오기
    hostname = socket.gethostname()

    if name == 'coco':
        """
        ── annotations
        │   ├── instances_train2017.json
        │   ├── instances_val2017.json
        ├── test2017
        │   └── *.jpg
        ├── train2017
        │   └── *.jpg
        └── val2017
            └── *.jpg

        """
        return '/data_path/coco_2017'

    if name == 'PCD_raw':
        """
        ├── GSV
        │   ├── mask
        │   │   └── *.png
        │   ├── README.txt
        │   ├── t0
        │   │   └── *.jpg
        │   └── t1
        │       └── *.jpg
        └── TSUNAMI
            ├── mask
            │   └── *.png
            ├── README.txt
            ├── t0
            │   └── *.jpg
            └── t1
                └── *.jpg
        """
        return '/data_path/pcd'

    if name == 'PCD_CV':
        """
        ├── set0
        │   ├── test
        │   │   ├── mask
        │   │   ├── t0
        │   │   └── t1
        │   ├── test.txt
        │   ├── train
        │   │   ├── mask
        │   │   ├── t0
        │   │   └── t1
        │   └── train.txt
        ├── set1
        ├── set2
        ├── set3
        └── set4
        """
        return '/data_path/pcd_5cv'

    if name == 'CMU_binary':
        """
        ├── test
        │   ├── mask
        │   ├── t0
        │   └── t1
        └── train
            ├── mask
            ├── t0
            └── t1
        """
        # Perform different actions according to the hostname
        if hostname == 'C-3PO_default':  # C-3PO default
            return '/data_path/VL-CMU-CD-binary255'
        else:
            return '/workspace/C-3PO_dataset/VL-CMU-CD-binary255'

    if name == 'CMU_raw':
        """
        ├── raw
        ├── test
        └── train
        """
        # Perform different actions according to the hostname
        if hostname == 'C-3PO_default':  # C-3PO default
            return '/data_path/VL-CMU-CD'
        else:
            return '/workspace/C-3PO_dataset/VL-CMU-CD'

    if name == 'ChangeSim':
        """
        ├── Query
        │   ├── Query_Seq_Test
        │   └── Query_Seq_Train
        └── Reference
            ├── Ref_Seq_Test
            └── Ref_Seq_Train
        """
        # Perform different actions according to the hostname
        if hostname == 'C-3PO_default':  # C-3PO default
            return '/data_path/ChangeSim'
        else:
            return '/workspace/C-3PO_dataset/ChangeSim'
