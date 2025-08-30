# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# ------------------------------------------------------------------------------

from .COCOKeypoints import CocoKeypoints as coco
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose
from .build import make_dataloader
from .build import make_test_dataloader

# dataset dependent configuration for visualization
# 更新为文档四个角点的标签
coco_part_labels = [
    'LT', 'RT','RD','LD'
]

# 更新为角点的索引映射
coco_part_idx = {
    b: a for a, b in enumerate(coco_part_labels)
}

# 更新为连接顺序，连接顺序为左上、右上、右下、左下
coco_part_orders = [
    ('LT', 'RT'),  # 左上到右上
    ('RT', 'RD'),  # 右上到右下
    ('RD', 'LD'),  # 右下到左下
    ('LD', 'LT'),  # 左下到左上
]


crowd_pose_part_labels = [
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'head', 'neck'
]
crowd_pose_part_idx = {
    b: a for a, b in enumerate(crowd_pose_part_labels)
}
crowd_pose_part_orders = [
    ('head', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle')
]

VIS_CONFIG = {
    'COCO': {
        'part_labels': coco_part_labels,
        'part_idx': coco_part_idx,
        'part_orders': coco_part_orders
    },
    'CROWDPOSE': {
        'part_labels': crowd_pose_part_labels,
        'part_idx': crowd_pose_part_idx,
        'part_orders': crowd_pose_part_orders
    }
}
