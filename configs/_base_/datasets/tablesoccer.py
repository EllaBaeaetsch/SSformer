import os

# dataset settings
dataset_type = 'TableSoccerDataset'
# data_root = 'data/tablesoccer.v2i.coco-mmdetection'
# data folder
data_root = "C:/Users/Ella/Documents/Masterarbeit/SSFormer/SSformer/data/tablesoccer.v2i.coco"
train_img_dir = os.path.join(data_root, 'train')
train_ann_dir = os.path.join(train_img_dir, 'segmentation_masks')

valid_img_dir = os.path.join(data_root, 'valid')
valid_ann_dir = os.path.join(valid_img_dir, 'segmentation_masks')

test_img_dir = os.path.join(data_root, 'test')
test_ann_dir = os.path.join(test_img_dir, 'segmentation_masks')

# Image Normalization Configuration
# mean and std of the three color channels
img_norm_cfg = dict(
    mean=[92.851, 129.125, 96.417], std=[77.589, 62.468, 64.544], to_rgb=True)
# crop_size = (512, 512)
# crop_size crops images


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='LoadAnnotations', reduce_zero_label=False),
    # reduce_zero_label: Class labels are reduced by one. So 1 becomes 0 and 2 becomes 1

    dict(type='Resize', ratio_range=(0.5, 2.0)),
    # img_scale: scale image to a specified size
    # ratio_range: vary aspect ratios in a specific area

    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # data augmentation operation, crop_size defines size of the crop,
    # cat_max_ratio specifies the maximum ratio between the largest object size and the image area

    dict(type='RandomFlip', prob=0.0),
    # data-augmentation operation, 50% chance to flip a image

    # dict(type='PhotoMetricDistortion'),
    # data-augmentation operation, it generates photometric distortions in images

    dict(type='Normalize', **img_norm_cfg),
    # Pre-processing operation,
    # normalizes image pixels before training, Standardize data in a more uniform area

    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    # Padding, pad_val=0 means that the added pixels are black,
    # black borders generally do not distort the image information
    # seg_pad_val=255 padding-value for segmentation masks

    dict(type='DefaultFormatBundle'),
    # Data (images, annotations, masks) are converted into a standardized format

    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    # determines which data points / elements from the pre-processed and enriched data collection are fed into the model
    # keys indicates which elements are collected
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    #    dict(type='Normalize', **img_norm_cfg),
    #    dict(type='ImageToTensor', keys=['img']),
    #    dict(type='Collect', keys=['img']),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(640, 295), (295, 640)],
        flip=False,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        transforms=[
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Datensatzaufbau
data = dict(
    # indirect defining batch size
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=train_img_dir,
        ann_dir=train_ann_dir,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=valid_img_dir,
        ann_dir=valid_ann_dir,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=test_img_dir,
        ann_dir=test_ann_dir,
        pipeline=test_pipeline))
