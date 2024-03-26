import os
import json
import numpy as np
import cv2
from PIL import Image
from pycocotools.coco import COCO


def create_segmentation_masks(annotation_file, img_dir, mask_dir):
    coco = COCO(annotation_file)
    catIds = coco.getCatIds(catNms=['ball'])
    imgIds = coco.getImgIds(catIds=catIds)
    all_imgIds = coco.getImgIds()

    for imgId in all_imgIds:
        img_info = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # Initialisiere eine leere Maske für jedes Bild
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                for segmentation in ann['segmentation']:
                    if isinstance(segmentation, list):  # check if segmentation is in polygon format
                        polygon = np.array(segmentation).reshape((-1, 2))
                        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)

        mask_img = Image.fromarray(mask).convert('L')
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        mask_filename = img_info['file_name'].replace('.jpg', '.png')
        mask_path = os.path.join(mask_dir, mask_filename)
        mask_img.save(mask_path)


"""
# Train Images
annotation_file = 'C:/Users/Ella/Documents/Masterarbeit/SSFormer/SSformer/data/tablesoccer.v2i.coco/train/_annotations.coco.json'
img_dir = 'C:/Users/Ella/Documents/Masterarbeit/SSFormer/SSformer/data/tablesoccer.v2i.coco/train'
mask_dir = 'C:/Users/Ella/Documents/Masterarbeit/SSFormer/SSformer/data/tablesoccer.v2i.coco/train/segmentation_masks'
"""
"""
# Validation Images
annotation_file = ('C:/Users/Ella/Documents/Masterarbeit/SSFormer/SSformer/data/tablesoccer.v2i.coco/valid'
                   '/_annotations.coco.json')
img_dir = 'C:/Users/Ella/Documents/Masterarbeit/SSFormer/SSformer/data/tablesoccer.v2i.coco/valid'
mask_dir = 'C:/Users/Ella/Documents/Masterarbeit/SSFormer/SSformer/data/tablesoccer.v2i.coco/valid/segmentation_masks'
"""

# Test Images
annotation_file = ('C:/Users/Ella/Documents/Masterarbeit/SSFormer/SSformer/data/tablesoccer.v2i.coco/test/_annotations'
                   '.coco.json')
img_dir = 'C:/Users/Ella/Documents/Masterarbeit/SSFormer/SSformer/data/tablesoccer.v2i.coco/test'
mask_dir = 'C:/Users/Ella/Documents/Masterarbeit/SSFormer/SSformer/data/tablesoccer.v2i.coco/test/segmentation_masks'

create_segmentation_masks(annotation_file, img_dir, mask_dir)
