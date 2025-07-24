import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import numpy as np
from pycocotools.coco import COCO
import torchvision.transforms.functional as F
from PIL import Image
from transforms.mask_rcnn_transforms import Compose, build_mask_rcnn_transforms


class COCODataset(Dataset):
    """
    标准 COCO 格式数据集加载器，适用于目标检测 + 实例分割（Mask R-CNN）。
    """

    def __init__(self, image_dir, ann_file, transforms=None, train=True):
        self.root = image_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.train = train

        # 类别信息
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = tuple(['__background__'] + [c['name'] for c in cats])
        self.num_classes = len(self.classes)
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_to_continuous_id = {
            coco_id: i + 1 for i, coco_id in enumerate(self.cat_ids)
        }

        print(f"加载了 {len(self.ids)} 张图像，{len(self.cat_ids)} 个类别")

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']

        # 使用 PIL 加载图像
        img_path = os.path.join(self.root, path)
        img = Image.open(img_path).convert('RGB')
        img = F.to_tensor(img)

        # 获取原始尺寸（从 COCO 元数据）
        width = img_info['width']
        height = img_info['height']

        boxes = []
        labels = []
        masks = []
        iscrowd = []
        area = []

        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            if len(ann['bbox']) == 0 or len(ann['segmentation']) == 0:
                continue

            cat_id = ann['category_id']
            class_id = self.cat_id_to_continuous_id[cat_id]

            x, y, w, h = ann['bbox']
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)

            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
            mask = coco.annToMask(ann)
            masks.append(mask)
            iscrowd.append(ann.get('iscrowd', 0))
            area.append(ann.get('area', w * h))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)

        image_id = torch.tensor([img_id])

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([int(height), int(width)]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def get_coco_api_from_dataset(dataset):
    return dataset.coco


def build_coco_dataset(image_dir, ann_file, transforms=None, train=True):
    return COCODataset(image_dir, ann_file, transforms, train)


def build_dataloader(dataset, batch_size=2, num_workers=4, train=True):
    from transforms.mask_rcnn_transforms import collate_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def create_coco_dataloader(config, train=True):
    root_dir = config['dataset']['root_dir']
    if train:
        image_dir = os.path.join(root_dir, config['dataset']['train_images'])
        ann_file = os.path.join(root_dir, config['dataset']['train_ann'])
    else:
        image_dir = os.path.join(root_dir, config['dataset']['val_images'])
        ann_file = os.path.join(root_dir, config['dataset']['val_ann'])

    transforms = build_mask_rcnn_transforms(
        train=train,
        min_size=config['transforms']['resize_min'],
        max_size=config['transforms']['resize_max']
    )

    dataset = build_coco_dataset(image_dir, ann_file, transforms, train)
    batch_size = config['training']['batch_size'] if train else 1
    num_workers = config['training']['workers']

    dataloader = build_dataloader(dataset, batch_size, num_workers, train)
    return dataset, dataloader