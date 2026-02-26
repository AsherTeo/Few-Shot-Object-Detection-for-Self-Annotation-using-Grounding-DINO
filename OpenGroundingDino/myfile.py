import json
import os
import torch
import shutil
import glob
import cv2
import random
from collections import defaultdict
from PIL import ImageDraw, ImageFont
import numpy as np
from collections import Counter, defaultdict

random.seed(42)
idx = 0
image_idx = 0
set_img = set()

def create_json_categories(cls_dict):
    json_cat = []

    for index, cls_name in cls_dict.items():
        cat_dict = {
                    "id": index,
                    "name": cls_name,
                    "supercategory": "none"
                }
        json_cat.append(cat_dict)
    return json_cat


def create_json_images(set_img, file_name, img_width, img_height):
    global idx

    if file_name in set_img:
        return None
    
    img_dict = {
                "id": idx,
                "file_name": file_name,
                "width": img_width,
                "height": img_height
            }
    
    set_img.add(file_name)
    idx += 1

    return img_dict

def create_json_annotation(cls, bbox):
    global image_idx

    x_min, y_min, bbox_w, bbox_h = bbox
    
    anno_dict = {
                "id": int(image_idx),
                "image_id": int(idx) - 1,
                "category_id": int(cls),
                "bbox": [
                    float(x_min),
                    float(y_min),
                    float(bbox_w),
                    float(bbox_h)
                ],
                "area": float(bbox_w * bbox_h),
                "iscrowd": 0
            }

    image_idx += 1
    return anno_dict

def draw_annotation(img, bbox, label):
    x1, y1, x2, y2 = bbox

    draw = ImageDraw.Draw(img)
    color = tuple(np.random.randint(0, 255, size=3).tolist())

    font = ImageFont.load_default()
    bbox = draw.textbbox((x1, y1), str(label), font)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
    draw.rectangle(bbox, fill=color)
    draw.text((x1, y1), str(label), fill="white")

    return img

def annotation_images(root_path, save_path, cls, status):
    root_path = os.path.join(root_path, cls)

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    root_path1 = os.path.join(root_path, status)
    if not os.path.exists(root_path1):
        os.makedirs(root_path1)

    label_path = os.path.join(root_path1, '_annotations.coco.json')

    with open(label_path, 'r') as f:
        coco = json.load(f)

    img_mapping = {img['id']:img['file_name'] for img in coco['images']}
    cat_mapping = {cat['id']:cat['name'] for cat in coco['categories']}

    images_list = defaultdict(list)
    for anno in coco['annotations']:
        images_list[anno['image_id']].append(anno)

    sample_annotation = list(images_list.keys())

    for image_id in sample_annotation:
        image_path = img_mapping[image_id]
        saves_path = os.path.join(root_path, 'w_label', status)

        if not os.path.exists(saves_path):
            os.makedirs(saves_path)

        saves_path = os.path.join(saves_path, image_path)

        IMAGE_PATH = os.path.join(root_path1, image_path)

        images = cv2.imread(IMAGE_PATH)

        for anno in images_list[image_id]:
            x, y, w, h = map(int, anno['bbox'])
            cat_name = cat_mapping[anno['category_id']]
            cv2.rectangle(images, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(images,cat_name,(x, max(y - 5, 15)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0), 2)

        cv2.imwrite(saves_path, images)

class ExtractClass():
    def __init__(self,  root_path, target_path, status, cls_name, yolo_format = True):
        self.root_path = root_path
        self.target_path = target_path
        self.status =  status
        self.cls_name =  cls_name
        self.yolo_format = yolo_format
        self.srs_path = None
        self.srs_label = None
        self.dst_path = None
        self.dst_label = None

    def status_and_name(self):
        
        self.srs_path = os.path.join(self.root_path, self.status)
        if self.yolo_format:
            self.srs_path = os.path.join(self.srs_path, 'images')
        self.srs_label = self.srs_path + '/' + '_annotations.coco.json'

        self.cls_path = os.path.join(self.target_path, self.cls_name) 
        if not os.path.exists(self.cls_path):
            os.makedirs(self.cls_path)

        self.dst_path = self.cls_path + '/' + self.status
        if not os.path.exists(self.dst_path):
            os.makedirs(self.dst_path)
        self.dst_label = self.dst_path + '/' + '_annotations.coco.json'

        return self.srs_path, self.srs_label, self.dst_path, self.dst_label

    def filter_class_from_json(self):
        
        self.status_and_name() 
        
        with open(self.srs_label, 'r') as f:
            coco = json.load(f)

        name_id = None

        for cat in coco['categories']:
            if cat['name'] == self.cls_name:
                name_id = cat['id']
                target_category = cat
   
        img_mapping = {img['id']:img['file_name'] for img in coco['images']}
        cat_mapping = {cat['id']:cat['name'] for cat in coco['categories']}

        target_annotation = []
        image_ids = set()

        for anno in coco['annotations']:
            if anno['category_id'] == name_id:
                target_annotation.append(anno)
                image_ids.add(anno['image_id'])
        
        images_list = []

        for img_id in image_ids:
            image_path = img_mapping[img_id]
            images_list.append(image_path)
            
            srs_path1 = os.path.join(self.srs_path, image_path)
            dst_path1 = os.path.join(self.dst_path, image_path)
            shutil.copy2(srs_path1, dst_path1)

        target_images = []

        for img in coco['images']:
            if img['file_name'] in images_list:
                target_images.append(img)

        new_label = {
            "categories": [target_category],
            "images": target_images,
            "annotations": target_annotation,
            
        }

        with open(self.dst_label, 'w') as f:
            new_label = json.dump(new_label, f, indent=5)
    
class Merge_Train_CLS():

    json_cat = []
    json_img = []
    json_annotations = []

    def __init__(self, path, cls_img, cls_json, merged_train):
        self.path = path
        self.cls_img = cls_img
        self.cls_json = cls_json
        self.merged_train = merged_train
        self.img_list = []
        self.coco = None
        self.added_files = set()

    def save_name_of_images(self):
        for img_path in glob.glob(self.path + '/' + '*.jpg'):
            file_name = os.path.basename(img_path)
            self.img_list.append(file_name)

    def copy_images(self):

        self.save_name_of_images()
        
        with open(self.cls_json, "r") as f:
            self.coco = json.load(f)

        for img in self.coco['images']:
            if img['file_name'] in self.img_list and img['file_name'] not in self.added_files:
                cls_path = os.path.join(self.cls_img, img['file_name'])
                Merge_Train_CLS.json_img.append(img)
                self.added_files.add(img['file_name'])
                shutil.copy2(cls_path, self.merged_train)

    def save_annotations(self, dict_classes, re_mapping):

        image_ids = {img['id'] for img in Merge_Train_CLS.json_img}
        for anno in self.coco['annotations']:
            if anno['image_id'] in image_ids and anno['category_id'] in re_mapping:
                anno['category_id'] = re_mapping[anno['category_id']]
                Merge_Train_CLS.json_annotations.append(anno)
        json_cat = create_json_categories(dict_classes)
        Merge_Train_CLS.json_cat += json_cat

    @classmethod
    def save_json(cls, output_path):
        new_json = {
            "categories": Merge_Train_CLS.json_cat,
            "images": Merge_Train_CLS.json_img,
            "annotations": Merge_Train_CLS.json_annotations, 
        }

        with open(output_path, 'w') as f:
            json.dump(new_json, f, indent = 5)
            
    @classmethod
    def reset(cls):
        cls.json_cat.clear()
        cls.json_annotations.clear()
        cls.json_img.clear()

    @classmethod
    def count_cls_bbox(cls):
        counter = Counter()

        for anno in cls.json_annotations:
            counter[anno['category_id']] += 1
        return counter
    
    @classmethod
    def img_lvl_cls_counting(cls):
        counter = Counter()
        image_to_classes = defaultdict(set)

        for ann in cls.json_annotations:
            image_id = ann['image_id']
            cat_id = ann['category_id']
            image_to_classes[image_id].add(cat_id)

        for class_set in image_to_classes.values():
            for cat_id in class_set:
                counter[cat_id] += 1

        return counter

    @staticmethod
    def print_bbox(dict, counter):
        for cls, cnt in counter.items():
            print(f'For {dict[cls]}, there are {cnt} bbox')
        
# class Merge_Train_CLS():

#     json_cat = []
#     json_img = []
#     json_annotations = []

#     def __init__(self, path, cls_img, cls_json, merged_train):
#         self.path = path
#         self.cls_img = cls_img
#         self.cls_json = cls_json
#         self.merged_train = merged_train
#         self.img_list = []
#         self.coco = None
#         self.added_files = set()

#     def save_name_of_images(self):
#         for img_path in glob.glob(self.path + '/' + '*.jpg'):
#             file_name = os.path.basename(img_path)
#             self.img_list.append(file_name)

#     def copy_images(self):

#         self.save_name_of_images()
        
#         with open(self.cls_json, "r") as f:
#             self.coco = json.load(f)

#         for img in self.coco['images']:
#             if img['file_name'] in self.img_list and img['file_name'] not in self.added_files:
#                 cls_path = os.path.join(self.cls_img, img['file_name'])
#                 Merge_Train_CLS.json_img.append(img)
#                 self.added_files.add(img['file_name'])
#                 shutil.copy2(cls_path, self.merged_train)

#     def save_annotations(self, idx):

#         image_ids = {img['id'] for img in self.json_img}

#         for anno in self.coco['annotations']:
#             if anno['image_id'] in image_ids:
#                 anno['category_id'] = idx
#                 for cat in self.coco['categories']:
#                     cat['id'] = idx
#                 Merge_Train_CLS.json_annotations.append(anno)
        
#         Merge_Train_CLS.json_cat += self.coco['categories']

#     @classmethod
#     def save_json(cls, output_path):
#         new_json = {
#             "categories": Merge_Train_CLS.json_cat,
#             "images": Merge_Train_CLS.json_img,
#             "annotations": Merge_Train_CLS.json_annotations, 
#         }

#         with open(output_path, 'w') as f:
#             json.dump(new_json, f, indent = 5)
            
#     @classmethod
#     def reset(cls):
#         cls.json_cat.clear()
#         cls.json_annotations.clear()
#         cls.json_img.clear()

class Random_select():

    json_cat = []
    json_img = []
    json_annotation = []

    def __init__(self, root_path, target_path, cls, status):

        self.root_path = root_path
        self.target_path = target_path
        self.cls = cls
        self.status = status
        self.label_path = None
        self.target_label = None
        self.path = None
        self.coco = None
        self.sample_id = []

    @staticmethod
    def create(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def folder(self, dir):

        if dir == self.root_path:

            dir = os.path.join(dir, self.cls) 
            self.create(dir)

        dir = dir + '/' + self.status
        self.create(dir)

        dir_label = dir + '/' + '_annotations.coco.json'
        return dir, dir_label
    
    def create_folder(self):

        self.path, self.label_path = self.folder(self.root_path)
        self.target_path, self.target_label = self.folder(self.target_path)

    def save_images(self):

        self.create_folder()

        with open(self.label_path, 'r') as f:
            self.coco = json.load(f)

        Random_select.json_img += self.coco['images']

        for img in self.coco['images']:
            file_name = img['file_name']
            self.sample_id.append(img['id'])
            srs_path = os.path.join(self.path, file_name)
            dst_path = os.path.join(self.target_path, file_name)
            shutil.copy2(srs_path, dst_path)

    def random_save_images(self, rdm_number):

        self.create_folder()

        with open(self.label_path, 'r') as f:
            self.coco = json.load(f)

        img = self.coco['images']
        sample_images = random.sample(img, rdm_number)
        Random_select.json_img += sample_images

        for anno in sample_images:
            file_name = anno['file_name']
            self.sample_id.append(anno['id'])
            srs_path = os.path.join(self.path, file_name)
            dst_path = os.path.join(self.target_path, file_name)

            shutil.copy2(srs_path, dst_path)

    def save_json(self, idx):

        Random_select.json_cat += self.coco['categories']       

        for anno in self.coco['annotations']:
            if anno['image_id'] in self.sample_id:
                anno['category_id'] = idx
                for cat in self.coco['categories']:
                    cat['id'] = idx
                Random_select.json_annotation.append(anno)

    @classmethod
    def new_json(cls, output):
        new_coco = {
            'categories' : Random_select.json_cat,
            'images': Random_select.json_img,
            'annotations': Random_select.json_annotation
        }
            
        with open(output, 'w') as f:
            json.dump(new_coco, f, indent = 5)

    @classmethod
    def reset(cls):
        Random_select.json_cat.clear()
        Random_select.json_img.clear()
        Random_select.json_annotation.clear()

    @classmethod
    def count_cls(cls):
        counter = Counter()

        for anno in cls.json_annotation:
            counter[anno['category_id']] += 1
        return counter
    
def coco2xyxy(bbox):
    x,y,w,h = bbox
    return (x, y, x + w, y + h)

def cxcywh2xyxy(bbox, img_w , img_h):
    cx, cy, w, h = bbox
    x_min = (cx - w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    x_max = (cx + w / 2) * img_w
    y_max = (cy + h / 2) * img_h

    #x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
    
    return (x_min, y_min, x_max, y_max)

def get_gt_info(path, json_path):
    gt_file = []

    target_file = os.path.basename(path)
    with open(json_path, "r") as f:
        coco = json.load(f)

    img_mapping = {img['id']:img['file_name'] for img in coco['images']}

    for anno in coco['annotations']:
        file_name= img_mapping[anno['image_id']]
        if file_name == target_file:
            bbox = coco2xyxy(anno['bbox'])
            cls = anno['category_id']
            gt_file.append((bbox, cls))
    return gt_file

def get_gt_id(path, json_path):
    gt_id = []

    target_file = os.path.basename(path)
    with open(json_path, "r") as f:
        coco = json.load(f)

    img_mapping = {img['id']:img['file_name'] for img in coco['images']}

    for anno in coco['annotations']:
        file_name= img_mapping[anno['image_id']]
        if file_name == target_file:
            id = anno['id']
            image_id = anno['image_id']
            gt_id.append((id, image_id))
    return gt_id

class BBOX():
    def __init__(self, gt_bbox, pred_bbox, device):

        if not isinstance(gt_bbox, torch.Tensor):
            gt_bbox = torch.tensor(gt_bbox, dtype = torch.float32, device = device)

        if not isinstance(pred_bbox, torch.Tensor):
            pred_bbox = torch.tensor(pred_bbox, dtype = torch.float32, device = device)

        self.gt = gt_bbox
        self.pred = pred_bbox

        if self.gt.ndim == 1:
            self.gt = self.gt.unsqueeze(0)
        if self.pred.ndim == 1:
            self.pred = self.pred.unsqueeze(0)


    def intersection(self):
        
        top_left = torch.maximum(self.gt[:, None, :2],self.pred[None, :, :2])
        bottom_right = torch.minimum(self.gt[:, None, 2:],self.pred[None, :, 2:])

        wh = (bottom_right - top_left).clamp(min=0)
        return wh[..., 0] * wh[...,1]
    
    def find_area(self, box):
        area = (box[..., 2:] - box[..., :2]).clamp(min=0)
        return area[..., 0] * area[..., 1]
    
    def union(self):
        gt_area = self.find_area(self.gt)[:, None]
        pred_area =self.find_area(self.pred)[None, :]
        return gt_area + pred_area - self.intersection()
    
    def compute_iou(self, eps = 1e-6):
        return self.intersection()/(self.union() + eps)

class Metrics():
    def __init__(self, TP, FP, FN):
        self.TP = TP
        self.FP = FP
        self.FN = FN

    @property
    def precision(self):
        return self.TP/ (self.TP + self.FP + 1e-6)
    
    @property
    def recall(self):
        return self.TP/ (self.TP + self.FN + 1e-6)

def matching_bbox(gt_bbox, pred_boxes, threshold, gt_cls, pred_cls, device):
        global TP, FP, FN
        bboxx = BBOX(gt_bbox, pred_boxes, device)
        iou_matrix = bboxx.compute_iou()

        matched_gt = set()
        matched_pred = set()

        for i in range(len(gt_bbox)):
            best_iou = 0
            best_j = -1

            for j in range(len(pred_boxes)):
                if j in matched_pred:
                    continue

                if gt_cls[i] != pred_cls[j]:
                    continue

                iou = iou_matrix[i, j]

                if iou >= threshold and iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j != -1:
                TP += 1
                matched_gt.add(i)
                matched_pred.add(best_j)

        FP += len(pred_boxes) - len(matched_pred)
        FN += len(gt_bbox) - len(matched_gt)