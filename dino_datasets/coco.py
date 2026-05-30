<<<<<<< HEAD
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from torchvision import transforms
from dino_datasets.data_util import preparing_dataset
import dino_datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou
import dino_datasets.sltransform as SLT
__all__ = ['build']


class label2compat():
    def __init__(self) -> None:
        self.category_map_str = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20} # "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
        self.category_map = {int(k):v for k,v in self.category_map_str.items()}

    def __call__(self, target, img=None):
        labels = target['labels']
        res = torch.zeros(labels.shape, dtype=labels.dtype)
        for idx, item in enumerate(labels):
            res[idx] = self.category_map[item.item()] - 1
        target['label_compat'] = res
        if img is not None:
            return target, img
        else:
            return target


class label_compat2onehot():
    def __init__(self, num_class=80, num_output_objs=1):
        self.num_class = num_class
        self.num_output_objs = num_output_objs
        if num_output_objs != 1:
            raise DeprecationWarning("num_output_objs!=1, which is only used for comparison")

    def __call__(self, target, img=None):
        labels = target['label_compat']
        place_dict = {k:0 for k in range(self.num_class)}
        if self.num_output_objs == 1:
            res = torch.zeros(self.num_class)
            for i in labels:
                itm = i.item()
                res[itm] = 1.0
        else:
            # compat with baseline
            res = torch.zeros(self.num_class, self.num_output_objs)
            for i in labels:
                itm = i.item()
                res[itm][place_dict[itm]] = 1.0
                place_dict[itm] += 1
        target['label_compat_onehot'] = res
        if img is not None:
            return target, img
        else:
            return target


class box_label_catter():
    def __init__(self):
        pass

    def __call__(self, target, img=None):
        labels = target['label_compat']
        boxes = target['boxes']
        box_label = torch.cat((boxes, labels.unsqueeze(-1)), 1)
        target['box_label'] = box_label
        if img is not None:
            return target, img
        else:
            return target


class RandomSelectBoxlabels():
    def __init__(self, num_classes, leave_one_out=False, blank_prob=0.8,
                    prob_first_item = 0.0,
                    prob_random_item = 0.0,
                    prob_last_item = 0.8,
                    prob_stop_sign = 0.2
                ) -> None:
        self.num_classes = num_classes
        self.leave_one_out = leave_one_out
        self.blank_prob = blank_prob

        self.set_state(prob_first_item, prob_random_item, prob_last_item, prob_stop_sign)
        

    def get_state(self):
        return [self.prob_first_item, self.prob_random_item, self.prob_last_item, self.prob_stop_sign]

    def set_state(self, prob_first_item, prob_random_item, prob_last_item, prob_stop_sign):
        sum_prob = prob_first_item + prob_random_item + prob_last_item + prob_stop_sign
        assert sum_prob - 1 < 1e-6, \
            f"Sum up all prob = {sum_prob}. prob_first_item:{prob_first_item}" \
            + f"prob_random_item:{prob_random_item}, prob_last_item:{prob_last_item}" \
            + f"prob_stop_sign:{prob_stop_sign}"

        self.prob_first_item = prob_first_item
        self.prob_random_item = prob_random_item
        self.prob_last_item = prob_last_item
        self.prob_stop_sign = prob_stop_sign
        

    def sample_for_pred_first_item(self, box_label: torch.FloatTensor):
        box_label_known = torch.Tensor(0,5)
        box_label_unknown = box_label
        return box_label_known, box_label_unknown

    def sample_for_pred_random_item(self, box_label: torch.FloatTensor):
        n_select = int(random.random() * box_label.shape[0])
        box_label = box_label[torch.randperm(box_label.shape[0])]
        box_label_known = box_label[:n_select]
        box_label_unknown = box_label[n_select:]
        return box_label_known, box_label_unknown

    def sample_for_pred_last_item(self, box_label: torch.FloatTensor):
        box_label_perm = box_label[torch.randperm(box_label.shape[0])]
        known_label_list = []
        box_label_known = []
        box_label_unknown = []
        for item in box_label_perm:
            label_i = item[4].item()
            if label_i in known_label_list:
                box_label_known.append(item)
            else:
                # first item
                box_label_unknown.append(item)
                known_label_list.append(label_i)
        box_label_known = torch.stack(box_label_known) if len(box_label_known) > 0 else torch.Tensor(0,5)
        box_label_unknown = torch.stack(box_label_unknown) if len(box_label_unknown) > 0 else torch.Tensor(0,5)
        return box_label_known, box_label_unknown

    def sample_for_pred_stop_sign(self, box_label: torch.FloatTensor):
        box_label_unknown = torch.Tensor(0,5)
        box_label_known = box_label
        return box_label_known, box_label_unknown

    def __call__(self, target, img=None):
        box_label = target['box_label'] # K, 5

        dice_number = random.random()

        if dice_number < self.prob_first_item:
            box_label_known, box_label_unknown = self.sample_for_pred_first_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item:
            box_label_known, box_label_unknown = self.sample_for_pred_random_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item + self.prob_last_item:
            box_label_known, box_label_unknown = self.sample_for_pred_last_item(box_label)
        else:
            box_label_known, box_label_unknown = self.sample_for_pred_stop_sign(box_label)

        target['label_onehot_known'] = label2onehot(box_label_known[:,-1], self.num_classes)
        target['label_onehot_unknown'] = label2onehot(box_label_unknown[:, -1], self.num_classes)
        target['box_label_known'] = box_label_known
        target['box_label_unknown'] = box_label_unknown

        return target, img


class RandomDrop():
    def __init__(self, p=0.2) -> None:
        self.p = p

    def __call__(self, target, img=None):
        known_box = target['box_label_known']
        num_known_box = known_box.size(0)
        idxs = torch.rand(num_known_box)
        # indices = torch.randperm(num_known_box)[:int((1-self).p*num_known_box + 0.5 + random.random())]
        target['box_label_known'] = known_box[idxs > self.p]
        return target, img


class BboxPertuber():
    def __init__(self, max_ratio = 0.02, generate_samples = 1000) -> None:
        self.max_ratio = max_ratio
        self.generate_samples = generate_samples
        self.samples = self.generate_pertube_samples()
        self.idx = 0

    def generate_pertube_samples(self):
        import torch
        samples = (torch.rand(self.generate_samples, 5) - 0.5) * 2 * self.max_ratio
        return samples

    def __call__(self, target, img):
        known_box = target['box_label_known'] # Tensor(K,5), K known bbox
        K = known_box.shape[0]
        known_box_pertube = torch.zeros(K, 6) # 4:bbox, 1:prob, 1:label
        if K == 0:
            pass
        else:
            if self.idx + K > self.generate_samples:
                self.idx = 0
            delta = self.samples[self.idx: self.idx + K, :]
            known_box_pertube[:, :4] = known_box[:, :4] + delta[:, :4]
            iou = (torch.diag(box_iou(box_cxcywh_to_xyxy(known_box[:, :4]), box_cxcywh_to_xyxy(known_box_pertube[:, :4]))[0])) * (1 + delta[:, -1])
            known_box_pertube[:, 4].copy_(iou)
            known_box_pertube[:, -1].copy_(known_box[:, -1])

        target['box_label_known_pertube'] = known_box_pertube
        return target, img


class RandomCutout():
    def __init__(self, factor=0.5) -> None:
        self.factor = factor

    def __call__(self, target, img=None):
        unknown_box = target['box_label_unknown']           # Ku, 5
        known_box = target['box_label_known_pertube']       # Kk, 6
        Ku = unknown_box.size(0)

        known_box_add = torch.zeros(Ku, 6) # Ku, 6
        known_box_add[:, :5] = unknown_box
        known_box_add[:, 5].uniform_(0.5, 1) 
        

        known_box_add[:, :2] += known_box_add[:, 2:4] * (torch.rand(Ku, 2) - 0.5) / 2
        known_box_add[:, 2:4] /= 2

        target['box_label_known_pertube'] = torch.cat((known_box, known_box_add))
        return target, img


class RandomSelectBoxes():
    def __init__(self, num_class=80) -> None:
        Warning("This is such a slow function and will be deprecated soon!!!")
        self.num_class = num_class

    def __call__(self, target, img=None):
        boxes = target['boxes']
        labels = target['label_compat']

        # transform to list of tensors
        boxs_list = [[] for i in range(self.num_class)]
        for idx, item in enumerate(boxes):
            label = labels[idx].item()
            boxs_list[label].append(item)
        boxs_list_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in boxs_list]

        # random selection
        box_known = []
        box_unknown = []
        for idx, item in enumerate(boxs_list_tensor):
            ncnt = item.shape[0]
            nselect = int(random.random() * ncnt) # close in both sides, much faster than random.randint

            item = item[torch.randperm(ncnt)]
            # random.shuffle(item)
            box_known.append(item[:nselect])
            box_unknown.append(item[nselect:])

        # box_known_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_known]
        # box_unknown_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_unknown]
        # print('box_unknown_tensor:', box_unknown_tensor)
        target['known_box'] = box_known
        target['unknown_box'] = box_unknown
        return target, img


def label2onehot(label, num_classes):
    """
    label: Tensor(K)
    """
    res = torch.zeros(num_classes)
    for i in label:
        itm = int(i.item())
        res[itm] = 1.0
    return res


class MaskCrop():
    def __init__(self) -> None:
        pass

    def __call__(self, target, img):
        known_box = target['known_box']
        h,w = img.shape[1:] # h,w
        # imgsize = target['orig_size'] # h,w

        scale = torch.Tensor([w, h, w, h])

        # _cnt = 0
        for boxes in known_box:
            if boxes.shape[0] == 0:
                continue
            box_xyxy = box_cxcywh_to_xyxy(boxes) * scale
            for box in box_xyxy:
                x1, y1, x2, y2 = [int(i) for i in box.tolist()]
                img[:, y1:y2, x1:x2] = 0
                # _cnt += 1
        # print("_cnt:", _cnt)
        return target, img


dataset_hook_register = {
    'label2compat': label2compat,
    'label_compat2onehot': label_compat2onehot,
    'box_label_catter': box_label_catter,
    'RandomSelectBoxlabels': RandomSelectBoxlabels,
    'RandomSelectBoxes': RandomSelectBoxes,
    'MaskCrop': MaskCrop,
    'BboxPertuber': BboxPertuber,
}


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, aux_target_hacks=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks

    def change_hack_attr(self, hackclassname, attrkv_dict):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                for k,v in attrkv_dict.items():
                    setattr(item, k, v)

    def get_hack(self, hackclassname):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                return item

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        prompt=[]
        
        img, target,prompt = self.prepare(img, target, prompt)
        if target.get("segmentation") is None or target["segmentation"].item() != 1:
            if self._transforms is not None:
                img, target = self._transforms(img, target)

        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)
        if target.get('segmentation', torch.tensor(0)).item() == 1:
            # img = transforms.ToTensor()(img)
            img, target = self._transforms(img, target)
            prompt= prompt
        
        return img, target, prompt


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, prompt):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        new_anno = []
        anno = target["annotations"]
        for ann in anno:
            if "mask_name" in ann and not "class_name" in ann:
                target = {}
                target["segmentation"] = torch.tensor(1, dtype=torch.long)
                target["mask"] = torch.tensor(ann["image_id"], dtype=torch.long)

                # Read the binary mask image (grayscale mode)
                mask_path = ann["mask_name"]  # Full path to the binary mask image
                binary_mask = Image.open("/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/Segmentation/"+mask_path).convert("L")  # 'L' mode = 8-bit pixels, black and white
                
                # binary_mask = Image.open(
                #     "/home/iml/DINO/coco_data/merge_train/Segmentation_test/dse_data/" 
                #     + os.path.splitext(mask_path)[0] + ".png"
                # ).convert("L")
                
                binary_mask = binary_mask.resize((512, 512), resample=Image.NEAREST)
                binary_mask_tensor = transforms.ToTensor()(binary_mask)
                target["binary_mask"] = binary_mask_tensor
                # target["category_id"] = torch.tensor(34, dtype=torch.long)
                prompt= ann["prompt"]
            elif "class_name" in ann:
                target = {}
                target["classification"] = torch.tensor(2, dtype=torch.long)
                target["category_id"] = torch.tensor(ann["category_id"], dtype=torch.long)

                # Read the binary mask image (grayscale mode)
                # mask_path = ann["mask_name"]  # Full path to the binary mask image
                # binary_mask = Image.open("/home/iml/DINO/coco_data/merge_train/"+mask_path).convert("L")  # 'L' mode = 8-bit pixels, black and white
                # binary_mask = binary_mask.resize((w, h), resample=Image.NEAREST)
                # binary_mask_tensor = transforms.ToTensor()(binary_mask)
                # target["class_name"] = ann["class_name"]
                prompt= ann["class_name"]
                  # or any action if key exists
            elif "masked_comment" in ann:
                target = {}
                target["masked_traning"] = torch.tensor(1, dtype=torch.long)
                target["prompt"] = ann["masked_comment"]
                target["completed_text"] = ann["completed_text"]

                # Read the binary mask image (grayscale mode)
                # mask_path = ann["mask_name"]  # Full path to the binary mask image
                # binary_mask = Image.open("/home/iml/DINO/coco_data/merge_train/"+mask_path).convert("L")  # 'L' mode = 8-bit pixels, black and white
                # binary_mask = binary_mask.resize((w, h), resample=Image.NEAREST)
                # binary_mask_tensor = transforms.ToTensor()(binary_mask)
                # target["class_name"] = ann["class_name"]
                prompt= ann["masked_comment"]
                prompt= "mask: "+ prompt
                  # or any action if key exists
            elif "Question" in ann:
                target = {}
                target["masked_traning"] = torch.tensor(1, dtype=torch.long)
                target["prompt"] = ann["Question"]
                target["completed_text"] = ann["Answer"]

                # Read the binary mask image (grayscale mode)
                # mask_path = ann["mask_name"]  # Full path to the binary mask image
                # binary_mask = Image.open("/home/iml/DINO/coco_data/merge_train/"+mask_path).convert("L")  # 'L' mode = 8-bit pixels, black and white
                # binary_mask = binary_mask.resize((w, h), resample=Image.NEAREST)
                # binary_mask_tensor = transforms.ToTensor()(binary_mask)
                # target["class_name"] = ann["class_name"]
                prompt= ann["Question"]
                  # or any action if key exists
            else:
                new_anno.append(ann)
        if new_anno:
            anno = [obj for obj in new_anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
            

            boxes = [obj["bbox"] for obj in new_anno]
            # guard against no boxes via resizing
            if not boxes:
                print("Boxes empty")
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

            classes = [obj["category_id"] for obj in new_anno]
            classes = torch.tensor(classes, dtype=torch.int64)
            prompt = [obj["prompt"] for obj in new_anno]
            # prompt = [obj["category_name"] for obj in new_anno]
            prompt = list(dict.fromkeys(prompt))
            
            morphology_attributes = []
            for obj in new_anno:
                # Check if 'nuclear_shape' exists in the current object
                if "nuclear_shape" in obj and obj["nuclear_shape"] is not None:
                    attributes = [
                        obj.get("nuclear_chromatio", 0),  # Replace with a default value if the key is missing
                        obj["nuclear_shape"],
                        obj.get("nucleolus", 0),
                        obj.get("cytoplasm", 0),
                        obj.get("cytoplasmic_basophilia", 0),
                        obj.get("cytoplasmic_vacuoles", 0)
                    ]
                else:
                    # If 'nuclear_shape' is not present, use an empty tensor as the attribute
                    attributes = torch.full((1, 6), 4, dtype=torch.long)
                    attributes = attributes.squeeze(0) 

                # Convert the attributes to a tensor and append to the list
                attributes = torch.tensor(attributes, dtype=torch.long)
                morphology_attributes.append(attributes)

            # Stack the list of attribute tensors into a single tensor
            if len(morphology_attributes) > 0:
                morphology_attributes = torch.stack(morphology_attributes)
            else:
                print("Warning: morphology_attributes is empty!")
                # morphology_attributes = torch.tensor([]) 

            if self.return_masks:
                segmentations = [obj["segmentation"] for obj in new_anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)

            keypoints = None
            if new_anno and "keypoints" in new_anno[0]:
                keypoints = [obj["keypoints"] for obj in new_anno]
                keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
                num_keypoints = keypoints.shape[0]
                if num_keypoints:
                    keypoints = keypoints.view(num_keypoints, -1, 3)

            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]
            # prompt=prompt[keep]
            if self.return_masks:
                masks = masks[keep]
            if keypoints is not None:
                keypoints = keypoints[keep]



            morphology_attributes = morphology_attributes[keep]


            target = {}
            target["boxes"] = boxes
            target["labels"] = classes
            if self.return_masks:
                target["masks"] = masks
            target["image_id"] = image_id
            if keypoints is not None:
                target["keypoints"] = keypoints
                
            
            target["morphology"] = morphology_attributes

            # for conversion to coco api
            area = torch.tensor([obj["area"] for obj in new_anno])
            iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in new_anno])
            target["area"] = area[keep]
            target["iscrowd"] = iscrowd[keep]

            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target, prompt


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    scales3_crop = [184, 200]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))
    # if image_set == 'train_4':
    #     return T.Compose([  # Uses your new fixed Resize class
    #         normalize,
    #     ])

    if image_set == 'train_4':
            # return T.Compose([
            #     T.RandomHorizontalFlip(),
            #     # T.RandomResize([(max_size, max(scales))]),
            #     T.RandomResize([(256, 256)]),
            #     normalize,
            # ])
            import dino_datasets.sltransform as SLT
            return T.Compose([
                T.RandomHorizontalFlip(),
                # T.RandomResize([(max_size, max(scales))]),
                T.RandomResize([(256, 256)]),
                # normalize,
                SLT.RandomSelectMulti([
                    # SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                    # SLT.AlbumentationsClassification()
                ]),
                normalize,
            ])
    elif image_set == 'train_3': 
            import dino_datasets.sltransform as SLT
            return T.Compose([
                T.RandomHorizontalFlip(),
                # T.RandomResize([(max_size, max(scales))]),
                T.RandomResize([(512, 512)]),
                # normalize,
                SLT.RandomSelectMulti([
                    # SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                    # SLT.AlbumentationsClassification()
                ]),
                normalize,
            ])
        # return T.Compose([
        #     T.Resize((512, 512)),  # Uses your new fixed Resize class
        #     normalize,
        # ])

    elif image_set == 'train_2':
            return T.Compose([
                # T.RandomHorizontalFlip(),
                # T.RandomResize([(max_size, max(scales))]),
                T.RandomResize([(512, 512)]),
                normalize,
            ])
            # import dino_datasets.sltransform as SLT
            # return T.Compose([
            #     T.RandomHorizontalFlip(),
            #     # T.RandomResize([(max_size, max(scales))]),
            #     T.RandomResize([(256, 256)]),
            #     # normalize,
            #     SLT.RandomSelectMulti([
            #         # SLT.RandomCrop(),
            #         SLT.LightingNoise(),
            #         SLT.AdjustBrightness(2),
            #         SLT.AdjustContrast(2),
            #         # SLT.AlbumentationsClassification()
            #     ]),
            #     normalize,
            # ])
    elif image_set in ['train']:
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                # T.RandomResize([(448, 448)]),
                normalize,
            ])
    

        if strong_aug:
            import dino_datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])
    valid_sets = {'val', 'eval_debug', 'train_reg', 'test'}

    if args.eval_type == "det" and image_set in valid_sets:
        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])
        
    # for classification 
    if args.eval_type == "cls" and image_set in valid_sets:
        return T.Compose([
           T.RandomResize([(512, 512)]),  # Resize to fixed size
            normalize,
        ])  
    

    if args.eval_type == "text" and image_set in valid_sets:
        return T.Compose([
             T.Resize((256,256)),  # Uses your new fixed Resize class
            normalize,
        ])
    if args.eval_type == "seg" and image_set in valid_sets:
        return T.Compose([
            T.Resize((512, 512)),  # Uses your new fixed Resize class
            normalize,
        ])


    # raise ValueError(f'unknown {image_set}')


def get_aux_target_hacks_list(image_set, args):
    if args.modelname in ['q2bs_mask', 'q2bs']:
        aux_target_hacks_list = [
            label2compat(), 
            label_compat2onehot(), 
            RandomSelectBoxes(num_class=args.num_classes)
        ]
        if args.masked_data and image_set == 'train':
            # aux_target_hacks_list.append()
            aux_target_hacks_list.append(MaskCrop())
    elif args.modelname in ['q2bm_v2', 'q2bs_ce', 'q2op', 'q2ofocal', 'q2opclip', 'q2ocqonly']:
        aux_target_hacks_list = [
            label2compat(),
            label_compat2onehot(),
            box_label_catter(),
            RandomSelectBoxlabels(num_classes=args.num_classes,
                                    prob_first_item=args.prob_first_item,
                                    prob_random_item=args.prob_random_item,
                                    prob_last_item=args.prob_last_item,
                                    prob_stop_sign=args.prob_stop_sign,
                                    ),
            BboxPertuber(max_ratio=0.02, generate_samples=1000),
        ]
    elif args.modelname in ['q2omask', 'q2osa']:
        if args.coco_aug:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                        prob_first_item=args.prob_first_item,
                                        prob_random_item=args.prob_random_item,
                                        prob_last_item=args.prob_last_item,
                                        prob_stop_sign=args.prob_stop_sign,
                                        ),
                RandomDrop(p=0.2),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
                RandomCutout(factor=0.5)
            ]
        else:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                        prob_first_item=args.prob_first_item,
                                        prob_random_item=args.prob_random_item,
                                        prob_last_item=args.prob_last_item,
                                        prob_stop_sign=args.prob_stop_sign,
                                        ),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
            ]
    else:
        aux_target_hacks_list = None

    return aux_target_hacks_list


def build(image_set, args):
    root = Path(args.coco_path)
    mode = 'instances'
    PATHS = {
        #Segmentation traning  (coco_data/annotations/segmentation/segmentation_train_v5_elsify_20.json)
        
        "train_2": (root / "merge_train/Segmentation", root / "annotations/segmentation" / f'segmentation_train_v5_elsify_20.json'),
        
       # detetion Traning 
       
        "train": (root / "merge_train/Detection", root / "annotations" / f'detection/detetction_complet_v5.json'),
        # "train":("/media/iml/Abdul_2/raabin_WBC/raabin_label1_m1/images/", '/media/iml/Abdul_2/raabin_WBC/raabin_label1_m1/microscope_1_train.json'), #
        
        ##Classification
        "train_3":(root / "merge_train/classification", root /"annotations/classification/11_9_3_4_5_6_7_12_13_14_8-t_v6-krdc-bccd.json"),
        
        ## Text data 
        "train_4":(root / "merge_train/Detection", root /"annotations/captions/Final_version/Q_A_val_train_MLM_train.json"),
        
        ## language val 
        
        # "val": (root / "merge_train/Detection", root / "annotations/captions/Final_version/Mask_test_fixed.json"), #MLM
        # "val": (root / "merge_train/Detection", root / "annotations/captions/Final_version/Q_A_test_small.json"), #Leukemia
        # "val": (root / "merge_train/Detection", root / "annotations/captions/Final_version/Q_A_test_large.json"), #Leukemia
        # 
        # "train": (root / "merge_train/Detection", root / "annotations" / f'detection/filtered_annotations_reset.json'),
        # "train": (root / "merge_train", root / "annotations" / f'merged_file_with_leuk_m5-high-low_para_plate_sickle.json'), #hematology 5 classes
        # "train": (root / "train/", root /"train/" f'coco_p4.json'), #FLIR FLIR_Data/images_thermal_train
        # "train": (root / "images_thermal_train/", root /"images_thermal_train/" f'coco_p4.json'), #thermal 
        # "val":(root / "images_rgb_val", root / f"images_rgb_val" / f'coco_p3.json'),   # FLIR_spatial
        # "val":(root / "images_thermal_val", root / f"images_thermal_val" / f'coco_p3.json'),   # FLIR_Thermalqw
        
        # "train_reg": (root / "train2017", root / "annotations" / f'{mode}_p_train2017.json'),
        
        
        
        # "train_3":(root / "merge_train/classification", root /"annotations/classification/" f'raabin_Train_update.json'),
        # "train_3":(root / "merge_train/", root /"annotations/classification/" f'test.json'), #FLIR FLIR_Data/images_thermal_traine
        # "val":(root / "merge_train/", root /"annotations/classification/" f'AML_Metak_test.json'), #FLIR FLIR_Data/images_thermal_traine
        # "val":(root / "merge_train/classification", root /"annotations/classification/" f'Raabin_testA_updated.json'),#/home/iml/DINO/coco_data/annotations/classification/test_Acevedo_update_20.json
        # "val":(root / "merge_train/classification", root /"annotations/classification/" f'/test_Acevedo_update_20.json'),#
        # "train":(root / "merge_train/", root /"annotations/captions/" f'HCM_100x_c2_train_text_train80.json'), #/home/iml/DINO/coco_data/annotations/captions/HCM_100x_c2_train_text_only_masked.json
        # "val":(root / "merge_train/", root /"annotations/captions/" f'HCM_100x_c2_train_text_val20.json'), 
       
       ##Segmentation val"
     
        "val":(root / "merge_train/Segmentation_test/", root /"annotations/segmentation_test/" f'Malaria-Detection-2019_test_data.json'), #FLIR FLIR_Data/images_thermal_train
        # "val":(root / "merge_train/Segmentation/", root /"annotations/segmentation_test/" f'AneRBC-II_Anemic_individuals_test.json'),  
        # "val":(root / "merge_train/Segmentation/", root /"annotations/segmentation_test/" f'AneRBC-II_Healthy_individuals_test.json'),  
        # "val":(root / "merge_train/Segmentation/", root /"annotations/segmentation_test/" f'Elsafty_RBCs_Cellular_Images_and_Masks_test.json'), 
    #    "val":(root / "merge_train/Segmentation/", root /"annotations/segmentation_test/" f'KRD_test_data.json'), 
        # "val":(root / "merge_train/Segmentation/", root /"annotations/segmentation_test/" f'test.json'), 
        # "val":(root / "merge_train/Segmentation/avecoda_seg/", root /"annotations/segmentation_test/" f'ava_test_data.json'), 
      
        
       
       # Detetion Dataset Valadation
        #Leukemia 4
        # "val": (root / "merge_train/Detection/HCM_100x_c2_test", root / "annotations/Detection_test" / f'{mode}_p_val2017_22_update.json'), #Leukemia
        # "val": (root / "merge_train/Detection", root / "annotations/leukemia/" / f'hcm_100x_c1_test_update.json'), #Leukemia h_c1_100x
        # "val": (root / "merge_train/Detection/", root / "annotations/leukemia" / f'hcm_40x_c2_m_update.json'), #Leukemia h_40x_c2
        # "val": (root / "merge_train/Detection/", root / "annotations/leukemia" / f'h_40x_c1_test_update.json'), #Leukemia h_40x_c1
        # "val": (root / "merge_train/Detection/", root / "annotations/leukemia" / f'l_100x_c2_test_update.json'), #Leukemia l_100x_c2
        # "val": (root / "merge_train/Detection/", root / "annotations/Detection_test" / f'l_100x_c1_test.json'), #Leukemia l_100x_c1
        # "val": (root / "merge_train/Detection/", root / "annotations/leukemia" / f'lcm_40x_c2_m_update.json'), #Leukemia l_40x_c2hcm_40x_c1
        # "val": (root / "merge_train/Detection/", root / "annotations/leukemia" / f'l_40x_c1_test_update.j88.3s97.7o71.2n95.2'90.6), #Leukemia l_40x_c1
        
        #M5 
        # "val":(root / "M5_coco/test/", root / "annotations/Detection_test" / f'hcm_test_p_1000x_22.json'), #M5 hcm /home/iml/DINO/coco_data/annotations/Malariae_test_p.json
        # "val":(root / "M5_coco/test/", root /'M5_coco/annotations/hcm_test_400x_v2.json'), #M5 hcm /home/iml/DINO/coco_data/annotations/Malariae_test_p.json
        # "val":(root / "M5_coco/test/",  root /'M5_coco/annotations/lcm_test_1000x_v2.json'),
        # "val":(root / "M5_coco/test/", root /'M5_coco/annotations/lcm_test_400x_v2.json'),
    
        
        #MP-IDB-The-Malaria-Parasite- (UNSEEN)
        # "val":(root / "merge_train/Detection", root / "annotations/Detection_test" / f'Malariae_test_p.json'), #Malariae 
        
        
        # Sickle cell 
        # "val":(root / "merge_train/Detection/positive_sickle_test", root / "annotations/Detection_test" / f'sickle_cell_test_2.json'), #scikle ce
        
        
        # leishman parasite detection 
        # "val":(root / "merge_train/Detection/parasite_detection_test", root / "annotations/Detection_test" / f'parasite_detection_test_las.json'), #parasites
        
        #Platelet
        # "val":(root / "merge_train/Detection/TXL_val", root / "annotations/Detection_test" / f'CRC_val_22.json'),   # platelet TXL
        
        # CBC 
        # "val":(root/"annotations/detection/baseline_results_detetion/test_images", root/'annotations/detection/baseline_results_detetion/test_annotation/unified/BCCD_test_v2.json'), #parasites
        # "val":(root/"annotations/detection/baseline_results_detetion/test_images", root/'annotations/detection/baseline_results_detetion/test_annotation/unified/TXL_test_v2.json'), #
        # "val":("/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_images", '/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_annotation/unified/Boi_net_test_v2_unssen.json'), #
        # "val":("/media/iml/Abdul_2/raabin_WBC/raabin_label1_m1/images/", '/media/iml/Abdul_2/raabin_WBC/raabin_label1_m1/microscope_1_test.json'), #
        # "train": (root / "merge_train", root / "annotations" / f'parasite_detection_train_p.json'),
        # "val":(root / "parasite_detetion_test", root / "annotations" / f'parasite_detection_test.json'), #parasites
        
         # "train": (root / "merge_train", root / "annotations" / f'CRC_train3.json'),#/home/iml/DINO/coco_data/annotations/orignal labels/CRC_train2.json
        # "val":(root / "CRC_val", root / "annotations" / f'CRC_val_1.json'),   # platelet 
        
        
        # "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_p_val2017.json'),
        # "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' ),
        # "eval_debug": (root / "test", root / "annotations" / f'hcm_test_p_1000x.json'),
        # "test": (root / "test", root / "annotations" / 'hcm_test_p_1000x.json' ),Pl
        
        # Fine_tune
        #  "train": ("/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/train_data", '/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/train_annotation/BCCD_train.json'),#/home/iml/DINO/coco_data/annotations/orignal labels/CRC_train2.json
        # "val":("/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_images", '/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_annotation/unified/BCCD_test_v2.json'),   # platelet 

        #  "train": ("/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/train_data", '/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/train_annotation/Boi_net_train.json'),
        # "val":("/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_images", '/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_annotation/unified/Boi_net_test_v2_unssen.json'),   # platelet 
    }

    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]
    print(img_folder)

    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    dataset = CocoDetection(img_folder, ann_file,
            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args), 
            return_masks=args.masks,
            aux_target_hacks=aux_target_hacks_list,
        )

    return dataset



if __name__ == "__main__":
    # Objects365 Val example
    dataset_o365 = CocoDetection(
            '/path/Objects365/train/',
            "/path/Objects365/slannos/anno_preprocess_train_v2.json",
            transforms=None,
            return_masks=False,
        )
    print('len(dataset_o365):', len(dataset_o365))




# class LearnableUpsample(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, output_size=(384, 384)):
#         super().__init__()
#         self.output_size = output_size
        
#         # This kernel & stride combination roughly scales 56 → 384
#         # 384 / 56 ≈ 6.857, so stride ~7
#         # Adjust padding and kernel_size for exact output
#         self.upsample = nn.Sequential(
#             # 56 → 112
#             nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),

#             # 112 → 224
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             # 224 → 384
#             nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = x.unsqueeze(1) 
#         out = self.upsample(x)
#         # Ensure exact output size
#         # out = F.interpolate(out, size=self.output_size, mode='bilinear', align_corners=False)
#         return out.squeeze(1) 




=======
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
if __name__=="__main__":
    # for debug only
    import os, sys
    sys.path.append(os.path.dirname(sys.path[0]))

import json
from pathlib import Path
import random
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from torchvision import transforms
from dino_datasets.data_util import preparing_dataset
import dino_datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy, box_iou
import dino_datasets.sltransform as SLT
__all__ = ['build']


class label2compat():
    def __init__(self) -> None:
        self.category_map_str = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20} # "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
        self.category_map = {int(k):v for k,v in self.category_map_str.items()}

    def __call__(self, target, img=None):
        labels = target['labels']
        res = torch.zeros(labels.shape, dtype=labels.dtype)
        for idx, item in enumerate(labels):
            res[idx] = self.category_map[item.item()] - 1
        target['label_compat'] = res
        if img is not None:
            return target, img
        else:
            return target


class label_compat2onehot():
    def __init__(self, num_class=80, num_output_objs=1):
        self.num_class = num_class
        self.num_output_objs = num_output_objs
        if num_output_objs != 1:
            raise DeprecationWarning("num_output_objs!=1, which is only used for comparison")

    def __call__(self, target, img=None):
        labels = target['label_compat']
        place_dict = {k:0 for k in range(self.num_class)}
        if self.num_output_objs == 1:
            res = torch.zeros(self.num_class)
            for i in labels:
                itm = i.item()
                res[itm] = 1.0
        else:
            # compat with baseline
            res = torch.zeros(self.num_class, self.num_output_objs)
            for i in labels:
                itm = i.item()
                res[itm][place_dict[itm]] = 1.0
                place_dict[itm] += 1
        target['label_compat_onehot'] = res
        if img is not None:
            return target, img
        else:
            return target


class box_label_catter():
    def __init__(self):
        pass

    def __call__(self, target, img=None):
        labels = target['label_compat']
        boxes = target['boxes']
        box_label = torch.cat((boxes, labels.unsqueeze(-1)), 1)
        target['box_label'] = box_label
        if img is not None:
            return target, img
        else:
            return target


class RandomSelectBoxlabels():
    def __init__(self, num_classes, leave_one_out=False, blank_prob=0.8,
                    prob_first_item = 0.0,
                    prob_random_item = 0.0,
                    prob_last_item = 0.8,
                    prob_stop_sign = 0.2
                ) -> None:
        self.num_classes = num_classes
        self.leave_one_out = leave_one_out
        self.blank_prob = blank_prob

        self.set_state(prob_first_item, prob_random_item, prob_last_item, prob_stop_sign)
        

    def get_state(self):
        return [self.prob_first_item, self.prob_random_item, self.prob_last_item, self.prob_stop_sign]

    def set_state(self, prob_first_item, prob_random_item, prob_last_item, prob_stop_sign):
        sum_prob = prob_first_item + prob_random_item + prob_last_item + prob_stop_sign
        assert sum_prob - 1 < 1e-6, \
            f"Sum up all prob = {sum_prob}. prob_first_item:{prob_first_item}" \
            + f"prob_random_item:{prob_random_item}, prob_last_item:{prob_last_item}" \
            + f"prob_stop_sign:{prob_stop_sign}"

        self.prob_first_item = prob_first_item
        self.prob_random_item = prob_random_item
        self.prob_last_item = prob_last_item
        self.prob_stop_sign = prob_stop_sign
        

    def sample_for_pred_first_item(self, box_label: torch.FloatTensor):
        box_label_known = torch.Tensor(0,5)
        box_label_unknown = box_label
        return box_label_known, box_label_unknown

    def sample_for_pred_random_item(self, box_label: torch.FloatTensor):
        n_select = int(random.random() * box_label.shape[0])
        box_label = box_label[torch.randperm(box_label.shape[0])]
        box_label_known = box_label[:n_select]
        box_label_unknown = box_label[n_select:]
        return box_label_known, box_label_unknown

    def sample_for_pred_last_item(self, box_label: torch.FloatTensor):
        box_label_perm = box_label[torch.randperm(box_label.shape[0])]
        known_label_list = []
        box_label_known = []
        box_label_unknown = []
        for item in box_label_perm:
            label_i = item[4].item()
            if label_i in known_label_list:
                box_label_known.append(item)
            else:
                # first item
                box_label_unknown.append(item)
                known_label_list.append(label_i)
        box_label_known = torch.stack(box_label_known) if len(box_label_known) > 0 else torch.Tensor(0,5)
        box_label_unknown = torch.stack(box_label_unknown) if len(box_label_unknown) > 0 else torch.Tensor(0,5)
        return box_label_known, box_label_unknown

    def sample_for_pred_stop_sign(self, box_label: torch.FloatTensor):
        box_label_unknown = torch.Tensor(0,5)
        box_label_known = box_label
        return box_label_known, box_label_unknown

    def __call__(self, target, img=None):
        box_label = target['box_label'] # K, 5

        dice_number = random.random()

        if dice_number < self.prob_first_item:
            box_label_known, box_label_unknown = self.sample_for_pred_first_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item:
            box_label_known, box_label_unknown = self.sample_for_pred_random_item(box_label)
        elif dice_number < self.prob_first_item + self.prob_random_item + self.prob_last_item:
            box_label_known, box_label_unknown = self.sample_for_pred_last_item(box_label)
        else:
            box_label_known, box_label_unknown = self.sample_for_pred_stop_sign(box_label)

        target['label_onehot_known'] = label2onehot(box_label_known[:,-1], self.num_classes)
        target['label_onehot_unknown'] = label2onehot(box_label_unknown[:, -1], self.num_classes)
        target['box_label_known'] = box_label_known
        target['box_label_unknown'] = box_label_unknown

        return target, img


class RandomDrop():
    def __init__(self, p=0.2) -> None:
        self.p = p

    def __call__(self, target, img=None):
        known_box = target['box_label_known']
        num_known_box = known_box.size(0)
        idxs = torch.rand(num_known_box)
        # indices = torch.randperm(num_known_box)[:int((1-self).p*num_known_box + 0.5 + random.random())]
        target['box_label_known'] = known_box[idxs > self.p]
        return target, img


class BboxPertuber():
    def __init__(self, max_ratio = 0.02, generate_samples = 1000) -> None:
        self.max_ratio = max_ratio
        self.generate_samples = generate_samples
        self.samples = self.generate_pertube_samples()
        self.idx = 0

    def generate_pertube_samples(self):
        import torch
        samples = (torch.rand(self.generate_samples, 5) - 0.5) * 2 * self.max_ratio
        return samples

    def __call__(self, target, img):
        known_box = target['box_label_known'] # Tensor(K,5), K known bbox
        K = known_box.shape[0]
        known_box_pertube = torch.zeros(K, 6) # 4:bbox, 1:prob, 1:label
        if K == 0:
            pass
        else:
            if self.idx + K > self.generate_samples:
                self.idx = 0
            delta = self.samples[self.idx: self.idx + K, :]
            known_box_pertube[:, :4] = known_box[:, :4] + delta[:, :4]
            iou = (torch.diag(box_iou(box_cxcywh_to_xyxy(known_box[:, :4]), box_cxcywh_to_xyxy(known_box_pertube[:, :4]))[0])) * (1 + delta[:, -1])
            known_box_pertube[:, 4].copy_(iou)
            known_box_pertube[:, -1].copy_(known_box[:, -1])

        target['box_label_known_pertube'] = known_box_pertube
        return target, img


class RandomCutout():
    def __init__(self, factor=0.5) -> None:
        self.factor = factor

    def __call__(self, target, img=None):
        unknown_box = target['box_label_unknown']           # Ku, 5
        known_box = target['box_label_known_pertube']       # Kk, 6
        Ku = unknown_box.size(0)

        known_box_add = torch.zeros(Ku, 6) # Ku, 6
        known_box_add[:, :5] = unknown_box
        known_box_add[:, 5].uniform_(0.5, 1) 
        

        known_box_add[:, :2] += known_box_add[:, 2:4] * (torch.rand(Ku, 2) - 0.5) / 2
        known_box_add[:, 2:4] /= 2

        target['box_label_known_pertube'] = torch.cat((known_box, known_box_add))
        return target, img


class RandomSelectBoxes():
    def __init__(self, num_class=80) -> None:
        Warning("This is such a slow function and will be deprecated soon!!!")
        self.num_class = num_class

    def __call__(self, target, img=None):
        boxes = target['boxes']
        labels = target['label_compat']

        # transform to list of tensors
        boxs_list = [[] for i in range(self.num_class)]
        for idx, item in enumerate(boxes):
            label = labels[idx].item()
            boxs_list[label].append(item)
        boxs_list_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in boxs_list]

        # random selection
        box_known = []
        box_unknown = []
        for idx, item in enumerate(boxs_list_tensor):
            ncnt = item.shape[0]
            nselect = int(random.random() * ncnt) # close in both sides, much faster than random.randint

            item = item[torch.randperm(ncnt)]
            # random.shuffle(item)
            box_known.append(item[:nselect])
            box_unknown.append(item[nselect:])

        # box_known_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_known]
        # box_unknown_tensor = [torch.stack(i) if len(i) > 0 else torch.Tensor(0,4) for i in box_unknown]
        # print('box_unknown_tensor:', box_unknown_tensor)
        target['known_box'] = box_known
        target['unknown_box'] = box_unknown
        return target, img


def label2onehot(label, num_classes):
    """
    label: Tensor(K)
    """
    res = torch.zeros(num_classes)
    for i in label:
        itm = int(i.item())
        res[itm] = 1.0
    return res


class MaskCrop():
    def __init__(self) -> None:
        pass

    def __call__(self, target, img):
        known_box = target['known_box']
        h,w = img.shape[1:] # h,w
        # imgsize = target['orig_size'] # h,w

        scale = torch.Tensor([w, h, w, h])

        # _cnt = 0
        for boxes in known_box:
            if boxes.shape[0] == 0:
                continue
            box_xyxy = box_cxcywh_to_xyxy(boxes) * scale
            for box in box_xyxy:
                x1, y1, x2, y2 = [int(i) for i in box.tolist()]
                img[:, y1:y2, x1:x2] = 0
                # _cnt += 1
        # print("_cnt:", _cnt)
        return target, img


dataset_hook_register = {
    'label2compat': label2compat,
    'label_compat2onehot': label_compat2onehot,
    'box_label_catter': box_label_catter,
    'RandomSelectBoxlabels': RandomSelectBoxlabels,
    'RandomSelectBoxes': RandomSelectBoxes,
    'MaskCrop': MaskCrop,
    'BboxPertuber': BboxPertuber,
}


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, aux_target_hacks=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.aux_target_hacks = aux_target_hacks

    def change_hack_attr(self, hackclassname, attrkv_dict):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                for k,v in attrkv_dict.items():
                    setattr(item, k, v)

    def get_hack(self, hackclassname):
        target_class = dataset_hook_register[hackclassname]
        for item in self.aux_target_hacks:
            if isinstance(item, target_class):
                return item

    def __getitem__(self, idx):
        """
        Output:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 4]. \
                    Init type: x0,y0,x1,y1. unnormalized data.
                    Final type: cx,cy,w,h. normalized data. 
        """
        try:
            img, target = super(CocoDetection, self).__getitem__(idx)
        except:
            print("Error idx: {}".format(idx))
            idx += 1
            img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        prompt=[]
        
        img, target,prompt = self.prepare(img, target, prompt)
        if target.get("segmentation") is None or target["segmentation"].item() != 1:
            if self._transforms is not None:
                img, target = self._transforms(img, target)

        # convert to needed format
        if self.aux_target_hacks is not None:
            for hack_runner in self.aux_target_hacks:
                target, img = hack_runner(target, img=img)
        if target.get('segmentation', torch.tensor(0)).item() == 1:
            # img = transforms.ToTensor()(img)
            img, target = self._transforms(img, target)
            prompt= prompt
        
        return img, target, prompt


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, prompt):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        new_anno = []
        anno = target["annotations"]
        for ann in anno:
            if "mask_name" in ann and not "class_name" in ann:
                target = {}
                target["segmentation"] = torch.tensor(1, dtype=torch.long)
                target["mask"] = torch.tensor(ann["image_id"], dtype=torch.long)

                # Read the binary mask image (grayscale mode)
                mask_path = ann["mask_name"]  # Full path to the binary mask image
                binary_mask = Image.open("/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/Segmentation/"+mask_path).convert("L")  # 'L' mode = 8-bit pixels, black and white
                
                # binary_mask = Image.open(
                #     "/home/iml/DINO/coco_data/merge_train/Segmentation_test/dse_data/" 
                #     + os.path.splitext(mask_path)[0] + ".png"
                # ).convert("L")
                
                binary_mask = binary_mask.resize((512, 512), resample=Image.NEAREST)
                binary_mask_tensor = transforms.ToTensor()(binary_mask)
                target["binary_mask"] = binary_mask_tensor
                # target["category_id"] = torch.tensor(34, dtype=torch.long)
                prompt= ann["prompt"]
            elif "class_name" in ann:
                target = {}
                target["classification"] = torch.tensor(2, dtype=torch.long)
                target["category_id"] = torch.tensor(ann["category_id"], dtype=torch.long)

                # Read the binary mask image (grayscale mode)
                # mask_path = ann["mask_name"]  # Full path to the binary mask image
                # binary_mask = Image.open("/home/iml/DINO/coco_data/merge_train/"+mask_path).convert("L")  # 'L' mode = 8-bit pixels, black and white
                # binary_mask = binary_mask.resize((w, h), resample=Image.NEAREST)
                # binary_mask_tensor = transforms.ToTensor()(binary_mask)
                # target["class_name"] = ann["class_name"]
                prompt= ann["class_name"]
                  # or any action if key exists
            elif "masked_comment" in ann:
                target = {}
                target["masked_traning"] = torch.tensor(1, dtype=torch.long)
                target["prompt"] = ann["masked_comment"]
                target["completed_text"] = ann["completed_text"]

                # Read the binary mask image (grayscale mode)
                # mask_path = ann["mask_name"]  # Full path to the binary mask image
                # binary_mask = Image.open("/home/iml/DINO/coco_data/merge_train/"+mask_path).convert("L")  # 'L' mode = 8-bit pixels, black and white
                # binary_mask = binary_mask.resize((w, h), resample=Image.NEAREST)
                # binary_mask_tensor = transforms.ToTensor()(binary_mask)
                # target["class_name"] = ann["class_name"]
                prompt= ann["masked_comment"]
                prompt= "mask: "+ prompt
                  # or any action if key exists
            elif "Question" in ann:
                target = {}
                target["masked_traning"] = torch.tensor(1, dtype=torch.long)
                target["prompt"] = ann["Question"]
                target["completed_text"] = ann["Answer"]

                # Read the binary mask image (grayscale mode)
                # mask_path = ann["mask_name"]  # Full path to the binary mask image
                # binary_mask = Image.open("/home/iml/DINO/coco_data/merge_train/"+mask_path).convert("L")  # 'L' mode = 8-bit pixels, black and white
                # binary_mask = binary_mask.resize((w, h), resample=Image.NEAREST)
                # binary_mask_tensor = transforms.ToTensor()(binary_mask)
                # target["class_name"] = ann["class_name"]
                prompt= ann["Question"]
                  # or any action if key exists
            else:
                new_anno.append(ann)
        if new_anno:
            anno = [obj for obj in new_anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
            

            boxes = [obj["bbox"] for obj in new_anno]
            # guard against no boxes via resizing
            if not boxes:
                print("Boxes empty")
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

            classes = [obj["category_id"] for obj in new_anno]
            classes = torch.tensor(classes, dtype=torch.int64)
            prompt = [obj["prompt"] for obj in new_anno]
            # prompt = [obj["category_name"] for obj in new_anno]
            prompt = list(dict.fromkeys(prompt))
            
            morphology_attributes = []
            for obj in new_anno:
                # Check if 'nuclear_shape' exists in the current object
                if "nuclear_shape" in obj and obj["nuclear_shape"] is not None:
                    attributes = [
                        obj.get("nuclear_chromatio", 0),  # Replace with a default value if the key is missing
                        obj["nuclear_shape"],
                        obj.get("nucleolus", 0),
                        obj.get("cytoplasm", 0),
                        obj.get("cytoplasmic_basophilia", 0),
                        obj.get("cytoplasmic_vacuoles", 0)
                    ]
                else:
                    # If 'nuclear_shape' is not present, use an empty tensor as the attribute
                    attributes = torch.full((1, 6), 4, dtype=torch.long)
                    attributes = attributes.squeeze(0) 

                # Convert the attributes to a tensor and append to the list
                attributes = torch.tensor(attributes, dtype=torch.long)
                morphology_attributes.append(attributes)

            # Stack the list of attribute tensors into a single tensor
            if len(morphology_attributes) > 0:
                morphology_attributes = torch.stack(morphology_attributes)
            else:
                print("Warning: morphology_attributes is empty!")
                # morphology_attributes = torch.tensor([]) 

            if self.return_masks:
                segmentations = [obj["segmentation"] for obj in new_anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)

            keypoints = None
            if new_anno and "keypoints" in new_anno[0]:
                keypoints = [obj["keypoints"] for obj in new_anno]
                keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
                num_keypoints = keypoints.shape[0]
                if num_keypoints:
                    keypoints = keypoints.view(num_keypoints, -1, 3)

            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]
            # prompt=prompt[keep]
            if self.return_masks:
                masks = masks[keep]
            if keypoints is not None:
                keypoints = keypoints[keep]



            morphology_attributes = morphology_attributes[keep]


            target = {}
            target["boxes"] = boxes
            target["labels"] = classes
            if self.return_masks:
                target["masks"] = masks
            target["image_id"] = image_id
            if keypoints is not None:
                target["keypoints"] = keypoints
                
            
            target["morphology"] = morphology_attributes

            # for conversion to coco api
            area = torch.tensor([obj["area"] for obj in new_anno])
            iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in new_anno])
            target["area"] = area[keep]
            target["iscrowd"] = iscrowd[keep]

            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target, prompt


def make_coco_transforms(image_set, fix_size=False, strong_aug=False, args=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # config the params for data aug
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    max_size = 1333
    scales2_resize = [400, 500, 600]
    scales2_crop = [384, 600]
    scales3_crop = [184, 200]
    
    # update args from config files
    scales = getattr(args, 'data_aug_scales', scales)
    max_size = getattr(args, 'data_aug_max_size', max_size)
    scales2_resize = getattr(args, 'data_aug_scales2_resize', scales2_resize)
    scales2_crop = getattr(args, 'data_aug_scales2_crop', scales2_crop)

    # resize them
    data_aug_scale_overlap = getattr(args, 'data_aug_scale_overlap', None)
    if data_aug_scale_overlap is not None and data_aug_scale_overlap > 0:
        data_aug_scale_overlap = float(data_aug_scale_overlap)
        scales = [int(i*data_aug_scale_overlap) for i in scales]
        max_size = int(max_size*data_aug_scale_overlap)
        scales2_resize = [int(i*data_aug_scale_overlap) for i in scales2_resize]
        scales2_crop = [int(i*data_aug_scale_overlap) for i in scales2_crop]

    datadict_for_print = {
        'scales': scales,
        'max_size': max_size,
        'scales2_resize': scales2_resize,
        'scales2_crop': scales2_crop
    }
    print("data_aug_params:", json.dumps(datadict_for_print, indent=2))
    # if image_set == 'train_4':
    #     return T.Compose([  # Uses your new fixed Resize class
    #         normalize,
    #     ])

    if image_set == 'train_4':
            # return T.Compose([
            #     T.RandomHorizontalFlip(),
            #     # T.RandomResize([(max_size, max(scales))]),
            #     T.RandomResize([(256, 256)]),
            #     normalize,
            # ])
            import dino_datasets.sltransform as SLT
            return T.Compose([
                T.RandomHorizontalFlip(),
                # T.RandomResize([(max_size, max(scales))]),
                T.RandomResize([(256, 256)]),
                # normalize,
                SLT.RandomSelectMulti([
                    # SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                    # SLT.AlbumentationsClassification()
                ]),
                normalize,
            ])
    elif image_set == 'train_3': 
            import dino_datasets.sltransform as SLT
            return T.Compose([
                T.RandomHorizontalFlip(),
                # T.RandomResize([(max_size, max(scales))]),
                T.RandomResize([(512, 512)]),
                # normalize,
                SLT.RandomSelectMulti([
                    # SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                    # SLT.AlbumentationsClassification()
                ]),
                normalize,
            ])
        # return T.Compose([
        #     T.Resize((512, 512)),  # Uses your new fixed Resize class
        #     normalize,
        # ])

    elif image_set == 'train_2':
            return T.Compose([
                # T.RandomHorizontalFlip(),
                # T.RandomResize([(max_size, max(scales))]),
                T.RandomResize([(512, 512)]),
                normalize,
            ])
            # import dino_datasets.sltransform as SLT
            # return T.Compose([
            #     T.RandomHorizontalFlip(),
            #     # T.RandomResize([(max_size, max(scales))]),
            #     T.RandomResize([(256, 256)]),
            #     # normalize,
            #     SLT.RandomSelectMulti([
            #         # SLT.RandomCrop(),
            #         SLT.LightingNoise(),
            #         SLT.AdjustBrightness(2),
            #         SLT.AdjustContrast(2),
            #         # SLT.AlbumentationsClassification()
            #     ]),
            #     normalize,
            # ])
    elif image_set in ['train']:
        if fix_size:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([(max_size, max(scales))]),
                # T.RandomResize([(448, 448)]),
                normalize,
            ])
    

        if strong_aug:
            import dino_datasets.sltransform as SLT
            
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose([
                        T.RandomResize(scales2_resize),
                        T.RandomSizeCrop(*scales2_crop),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                SLT.RandomSelectMulti([
                    SLT.RandomCrop(),
                    SLT.LightingNoise(),
                    SLT.AdjustBrightness(2),
                    SLT.AdjustContrast(2),
                ]),
                normalize,
            ])
        
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])
    valid_sets = {'val', 'eval_debug', 'train_reg', 'test'}

    if args.eval_type == "det" and image_set in valid_sets:
        if os.environ.get("GFLOPS_DEBUG_SHILONG", False) == 'INFO':
            print("Under debug mode for flops calculation only!!!!!!!!!!!!!!!!")
            return T.Compose([
                T.ResizeDebug((1280, 800)),
                normalize,
            ])   

        return T.Compose([
            T.RandomResize([max(scales)], max_size=max_size),
            normalize,
        ])
        
    # for classification 
    if args.eval_type == "cls" and image_set in valid_sets:
        return T.Compose([
           T.RandomResize([(512, 512)]),  # Resize to fixed size
            normalize,
        ])  
    

    if args.eval_type == "text" and image_set in valid_sets:
        return T.Compose([
             T.Resize((256,256)),  # Uses your new fixed Resize class
            normalize,
        ])
    if args.eval_type == "seg" and image_set in valid_sets:
        return T.Compose([
            T.Resize((512, 512)),  # Uses your new fixed Resize class
            normalize,
        ])


    # raise ValueError(f'unknown {image_set}')


def get_aux_target_hacks_list(image_set, args):
    if args.modelname in ['q2bs_mask', 'q2bs']:
        aux_target_hacks_list = [
            label2compat(), 
            label_compat2onehot(), 
            RandomSelectBoxes(num_class=args.num_classes)
        ]
        if args.masked_data and image_set == 'train':
            # aux_target_hacks_list.append()
            aux_target_hacks_list.append(MaskCrop())
    elif args.modelname in ['q2bm_v2', 'q2bs_ce', 'q2op', 'q2ofocal', 'q2opclip', 'q2ocqonly']:
        aux_target_hacks_list = [
            label2compat(),
            label_compat2onehot(),
            box_label_catter(),
            RandomSelectBoxlabels(num_classes=args.num_classes,
                                    prob_first_item=args.prob_first_item,
                                    prob_random_item=args.prob_random_item,
                                    prob_last_item=args.prob_last_item,
                                    prob_stop_sign=args.prob_stop_sign,
                                    ),
            BboxPertuber(max_ratio=0.02, generate_samples=1000),
        ]
    elif args.modelname in ['q2omask', 'q2osa']:
        if args.coco_aug:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                        prob_first_item=args.prob_first_item,
                                        prob_random_item=args.prob_random_item,
                                        prob_last_item=args.prob_last_item,
                                        prob_stop_sign=args.prob_stop_sign,
                                        ),
                RandomDrop(p=0.2),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
                RandomCutout(factor=0.5)
            ]
        else:
            aux_target_hacks_list = [
                label2compat(),
                label_compat2onehot(),
                box_label_catter(),
                RandomSelectBoxlabels(num_classes=args.num_classes,
                                        prob_first_item=args.prob_first_item,
                                        prob_random_item=args.prob_random_item,
                                        prob_last_item=args.prob_last_item,
                                        prob_stop_sign=args.prob_stop_sign,
                                        ),
                BboxPertuber(max_ratio=0.02, generate_samples=1000),
            ]
    else:
        aux_target_hacks_list = None

    return aux_target_hacks_list


def build(image_set, args):
    root = Path(args.coco_path)
    mode = 'instances'
    PATHS = {
        #Segmentation traning  (coco_data/annotations/segmentation/segmentation_train_v5_elsify_20.json)
        
        "train_2": (root / "merge_train/Segmentation", root / "annotations/segmentation" / f'segmentation_train_v5_elsify_20.json'),
        
       # detetion Traning 
       
        "train": (root / "merge_train/Detection", root / "annotations" / f'detection/detetction_complet_v5.json'),
        # "train":("/media/iml/Abdul_2/raabin_WBC/raabin_label1_m1/images/", '/media/iml/Abdul_2/raabin_WBC/raabin_label1_m1/microscope_1_train.json'), #
        
        ##Classification
        "train_3":(root / "merge_train/classification", root /"annotations/classification/11_9_3_4_5_6_7_12_13_14_8-t_v6-krdc-bccd.json"),
        
        ## Text data 
        "train_4":(root / "merge_train/Detection", root /"annotations/captions/Final_version/Q_A_val_train_MLM_train.json"),
        
        ## language val 
        
        # "val": (root / "merge_train/Detection", root / "annotations/captions/Final_version/Mask_test_fixed.json"), #MLM
        # "val": (root / "merge_train/Detection", root / "annotations/captions/Final_version/Q_A_test_small.json"), #Leukemia
        # "val": (root / "merge_train/Detection", root / "annotations/captions/Final_version/Q_A_test_large.json"), #Leukemia
        # 
        # "train": (root / "merge_train/Detection", root / "annotations" / f'detection/filtered_annotations_reset.json'),
        # "train": (root / "merge_train", root / "annotations" / f'merged_file_with_leuk_m5-high-low_para_plate_sickle.json'), #hematology 5 classes
        # "train": (root / "train/", root /"train/" f'coco_p4.json'), #FLIR FLIR_Data/images_thermal_train
        # "train": (root / "images_thermal_train/", root /"images_thermal_train/" f'coco_p4.json'), #thermal 
        # "val":(root / "images_rgb_val", root / f"images_rgb_val" / f'coco_p3.json'),   # FLIR_spatial
        # "val":(root / "images_thermal_val", root / f"images_thermal_val" / f'coco_p3.json'),   # FLIR_Thermalqw
        
        # "train_reg": (root / "train2017", root / "annotations" / f'{mode}_p_train2017.json'),
        
        
        
        # "train_3":(root / "merge_train/classification", root /"annotations/classification/" f'raabin_Train_update.json'),
        # "train_3":(root / "merge_train/", root /"annotations/classification/" f'test.json'), #FLIR FLIR_Data/images_thermal_traine
        # "val":(root / "merge_train/", root /"annotations/classification/" f'AML_Metak_test.json'), #FLIR FLIR_Data/images_thermal_traine
        # "val":(root / "merge_train/classification", root /"annotations/classification/" f'Raabin_testA_updated.json'),#/home/iml/DINO/coco_data/annotations/classification/test_Acevedo_update_20.json
        # "val":(root / "merge_train/classification", root /"annotations/classification/" f'/test_Acevedo_update_20.json'),#
        # "train":(root / "merge_train/", root /"annotations/captions/" f'HCM_100x_c2_train_text_train80.json'), #/home/iml/DINO/coco_data/annotations/captions/HCM_100x_c2_train_text_only_masked.json
        # "val":(root / "merge_train/", root /"annotations/captions/" f'HCM_100x_c2_train_text_val20.json'), 
       
       ##Segmentation val"
     
        "val":(root / "merge_train/Segmentation_test/", root /"annotations/segmentation_test/" f'Malaria-Detection-2019_test_data.json'), #FLIR FLIR_Data/images_thermal_train
        # "val":(root / "merge_train/Segmentation/", root /"annotations/segmentation_test/" f'AneRBC-II_Anemic_individuals_test.json'),  
        # "val":(root / "merge_train/Segmentation/", root /"annotations/segmentation_test/" f'AneRBC-II_Healthy_individuals_test.json'),  
        # "val":(root / "merge_train/Segmentation/", root /"annotations/segmentation_test/" f'Elsafty_RBCs_Cellular_Images_and_Masks_test.json'), 
    #    "val":(root / "merge_train/Segmentation/", root /"annotations/segmentation_test/" f'KRD_test_data.json'), 
        # "val":(root / "merge_train/Segmentation/", root /"annotations/segmentation_test/" f'test.json'), 
        # "val":(root / "merge_train/Segmentation/avecoda_seg/", root /"annotations/segmentation_test/" f'ava_test_data.json'), 
      
        
       
       # Detetion Dataset Valadation
        #Leukemia 4
        # "val": (root / "merge_train/Detection/HCM_100x_c2_test", root / "annotations/Detection_test" / f'{mode}_p_val2017_22_update.json'), #Leukemia
        # "val": (root / "merge_train/Detection", root / "annotations/leukemia/" / f'hcm_100x_c1_test_update.json'), #Leukemia h_c1_100x
        # "val": (root / "merge_train/Detection/", root / "annotations/leukemia" / f'hcm_40x_c2_m_update.json'), #Leukemia h_40x_c2
        # "val": (root / "merge_train/Detection/", root / "annotations/leukemia" / f'h_40x_c1_test_update.json'), #Leukemia h_40x_c1
        # "val": (root / "merge_train/Detection/", root / "annotations/leukemia" / f'l_100x_c2_test_update.json'), #Leukemia l_100x_c2
        # "val": (root / "merge_train/Detection/", root / "annotations/Detection_test" / f'l_100x_c1_test.json'), #Leukemia l_100x_c1
        # "val": (root / "merge_train/Detection/", root / "annotations/leukemia" / f'lcm_40x_c2_m_update.json'), #Leukemia l_40x_c2hcm_40x_c1
        # "val": (root / "merge_train/Detection/", root / "annotations/leukemia" / f'l_40x_c1_test_update.j88.3s97.7o71.2n95.2'90.6), #Leukemia l_40x_c1
        
        #M5 
        # "val":(root / "M5_coco/test/", root / "annotations/Detection_test" / f'hcm_test_p_1000x_22.json'), #M5 hcm /home/iml/DINO/coco_data/annotations/Malariae_test_p.json
        # "val":(root / "M5_coco/test/", root /'M5_coco/annotations/hcm_test_400x_v2.json'), #M5 hcm /home/iml/DINO/coco_data/annotations/Malariae_test_p.json
        # "val":(root / "M5_coco/test/",  root /'M5_coco/annotations/lcm_test_1000x_v2.json'),
        # "val":(root / "M5_coco/test/", root /'M5_coco/annotations/lcm_test_400x_v2.json'),
    
        
        #MP-IDB-The-Malaria-Parasite- (UNSEEN)
        # "val":(root / "merge_train/Detection", root / "annotations/Detection_test" / f'Malariae_test_p.json'), #Malariae 
        
        
        # Sickle cell 
        # "val":(root / "merge_train/Detection/positive_sickle_test", root / "annotations/Detection_test" / f'sickle_cell_test_2.json'), #scikle ce
        
        
        # leishman parasite detection 
        # "val":(root / "merge_train/Detection/parasite_detection_test", root / "annotations/Detection_test" / f'parasite_detection_test_las.json'), #parasites
        
        #Platelet
        # "val":(root / "merge_train/Detection/TXL_val", root / "annotations/Detection_test" / f'CRC_val_22.json'),   # platelet TXL
        
        # CBC 
        # "val":(root/"annotations/detection/baseline_results_detetion/test_images", root/'annotations/detection/baseline_results_detetion/test_annotation/unified/BCCD_test_v2.json'), #parasites
        # "val":(root/"annotations/detection/baseline_results_detetion/test_images", root/'annotations/detection/baseline_results_detetion/test_annotation/unified/TXL_test_v2.json'), #
        # "val":("/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_images", '/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_annotation/unified/Boi_net_test_v2_unssen.json'), #
        # "val":("/media/iml/Abdul_2/raabin_WBC/raabin_label1_m1/images/", '/media/iml/Abdul_2/raabin_WBC/raabin_label1_m1/microscope_1_test.json'), #
        # "train": (root / "merge_train", root / "annotations" / f'parasite_detection_train_p.json'),
        # "val":(root / "parasite_detetion_test", root / "annotations" / f'parasite_detection_test.json'), #parasites
        
         # "train": (root / "merge_train", root / "annotations" / f'CRC_train3.json'),#/home/iml/DINO/coco_data/annotations/orignal labels/CRC_train2.json
        # "val":(root / "CRC_val", root / "annotations" / f'CRC_val_1.json'),   # platelet 
        
        
        # "eval_debug": (root / "val2017", root / "annotations" / f'{mode}_p_val2017.json'),
        # "test": (root / "test2017", root / "annotations" / 'image_info_test-dev2017.json' ),
        # "eval_debug": (root / "test", root / "annotations" / f'hcm_test_p_1000x.json'),
        # "test": (root / "test", root / "annotations" / 'hcm_test_p_1000x.json' ),Pl
        
        # Fine_tune
        #  "train": ("/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/train_data", '/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/train_annotation/BCCD_train.json'),#/home/iml/DINO/coco_data/annotations/orignal labels/CRC_train2.json
        # "val":("/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_images", '/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_annotation/unified/BCCD_test_v2.json'),   # platelet 

        #  "train": ("/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/train_data", '/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/train_annotation/Boi_net_train.json'),
        # "val":("/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_images", '/home/iml/DINO/coco_data/annotations/detection/baseline_results_detetion/test_annotation/unified/Boi_net_test_v2_unssen.json'),   # platelet 
    }

    # add some hooks to datasets
    aux_target_hacks_list = get_aux_target_hacks_list(image_set, args)
    img_folder, ann_file = PATHS[image_set]
    print(img_folder)

    # copy to local path
    if os.environ.get('DATA_COPY_SHILONG') == 'INFO':
        preparing_dataset(dict(img_folder=img_folder, ann_file=ann_file), image_set, args)

    try:
        strong_aug = args.strong_aug
    except:
        strong_aug = False
    dataset = CocoDetection(img_folder, ann_file,
            transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug, args=args), 
            return_masks=args.masks,
            aux_target_hacks=aux_target_hacks_list,
        )

    return dataset



if __name__ == "__main__":
    # Objects365 Val example
    dataset_o365 = CocoDetection(
            '/path/Objects365/train/',
            "/path/Objects365/slannos/anno_preprocess_train_v2.json",
            transforms=None,
            return_masks=False,
        )
    print('len(dataset_o365):', len(dataset_o365))




# class LearnableUpsample(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, output_size=(384, 384)):
#         super().__init__()
#         self.output_size = output_size
        
#         # This kernel & stride combination roughly scales 56 → 384
#         # 384 / 56 ≈ 6.857, so stride ~7
#         # Adjust padding and kernel_size for exact output
#         self.upsample = nn.Sequential(
#             # 56 → 112
#             nn.ConvTranspose2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),

#             # 112 → 224
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             # 224 → 384
#             nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = x.unsqueeze(1) 
#         out = self.upsample(x)
#         # Ensure exact output size
#         # out = F.interpolate(out, size=self.output_size, mode='bilinear', align_corners=False)
#         return out.squeeze(1) 




>>>>>>> e8a8ec0028059ae8e36eaba4f8a1954505fd2f66
# predictions_mask= self.upsampler(predictions_mask[-1][:, 0])