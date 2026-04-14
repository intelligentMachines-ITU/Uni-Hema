import os
import contextlib
import copy
import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from util.misc import all_gather

class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, useCats=True):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].useCats = useCats

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.useCats = useCats

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.useCats = self.useCats
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def print_individual_class_ap(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.accumulate()
            precision = coco_eval.eval['precision']  # Shape: (10, #categories, #area ranges, #max detections)
            cat_ids = coco_eval.params.catIds

            for cat_idx, cat_id in enumerate(cat_ids):
                ap = np.mean(precision[:, cat_idx, 0, -1])
                print(f"Class {cat_id} mAP: {ap:.4f}")
    def print_individual_class_re(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.accumulate()
            precision = coco_eval.eval['recall']  # Shape: (10, #categories, #area ranges, #max detections)
            cat_ids = coco_eval.params.catIds

            for cat_idx, cat_id in enumerate(cat_ids):
                ap = np.mean(precision[:, cat_idx, 0, -1])
                print(f"Class {cat_id} mAP: {ap:.4f}")
    def count_false_positives(self, iou_type="bbox"):
        assert iou_type in self.coco_eval, f"{iou_type} not found in coco_eval"
        coco_eval = self.coco_eval[iou_type]
        evalImgs = coco_eval.evalImgs

        false_positives = 0
        true_positives = 0

        for eval_img in evalImgs:
            if eval_img is None:
                continue
            # dtMatches is [T, D], where T is IoU thresholds, D is detections
            # We're only interested in the first IoU threshold (0.5)
            dt_matches = eval_img['dtMatches'][0]  # [D] or [T, D]
            dt_ignore = eval_img['dtIgnore'][0]    # [D]
            for matched, ignored in zip(dt_matches, dt_ignore):
                if ignored:
                    continue
                if matched > 0:
                    true_positives += 1
                else:
                    false_positives += 1

        print(f"\nFalse Positives (FP) at IoU 0.5: {false_positives}")
        print(f"True Positives (TP) at IoU 0.5: {true_positives}")
        return false_positives
    def count_false_positives_far(self, iou_type="bbox", iou_threshold=0.1, distance_threshold=50):
        """
        Count predicted boxes that are not matched to any GT box (false positives).
        Includes an optional distance-based check for 'far' false positives.
        """
        import torch

        assert iou_type in self.iou_types, f"IoU type {iou_type} not found in evaluator."

        coco_eval = self.coco_eval[iou_type]
        coco_dt = coco_eval.cocoDt
        coco_gt = coco_eval.cocoGt

        false_positive_count = 0
        far_fp_count = 0

        for img_id in self.img_ids:
            dt_ann = coco_dt.imgToAnns.get(img_id, [])
            gt_ann = coco_gt.imgToAnns.get(img_id, [])

            gt_boxes = torch.tensor([ann["bbox"] for ann in gt_ann], dtype=torch.float32)
            dt_boxes = torch.tensor([ann["bbox"] for ann in dt_ann], dtype=torch.float32)

            if len(dt_boxes) == 0:
                continue
            if len(gt_boxes) == 0:
                false_positive_count += len(dt_boxes)
                far_fp_count += len(dt_boxes)
                continue

            # Convert to xyxy for IoU computation
            def xywh_to_xyxy(boxes):
                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 0] + boxes[:, 2]
                y2 = boxes[:, 1] + boxes[:, 3]
                return torch.stack([x1, y1, x2, y2], dim=1)

            dt_xyxy = xywh_to_xyxy(dt_boxes)
            gt_xyxy = xywh_to_xyxy(gt_boxes)

            ious = box_iou(dt_xyxy, gt_xyxy)

            matched = (ious > iou_threshold).any(dim=1)
            false_positive_mask = ~matched
            false_positive_count += false_positive_mask.sum().item()

            # Optional: check if any are FAR false positives
            dt_centers = (dt_xyxy[:, :2] + dt_xyxy[:, 2:]) / 2
            gt_centers = (gt_xyxy[:, :2] + gt_xyxy[:, 2:]) / 2

            for i, is_fp in enumerate(false_positive_mask):
                if not is_fp:
                    continue
                d = torch.norm(gt_centers - dt_centers[i], dim=1)
                if torch.min(d) > distance_threshold:
                    far_fp_count += 1

        print(f"❌ Total False Positives (IoU < {iou_threshold}): {false_positive_count}")
        print(f"🔴 Far False Positives (distance > {distance_threshold}): {far_fp_count}")
        return false_positive_count, far_fp_count
def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou
# CocoEvaluator.print_individual_class_ap = print_individual_class_ap

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs

def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

def evaluate(self):
    p = self.params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)

    return p.imgIds, evalImgs

# Example usage:
# evaluator = CocoEvaluator(coco_gt, ['bbox', 'segm'])
# evaluator.print_individual_class_ap()
