# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
from PIL import Image
import os
import sys
from typing import Iterable
import numpy as np
from util.utils import slprint, to_device
from sklearn.metrics import accuracy_score
import numpy as np
from itertools import zip_longest
import torch
# from medpy import metric
from compute_rouge import compute_rouge
import util.misc as utils
from dino_datasets.coco_eval import CocoEvaluator
from dino_datasets.panoptic_eval import PanopticEvaluator
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from nltk.translate.bleu_score import corpus_bleu
# import nltk
# from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
# from nltk.tokenize import word_tokenize
from itertools import zip_longest
# from evaluate import load
# Only needed once
# nltk.download('punkt', download_dir='nltk_data')
# nltk.download('punkt', download_dir='/home/iml_abdul/nltk_data')



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable,data_loader_2: Iterable,data_loader_3: Iterable,data_loader_4: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None,  args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    _cnt = 0
    
    loader1 = iter(metric_logger.log_every(data_loader, print_freq, header, logger=logger))
    loader2 = iter(metric_logger.log_every(data_loader_2, print_freq, header, logger=logger))
    loader3 = iter(metric_logger.log_every(data_loader_3, print_freq, header, logger=logger))
    loader4 = iter(metric_logger.log_every(data_loader_4, print_freq, header, logger=logger))
    

    # Select loader based on training step
    if args.traning_step == 1:
        data_iter = zip_longest(loader3, fillvalue=None)

    elif args.traning_step == 3:
        data_iter = zip(loader1, loader2, loader3)

    elif args.traning_step == 4:
        data_iter = zip_longest(loader1, fillvalue=None)

    elif args.traning_step == 5:
        data_iter = zip_longest(loader2, fillvalue=None)

    elif args.traning_step == 6:
        data_iter = zip_longest(loader4, fillvalue=None)

    else:
        raise ValueError("Invalid training step")


    for batches in data_iter:

        losses = torch.as_tensor(0.).to(device)
        loss_value = torch.as_tensor(0.).to(device)
        combined_loss_dict_scaled = {}
        combined_loss_dict_unscaled = {}

        # Handle multi-loader case (step 3)
        if args.traning_step == 3:
            batch_list = list(batches)
        else:
            batch_list = [batches]

        for i, batch in enumerate(batch_list):

            if batch is None:
                continue

            # Unpack data
            if args.traning_step == 3:
                samples, targets, prompt = batch
            else:
                samples, targets, prompt = batch[0]

            samples = samples.to(device)

            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in targets
            ]

            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(samples, prompt, targets) if need_tgt_for_training else model(samples, prompt)

            # Compute loss dict
            # loss_dict = criterion(outputs, targets)
            # weight_dict = criterion.weight_dict

            # # Reduce across distributed processes
            # loss_dict_reduced = utils.reduce_dict(loss_dict)

            # # Unscaled loss components (for logging)
            # if len(loss_dict_reduced)>1:
            #     for k, v in loss_dict_reduced.items():
            #         combined_loss_dict_unscaled[f'{k}_unscaled_{i+1}'] = v

            #     # Scaled loss components
            #     loss_dict_reduced_scaled = {
            #         k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
            #     }

            # # Sum scaled losses
            # loss = sum(loss_dict_reduced_scaled.values())
            # losses += loss
            # loss_value += loss.item()
            # print("batch image 1", targets[0]['image_id'],"batch image 2", targets[1]['image_id'])
            loss_dict = criterion(outputs, targets,args)
            weight_dict = criterion.weight_dict

            # Reduce across distributed processes
            loss_dict_reduced = utils.reduce_dict(loss_dict)

            # Unscaled loss components (for logging)
            for k, v in loss_dict_reduced.items():
                combined_loss_dict_unscaled[f'{k}_unscaled_{i+1}'] = v

            # Scaled loss components if weights exist
            loss_dict_reduced_scaled = {
                k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict
            }

            # Final loss computation
            if loss_dict_reduced_scaled:
                loss = sum(loss_dict_reduced_scaled.values())
            else:
                # print(f"⚠️ No keys in loss_dict_reduced matched weight_dict at step {i+1}.")
                # Fallback: use unscaled loss
                loss = sum(loss_dict_reduced.values())

            # Accumulate loss
            losses= losses+loss
            
            loss_value = loss_value + loss.item()

            for k, v in loss_dict_reduced_scaled.items():
                combined_loss_dict_scaled[f'{k}_scaled_{i+1}'] = v
            
            # pred_morphology = outputs['pred_morphology']
            # logit = outputs['pred_logits']
            # pred_box = outputs['pred_boxes']
        # loss_dict = criterion(outputs, targets)
        
        # weight_dict = criterion.weight_dict

        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # loss_morphology = loss_dict.get('loss_morphology', None)
            # if loss_morphology is not None:
            #     print(f"Loss Morphology: {loss_morphology.item()}")


        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        # loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            # optimizer_bart.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            # scaler.step(optimizer_bart)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            # optimizer_bart.zero_grad()
            if losses != 0:
                losses.backward(retain_graph=True)
            # if max_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            # optimizer_bart.step()

        if args.onecyclelr:
            lr_scheduler.step()
            # lr_scheduler_bart.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **{
        **{k: v.item() for k, v in combined_loss_dict_unscaled.items()},
        **{k: v.item() for k, v in combined_loss_dict_scaled.items()}
    })
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])                                                                                   
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat
def plot_pred_mask(pred_mask, threshold=0.5, title="Predicted Mask"):
    # If batch dimension exists, remove it
    if pred_mask.dim() == 4:
        pred_mask = pred_mask[0, 0]
    elif pred_mask.dim() == 3:
        pred_mask = pred_mask[0]

    # Apply sigmoid and threshold if it's not binary yet
    pred_mask = pred_mask.sigmoid() if pred_mask.max() > 1 else pred_mask
    binary_mask = (pred_mask > threshold).float()

    # Convert to numpy for plotting
    binary_mask_np = binary_mask.detach().cpu().numpy()

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_mask_np, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
def dice_score2(pred, target, epsilon=1e-6):
    # def dice_score2(pred, target, epsilon=1e-6):
    # Ensure inputs are tensors (handle list of tensors)
    if isinstance(pred, list):
        pred = torch.stack(pred) if isinstance(pred[0], torch.Tensor) else torch.tensor(pred)
    if isinstance(target, list):
        target = torch.stack(target) if isinstance(target[0], torch.Tensor) else torch.tensor(target)

    pred = pred.float()
    target = target.float()

    # Ensure binary masks
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred.cpu() * target.cpu()).sum()
    union = pred.cpu().sum() + target.cpu().sum()

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.item()/len(pred)


def multiclass_dice_score(pred, target, num_classes, epsilon=1e-6):
    """
    pred: (B, C, H, W) raw logits or probabilities
    target: (B, H, W) class indices 0..C-1
    """

    # Softmax predictions → probabilities
    pred_soft = F.softmax(pred, dim=1)

    # Convert target to one-hot (B, C, H, W)
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    # Compute Dice per class
    dice_scores = []
    for c in range(num_classes):
        pred_c = pred_soft[:, c, :, :]
        target_c = target_onehot[:, c, :, :]

        intersection = (pred_c * target_c).sum(dim=(1,2))
        union = pred_c.sum(dim=(1,2)) + target_c.sum(dim=(1,2))

        dice = ((2 * intersection + epsilon) / (union + epsilon)).mean()  # mean over batch
        dice_scores.append(dice)

    # Mean Dice over all classes
    mean_dice = torch.mean(torch.stack(dice_scores))

    return mean_dice, dice_scores

def focal_loss(pred, target, alpha=0.25, gamma=2.):
        pred = pred.view(-1)
        target = target.view(-1)
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
@torch.no_grad()
def map_label(lbl):
    if 0 <= lbl <= 14:
        return 24
    elif lbl in [24, 29]:
        return 24
    elif 15 <= lbl <= 20:
        return 23
    elif lbl == 22:
        return 23
    elif 25 <= lbl <= 28:
        return 23
    else:
        return lbl 
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()
    
    all_gt_morphology = []
    all_pred_morphology = []
    seg_mask_t=[]
    classification_t=[]
    classification_p=[]
    # classification_t_feat=[]
    # classification_p_feat=[]
    Dice_score_all=[]
    text_t=[]
    text_p=[]
    prompt_text=[]

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    if args.eval_type == "det": 
        coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)             #coco eveluation for detetion change
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets,prompt  in metric_logger.log_every(data_loader, 100, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [
                        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                        for t in targets]
        # targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        
        # prompt=(["myeloblast","lymphoblast","neutrophil","atypical lymphocyte","promonocyte","monoblast","lymphocyte","myelocyte","abnormal promyelocyte","monocyte","metamyelocyte","eosinophil","basophil","none","gametocyte","schizont","trophozoite","ring","concentrated_leishman_parasite","leishman_parasite","Platelet", "Sickle Cells","RBC", "WBC",],)
        # prompt= (["Detect for all Hematology"],)
        # prompt= (['neutrophil'],)
        # for t in targets:
        #     if t["labels"].numel() == 0:   # if no labels detetion for 
        #         print(f"No labels for image_id: {t['image_id'].item()}")
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples,prompt, targets)
            else:
                # outputs = model(samples,prompt, targets)
                outputs = model(samples,prompt)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets,args)
        weight_dict = criterion.weight_dict
        
        
        
        
        
        # Extract morphology predictions
        pred_morphology = outputs['pred_morphology']  # Shape: [batch_size, num_queries, total_morphology_classes]

        num_attributes = 6
        num_classes_per_attribute = 2  # Valid labels are 0 and 1

        pred_morphology = pred_morphology.view(
            pred_morphology.size(0),
            pred_morphology.size(1),
            num_attributes,
            num_classes_per_attribute
        )

        morphology_probs = F.softmax(pred_morphology, dim=-1)
        pred_morphology_labels = morphology_probs.argmax(-1)  # Shape: [batch_size, num_queries, num_attributes]

        # Collect ground truth and predicted morphology labels
        
        for i, target in enumerate(targets):
            if 'segmentation' in target:
                seg_mask_t.append(target['segmentation'])
                pred_masks = outputs['pred_mask'] # shape: (B, 64, 64)

                # Upsample to match ground truth
                # pred_masks = F.interpolate(pred_maskss.unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
                # shape: (B, 1, 512, 512) 
                # pred_probss = torch.sigmoid(pred_maskss)
                
                # seg_mask_p.append(pred_probs)
                # pred_np = (pred_probs > 0.5).detach().cpu().numpy().astype(np.bool_)
                # gt_np = target['binary_mask'].detach().cpu().numpy().astype(np.bool_)
                # Dice_score_all.append(metric.binary.dc(pred_np, gt_np))
                pred_probs = torch.sigmoid(pred_masks)
                Dice_score_all.append(dice_score2(pred_probs, target['binary_mask']))
                for j in range(pred_probs.shape[0]):
                    mask = (pred_probs[j,0] * 255).cpu().detach().numpy().astype('uint8')  # scale 0-255
                    img = Image.fromarray(mask)
                    
                    image_id = target['mask'].item()
                    img.save(os.path.join("Malaria-Detection-2019_test_data", f"pred_mask_{image_id}.png"))
                
            if 'classification' in target:
                logits_classification = outputs['pred_image_class'] 
                logits_classification_feat = outputs['pred_image_feat'] 
                # shape: [1, num_classes]
                probs = F.softmax(logits_classification, dim=1)        # get probabilities
                pred_class = probs.argmax(dim=1).item()  # get predicted class index

                target_class = target['category_id'].item()
                #target_class_feat = target['label_embeding'].item()
                # print(target['category_id'].item()," and pred is ", pred_class )
                classification_t.append(target_class)
                classification_p.append(pred_class)
                # classification_t_feat.append(target_class_feat)
                # classification_p_feat.append(logits_classification_feat)# shape: (B, 64, 64)
            if 'masked_traning' in target:
                pred_text = outputs['pred_text']  # shape: [1, num_classes]
                # probs = F.softmax(logits_classification, dim=1)        # get probabilities
                # pred_class = probs.argmax(dim=1).item()  # get predicted class index
                
                target_text = outputs['completed_text']
                # print(target['category_id'].item()," and pred is ", pred_class )
                text_t.extend(target_text)
                text_p.extend(pred_text) # shape: (B, 64, 64)
                prompt_text.append(target['prompt'])
                
                
            if 'morphology' in target: 
                gt_morphology = target['morphology']  # Shape: [num_objects, num_attributes]

                # Get matching indices
                indices = criterion.matcher(outputs, [target])[0]
                src_idx = criterion._get_src_permutation_idx([indices])[1]
                tgt_idx = criterion._get_tgt_permutation_idx([indices])[1]

                # Get matched predictions and ground truths
                pred_labels = pred_morphology_labels[i, src_idx]  # Shape: [num_matched_objects, num_attributes]
                gt_labels = gt_morphology[tgt_idx]                # Shape: [num_matched_objects, num_attributes]

                all_pred_morphology.append(pred_labels)
                all_gt_morphology.append(gt_labels)

        
        
        

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        if seg_mask_t:
            E_Score= 1
        elif classification_t:
            E_Score= 1
        elif text_t:
            E_Score= 1
        else:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            score_threshold = 0.1  # 👈 Change this value to your desired threshold

            # Apply the threshold
            for result in results:
                keep = result["scores"] > score_threshold
                for key in result.keys():
                    result[key] = result[key][keep]
            # [scores: [100], labels: [100], boxes: [100, 4]] x B
            if 'segm' in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            # if prompt == (['complete blood count'],):
            #     for img_id, output in res.items():
            #         raw_labels = output["labels"]
            #         print(raw_labels)
            #         mapped_labels = torch.tensor([map_label(int(lbl.item())) for lbl in raw_labels],
            #                                     device=raw_labels.device)
            #         res[img_id]["labels"] = mapped_labels
            if coco_evaluator is not None:
                coco_evaluator.update(res)

            if panoptic_evaluator is not None:
                res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
                for i, target in enumerate(targets):
                    image_id = target["image_id"].item()
                    file_name = f"{image_id:012d}.png"
                    res_pano[i]["image_id"] = image_id
                    res_pano[i]["file_name"] = file_name

                panoptic_evaluator.update(res_pano)
            
            if args.save_results:
                # res_score = outputs['res_score']
                # res_label = outputs['res_label']
                # res_bbox = outputs['res_bbox']
                # res_idx = outputs['res_idx']


                for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                    """
                    pred vars:
                        K: number of bbox pred
                        score: Tensor(K),
                        label: list(len: K),
                        bbox: Tensor(K, 4)
                        idx: list(len: K)
                    tgt: dict.

                    """
                    # compare gt and res (after postprocess)
                    gt_bbox = tgt['boxes']
                    gt_label = tgt['labels']
                    gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                    
                    # img_h, img_w = tgt['orig_size'].unbind()
                    # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                    # _res_bbox = res['boxes'] / scale_fct
                    _res_bbox = outbbox
                    _res_prob = res['scores']
                    _res_label = res['labels']
                    res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                    # import ipdb;ipdb.set_trace()

                    if 'gt_info' not in output_state_dict:
                        output_state_dict['gt_info'] = []
                    output_state_dict['gt_info'].append(gt_info.cpu())

                    if 'res_info' not in output_state_dict:
                        output_state_dict['res_info'] = []
                    output_state_dict['res_info'].append(res_info.cpu())

                # # for debug only
                # import random
                # if random.random() > 0.7:
                #     print("Now let's break")
                #     break

            _cnt += 1
            if args.debug:
                if _cnt % 15 == 0:
                    print("BREAK!"*5)
                    break
                
            
    
    
    # After all batches are processed
    # if len(all_pred_morphology) > 0:
    #     all_pred_morphology = torch.cat(all_pred_morphology, dim=0)
    #     all_gt_morphology = torch.cat(all_gt_morphology, dim=0)

    #     pred_labels_flat = all_pred_morphology.reshape(-1)
    #     gt_labels_flat = all_gt_morphology.reshape(-1)
    #     valid_mask = gt_labels_flat != 4  # Ignore labels equal to 4

    #     pred_labels_valid = pred_labels_flat[valid_mask]
    #     gt_labels_valid = gt_labels_flat[valid_mask]

    #     if pred_labels_valid.numel() > 0:
    #         # Compute overall accuracy
    #         accuracy = accuracy_score(gt_labels_valid.cpu().numpy(), pred_labels_valid.cpu().numpy())
    #     else:
    #         accuracy = float('nan')
    # else:
    #     accuracy = float('nan')

    # # Reduce accuracy across all processes
    # morphology_accuracy = torch.tensor([accuracy], device=device)
    # if utils.is_dist_avail_and_initialized():
    #     torch.distributed.all_reduce(morphology_accuracy)
    #     morphology_accuracy /= utils.get_world_size()
    
    if seg_mask_t:
            # pred_masks = outputs['pred_mask'][:, 0]  # shape: (B, 64, 64)

            # # Upsample to match ground truth
            # pred_masks = F.interpolate(pred_masks.unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
            # # shape: (B, 1, 512, 512) 
            # # seg_mask_t = seg_mask_t.to(pred_masks.device)
            # pred_probs = torch.sigmoid(pred_masks)
            # plot_pred_mask(seg_mask_p[0], title="Predicted Segmentation Mask")
            # plot_pred_mask(seg_mask_t[0], title="Predicted Segmentation Mask")
            D_Score=  sum(Dice_score_all) / len(Dice_score_all)
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
            stats['Dice_Score'] = D_Score
            return stats  , 1 
    elif text_t:
        
            # bleu = evaluate.load("bleu")
            # rouge = evaluate.load("rouge")
            # bertscore = evaluate.load("bertscore")
            tokenized_preds = [word_tokenize(pred.lower()) for pred in text_p]
            tokenized_refs = [[word_tokenize(ref.lower())] for ref in text_t]

            # Compute BLEU score with smoothing
            smoothing = SmoothingFunction().method4
            bleu_scores_4= corpus_bleu(tokenized_refs, tokenized_preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

            # bleu_scores_4 =  corpus_bleu(text_t, text_p, weights=(0.25, 0.25, 0.25, 0.25))+
            bleu_scores_2 = corpus_bleu(
            tokenized_refs, tokenized_preds,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothing
        )

            # BLEU-3
            bleu_scores_3 = corpus_bleu(
                tokenized_refs, tokenized_preds,
                weights=(0.33, 0.33, 0.33, 0),
                smoothing_function=smoothing
            )
            bleu_scores_1    = corpus_bleu(tokenized_refs, tokenized_preds, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            rouge_scores =  compute_rouge(text_p, text_t)
            with open("predictions_Backbone_QA.txt", "a", encoding="utf-8") as f:
                for i in range(len(prompt_text)):
                    f.write(f"Prompt: {prompt_text[i]}\n")
                    f.write(f"Prediction: {text_p[i]}\n")
                    f.write(f"Target: {text_t[i]}\n")
                    f.write("-" * 50 + "\n")
                print("Txt file save")
            # print(f"Avg BLEU Score: {sum(bleu_scores) / len(bleu_scores):.2f}")
            # Compute BLEU
            # bleu_result = sum(bleu_scores) / len(bleu_scores)
            # print("BLEU:", bleu_result)

            # Compute ROUGE
            # rouge_result = rouge.compute(predictions=text_p, references=text_t)
            # # print("ROUGE:", rouge_result)

            # # Compute BERTScore
            # bertscore_result = bertscore.compute(predictions=text_p, references=text_t, lang="en")
            # bertscore_avg = sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
            # print("BERTScore F1 (average):", bertscore_avg)
            metric_logger.synchronize_between_processes()
            
            print("Averaged stats:", metric_logger)
            print("\nROUGE Scores:")
            for k, v in rouge_scores.items():
                print(f"{k}: {v:.4f}")
            stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
            
            stats['bleu_scores_1_result'] = bleu_scores_1
            stats['bleu_scores_2_result'] = bleu_scores_2
            stats['bleu_scores_3_result'] = bleu_scores_3
            stats['bleu_scores_4_result'] = bleu_scores_4
            
            stats['rouge_scores'] = {k: round(v, 4) for k, v in rouge_scores.items()}
            # stats['rouge_result'] = rouge_result
            # stats['bertscore_avg'] = bertscore_avg
            return stats  , 1 
    elif classification_t:
        
            f1_score_classification = f1_score(classification_t, classification_p, average='weighted') 
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
            stats['F1_Score'] = f1_score_classification
            return stats  , 1 
        
        
         
    elif len(all_gt_morphology) > 0:
        all_pred_morphology = torch.cat(all_pred_morphology, dim=0)
        all_gt_morphology = torch.cat(all_gt_morphology, dim=0)

        pred_labels_flat = all_pred_morphology.reshape(-1)
        gt_labels_flat = all_gt_morphology.reshape(-1)
        valid_mask = gt_labels_flat != 4  # Ignore labels equal to 4

        pred_labels_valid = pred_labels_flat[valid_mask]
        gt_labels_valid = gt_labels_flat[valid_mask]

        if pred_labels_valid.numel() > 0:
            # Compute overall accuracy
            # accuracy = accuracy_score(gt_labels_valid.cpu().numpy(), pred_labels_valid.cpu().numpy())

            # Compute F1 Score for each class
            all_pred_morphology_np = all_pred_morphology.cpu().numpy()
            all_gt_morphology_np = all_gt_morphology.cpu().numpy()

            # Flatten predictions and ground truths for masking
            pred_labels_flat = all_pred_morphology_np.reshape(-1)
            gt_labels_flat = all_gt_morphology_np.reshape(-1)

            # Apply mask to exclude labels with value 4
            valid_mask = gt_labels_flat != 4
            pred_labels_valid = pred_labels_flat[valid_mask]
            gt_labels_valid = gt_labels_flat[valid_mask]

            # Initialize variables
            labels = ["NC", "NS", "N", "C", "CB", "CV"]
            f1_scores = {}

            # Convert valid predictions and ground truths back to 2D shape
            pred_labels_valid_reshaped = pred_labels_valid.reshape(-1, len(labels))
            gt_labels_valid_reshaped = gt_labels_valid.reshape(-1, len(labels))

            # Calculate F1 Score for each label
            for i, label in enumerate(labels):
                pred_label = pred_labels_valid_reshaped[:, i]
                gt_label = gt_labels_valid_reshaped[:, i]
                
                # Ensure binary nature for each column
                unique_values = set(pred_label).union(set(gt_label))
                if len(unique_values) <= 2:  # If binary
                    f1_s = f1_score(gt_label, pred_label, average="binary")
                else:  # If multiclass, calculate macro F1
                    f1_s = f1_score(gt_label, pred_label, average="macro")
                
                f1_scores[label] = f1_s

            # Calculate combined F1 score (macro-average across all labels)
            overall_f1 = f1_score(
                gt_labels_valid_reshaped, pred_labels_valid_reshaped, average="macro"
            )
            accuracy = accuracy_score(gt_labels_valid, pred_labels_valid)

            # # Print results
            # for label, f1 in per_label_f1.items():
            #     print(f"F1 Score for {label}: {f1:.4f}")
            # print(f"Combined Macro F1 Score: {combined_f1:.4f}")

        else:
            accuracy = float('nan')
            f1_scores = [0.0] * 6
            overall_f1 = float('nan')
    else:
        accuracy = float('nan')
        f1_scores = [0.0] * 6 
        overall_f1 = float('nan')
    
    # Reduce accuracy and F1 scores across all processes
    morphology_accuracy = torch.tensor([accuracy], device=device)
    if all(value == 0.0 for value in f1_scores):
        f1_scores_list = f1_scores
    else:
        f1_scores_list = list(f1_scores.values())

    # Now create a tensor from the list of F1 scores
    morphology_f1_scores = torch.tensor(f1_scores_list, device=device)
    overall_f1_tensor = torch.tensor([overall_f1], device=device)

    if utils.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(morphology_accuracy)
        torch.distributed.all_reduce(morphology_f1_scores)
        torch.distributed.all_reduce(overall_f1_tensor)

        morphology_accuracy /= utils.get_world_size()
        morphology_f1_scores /= utils.get_world_size()
        overall_f1_tensor /= utils.get_world_size()


    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        coco_evaluator.count_false_positives(iou_type="bbox")
        coco_evaluator.count_false_positives_far(iou_type="bbox", iou_threshold=0.7, distance_threshold=50)
    
        cocoeval = coco_evaluator.coco_eval['bbox']
        iou_index = np.where(cocoeval.params.iouThrs == 0.5)[0]
        area_index = cocoeval.params.areaRngLbl.index("all")
        maxdet_index = cocoeval.params.maxDets.index(100)
        print(cocoeval.eval["precision"][iou_index, :, :, area_index, maxdet_index].mean(axis = 1))
        # print(cocoeval.eval['recall'][iou_index, :, area_index, maxdet_index])
        area_index = 0       # 'all'
        maxdet_index = 2     # maxDets=1 (index 0 in [1, 10, 100])

        # Average recall over IoUs for all categories
        ar = cocoeval.eval['recall'][:, :, area_index, maxdet_index]  # shape: [10 IoUs, num_classes]
        valid = ar[ar > -1]  # Filter out invalid entries (=-1)
        average_recall = valid.mean()
        

        print("Average Recall @IoU=0.50:0.95 | area=all | maxDets=1 =", average_recall)
        print(cocoeval.eval['recall'][5, :, area_index, maxdet_index])

        panoptic_res = None
        if panoptic_evaluator is not None:
            panoptic_res = panoptic_evaluator.summarize()
        stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        if coco_evaluator is not None:
            if 'bbox' in postprocessors.keys():
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in postprocessors.keys():
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        if panoptic_res is not None:
            stats['PQ_all'] = panoptic_res["All"]
            stats['PQ_th'] = panoptic_res["Things"]
            stats['PQ_st'] = panoptic_res["Stuff"]
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    
    
    # Update stats
    
    stats['morphology_accuracy'] = morphology_accuracy.item()
    # if pred_labels_valid.numel() > 0:
    if any(pred_labels_valid):
        for i, label_class in enumerate(labels):  # Assuming `labels` is your list of class names
            stats[f"f1_{label_class}"] = morphology_f1_scores[i].item()

        stats['overall_f1'] = overall_f1_tensor.item()
        print("______ MORPHOLOGY RESULTS__________")
    # print(f"stats['morphology_accuracy'] = {morphology_accuracy.item()}")
    # print(f"stats['morphology_f1_scores'] = {morphology_f1_scores.tolist()}")
        for label, f1 in zip(labels, morphology_f1_scores):
             print(f"F1 Score for {label}  |   {f1:.4f}")

    # Print overall F1 score
    # print(f"stats['overall_f1'] = {overall_f1_tensor.item()}")
    

    # Optionally, update metric logger
        metric_logger.update(morphology_accuracy=morphology_accuracy.item())
        metric_logger.update(
        morphology_accuracy=morphology_accuracy.item(),
        overall_f1=overall_f1_tensor.item(),
    )
        for i, label_class in enumerate(labels):  # Assuming `labels` is your list of class names
            metric_logger.update(**{f"f1_{label_class}": morphology_f1_scores[i].item()})

    # for label_class in labels:
    #     print(f"F1 Score for {label_class}: {stats[f'f1_{label_class}']:.4f}")
    
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res
