# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils
from transformers import T5Tokenizer, T5ForConditionalGeneration
import dino_datasets
from dino_datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=False)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='coco_data')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='output_dir/1',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    parser.add_argument('--classification', action='store_true',
    help='If true, build classification head in the model')
    
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    # args.config_file = 'config/DINO/DINO_4scale.py'
    # args.pretrain_model_path = '/media/iml1/Disk2/Ali/Dino_changed_1/DIno_3sep/DINO-main/pretrained/checkpoint0011_4scale.pth'
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    args.eval = True
    # args.coco_path = '/media/iml1/Disk2/Ali/Dino_changed_1/Dino_o_1/DINO/coco_data'
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    # args.output_dir = 'output_dir'

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    # for param in model.parameters():
    #     param.requires_grad = False
  


    #  Freez backbone and encoder 
   
    # model.transformer.encoder.cls_token.requires_grad = False
    # model.transformer.encoder.cls_pos.requires_grad = T
            # print(f" is frozen: requires_grad = {param.requires_grad}")
    # ema
    # frozen visual decoder parameters 
    # for param in model.MultimodalQFormerWithTextCrossAttn.parameters():
    #     param.requires_grad = False
    #     print(f"Frozen: requires_grad = {param.requires_grad}")
    # for param in model.transformer.decoder.parameters(): #encoder parameter freez 
    #         param.requires_grad = False
    #         print(f" is frozen: requires_grad = {param.requires_grad}")
    # frozen text_encoder parameters 
    # for name, param in model.T5_model.encoder.named_parameters():
    #     param.requires_grad = False
        # Optional: print to verify
        # print(f"[Encoder] {name} frozen.")
    # for name, param in model.T5_model.decoder.named_parameters():
        # param.requires_grad = False
        
    # for name, param in model.bart_model.model.encoder.named_parameters():
    #     param.requires_grad = False
    #     # Optional: print to verify
    #     # print(f"[Encoder] {name} frozen.")
    # for name, param in model.bart_model.model.decoder.named_parameters():
    #     param.requires_grad = False
    
    # for param in model.parameters():
    #     param.requires_grad = False

    # # --- Unfreeze only class_embed ---
    # for layer in model.class_embed:
    #     for param in layer.parameters():
    #         param.requires_grad = True

    # # --- Double-check (optional) ---
    # print("Trainable parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    
    
    
    
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    # param_dicts = get_param_dict(args, model_without_ddp)


    # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
    #                               weight_decay=args.weight_decay)
    

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_train_2 = build_dataset(image_set='train_2', args=args)
    dataset_train_3 = build_dataset(image_set='train_3', args=args)
    dataset_train_4 = build_dataset(image_set='train_4', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_train_2 = DistributedSampler(dataset_train_2)
        sampler_train_3 = DistributedSampler(dataset_train_3)
        sampler_train_4 = DistributedSampler(dataset_train_4)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_train_2 = torch.utils.data.RandomSampler(dataset_train_2)
        sampler_train_3 = torch.utils.data.RandomSampler(dataset_train_3)
        sampler_train_4 = torch.utils.data.RandomSampler(dataset_train_4)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    batch_sampler_train_2 = torch.utils.data.BatchSampler(
        sampler_train_2, args.batch_size, drop_last=True)
    batch_sampler_train_3 = torch.utils.data.BatchSampler(
        sampler_train_3, args.batch_size, drop_last=True)
    batch_sampler_train_4 = torch.utils.data.BatchSampler(
        sampler_train_4, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_train_2 = DataLoader(dataset_train_2, batch_sampler=batch_sampler_train_2,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_train_3 = DataLoader(dataset_train_3, batch_sampler=batch_sampler_train_3,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_train_4 = DataLoader(dataset_train_4, batch_sampler=batch_sampler_train_4,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, 2, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # if args.llm_traing:
    #     lr_scheduler_bart = torch.optim.lr_scheduler.OneCycleLR(optimizer_bart, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    
   


    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = dino_datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)  
    
    # print("Reloading T5 model from bin...")
    # model.T5_tokenizer = T5Tokenizer.from_pretrained(
    #     "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/coco_data/t5_models/mm_leukemia_qa_wbcatt" 
    #     #"/home/iml/DINO/coco_data/coco_data/t5_models/mm_leukemia_qa_wbcatt_26_10_2nd_traing_step_2"
    # )
    # # # 
    # model.T5_model = T5ForConditionalGeneration.from_pretrained(
    #     "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/coco_data/t5_models/mm_leukemia_qa_wbcatt"
    # #    "/home/iml/DINO/coco_data/coco_data/t5_models/mm_leukemia_qa_wbcatt_26_10_2nd_traing_step_2"
    #  ).cuda()
      
    for name, param in model.named_parameters():
        param.requires_grad = True
    # # segmentation 
    for name, param in list(model.decoder_norm.named_parameters()) + \
                  list(model.class_embed_seg.named_parameters()) + \
                  list(model.mask_embed.named_parameters()):
        param.requires_grad = False
    for name, param in model.upsampler.named_parameters():
        param.requires_grad = True


    # for name, param in model.forward_prediction_heads.named_parameters():
    #     param.requires_grad = True
    # 2. Unfreeze T5_model
    for name, param in model.T5_model.named_parameters():
        param.requires_grad = False

    # # Text
    # for name, param in model.T5_model.decoder.named_parameters():
    #     param.requires_grad = True

    # # # 3. Unfreeze xformer
    for name, param in model.xformer.named_parameters():
        param.requires_grad = False

    #detetcion 
    # for name, param in model.named_parameters():
    #     param.requires_grad =True
    # for param in model.transformer.encoder.parameters(): #encoder parameter freez 
    #         param.requires_grad = False
    # # for param in model.transformer.decoder.parameters(): #encoder parameter freez 
    # #         param.requires_grad = False
    for param in model.backbone.parameters(): #encoder parameter freez 
            param.requires_grad = False
    # for name, param in model.T5_model.named_parameters():
    #     param.requires_grad = False

    #all 
    for param in model.backbone.parameters(): #encoder parameter freez 
            param.requires_grad = False
            
            
   
    
    for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} -> trainable")
            else:
                print(f"{name} -> frozen")
    param_dicts = get_param_dict(args, model_without_ddp)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train_4), epochs=args.epochs, pct_start=0.2)
        
        # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train)+len(data_loader_train_2), epochs=args.epochs, pct_start=0.2)
        #  lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    print("=== Trainable Parameters ===")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} -> trainable")
    #     else:
    #         print(f"{name} -> frozen")
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # model_without_ddp.load_state_dict(checkpoint['model']) #change
        model.load_state_dict(checkpoint['model'], strict=False)
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            # for name, param in model.named_parameters():
                # param.requires_grad = False
            # for name, param in list(model.decoder_norm.named_parameters()) + \
            #             list(model.class_embed_seg.named_parameters()) + \
            #             list(model.mask_embed.named_parameters()):
            #     param.requires_grad = True
            # for name, param in model.upsampler.named_parameters():
            #     param.requires_grad = True
            
        # for name, param in model.named_parameters():
            # param.requires_grad = True
        # for name, param in model.T5_model.decoder.named_parameters():
            # param.requires_grad = False

        # # 3. Unfreeze xformer
        # for name, param in model.xformer.named_parameters():
            # param.requires_grad = False

        # # 2. Unfreeze T5_model
        # for name, param in model.T5_model.named_parameters():
        #     param.requires_grad = True
        # for name, param in model.named_parameters():
        #     param.requires_grad = False
        # for name, param in list(model.decoder_norm.named_parameters()) + \
        #             list(model.class_embed_seg.named_parameters()) + \
        #             list(model.mask_embed.named_parameters()):
        #     param.requires_grad = True
        # for name, param in model.upsampler.named_parameters():
        #     param.requires_grad = True
        # # 3. Unfreeze xformer
        # for name, param in model.xformer.named_parameters():
        #     param.requires_grad = True
        # for name, param in model.T5_model.named_parameters():
        #     param.requires_grad = False

        # # 3. Unfreeze xformer
        # for name, param in model.xformer.named_parameters():
        #     param.requires_grad = False
        # for param in model.transformer.encoder.parameters(): #encoder parameter freez 
        #         param.requires_grad = False
        # for param in model.backbone.parameters(): #encoder parameter freez 
        #         param.requires_grad = False
        # for param in model.transformer.encoder.parameters(): #encoder parameter freez 
        #     param.requires_grad = True
        # for param in model.backbone.parameters(): #encoder parameter freez 
        #     param.requires_grad = True
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name} -> trainable")
        #     else:
        #         print(f"{name} -> frozen")
    
        param_dicts = get_param_dict(args, model_without_ddp)
        # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            if coco_evaluator != 1:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        if coco_evaluator == 1:
            print(log_stats)

        return

    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,data_loader_train_2,data_loader_train_3,data_loader_train_4, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

        if not args.onecyclelr:
            lr_scheduler.step()
            # lr_scheduler_bart.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)
                
        # eval of the model 
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )
        #map_regular = (
        ##    test_stats.get('coco_eval_bbox', [None])[0]
         #   or test_stats.get('Dice_Score')
         #   or test_stats.get('bleu_scores_1_result')
       # )
        if args.classification_head:
            # map_regular = test_stats['F1_Score']
            map_regular = test_stats['coco_eval_bbox'][0]
            # map_regular= test_stats['Dice_Score']
            # map_regular= test_stats['bleu_scores_1_result']
        # map_regular = test_stats['coco_eval_bbox'][0]
        # map_regular_2= test_stats['morphology_accuracy']
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
        }

        # eval ema
        # if args.use_ema:
        #     ema_test_stats, ema_coco_evaluator = evaluate(
        #         ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
        #         wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        #     )
        #     log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
        #     map_ema = ema_test_stats['coco_eval_bbox'][0]
        #     _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
        #     if _isbest:
        #         checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
        #         utils.save_on_master({
        #             'model': ema_m.module.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             'epoch': epoch,
        #             'args': args,
        #         }, checkpoint_path)
        log_stats.update(best_map_holder.summary())

        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                # if "bbox" in coco_evaluator.coco_eval:
                filenames = ['latest.pth']
                if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                # for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
        # print("Dice Score Test: ",log_stats['test_Dice_Score'])
        # print(" Score Test: ",log_stats['test_F1_Score'])  #test_stats['F1_Score']
        # print(" Score Test: ",log_stats['test_bleu_scores_1_result'])
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from dino_datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
