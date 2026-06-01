coco_path=$1
python main.py \
	--output_dir logs/DINO/R50-MS4_1_4_2.0 -c config/DINO/DINO_4scale.py --coco_path /home/iml/DINO/coco_data \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	--pretrain_model_path /home/iml/DINO/checkpoint0011_4scale.pth \
	--finetune_ignore label_enc.weight class_embed
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
