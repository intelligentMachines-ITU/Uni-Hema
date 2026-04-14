coco_path=$1
checkpoint=$2
python main.py \
  --output_dir logs/DINO/R50-MS4_1_4_sample_4_sample_5 \
	-c config/DINO/DINO_4scale.py --coco_path /home/iml/DINO/coco_data  \
	--eval --resume logs/DINO/R50-MS4_1_4_sample_4_sample_5/checkpoint0023.pth \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
