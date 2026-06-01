import json
import os
import torch
from PIL import Image

from main import build_model_main
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from util import box_ops
import dino_datasets.transforms as T



def separate_domain_bit(outputs):
        # Separate the last column from pred_logits
        pred_logits = outputs['pred_logits']
        domain_bit = pred_logits[..., -1]  # Extract the last column (class + 1)
        pred_logits = pred_logits[..., :-1]  # Extract everything except the last column
        # Store the results for pred_logits
        separated_dict = {'domain_bit': domain_bit,'pred_logits': pred_logits}
        for key, value in outputs.items():
            if key not in ['pred_logits', 'aux_outputs']:
                separated_dict[key] = value  # Copy other keys as-is
        # Process aux_outputs (list of dictionaries)
        if 'aux_outputs' in outputs:
            aux_domain_bit = []
            aux_outputs = []
            for aux_dict in outputs['aux_outputs']:
                aux_pred_logits = aux_dict['pred_logits']
                aux_domain_bit = aux_pred_logits[..., -1]  # Extract the last column
                aux_pred_logits = aux_pred_logits[..., :-1]  # Extract the rest
                # Create a new dictionary for each auxiliary output
                aux_dict_new = {'pred_logits': aux_pred_logits,'domain_bit': aux_domain_bit}
                # Copy any additional keys in each aux_dict
                for key, value in aux_dict.items():
                    if key != 'pred_logits':
                        aux_dict_new[key] = value
                aux_outputs.append(aux_dict_new)
            separated_dict['aux_outputs'] = aux_outputs
        return separated_dict

# Paths to the model config, checkpoint, and JSON file
#model_config_path = "logs/dn_DABDETR/R50_3/config.json"
#model_checkpoint_path = "logs/dn_DABDETR/R50_3/checkpoint0049.pth"
#json_file_path = "M5_coco/annotations/lcm_test_1000x.json"  # Replace with your JSON file path

model_config_path ="/media/iml/Abdul_1/UNI_X/upsamples_detetion_last_2/config_args_all.json" #"config.json" # change the path of the model config
#model_checkpoint_path = "logs/dn_DABDETR/R50_9/checkpoint0049.pth" #"checkpoint_optimized_44.7ap.pth" # change the path of the model checkpoint
model_checkpoint_path = "/media/iml/Abdul_1/UNI_X/upsamples_detetion_last_2/checkpoint0007.pth"
json_file_path ="/home/iml/DINO/coco_data/M5_coco/annotations/hcm_test_400x_v2.json" # "/home/iml/DINO/M5_coco/annotations/hcm_test_p_1000x.json"

output_dir = "/home/iml/DINO_X/logs/DINO_X/prompt/gt_hcm_40x_m5" 
output_dir_2 = "/home/iml/DINO_X/logs/DINO_X/prompt/pr_hcm_40x_m5" # Replace with the directory where you want to save the results

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_2, exist_ok=True)

# Load model
args = SLConfig.fromfile(model_config_path)
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

# Initialize the visualizer
vslzr = COCOVisualizer()

# Load JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extract image information
#id2name = {0: 'gametocyte', 1: 'schizont', 2: 'trophozoite', 3: 'ring'}
id2name = {
 '1': 'myeloblast',
 '2': 'lymphoblast',
 '3': 'neutrophil',
 '4': 'atypical lymphocyte',
 '5': 'promonocyte',
 '6': 'monoblast',
 '7': 'lymphocyte',
 '8': 'myelocyte',
 '9': 'abnormal promyelocyte',
 '10': 'monocyte',
 '11': 'metamyelocyte',
 '12': 'eosinophil',
 '13': 'basophil',
 '14': 'none',
 '15': 'gametocyte',
 '16': 'schizont',
 '17': 'trophozoite',
 '18': 'ring',
 '19': 'concentrated_leishman_parasite',
 '20': 'leishman_parasite',
 '21': 'platelet',
 '22': 'Sickle Cells',
 '23': 'RBC',
 '24': 'WBC',
 '25': 'parasite',
 '26': 'plasmodium',
 '27': 'TBbacillus',
 '28': 'leukocyte',
 '29': 'difficult'
}

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Perform inference and save results
for img_info in data['images']:
    image_name = img_info['file_name']
    image_id = img_info['id']

    # Load and preprocess image
    image_path = os.path.join("/home/iml/DINO/coco_data/M5_coco/test/", image_name)
    
    if not os.path.exists(image_path):
        print(f"Skipping missing file: {image_path}")
        continue   # 👉 goes to the next image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image, _ = transform(image, None)

    # Perform inference
    output = model.cuda()(image[None].cuda(), (["Malaria"],))
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    # Visualize outputs
    threshold = 0.25
    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > threshold

    # Prediction boxes and labels
    pred_box_label = [id2name[str(int(item))] for item in labels[select_mask]]
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': torch.Tensor([image.shape[1], image.shape[2]]),
        'box_label': pred_box_label,
        'image_id': image_id,
        'image_name':f"pr_{image_name}"
    }

    # Extract Ground Truth boxes and labels
    gt_boxes = []
    gt_labels = []
    img_h = img_info['height']
    img_w = img_info['width']
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            # Convert to cxcywh and normalize
            bbox = torch.tensor(ann['bbox'])
            bbox[0] = bbox[0] + bbox[2] / 2  # x_center
            bbox[1] = bbox[1] + bbox[3] / 2  # y_center
            bbox = bbox / torch.tensor([img_w, img_h, img_w, img_h])  # Normalize
            gt_boxes.append(bbox)
            gt_labels.append(id2name[str(ann['category_id']+0)])

    gt_dict = {
         'boxes': torch.stack(gt_boxes),
        'size': torch.Tensor([image.shape[1], image.shape[2]]),
        'box_label': gt_labels,
        'image_id': image_id,
        'image_name':f"gt_{image_name}"
    }

    # Save the result with Ground Truth in red
    save_path = os.path.join(output_dir, image_name)
    vslzr.visualize(image, gt_dict, savedir=output_dir, show_in_console=False)
    # vslzr.visualize(image, pred_dict, color='green', savedir=output_dir, show_in_console=False)
    vslzr.visualize(image, pred_dict, savedir=output_dir_2, show_in_console=False)


