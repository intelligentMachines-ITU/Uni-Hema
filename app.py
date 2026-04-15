import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import build_model_main
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer
from util import box_ops
import dino_datasets.transforms as T
import torch.nn.functional as F
# # -----------------------------
# # CONFIG PATHS
# # -----------------------------
# model_config_path = "Model/config_args_all.json"
# model_checkpoint_path = "Model/checkpoint0023.pth"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# args = SLConfig.fromfile(model_config_path)
# model, criterion, postprocessors = build_model_main(args)

# checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
# model.load_state_dict(checkpoint["model"], strict=False)

# model.to(device)
# model.eval()
import torch
from huggingface_hub import hf_hub_download
from main import build_model_main
from util.slconfig import SLConfig

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# DOWNLOAD FILES FROM HF
# -----------------------------
repo_id = "ryhm/Uni-hema"

config_path = hf_hub_download(
    repo_id=repo_id,
    filename="config_args_all.json"
)

checkpoint_path = hf_hub_download(
    repo_id=repo_id,
    filename="checkpoint0023.pth"   # or pytorch_model.bin if you switch
)

cloze_samples_seg = [
    ["sample_seg/10.jpg"],
    ["sample_seg/0015_05_a.png"],
    ["sample_seg/0019_11_h.png"],
    ["sample_seg/95-8-27-1_223_5.bmp"],
    ["sample_seg/326Slide 1 (142,886,44,953,93).png"],
    ["sample_seg/Trip 053 Day 2 19-11-05 Image 10 add_4.png"],

]

cloze_samples_cls = [
    ["sample_cls/95-8-27-1_154_1.jpg"],
    ["sample_cls/95-8-6-1_61_2.jpg"],
    ["sample_cls/95-8-6-1_741_1.jpg"],
    ["sample_cls/95-8-6-1_565_1.jpg"],
    ["sample_cls/20190531_114409_2.jpg"],
]

cloze_samples_det = [
    ["sample_det/8_18_1000_ALL.png"],
    ["sample_det/0184a270-648e-5e6d-bc61-a9e8f7b0f6e4.jpg"],
    ["sample_det/187f92b2-71ed-484a-9bd3-599ca0d2636f.png"],
    ["sample_det/193.jpg"],
    ["sample_det/Malaria_CM2_21Jun2021105854_1000_135.0_17.0_100x.png"],
    ["sample_det/Malaria_CM10_14Jul2021122109_0001_137.0_14.0_1000x.png"],
]
cloze_samples_QA = [
    ["sample_qa/BNE_35858.jpg", "Q: What is the texture of the cytoplasm?"],
    ["sample_qa/EO_85775.jpg", "Q: What can you observe in the image of the cell?"],
    ["sample_qa/EO_960399.jpg", "Q: What is the cell classified as due to its specific morphological features?"],
    ["sample_qa/LY_810594.jpg", "Q: What does a normal lymphocyte look like?"],
    ["sample_qa/MO_614447.jpg", "Q: What is the typical shape of monocytes?"],
    ["sample_qa/SNE_185225.jpg", "Q: What are the typical features of a neutrophil cell?"],
]
cloze_samples_dm = [
    ["sample_dm/18_33_1000_ALL.png"],
    ["sample_dm/8_18_1000_ALL.png"],
    ["sample_dm/15_20_1000_AML.png"],
    ["sample_dm/21_32_1000_CLL.png"],
    ["sample_dm/28_24_1000_CML.png"],
    ["sample_dm/31_23_1000_CML.png"],
    ["sample_dm/31_34_1000_CML.png"],
    ["sample_dm/23_40_1000_APML.png"],
]
cloze_samples_mlm = [
    ["sample_mlm/46_37_1000_AML.png",
     "Mask:Neutrophils present with <extra_id_0> chromatin, irregular shape, <extra_id_0> basophilia, indicating <extra_id_0>."],

    ["sample_mlm/30_44_1000_CML.png",
     "Mask:WBC count shows a <extra_id_0> number of myeloblasts, neutrophils, myelocytes and monocyte are also noted."],

    ["sample_mlm/22_1_1000_AML.png",
     "Mask:Myeloblasts are <extra_id_0> in number, suggesting a possible hematological malignancy."],

    ["sample_mlm/25_9_1000_AML.png",
     "Mask:Myeloblasts show <extra_id_0> chromatin, <extra_id_0> nuclear shape and <extra_id_0> nucleoli; suggestive of <extra_id_0>."],

    ["sample_mlm/5_45_1000_ALL.png",
     "Mask:Neutrophils present with <extra_id_0> chromatin, <extra_id_0> basophilia and <extra_id_0> cytoplasm."]
]
# cloze_samples_QA = [
#     ["sample_qa/BNE_35858.jpg"],
#     ["sample_qa/EO_85775.jpg"],
#     ["sample_qa/EO_960399.jpg"],
#     ["sample_qa/LY_810594.jpg"],
#     ["sample_qa/MO_614447.jpg"],
#     ["sample_qa/SNE_185225.jpg"],
# ]
# Visualizer (optional)
vslzr = COCOVisualizer()

# -----------------------------
# LABEL MAP
# -----------------------------
id2name = {
 '1': 'myeloblast','2': 'lymphoblast','3': 'neutrophil','4': 'atypical lymphocyte',
 '5': 'promonocyte','6': 'monoblast','7': 'lymphocyte','8': 'myelocyte',
 '9': 'abnormal promyelocyte','10': 'monocyte','11': 'metamyelocyte',
 '12': 'eosinophil','13': 'basophil','14': 'Unidentified','15': 'gametocyte',
 '16': 'schizont','17': 'trophozoite','18': 'ring','19': 'concentrated_leishman_parasite',
 '20': 'leishman_parasite','21': 'platelet','22': 'Sickle Cells','23': 'RBC',
 '24': 'WBC','25': 'parasite','26': 'plasmodium','27': 'TBbacillus',
 '28': 'leukocyte','29': 'difficult'
}
id2class = {
 '0': 'abnormal eosinophil',
 '1': 'artefact',
 '2': 'basophil',
 '3': 'blast',
 '4': 'erythroblast',
 '5': 'eosinophil',
 '6': 'faggot cell',
 '7': 'hairy cell',
 '8': 'smudge cell',
 '9': 'immature lymphocyte',
 '10': 'lymphocyte',
 '11': 'metamyelocyte',
 '12': 'monocyte',
 '13': 'myelocyte',
 '14': 'band neutrophil',
 '15': 'segmented neutrophil',
 '16': 'not identifiable',
 '17': 'other cell',
 '18': 'proerythroblast',
 '19': 'plasma cell',
 '20': 'promyelocyte',
 '21': 'myeloblast',
 '22': 'monoblast',
 '23': 'atypical lymphocyte',
 '24': 'neutrophil',
 '25': 'lymphocyte variant',
 '26': 'giant thrombocyte',
 '27': 'blast no lineage',
 '28': 'promonocyte',
 '29': 'thrombocyte aggregation',
 '30': 'unidentified',
 '31': 'normoblast',
 '32': 'prolymphocyte',
 '33': 'wbc',
 '34': 'rbc',
 '35': 'platelet',
 '36': 'lymphoblast',
 '37': 'neutrophil band',
 '38': 'babesia',
 '39': 'leishmania',
 '40': 'leukocyte',
 '41': 'plasmodium',
 '42': 'toxoplasma',
 '43': 'trichomonad',
 '44': 'trypanosome'
}
# -----------------------------
# TRANSFORM
# -----------------------------
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
tarnsform_seg= T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


def get_morphology_result(filtered_morphology):
    """
    filtered_morphology: torch.Tensor of shape [N, 6]
    Each row is one detection, each column is a morphology feature (0 or 1).
    Majority vote across all detections per feature.
    """
    result_list = []
    MORPHOLOGY_COLS = [
    'Nuclear Chromatin',
    'Nuclear Shape', 
    'Nucleus',
    'Cytoplasm',
    'Cytoplasmic Basophilia',
    'Cytoplasmic Vacuoles'
]

    # Label mappings for each morphology feature (0 vs 1)
    MORPHOLOGY_LABELS = {
        'Nuclear Chromatin':      {0: "open",          1: "Coarse"},
        'Nuclear Shape':          {0: "regular",        1: "irregular"},
        'Nucleus':                {0: "inconspicuous",  1: "prominent"},
        'Cytoplasm':              {0: "scanty",         1: "abundant"},
        'Cytoplasmic Basophilia': {0: "slight",         1: "moderate"},
        'Cytoplasmic Vacuoles':   {0: "absent",         1: "prominent"},
    }

    for col_idx, col_name in enumerate(MORPHOLOGY_COLS):
        col_values = filtered_morphology[:, col_idx]  # shape: [N]
        
        count_0 = (col_values == 0).sum().item()
        count_1 = (col_values == 1).sum().item()
        
        # Majority vote: 0 wins ties
        predicted_class = 0 if count_0 >= count_1 else 1
        result_list.append(MORPHOLOGY_LABELS[col_name][predicted_class])

    # Fill in the template
    text = (
        f"Mostly WBC's are, {result_list[0]} chromatin, "
        f"and {result_list[1]} shaped nuclei. "
        f"The nucleoli are {result_list[2]}, "
        f"and the cytoplasm is {result_list[3]} "
        f"with {result_list[4]} basophilia. "
        f"Cytoplasmic vacuoles are {result_list[5]}."
    )

    return result_list, text




# -----------------------------
# DRAW BOXES MANUALLY
# -----------------------------
# def draw_boxes(image, boxes, labels, scores, threshold=0.25):
#     image = image.copy()
#     draw = ImageDraw.Draw(image)

#     w, h = image.size

#     for box, label, score in zip(boxes, labels, scores):
#         if score < threshold:
#             continue

#         # box is cxcywh normalized → convert to xyxy in pixels
#         cx, cy, bw, bh = box
#         x1 = (cx - bw / 2) * w
#         y1 = (cy - bh / 2) * h
#         x2 = (cx + bw / 2) * w
#         y2 = (cy + bh / 2) * h

#         # Draw rectangle
#         draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

#         # Label
#         text = f"{label}: {score:.2f}"
#         draw.text((x1, y1), text, fill="red")

#     return image
def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


def draw_boxes(image, boxes, labels, scores, threshold=0.25, iou_thresh=0.5):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    w, h = image.size

    # Convert boxes (cxcywh normalized → xyxy pixel)
    boxes_xyxy = []
    for box in boxes:
        cx, cy, bw, bh = box
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes_xyxy.append([x1, y1, x2, y2])

    boxes_xyxy = np.array(boxes_xyxy)

    # Convert scores to numpy (safe)
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.array(scores)

    # Sort indices by score (descending)
    order = np.argsort(-scores_np)

    keep = np.ones(len(scores_np), dtype=bool)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    # 🔥 NMS-like suppression
    for i in range(len(order)):
        idx_i = order[i]

        if not keep[idx_i]:
            continue

        for j in range(i + 1, len(order)):
            idx_j = order[j]

            if not keep[idx_j]:
                continue

            iou = compute_iou(boxes_xyxy[idx_i], boxes_xyxy[idx_j])

            if iou > iou_thresh:
                # Suppress lower score box
                keep[idx_j] = False
                scores_np[idx_j] = 0.0  # 🔥 update score

    # 🔥 Draw only valid boxes
    for i in range(len(boxes_xyxy)):
        if not keep[i] or scores_np[i] < threshold:
            continue

        x1, y1, x2, y2 = boxes_xyxy[i]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text = f"{labels[i]}" #: {scores_np[i]:.2f}"
        draw.text((x1+5, y1+5), text, fill="red",font= font)

    # Convert scores back to torch (optional, keeps pipeline consistent)
    updated_scores = torch.tensor(scores_np)

    return image, updated_scores

# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def dino_inference(pil_image,task, prompt):
    try:
        image = pil_image.convert("RGB")
        w, h = image.size

        # Transform
        if task in ["Detection", "Detection with Morphology"]:
            image_tensor, _ = transform(image, None)
        else:
            image_tensor,_= tarnsform_seg(image, None)
        if task in ["Masked Language Modeling", "Question Answering"]:
            targets = [
                    {"completed_text": "cell A"}]
            outputs = model(image_tensor.unsqueeze(0).to(device), [prompt], targets)
        else:
        # Forward pass
            outputs = model(image_tensor.unsqueeze(0).to(device), [prompt])
        # pred_morphology = outputs['pred_morphology']  # Shape: [batch_size, num_queries, total_morphology_classes]
        threshold = 0.35
        # num_attributes = 6
        # num_classes_per_attribute = 2  # Valid labels are 0 and 1

        # pred_morphology = pred_morphology.view(
        #     pred_morphology.size(0),
        #     pred_morphology.size(1),
        #     num_attributes,
        #     num_classes_per_attribute
        # )

        # morphology_probs = F.softmax(pred_morphology, dim=-1)
        # pred_morphology_labels = morphology_probs.argmax(-1)
        # Post-process
        output = postprocessors['bbox'](
            outputs, torch.Tensor([[1.0, 1.0]]).to(device)
        )[0]
        

        scores = output['scores'].detach().cpu()
        labels = output['labels'].detach().cpu()
        boxes = output['boxes'].detach().cpu()
        # if outputs['morphology_labels']:
        
        # Convert labels to names
       

        
        label_names = [id2name[str(int(l))] for l in labels]

        # Convert boxes to cxcywh normalized
        boxes_cxcywh = box_ops.box_xyxy_to_cxcywh(boxes)

        # Draw boxes
        annotated_image,scores = draw_boxes(
            image,
            boxes_cxcywh,
            label_names,
            scores,threshold
        )

        # Output text
        num_detections = sum(scores > threshold)

        score_mask = scores > threshold
        if task == "Detection with Morphology":
            morphology_labels= output['morphology_labels'].detach().cpu()
            filtered_morphology = morphology_labels[score_mask] if morphology_labels is not None else None
        # if filtered_morphology is not None and filtered_morphology.shape[0] > 0:
            result_list, morphology_text = get_morphology_result(filtered_morphology)
            text_output = f"{num_detections} Cells Detected  and  {morphology_text}"
            output_image = annotated_image
        elif task == "Detection":
            morphology_text = "No morphology predictions available."
            result_list = []
            text_output = f"{num_detections} Cells Detected"
            output_image = annotated_image
        elif task == "Segmentation":
            pred_masks = outputs['pred_mask'] # shape: (B, 64, 64)

            pred_probs = torch.sigmoid(pred_masks)
                
                
            output_image = (pred_probs[0,0] * 255).cpu().detach().numpy().astype('uint8')  # scale 0-255
            # output_image = Image.fromarray(mask)

            text_output = f"Binary Segmented Image"
        elif task in ["Masked Language Modeling", "Question Answering"]:
            
            text_output = outputs['pred_text'] # shape: (B, 64, 64)
            
            output_image = image
        elif task in ["Classification"]:
            
            logits = outputs['pred_image_class']

            probs = F.softmax(logits, dim=1)
            pred_id = probs.argmax(dim=1).item()
            confidence = probs[0, pred_id].item()

            text_output = id2class[str(pred_id)] # shape: (B, 64, 64)

            output_image = image
        
        return output_image, text_output

    except Exception as e:
        return None, f"Error: {str(e)}"

# -----------------------------
# GRADIO UI
# -----------------------------
# with gr.Blocks() as demo:
#     gr.Markdown("# 🧠 DINO Object Detection (Inference Only)")

#     with gr.Row():
#         with gr.Column():
#             input_image = gr.Image(type="pil", label="Upload Image")
#             run_btn = gr.Button("Run Inference")

#         with gr.Column():
#             output_image = gr.Image(type="pil", label="Prediction Output")
#             output_text = gr.Textbox(label="Info")

#     run_btn.click(
#         fn=dino_inference,
#         inputs=input_image,
#         outputs=[output_image, output_text]
#     )

# demo.launch()
custom_css = """
/* General text */
body {
    color: #111827;
}

/* Labels (default styling) */
.gradio-container label {
    color: #111827 !important;
    font-weight: 600;
}

/* Component titles */
.gradio-container .label {
    color: #111827 !important;
    font-weight: 600;
}

/* Radio + textbox labels */
.gr-form .label, 
.gr-box .label {
    color: #111827 !important;
    font-weight: 600;
}

/* Textbox input text */
textarea, input {
    color: #111827 !important;
    font-weight: 600;
}

/* Placeholder text */
textarea::placeholder, input::placeholder {
    color: #6b7280 !important;
    font-weight: 600;
}

/* Radio text */
.gradio-radio label {
    color: #111827 !important;
    font-weight: 600;
}

/* ============================= */
/* 🔥 Highlight Select Task */
/* ============================= */
.select-task label {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: #ffffff !important;

    /* Typography */
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;

    /* Layout */
    padding: 8px 14px;
    border-radius: 10px;
    display: inline-block;

    /* Visual depth */
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);

    /* Smooth look */
    transition: all 0.2s ease-in-out;
}

/* ============================= */
/* 🔥 Highlight Response */
/* ============================= */
.response-box .wrap label {
    background: linear-gradient(90deg, #16a34a, #15803d);
    color: white !important;

    font-size: 18px !important;
    font-weight: 700 !important;

    padding: 6px 12px;
    border-radius: 8px;
    display: inline-block;
}

/* Response textbox styling */
.response-box textarea {
    border: 2px solid #16a34a !important;
    border-radius: 8px;
    padding: 10px;
}

/* Optional: nicer radio spacing */
.select-task .wrap {
    gap: 8px;
}

/* Optional: button styling */
button {
    background: linear-gradient(90deg, #111827, #374151);
    color: white;
    font-weight: 600;
    border-radius: 8px;
}

button:hover {
    background: linear-gradient(90deg, #374151, #111827);
}
"""
header = """
<div style="
    text-align: center;
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(90deg, #e9d5ff, #dbeafe);
    color: #1f2937;
    margin-bottom: 20px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
">
<h1 style="
    margin:0;
    font-size: 32px;
    font-weight: 800;
    color: #111827;
">
Unified Model For Hematopathology (CVPR-26)
</h1>

<p style="
    margin-top:8px;
    color:#374151;
    opacity:0.9;
">
Multi-task AI for hematology analysis
</p>
</div>
"""
abstract = """
<div class="card">

<div style="font-size: 18px; font-weight: 600; margin-bottom: 12px;">
🤗 Demo of 
<b style="color:#7c3aed;">Uni-Hema: A Unified Multi-Task Model for Hematology Analysis</b>
</div>

<p>
🧠 This system performs <b>detection, morphology, segmentation, classification, MLM, and QA</b> in a unified framework.
</p>

<p>
⚡ It leverages multimodal representations combining microscopy images and textual prompts.
</p>

<p>
🔬 Designed for blood cell analysis and disease understanding in hematopathology.
</p>

<p>
🧪 Input images should be microscopy images (e.g., 100× magnification).
</p>

</div>
"""
footer = """
<div class="card" style="margin-top:20px;">

<h3>🦁 Developed by</h3>
Intelligent Machines Lab, ITU Punjab  
<a href="https://iml.itu.edu.pk/" target="_blank">🔗 Website</a>

<hr>

<h3>🧪 Paper</h3>
<a href="https://arxiv.org/abs/2511.13889" target="_blank">📄 Uni-Hema Paper</a>

<hr>

<h3>⭐ GitHub</h3>
<a href="https://github.com/intelligentMachines-ITU/Uni-Hema" target="_blank">
Uni-Hema Repository
</a>

<hr>

<h3>📧 Contact</h3>
Abdul Rehman — phdcs23002@itu.edu.pk

<hr>

<h3>📝 Citation</h3>
<pre>
@article{rehman2025uni,
  title={Uni-Hema: Unified Model for Digital Hematopathology},
  author={Rehman, Abdul and Rasool, Iqra and Imran, Ayisha and Ali, Mohsen and Sultani, Waqas},
  journal={arXiv preprint arXiv:2511.13889},
  year={2025}
}
</pre>

</div>
"""


samples_map = {
    "Segmentation": cloze_samples_seg,
    "Detection": cloze_samples_det,
    "Detection with Morphology": cloze_samples_dm,
    "Classification": cloze_samples_cls,
    "Masked Language Modeling": cloze_samples_mlm,
    "Question Answering": cloze_samples_QA,
}

# -------------------------
# Toggle function
# -------------------------
def toggle_input(task):
    task_samples = samples_map.get(task, [])
    show_cloze = len(task_samples) > 0

    # Dataset format
    if task == "Question Answering":
        dataset_update = gr.update(
            samples=task_samples,
            components=["image", "text"]
        )
    else:
        dataset_update = gr.update(
            samples=task_samples,
            components=["image"]
        )

    container_update = gr.update(visible=show_cloze)

    # Text behavior
    if task in ["Detection", "Detection with Morphology", "Segmentation"]:
        text_update = gr.update(
            visible=True,
            interactive=False,
            label="Disease Prompt",
            placeholder="e.g. leukemia, malaria..."
        )

    elif task == "Question Answering":
        text_update = gr.update(
            visible=True,
            interactive=False,
            label="Question"
        )

    elif task == "Masked Language Modeling":
        text_update = gr.update(
            visible=True,
            interactive=False,   # 🔒 fixed masked text
            label="Masked Input"
        )

    elif task == "Classification":
        text_update = gr.update(visible=False)

    else:
        text_update = gr.update(visible=True)

    return text_update, container_update, dataset_update


# -------------------------
# SAMPLE CLICK HANDLER
# -------------------------
def set_cloze_samples(example, task):
    if task in ["Question Answering", "Masked Language Modeling"]:
        return example[0], example[1]
    else:
        return example[0], ""


# -------------------------
# Dummy inference
# -------------------------



# -------------------------
# UI
# -------------------------
with gr.Blocks() as demo:

    gr.Markdown(header)
    gr.Markdown(abstract)

    with gr.Row():

        # -------- LEFT --------
        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="Upload Image 📤",
                height=320,
                width=320
            )

            task_selector = gr.Radio(
                choices=[
                    "Detection",
                    "Detection with Morphology",
                    "Segmentation",
                    "Classification",
                    "Masked Language Modeling",
                    "Question Answering"
                ],
                value="Detection",
                label="Select Task"
            )

            text_input = gr.Textbox(
                label="Input",
                placeholder="Enter disease / question...",
                visible=True
            )

            run_btn = gr.Button("Run Inference")

        # -------- RIGHT --------
        with gr.Column():

            output_image = gr.Image(
                type="pil",
                label="Prediction Output 🧾",
                height=480,
                width=480
            )

            output_text = gr.Textbox(label="Response")

            with gr.Column(visible=False) as cloze_container:
                cloze_examples = gr.Dataset(
                    label="Samples",
                    components=["image"],   # default
                    samples=[]
                )

    # ---------------- EVENTS ----------------

    # Task change
    task_selector.change(
        fn=toggle_input,
        inputs=task_selector,
        outputs=[text_input, cloze_container, cloze_examples]
    )

    # Run inference
    run_btn.click(
        fn=dino_inference,
        inputs=[input_image, task_selector, text_input],
        outputs=[output_image, output_text]
    )

    # Sample click
    cloze_examples.click(
        fn=set_cloze_samples,
        inputs=[cloze_examples, task_selector],
        outputs=[input_image, text_input]
    )
    gr.Markdown(footer)

# -------------------------
# Launch
# -------------------------

demo.launch()