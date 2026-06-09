import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from main import build_model_main
from util.slconfig import SLConfig
import dino_datasets.transforms as T

# === CONFIG ===
# train_json = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/BM_cytomorphology_data_train.json"
# test_json  = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/BM_cytomorphology_data_test.json"
root_images = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification"  # where file_name in JSON is relative to BMCD_FGCD_train.json/'
# root_images = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/bloodmnist_224/"
# train_json = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/BMCD_FGCD_train.json"
# test_json  = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/BMCD_FGCD_test.json"
# train_json = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/bloodmnist_224/mnist_train_annotations.json"
# test_json  = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/bloodmnist_224/mnist_test_annotations.json"
# root_images = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/bloodmnist_224/"
# train_json = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/PKG - C-NMC 2019/c_nmc_all_folds_train.json"
# test_json  = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/PKG - C-NMC 2019/c_nmc_all_folds_test.json"
# root_images = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/PKG - C-NMC 2019"
train_json = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/raabin_Train_update.json"
test_json  = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/Raabin_testA_updated.json"
# /media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/PKG - C-NMC 2019/c_nmc_all_folds_train.json
# train_json = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/train_Acevedo_update_20.json"
# test_json  = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/test_Acevedo_update_20.json"
# /media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/PKG - C-NMC 2019/c_nmc_all_folds_train.json
# train_json = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/acevedo_2_class/train_Acevedo_update_2_class.json"
# test_json  = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/acevedo_2_class/test_Acevedo_update_2_class.json"
# root_images = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/"
# train_json= "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/Parasite Data Set_train.json"
# test_json= "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/annotations/classification/Parasite Data Set_test.json"
#root_images = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/"
# train_json= "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/RV_Pbs/classification_data/train_annotations.json"
# test_json = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/RV_Pbs/classification_data/test_annotations.json"
# root_images = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/"
# train_json = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/RV_Pbs/PBC_8_DA/train_annotations.json"
# test_json = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/RV_Pbs/PBC_8_DA/test_annotations.json"
# root_images = "/media/iml_abdul/40a71b91-5aab-40ee-a60a-16e92f17d93f/home/iml/DINO/coco_data/merge_train/classification/Dunseen_data/"
model_config_path = "/home/iml_abdul/Uni_hema/check_step4_det/config_args_all.json"
model_checkpoint_path = "/home/iml_abdul/Uni_hema/check_step4_det/checkpoint0005.pth"
prompt_text = (["myeloblast"],)   # adjust if needed


input_dim = 368  # should match encoder_class_feat dim
hidden_dim = 256
epochs = 10
lr = 1e-5
batch_size = 8

# === LOAD MODEL ===
print("🚀 Loading detection/feature model...")
args = SLConfig.fromfile(model_config_path)
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
# Count backbone parameters
backbone_params = sum(p.numel() for p in model.backbone.parameters())
print(f"Backbone parameters: {backbone_params:,}")

# Count encoder parameters
encoder_params = sum(p.numel() for p in model.transformer.encoder.parameters())
print(f"Encoder parameters: {encoder_params:,}")
model.cuda().eval()


print("✅ Model loaded.")

# === PARSE COCO JSON ===
def load_coco(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    id2file = {img["id"]: img["file_name"] for img in data["images"]}
    samples = []
    for ann in data["annotations"]:
        file_name = id2file[ann["image_id"]]
        full_path = os.path.join(root_images, file_name)
        label = ann["class_name"]
        samples.append((full_path, label))
    return samples

train_samples = load_coco(train_json)
test_samples  = load_coco(test_json)

# === Label to index mapping ===
all_labels = sorted(set(label for _, label in train_samples + test_samples))
label2idx = {label: idx for idx, label in enumerate(all_labels)}
num_classes = len(label2idx)
print(f"✅ Found {num_classes} classes.")

# === TRANSFORM ===
transform = T.Compose([
    T.RandomResize([(512, 512)]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === DATASET ===
class DinoFeatureDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except:
            print(f"⚠️ Could not load image: {path}")
            image = Image.new("RGB", (256,256))
        img_t, _ = transform(image, None)
        return img_t, label2idx[label]

train_ds = DinoFeatureDataset(train_samples)
test_ds  = DinoFeatureDataset(test_samples)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# === CLASSIFIER ===
# class ImageClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, output_dim)
#         )
#     def forward(self, x):
#         return self.net(x)
from sklearn.linear_model import LogisticRegression

# === EXTRACT TRAIN FEATURES ===
train_feats, train_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(train_loader, desc="Extracting train features"):
        imgs = imgs.cuda()
        prompt_text_batch = prompt_text * imgs.shape[0]
        output = model(imgs, prompt_text_batch)
        feats = output["encoder_class_feat"].squeeze()
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        train_feats.append(feats.cpu().numpy())
        train_labels.append(labels.numpy())

train_feats = np.concatenate(train_feats, axis=0)
train_labels = np.concatenate(train_labels, axis=0)

# === EXTRACT TEST FEATURES ===
test_feats, test_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Extracting test features"):
        imgs = imgs.cuda()
        prompt_text_batch = prompt_text * imgs.shape[0]
        output = model(imgs, prompt_text_batch)
        feats = output["encoder_class_feat"].squeeze()
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        test_feats.append(feats.cpu().numpy())
        test_labels.append(labels.numpy())

test_feats = np.concatenate(test_feats, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

# === CALCULATE C VALUE ===
n = train_feats.shape[0]     # number of training samples
c = num_classes              # number of classes
C_value = (c * n) / 100
print(f"🔧 LogisticRegression C value = {C_value}")

# === TRAIN LINEAR PROBE ===
clf = LogisticRegression(
    penalty='l2',
    C=C_value,
    solver='lbfgs',
    multi_class='multinomial',
    max_iter=2000,
    # n_jobs=-1
)
clf.fit(train_feats, train_labels)

# === EVALUATE ===
preds = clf.predict(test_feats)
acc = accuracy_score(test_labels, preds)
macro = f1_score(test_labels, preds, average='macro')
weighted = f1_score(test_labels, preds, average='weighted')

print(f"[Test] Acc={acc:.4f} MacroF1={macro:.4f} WeightedF1={weighted:.4f}")
print("✅ Linear probe evaluation complete.")
