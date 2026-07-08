# Uni-Hema: Unified Model for Digital Hematopathology
![architecture_AttriDet](models/1771700412172.jpeg)

**Authors:** Abdul Rehman, Iqra Rasool, Ayisha Imran, Mohsen Ali, Waqas Sultani


**CVPR 2026**
### Paper

📄 **PDF:** [Uni-Hema: Unified Model for Digital Hematopathology](https://openaccess.thecvf.com/content/CVPR2026/papers/Rehman_Uni-Hema_Unified_Model_for_Digital_Hematopathology_CVPR_2026_paper.pdf)

---

## Overview

Uni-Hema is a unified multi-task, multi-modal framework for comprehensive cell-level analysis across hematological diseases.  
It integrates detection, classification, segmentation, morphology prediction, and clinical reasoning — trained on 46 public datasets with 700K+ microscopy images.

---
## Installation

<details>
  <summary>Installation</summary>
  
  We use the same environment as DAB-DETR, DN-DETR, and DINO. If you have run DN-DETR or DAB-DETR, you can skip this step. 
  We test our models under ```python=3.7.3,pytorch=1.9.0,cuda=11.1```. Other versions might be available as well. Click the `Details` below for more details.

   1. Clone this repo
   ```sh
   git clone https://github.com/intelligentMachines-ITU/Uni-Hema.git
   cd Uni-Hema
   ```

   2. Install PyTorch and torchvision

   Follow the instructions on https://pytorch.org/get-started/locally/.
   ```sh
   # an example:
   conda install -c pytorch pytorch torchvision
   ```

   3. Install other needed packages
   ```sh
   pip install -r requirements.txt
   ```

   4. Compiling CUDA operators
   ```sh
   cd models/dino/ops
   python setup.py build install
   # unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```
</details>
  
## Demosntrarion  
Demonstration can be performed by running the following bash script:


```
python app.py
 
```

---
Step1: Select the task (detection, Segmentation, Classification, VQA )

Step2: Select the image or upload 

Step3: Run the Inference 

## Dataset  


| Task | Dataset | Dataset Link | Paper Link |
|------|---------|--------------|------------|
| **Segmentation** | Malaria-Detection-2019 | [Link](https://data.mendeley.com/public-api/zip/5bf2kmwvfn/download/1) | [Link](https://link.springer.com/article/10.1186/s13000-020-01040-9) |
|  | NuClick | [Link](https://github.com/navidstuv/NuClick) | [Link](https://arxiv.org/pdf/2005.14511.pdf) |
|  | KRD-WBC | [Link](https://data.mendeley.com/datasets/jzdj6h7gms/2) | [Link](https://data.mendeley.com/datasets/jzdj6h7gms/2) |
|  | WBC Image Dataset | [Link](https://github.com/zxaoyou/segmentation_WBC) | [Link](https://doi.org/10.1016/j.micron.2018.01.010) |
|  | White Blood Cell Dataset | [Link](https://www.raabindata.com/free-data/) | [Link](https://www.researchgate.net/profile/Amr-Guaily/publication/261072982_An_efficient_technique_for_white_blood_cells_nuclei_automatic_segmentation/links/56160c9908ae983c1b42476f/An-efficient-technique-for-white-blood-cells-nuclei-automatic-segmentation.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19) |
|  | ErythrocytesIDB | [Link](https://doi.org/10.5281/zenodo.18299473) | [Link](https://www.researchgate.net/publication/265607505_Red_Blood_Cell_Cluster_Separation_From_Digital_Images_for_Use_in_Sickle_Cell_Disease) |
|  | AneRBC-II-Anemic | [Link](https://data.mendeley.com/public-api/zip/hms3sjzt7f/download/1) | — |
|  | AneRBC-II-Healthy | [Link](https://data.mendeley.com/public-api/zip/hms3sjzt7f/download/1) | — |
|  | MP-IDB | [Link](https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis) | [Link](https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis) |
|  | Elsafty-RBC for AI | [Link](https://figshare.com/ndownloader/files/43957974) | [Link](https://www.researchgate.net/journal/Scientific-Data-2052-4463/publication/381921572_1_Million_Segmented_Red_Blood_Cells_With_240_K_Classified_in_9_Shapes_and_47_K_Patches_of_25_Manual_Blood_Smears/links/6684c6772aa57f3b8268db49/1-Million-Segmented-Red-Blood-Cells-With-240-K-Classified-in-9-Shapes-and-47-K-Patches-of-25-Manual-Blood-Smears.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InNpZ251cCIsInBhZ2UiOiJwdWJsaWNhdGlvbiJ9fQ) |
|  | BBBC041Seg | [Link](https://datasetninja.com/blood-cell-segmentation) | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0040816621001695) |
|  | SSL Seg | [Link](https://github.com/zxaoyou/segmentation_WBC) | [Link](#) |
| **Detection** | LeukemiaAttri | [Link](https://github.com/intelligentMachines-ITU/Blood-Cancer-Dataset-Lukemia-Attri-MICCAI-2024) | [Link](https://papers.miccai.org/miccai-2024/paper/4180_paper.pdf/) |
|  | M5 | [Link](https://github.com/intelligentMachines-ITU/LowCostMalariaDetection_CVPR_2022) | [Link](https://im.itu.edu.pk/m5-malaria-dataset/) |
|  | TXL PBC | — | — |
|  | BCCD | [Link](https://datasetninja.com/bccd#download) | [Link](#) |
|  | Sickle-cell | [Link](#) | [Link](https://www.researchgate.net/profile/Florence-Tushabe-2/publication/385820893_An_Image-based_Sickle_Cell_Detection_Method/links/6737306369c07a411447b951/An-Image-based-Sickle-Cell-Detection-Method.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19) |
|  | Plasmodium | — | — |
|  | Plasmodium Phonecamera | — | — |
|  | Tuberculosis Phonecamera | — | — |
|  | Vivax | [Link](https://datasetninja.com/malaria-bounding-boxes) | [Link](https://www.researchgate.net/profile/Anne-Carpenter-3/publication/324769547_Applying_Faster_R-CNN_for_Object_Detection_on_Malaria_Images/links/5b8594fb299bf1d5a72e9cea/Applying-Faster-R-CNN-for-Object-Detection-on-Malaria-Images.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19) |
|  | ThickBloodSmears | [Link](https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html) | [Link](https://www.researchgate.net/profile/Stefan-Jaeger-4/publication/336011001_Deep_Learning_for_Smartphone-Based_Malaria_Parasite_Detection_in_Thick_Blood_Smears/links/5dc98ca5299bf1a47b2f9d83/Deep-Learning-for-Smartphone-Based-Malaria-Parasite-Detection-in-Thick-Blood-Smears.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19) |
|  | NIH-NLM-Thick PV | [Link](https://data.lhncbc.nlm.nih.gov/public/Malaria/Thick_Smears_150/index.html) | [Link](#) |
|  | Parasite | [Link](https://www.researchgate.net/journal/Diagnostics-2075-4418/publication/355707523_Diagnosing_Malaria_Patients_with_Plasmodium_falciparum_and_vivax_Using_Deep_Learning_for_Thick_Smear_Images/links/68038a35bd3f1930dd6021cb/Diagnosing-Malaria-Patients-with-Plasmodium-falciparum-and-vivax-Using-Deep-Learning-for-Thick-Smear-Images.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19) | — |
|  | Acevedo | [Link](https://data.mendeley.com/datasets/snkd93bnjr/draft?a=d9582c71-9af0-4e59-9062-df30df05a121) | [Link](https://www.researchgate.net/publication/340522931_A_dataset_of_microscopic_peripheral_blood_cell_images_for_development_of_automatic_recognition_systems/fulltext/5e8e988fa6fdcca78901f7cd/A-dataset-of-microscopic-peripheral-blood-cell-images-for-development-of-automatic-recognition-systems.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19) |
|  | MP-IDB | [Link](https://github.com/andrealoddo/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis) | [Link](https://www.researchgate.net/profile/Andrea-Loddo-3/publication/331570908_MP-IDB_The_Malaria_Parasite_Image_Database_for_Image_Processing_and_Analysis/links/5ca2206f92851cf0aea6506c/MP-IDB-The-Malaria-Parasite-Image-Database-for-Image-Processing-and-Analysis.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InNpZ251cCIsInBhZ2UiOiJwdWJsaWNhdGlvbiJ9fQ) |
|  | Bio-Net | [Link](https://www.sciencedirect.com/science/article/abs/pii/S1079979624000019) | [Link](https://doi.org/10.1016/j.bcmd.2024.102823) |
|  | Raabin-M1 | [Link](https://universe.roboflow.com/raabintest/raabin-wbc-data) | [Link](https://www.researchgate.net/publication/353163296_Raabin-WBC_a_large_free_access_dataset_of_white_blood_cells_from_normal_peripheral_blood/fulltext/60ea715c1c28af34585e73af/Raabin-WBC-a-large-free-access-dataset-of-white-blood-cells-from-normal-peripheral-blood.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19) |
|  | Raabin-M2 | [Link](https://universe.roboflow.com/raabintest/raabin-wbc-data) | — |
| **Classification** | BMC | [Link](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101941770) | [Link](https://ashpublications.org/blood/article/138/20/1917/477932/Highly-accurate-differentiation-of-bone-marrow) |
|  | AML Matek | [Link](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/) | [Link](https://doi.org/10.7937/tcia.2019.36f5o9ld) |
|  | Raabin WBC | [Link](https://raabindata.com/free-data/) | [Link](https://www.nature.com/articles/s41598-021-04426-x) |
|  | Warty Pig | [Link](https://drive.google.com/drive/folders/1CsDoL448kvAtFVd5jowVJGKjFLv3qjz4) | [Link](https://ieee-dataport.org/documents/dataset-machine-learning-based-classification-white-blood-cells-juvenile-visayan-warty-pig) |
|  | LISC | [Link](http://users.cecs.anu.edu.au/~hrezatofighi/Data/Leukocyte%20Data.htm) | [Link](https://pubmed.ncbi.nlm.nih.gov/21300521/) |
|  | KRD-WBC | [Link](https://data.mendeley.com/datasets/jzdj6h7gms/2) | [Link](https://data.mendeley.com/datasets/jzdj6h7gms/2) |
|  | BCCD | [Link](https://www.kaggle.com/datasets/konstantinazov/bccd-dataset) | [Link](https://www.researchgate.net/publication/261072982_An_efficient_technique_for_white_blood_cells_nuclei_automatic_segmentation) |
|  | HRLS | — | — |
|  | APL_AML | [Link](https://www.kaggle.com/datasets/eugeneshenderov/acute-promyelocytic-leukemia-apl/data) | [Link](https://pubmed.ncbi.nlm.nih.gov/33990660/) |
|  | White-Blood-Cell-Dataset | [Link]([#](https://github.com/arbackes/White-Blood-Cell-dataset)) | [Link]([#](https://ieeexplore.ieee.org/abstract/document/6379408/)) |
|  | Acevedo | [Link](https://data.mendeley.com/public-api/zip/snkd93bnjr/download/1) | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0169260719303578?via%3Dihub) |
|  | RV PBS | [Link](https://figshare.com/ndownloader/articles/23804523/versions/2) | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0957417424005268) |
|  | PBC-8-DA | [Link]([#](https://figshare.com/ndownloader/articles/23804523/versions/2)) | — |
|  | C-NMC2019 | [Link](https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjczNCIsInBhc3Njb2RlIjoiNDM3ZmMzM2RkMzQ1ZmMzZjNjM2FlY2JmZWQ0MThlY2NjYTkzM2RmMiIsInBhY2thZ2VfaWQiOiI3MzQiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0=) | — |
|  | BloodMNIST | [Link](https://zenodo.org/records/5208230) | [Link](https://www.nature.com/articles/s41597-022-01721-8) |
|  | AML Hehr | [Link](https://www.cancerimagingarchive.net/collection/aml-cytomorphology_mll_helmholtz/) | [Link](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000187) |


1: All the datasets are publicly available, and the details are given in the supplementary. 
2: The annotations of all the training and testing datasets are available in the dataset_annotation folder. 

## Single Cell linear classifier model training and testing   
```
Set the train and test paths w.r.t the single-cell datasets
python uni_hema_Scc.py

```
## Traning 
```
python main_unihema.py \
    -c config/DINO/DINO_4scale.py \
    --output_dir ./step2/ \
    --coco_path ./  \
    --options embed_init_tgt=TRUE
```
---
## Citation

```bibtex
@inproceedings{rehman2026uni,
  title={Uni-Hema: Unified Model for Digital Hematopathology},
  author={Rehman, Abdul and Rasool, Iqra and Imran, Ayisha and Ali, Mohsen and Sultani, Waqas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={37578--37589},
  year={2026}
}
