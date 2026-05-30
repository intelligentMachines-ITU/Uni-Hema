🧬 Uni-Hema
A Unified Model for Digital Hematopathology

CVPR 2026

---

## Overview

Uni-Hema is a unified multi-task, multi-modal framework for comprehensive cell-level analysis across hematological diseases.  
It integrates detection, classification, segmentation, morphology prediction, and clinical reasoning — trained on 46 public datasets with 700K+ microscopy images.

---
## Installation

<details>
  <summary>Installation</summary>
  
  We use the environment same to DAB-DETR and DN-DETR to run DINO. If you have run DN-DETR or DAB-DETR, you can skip this step. 
  We test our models under ```python=3.7.3,pytorch=1.9.0,cuda=11.1```. Other versions might be available as well. Click the `Details` below for more details.

   1. Clone this repo
   ```sh
   git clone https://github.com/intelligentMachines-ITU/Uni-Hema.git
   cd DINO
   ```

   2. Install Pytorch and torchvision

   Follow the instruction on https://pytorch.org/get-started/locally/.
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
## Paper

📌 arXiv Preprint: https://arxiv.org/abs/2511.13889  
📌 Conference: CVPR 2026

---

## Citation

```bibtex
@article{rehman2025unihema,
  title   = {Uni-Hema: Unified Model for Digital Hematopathology},
  author  = {Rehman, Abdul and Rasool, Iqra and Imran, Ayisha and Ali, Mohsen and Sultani, Waqas},
  journal = {arXiv preprint arXiv:2511.13889},
  year    = {2025}
}
