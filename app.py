import os
import gradio as gr
# from app_util import ContextDetDemo
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils.my_model import MyCNN
from models.common import DetectMultiBackend
import numpy as np
import csv
import torch.nn.functional as F
from PIL import Image, ImageOps
from utils.augmentations import letterbox
from utils.general import (scale_boxes, non_max_suppression)
import pandas as pd
import os

from torchvision.ops import roi_align
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,get_fixed_xyxy)
# Initialize Model with Error Handling
# try:
    # model = DetectMultiBackend('best.pt')
    # model = DetectMultiBackend('best.pt')
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cell_attribute_model= MyCNN(num_classes=12, dropout_prob=0.5, in_channels=480).cpu()
# folder_name = '/home/iml1/AR/Sparse_Det_TMI/Attribute_model'
custom_weights_path = f"Attridet_weight/Attrihead_hcm_100x.pth"
custom_weights = torch.load(custom_weights_path,map_location=torch.device('cpu'))
cell_attribute_model.load_state_dict(custom_weights)
cell_attribute_model.eval().to(device)

model = DetectMultiBackend('Attridet_weight/last_300e_100x.pt')
# except Exception as e:
    # print(f"Error loading model: {e}")

header = """
<div align=center>
<h1 style="font-weight: 900; margin-bottom: 7px;">
Leukemia Detection with Morphology Attributes
</h1>
</div>
"""

abstract = """
🤗 This is the demo of the Paper  <b> Leveraging Sparse Annotations for Leukemia Diagnosis on the Large Leukemia Dataset</b>.

🆒 Our goal is to detect infected cells with Morphology for the bettre diagnosis explainabilty.

⚡ For faster inference, you may duplicate the space and use the GPU setting.

🧪 Note : Image size: 640×640 pixels, captured using a 100x microscope lens..
"""

footer = r"""
## 🦁 Developed by  
***Intelligent Machines Lab***, Information Technology University of Punjab  
<a href="https://im.itu.edu.pk/" target="_blank">🔗 website</a>  

## 🧪 Demo Paper  
Our demo paper is available at: Leveraging Sparse Annotations for Leukemia Diagnosis on the Large Leukemia Dataset
<a href="" target="_blank">📄 arXiv:2405.10803</a>  

## 🦁 Github Repository  
We would be grateful if you consider starring our  
<a href="https://github.com/intelligentMachines-ITU/Blood-Cancer-Dataset-Lukemia-Attri-MICCAI-2024" target="_blank">⭐ Blood Cancer Dataset Repository</a>  

## 🦁 Contact
If you have any questions, please feel free to contact Abdul Rehman <b>(phdcs23002@itu.edu.pk)</b>.

## 📝 Citation  
```bibtex
@inproceedings{rehman2025leveraging,
  title={Leveraging Sparse Annotations for Leukemia Diagnosis on the Large Leukemia Dataset},
  author={Rehman, Abdul and Meraj, Talha and Minhas, Aiman Mahmood and Imran, Ayisha and Ali, Mohsen and Sultani, Waqas and Shah, Mubarak},
  booktitle={},
  pages={},
  year={2025},
  organization={Springer}
}


"""

css = """
h1#title {
  text-align: right;
}
"""
cloze_samples = [
    ["sample/18_33_1000_ALL.png"],
    ["sample/8_18_1000_ALL.png"],
    ["sample/15_20_1000_AML.png"],
    ["sample/21_32_1000_CLL.png"],
    ["sample/28_24_1000_CML.png"],
    ["sample/31_23_1000_CML.png"],
    ["sample/31_34_1000_CML.png"],
    ["sample/23_40_1000_APML.png"],
]



def capture_image(pil_img):
        # if self.session_started:
        #     slide_number = self.slide_number_entry.text().strip()
        #     if slide_number:
                
        #         self.slide_dir = os.path.join(os.getcwd(), slide_number)
        #         # print(slide_dir)
                # image_path = os.path.join(self.slide_dir, f"image_{self.image_counter}.png")
                # ret, frame = self.camera.read()
                
                
                # self.image_counter_label.setText(f"{self.image_counter}")
                # cv2.imwrite(image_path, frame)
                
                conf_thres=0.1
                iou_thres=0.45
                max_det=1000
                hide_labels=False
                hide_conf=False
                
                all_predictions = []
                # pil_img = Image.fromarray(frame)
                image = pil_img.resize((640,640), Image.LANCZOS)
                im0 = np.array(image)
                annotated_img= im0
                filled_text= "White blood cells are not presented in the image."

                
                im = letterbox(im0, 640, 32, auto=True)[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(im)
                img= torch.from_numpy(img)
             
                
                
                

                # transform = transforms.Compose([
                # transforms.ToPILImage(),  # Convert numpy array to PIL Image
                # transforms.Resize((640, 640)),  # Resize image
                # transforms.ToTensor(),  # Convert PIL Image to tensor
                # # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
                # ])
                #  # Add batch dimension

                # # Inference
                # # pred, int_feats = model(img, augment=False, visualize=False)
                # frame=transform(frame)
                img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
                img /= 255

                # Inference
                img=img.unsqueeze(0)
                pred, int_feats,_ = model(img, augment=False, visualize=False)


                #attri

                int_feats_p2 = int_feats[0][0].to(torch.float32).unsqueeze(0)
                int_feats_p3 = int_feats[1][0].to(torch.float32).unsqueeze(0)
                in_channels = int_feats_p2.shape[1]+int_feats_p3.shape[1]
                

























                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)


                if (len(pred[0])>0):
                    all_top_indices_cell_pred = []
                    top_indices_cell_pred = []
                    pred_Nuclear_Chromatin_array = []
                    pred_Nuclear_Shape_array = []
                    pred_Nucleus_array = []
                    pred_Cytoplasm_array = []
                    pred_Cytoplasmic_Basophilia_array = []
                    pred_Cytoplasmic_Vacuoles_array = []

                    for i in range(len(pred[0])):
                        # if pred[0][i].numel() > 0:  # Check if the tensor is not empty

                            pred_tensor = pred[0][i][0:4]
                            
                            if pred[0][i][5] != 0:
                                
                                img_shape_tensor = torch.tensor([img.shape[2], img.shape[3],img.shape[2],img.shape[3]]).to(device)

                                normalized_xyxy=pred_tensor.to(device) / img_shape_tensor
                                p2_feature_shape_tensor = torch.tensor([int_feats[0].shape[1], int_feats[0].shape[2],int_feats[0].shape[1],int_feats[0].shape[2]]).to(device)                        # reduce_channels_layer = torch.nn.Conv2d(1280, 250, kernel_size=1).to(device)
                                p3_feature_shape_tensor = torch.tensor([int_feats[1].shape[1], int_feats[1].shape[2],int_feats[1].shape[1],int_feats[1].shape[2]]).to(device)             # reduce_channels_layer = torch.nn.Conv2d(1280, 250, kernel_size=1).to(device)
                            
                            
                                p2_normalized_xyxy = normalized_xyxy*p2_feature_shape_tensor
                                p3_normalized_xyxy = normalized_xyxy*p3_feature_shape_tensor
                                p2_x_min, p2_y_min, p2_x_max, p2_y_max = get_fixed_xyxy(p2_normalized_xyxy,int_feats_p2)
                                p3_x_min, p3_y_min, p3_x_max, p3_y_max = get_fixed_xyxy(p3_normalized_xyxy,int_feats_p3)

                                p2_roi = torch.tensor([p2_x_min, p2_y_min, p2_x_max, p2_y_max], device=device).float() 
                                p3_roi = torch.tensor([p3_x_min, p3_y_min, p3_x_max, p3_y_max], device=device).float() 

                                batch_index = torch.tensor([0], dtype=torch.float32, device = device)

                                # Concatenate the batch index to the bounding box coordinates
                                p2_roi_with_batch_index = torch.cat([batch_index, p2_roi])
                                p3_roi_with_batch_index = torch.cat([batch_index, p3_roi])
                                p2_resized_object = roi_align(int_feats_p2.to(device), p2_roi_with_batch_index.unsqueeze(0).to(device), output_size=(24, 30))
                                p3_resized_object = roi_align(int_feats_p3.to(device), p3_roi_with_batch_index.unsqueeze(0).to(device), output_size=(24, 30))
                                concat_box = torch.cat([p2_resized_object,p3_resized_object],dim=1)

                                output_cell_prediction= cell_attribute_model(concat_box)
                                output_cell_prediction_prob = F.softmax(output_cell_prediction.view(6,2), dim=1)
                                top_indices_cell_pred = torch.argmax(output_cell_prediction_prob, dim=1)
                                pred_Nuclear_Chromatin_array.append(top_indices_cell_pred[0].item())
                                pred_Nuclear_Shape_array.append(top_indices_cell_pred[1].item())
                                pred_Nucleus_array.append(top_indices_cell_pred[2].item())
                                pred_Cytoplasm_array.append(top_indices_cell_pred[3].item())
                                pred_Cytoplasmic_Basophilia_array.append(top_indices_cell_pred[4].item())
                                pred_Cytoplasmic_Vacuoles_array.append(top_indices_cell_pred[5].item())
                            # all_top_indices_cell_pred.append(top_indices_cell_pred.item())
                            else:
                                # top_indices_cell_pred = torch.tensor([0,0,0,0,0,0]).to(device)
                                pred_Nuclear_Chromatin_array.append(4)
                                pred_Nuclear_Shape_array.append(4)
                                pred_Nucleus_array.append(4)
                                pred_Cytoplasm_array.append(4)
                                pred_Cytoplasmic_Basophilia_array.append(4)
                                pred_Cytoplasmic_Vacuoles_array.append(4)




                    # Second-stage classifier (optional)
                    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                    # Define the path for the CSV file
                    df_predictions = pd.DataFrame(columns=['Image Name', 'Prediction', 'Confidence', 'Nuclear Chromatin',
                                        'Nuclear Shape', 'Nucleus', 'Cytoplasm', 'Cytoplasmic Basophilia',
                                        'Cytoplasmic Vacuoles', 'x_min', 'y_min', 'x_max', 'y_max'])

                    # Function to add data to the DataFrame and plot labels
                    def write_to_dataframe(img, name, predicts, confid, pred_NC, pred_NS, 
                                                    pred_N, pred_C, pred_CB, pred_CV,
                                                    x_min, y_min, x_max, y_max):
                        # global df_predictions

                        new_data = pd.DataFrame([{
                            'Image Name': name,
                            'Prediction': predicts,
                            'Confidence': confid,
                            'Nuclear Chromatin': pred_NC,
                            'Nuclear Shape': pred_NS,
                            'Nucleus': pred_N,
                            'Cytoplasm': pred_C,
                            'Cytoplasmic Basophilia': pred_CB,
                            'Cytoplasmic Vacuoles': pred_CV,
                            'x_min': x_min,
                            'y_min': y_min,
                            'x_max': x_max,
                            'y_max': y_max
                        }])

                        # df_predictions = pd.concat([df_predictions, new_data], ignore_index=True)

                        # Draw bounding box and label
                        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        # cv2.putText(img, predicts, (x_min, y_min - 10),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        return new_data

                    names = ["Unidentified", "Myeloblast", "Lymphoblast", "Neutrophil", "Atypical lymphocyte", 
                            "Promonocyte", "Monoblast", "Lymphocyte", "Myelocyte", "Abnormal promyelocyte", 
                            "Monocyte", "Metamyelocyte", "Eosinophil", "Basophil"]

                    # Process predictions
                    for i, det in enumerate(pred):  # per image

                        # img = cv2.imread("image.png")  # Load the image

                        for count, (*xyxy, conf, cls) in enumerate(det):
                            c = int(cls)  # integer class
                            label = names[c]
                            confidence = float(conf)
                            confidence_str = f'{confidence:.2f}'

                            x_min, y_min, x_max, y_max = xyxy
                            new_data_update = write_to_dataframe (im0 , "image.png", label, confidence_str,
                                                            pred_Nuclear_Chromatin_array[count],
                                                            pred_Nuclear_Shape_array[count],
                                                            pred_Nucleus_array[count],
                                                            pred_Cytoplasm_array[count],
                                                            pred_Cytoplasmic_Basophilia_array[count],
                                                            pred_Cytoplasmic_Vacuoles_array[count],
                                                            int(x_min.detach().cpu().item()),
                                                            int(y_min.detach().cpu().item()),
                                                            int(x_max.detach().cpu().item()),
                                                            int(y_max.detach().cpu().item()))
                            df_predictions = pd.concat([df_predictions, new_data_update], ignore_index=True)
                            
                        # Save or display the result
                        # cv2.imwrite("annotated_image.png", img)
                        # cv2.imshow("Annotated Image", img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                    # Optionally, display or export the DataFrame
                    result_list = []

                    # Conditions for each column
                    result_list.append("open" if (df_predictions['Nuclear Chromatin'] == 0).sum() > (df_predictions['Nuclear Chromatin'] == 1).sum() else "Coarse")
                    result_list.append("regular" if (df_predictions['Nuclear Shape'] == 0).sum() > (df_predictions['Nuclear Shape'] == 1).sum() else "irregular")
                    result_list.append("inconspicuous" if (df_predictions['Nucleus'] == 0).sum() > (df_predictions['Nucleus'] == 1).sum() else "prominent")
                    result_list.append("scanty" if (df_predictions['Cytoplasm'] == 0).sum() > (df_predictions['Cytoplasm'] == 1).sum() else "abundant")
                    result_list.append("slight" if (df_predictions['Cytoplasmic Basophilia'] == 0).sum() > (df_predictions['Cytoplasmic Basophilia'] == 1).sum() else "moderate")
                    result_list.append("absent" if (df_predictions['Cytoplasmic Vacuoles'] == 0).sum() > (df_predictions['Cytoplasmic Vacuoles'] == 1).sum() else "prominent")
                    # Sample text with <mask> placeholders
                    text = """These WBC’s are, <mask> chromatin, and <mask> shaped nuclei. The nucleoli are <mask>, and the cytoplasm is <mask> with <mask> basophilia. Cytoplasmic vacuoles are <mask>."""

                    # Replace <mask> with values from result_list
                    if not result_list:
                        filled_text = "No white blood cells are present in the image."
                    else:
                        filled_text = text.replace("<mask>", "{}").format(*result_list)
                    
                    
                    def plot_bboxes_from_dataframe(img, df_predictions):
                        # Iterate through the DataFrame
                        for _, row in df_predictions.iterrows():
                            # Extract coordinates (convert from string to int)
                            x_min, y_min, x_max, y_max = map(int, [row['x_min'], row['y_min'], row['x_max'], row['y_max']])
                            prediction = row['Prediction']
                            confidence = float(row['Confidence'])

                            # Skip predictions marked as 'None'
                            if prediction == "None":
                                continue
                            
                            # Draw the bounding box
                            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255, 0), 2)

                            # Display prediction with confidence
                            # label = f"{prediction} ({confidence:.2f})"
                            label = f"{prediction}"
                            cv2.putText(img, label, (x_min, max(0, y_min - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 2)

                        return img  # Return the annotated image
                    # df_predictions.to_csv("predictions.csv", index=False)  # Save if needed
                    annotated_img = plot_bboxes_from_dataframe(im0, df_predictions)
                    # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    # cv2.putText(img, predicts, (x_min, y_min - 10)),
                    # print(df_predictions)

                
                # else:
                #     QMessageBox.critical(self, "Error", "Please enter a slide number.")        
                # image_counter = 1 
                else:
                    annotated_img= im0
                    filled_text= "White blood cells are not presented in the image."
    
                return annotated_img ,filled_text
                # Process detections
                # for i, det in enumerate(pred):
                #     if len(det):
                #         det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                #         for *xyxy, conf, cls in reversed(det):
                #             c = int(cls)  # integer class
                #             label = None if self.hide_labels else (model.names[c] if self.hide_conf else f'{model.names[c]} {conf:.2f}')
                #             img0 = self.plot_one_box(xyxy, frame, label=label, color=(0,255,0))

                # # Save image with bounding boxes
                # output_path =  os.path.join(self.slide_dir, f"image_detection{self.image_counter}.png")

                
                
                # if len(det):
                #      cv2.imwrite(output_path, img0)
                # #QMessageBox.information(self, "Success", f"Image {self.image_counter} captured and saved.")
                # self.image_counter += 1
                # self.image_counter_label.setText(f"{self.image_counter}")
        
def inference_fn_select(image_input):
    try:
        # img = letterbox(image_input, (640, 640), stride=32, auto=True)[0]  # Resize and pad image
        # img = img.transpose(2, 0, 1)[::-1]  # Convert to channel-first format
        # img = np.ascontiguousarray(img)
        results,filled_text = capture_image(image_input) 
        state = 1# Model inference
        result_pil = Image.fromarray(results)
        return result_pil,filled_text
    except Exception as e:
        return None, f"Error in inference: {e}"
    
def set_cloze_samples(example: list) -> dict:
    return gr.update(value=example[0]), 'Cloze Test'

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(header)
    gr.Markdown(abstract)
    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=0.5, min_width=500):
            image_input = gr.Image(type="pil", interactive=True, label="Upload an image 📁", height=250)
        with gr.Column(scale=0.5, min_width=500):
            task_button = gr.Radio(label="Contextual Task type", interactive=True,
                                   choices=['Detect'],
                                   value='Detect')
            with gr.Row():
                submit_button = gr.Button(value="🏃 Run", interactive=True, variant="primary")
                clear_button = gr.Button(value="🔄 Clear", interactive=True)

    with gr.Row():
        with gr.Column(scale=0.5, min_width=500):
            image_output = gr.Image(type='pil', interactive=False, label="Detection output")
        with gr.Column(scale=0.5, min_width=500):
            chat_output = gr.Textbox(label="Text output")
    # with gr.Row():
        # with gr.Column(scale=0.5, min_width=500):
            with gr.Row():
                cloze_examples = gr.Dataset(
                    label='Sample Images',
                    components=[image_input],
                    samples=cloze_samples,
            )


    submit_button.click(
        inference_fn_select,
        [image_input],
        [image_output, chat_output],
    )

    clear_button.click(
        lambda: (None, None, "", [], [], 'Detect'),
        [],
        [image_input, image_output, chat_output, task_button],
        queue=False,
    )

    image_input.change(
        lambda: (None, "", []),
        [],
        [image_output, chat_output],
        queue=False,
    )
    cloze_examples.click(
        fn=set_cloze_samples,
        inputs=[cloze_examples],
        outputs=[image_input, chat_output],
    )

    gr.Markdown(footer)

demo.queue()  # Enable request queuing
demo.launch(share=False)