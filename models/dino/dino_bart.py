<<<<<<< HEAD
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]MemoryClassifier
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from torch.nn.utils.rnn import pad_sequence
import math
from typing import List
import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
from transformers import BartTokenizer, BartForConditionalGeneration
from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn,dn_post_process
# from deformable_transformer import forward_prediction_heads
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
class ImageClassifier2(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=45, dropout=0.1):
        super(ImageClassifier2, self).__init__()
        self.classifier2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        return self.classifier2(x)
# class classificationdecoder(nn.Module):
#     def __init__(self, input_dim=256, feature_dim=368, output_dim=45, num_heads=4, dropout=0.1):
#         super().__init__()

#         # Cross-attention
#         self.learnable_query = nn.Parameter(torch.zeros(1, 1, input_dim))
#         nn.init.trunc_normal_(self.learnable_query, std=0.02)
#         self.cross_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)

#         # Layer norms
#         self.norm1 = nn.LayerNorm(input_dim)

#         # === MLP 1: Feature extraction ===
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_dim, feature_dim),
#             # nn.LeakyReLU(negative_slope=0.01),
#             # nn.Dropout(dropout)
#         )

#         # === MLP 2: Classification ===
#         self.classifier = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim // 2),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(dropout),
#             nn.Linear(feature_dim // 2, output_dim)
#         )

#     def forward(self, cls_feature, encoder_features):
#         """
#         cls_feature: [B, 1, C] (learnable)
#         pooled_feature: [B, 1, C] (pooled DINO encoder features)
#         """
#         # Cross-attention
#         B = encoder_features.size(0)

#         # Repeat learnable query for batch
#         cls_query = self.learnable_query.expand(B, -1, -1)  # [B, 1, C]

#         # Cross-attention
        
#         attn_output, _ = self.cross_attn(
#             query=cls_query,
#             key=encoder_features,
#             value=encoder_features
#         )
#         # cls_feature = cls_query + attn_output  # residual
#         # cls_feature = torch.cat([cls_query, attn_output], dim=-1) 
#         cls_feature = self.norm1(attn_output)

#         # Flatten [B, 1, C] → [B, C]
#         cls_feature = cls_feature.squeeze(1)

#         # Feature extraction
#         features = self.feature_extractor(cls_feature)

#         # Classification
#         logits = self.classifier(features)
#         return logits, features
class ImageClassifier_feat(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=768, dropout=0.1):
        super(ImageClassifier_feat, self).__init__()
        self.classifier_feat = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.classifier_feat(x)
# class ImageClassifier2(nn.Module):
#     def __init__(self, input_dim=256, hidden_dim=256, output_dim=45, dropout=0.1):
#         super(ImageClassifier2, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.classifier(x)

class MultimodalQFormerWithTextCrossAttn(nn.Module):
    def __init__(self, vision_dim=256, text_dim=768, num_queries=64, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, vision_dim)).cuda()  # Learnable queries

        # Cross-attention: queries attend to vision features (encoder + decoder)
        self.vision_attn = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=num_heads, batch_first=True)

        # Project vision attended output from vision_dim → text_dim
        self.vision_proj = nn.Linear(vision_dim, text_dim)

        # Cross-attention: queries attend to text embeddings
        self.text_attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, batch_first=True)

        # Optional final projection or layer norm (add as needed)
        self.final_proj = nn.Linear(text_dim, text_dim)
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, encoder_output, decoder_output, text_embeddings):
        """
        Args:
            encoder_output: [B, N_enc, vision_dim] (e.g. [B, 1154, 256])
            decoder_output: [B, N_dec, vision_dim] (e.g. [B, 900, 256])
            text_embeddings: [B, num_queries, text_dim] (e.g. [B, 64, 768])

        Returns:
            Tensor of shape [B, num_queries, text_dim] — fused multimodal output
        """
        B = encoder_output.size(0)

        # Combine vision tokens
        # vision_feats
        vision_feats = encoder_output.unsqueeze(1).detach()#torch.cat([encoder_output, decoder_output], dim=1)  # [B, 2054, vision_dim]

        # Expand queries for batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, 64, vision_dim]

        # # Step 1: Cross-attention queries -> vision features
        vision_attended, _ = self.vision_attn(query=queries, key=vision_feats, value=vision_feats)  # [B, 64, vision_dim]

        # Project vision attended output to text embedding dim
        vision_proj = self.vision_proj(vision_attended)  # [B, 64, text_dim]

        # Step 2: Cross-attention queries (vision_proj) -> text embeddings
        text_attended, _ = self.text_attn(query=text_embeddings, key=vision_proj, value=vision_proj)  # [B, 64, text_dim]

        # Optional: residual + norm + final projection
        combined = self.norm(text_attended + vision_proj)
        output = self.final_proj(combined)  # [B, 64, text_dim]

        return output
    
class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, iter_update=False,
                    query_dim=2, 
                    random_refpoints_xy=False,
                    fix_refpoints_hw=-1,
                    num_feature_levels=1,
                    nheads=8,
                    # two stage
                    two_stage_type='no', # ['no', 'standard']
                    two_stage_add_query_num=0,
                    dec_pred_class_embed_share=True,
                    dec_pred_bbox_embed_share=True,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    decoder_sa_type = 'sa',
                    num_patterns = 0,
                    dn_number = 100,
                    dn_box_noise_scale = 0.4,
                    dn_label_noise_ratio = 0.5,
                    dn_labelbook_size = 100,
                    dn="seg",
                    noise_scale=0.4,
                    dn_num=100,
                    initial_pred=True,
                    classification_head=False
                    
        
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size
        # self.learn_tgt = learn_tgt
        self.dn=dn
        self.noise_scale=noise_scale
        self.dn_num=dn_num
        self.initial_pred = initial_pred
        self.class_embed_seg = nn.Linear(hidden_dim, num_classes+1)
        self.text_proj_dec = nn.Linear(768, 256).cuda()  # 768 -> 256
        self.mask_embed = MLP(hidden_dim, hidden_dim, 256, 3)
        self.dformer = MultimodalQFormerWithTextCrossAttn()
        for param in self.dformer.parameters():
            param.requires_grad = False
            print(f"Frozen: requires_grad = {param.requires_grad}")
        self.classifier_image = ImageClassifier2().cuda()
        # self.classification_decoder = classificationdecoder(
        #                     input_dim=256, 
        #                     feature_dim=368, 
        #                     output_dim=45, 
        #                     num_heads=4
        #                 ).cuda()
        self.classifier_feat =  ImageClassifier_feat().cuda()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.classification_head=classification_head
        
        # self.pool = nn.AdaptiveAvgPool1d(1)  # global pooling over tokens
        # self.classifier_image = nn.Sequential(
        #                         nn.Linear(512, 256),
        #                         nn.LeakyReLU(negative_slope=0.01),
        #                         nn.Dropout(0.25),
        #                         nn.Linear(256, 26)
        #                     ) # final linear layer
        # prepare input projection layers
        # self.image_classifier = MemoryClassifier(hidden_dim=512, num_classes=26)
        # self.bart_model = AutoModelForSeq2SeqLM.from_pretrained('GanjinZero/biobart-v2-base').cuda()
        # self.bart_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/biobart-v2-base')
        self.bart_tokenizer  = BartTokenizer.from_pretrained("/home/iml/DINO/coco_data/bart_models/finetuned_bart_manual")
        self.bart_model = BartForConditionalGeneration.from_pretrained("/home/iml/DINO/coco_data/bart_models/finetuned_bart_manual").cuda()
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.class_tokens = nn.ParameterList([
    nn.Parameter(torch.randn(1, 256, 1, 1)) for _ in range(self.num_feature_levels)
])
        self.class_pos_tokens = nn.ParameterList([
    nn.Parameter(torch.randn(1, 256, 1, 1)) for _ in range(self.num_feature_levels)
])
        
        
        # self.num_morphology_features = 6
        # self.num_classes_per_feature = 3
        # self.morphology_embed = nn.ModuleList([
        # nn.Linear(self.hidden_dim, self.num_classes_per_feature)
        # for _ in range(self.num_morphology_features)
        # ])
        
        num_morph_attributes = 6
        num_morph_classes = 2
        
              
        total_morphology_classes = num_morph_attributes * num_morph_classes  # 6 * 3 = 18

        _morphology_embed = nn.Linear(hidden_dim, total_morphology_classes)

        if dec_pred_class_embed_share:
            morphology_embed_layerlist = [_morphology_embed for _ in range(transformer.num_decoder_layers)]
        else:
            morphology_embed_layerlist = [copy.deepcopy(_morphology_embed) for _ in range(transformer.num_decoder_layers)]

        self.morphology_embed = nn.ModuleList(morphology_embed_layerlist)
                    
                    
            
        
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value  # did not add for the morphology 
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        # self.class_embed = nn.ModuleList([
        #     nn.Linear(64, num_classes)
        #     for _ in range(transformer.num_decoder_layers)
        # ])
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
    
            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
    
            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
         # Initialize morphology embedding layers
        # for morphology_layer_list in self.morphology_embed:
        #     for morphology_layer in morphology_layer_list:
        #         nn.init.normal_(morphology_layer.weight, std=0.01)
        #         nn.init.constant_(morphology_layer.bias, 0)
    def dn_post_process_seg(self,outputs_class,outputs_coord,mask_dict,outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1],'pred_masks': output_known_mask[-1]}

        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask,output_known_coord)
        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class, outputs_coord, outputs_mask
    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:

            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))
    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num,self.noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num)>0:
                scalar = scalar//(int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of bbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to('cuda')
            input_label_embed = self.label_enc(m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
            padding_bbox = torch.zeros(pad_size, 4).cuda()

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label=padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label=None
                input_query_bbox=None
            attn_mask = None
            mask_dict=None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label,input_query_bbox,attn_mask,mask_dict
    
    def get_prompt_embeddings(self,prompts, device='cpu'):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

        # Tokenize the list of prompts
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            # Get the [CLS] token embeddings for each prompt
            text_embeddings = bert_model(**tokenized_prompts).last_hidden_state[:, 0, :]  # shape: [N, 768]

        return text_embeddings  # [num_prompts, 768]
    def shift_right(self,input_ids, bos_token_id):
        # Shift tokens one step to the right and insert BOS token at the front
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 0] = bos_token_id
        shifted[:, 1:] = input_ids[:, :-1]
        return shifted
    
    def repeat_to_fill(self, text, target_len=64, tokenizer=None):
        if not isinstance(text, str):
            text = ", ".join(text)

        # Get token count (excluding special tokens)
        base_tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        token_len = len(base_tokens)

        if token_len >= target_len:
            return text

        # Repeat the string enough times and join with commas
        repeat_count = (target_len // token_len) + 1
        repeated_text = ", ".join([text] * repeat_count)

        return repeated_text
    def get_meaningful_embeddings(self,text_embeddings, attention_mask):
        batch = []
        for i in range(text_embeddings.size(0)):
            valid_embeddings = text_embeddings[i][attention_mask[i].bool()]  # (valid_len, hidden_dim)
            batch.append(valid_embeddings)
        return pad_sequence(batch, batch_first=True)
    def encode_prompts(self, input_text,target_text, bart_model, bart_tokenizer, device='cuda'):
        # bart_model.eval()  # No gradients for encoder
        # processed_text = [
        #     text if isinstance(text, str) else ", ".join(text)
        #     for text in input_text
        # ]
        processed_text = [
                            f"This image is for the diagnosis of {text if isinstance(text, str) else ', '.join(text)}"
                            for text in input_text
]
#         processed_text = [
#             self.repeat_to_fill(text, target_len=64, tokenizer=bart_tokenizer)
#             for text in input_text
# ]

        input_enc = bart_tokenizer(
            processed_text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        if target_text:
            # Join target_text entries if they are not strings
            target_text = [
                text if isinstance(text, str) else ", ".join(text)
                for text in target_text
            ]
            with bart_tokenizer.as_target_tokenizer():
                target_enc = bart_tokenizer(
                    target_text,
                    max_length=64,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

            labels = target_enc["input_ids"]
            labels[labels == bart_tokenizer.pad_token_id] = -100
        else:
            labels = None

        input_ids = input_enc["input_ids"].to(device)
        attention_mask = input_enc["attention_mask"].to(device)
        if labels is not None:
            labels = labels.to(device)

        # Step 1: Encoder forward
        with torch.no_grad():
            encoder_outputs = bart_model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Step 2: Prepare decoder inputs (only if labels exist)
        if labels is not None:
            decoder_input_ids = bart_model.prepare_decoder_input_ids_from_labels(labels)
        else:
            decoder_input_ids = None  # or handle as needed

        # meaningful_embeddings = self.get_meaningful_embeddings(encoder_hidden_states, attention_mask)

        return encoder_hidden_states, attention_mask, decoder_input_ids,labels
    def decode_from_encoder(
                self, 
            encoder_hidden_states, 
            attention_mask, 
            decoder_input_ids,  # decoder inputs for teacher forcing
            labels,             # target output tokens for loss
            model, 
            tokenizer, 
            device='cuda'
        ):
            batch_size = encoder_hidden_states.size(0)
            max_length=64
            # Wrap encoder outputs
            decoder_outputs = model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask
        )

            # Step 4: Compute logits and loss
            logits = model.lm_head(decoder_outputs.last_hidden_state)
            #labels_for_loss=labels.clone()
            #labels_for_loss[labels_for_loss == -100]=tokenizer.pad_token_id
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            generated_ids = model.generate(
                input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            # labels = labels.clone()
            # labels[labels == -100] = tokenizer.pad_token_id
# decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_labels= tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            return  loss,decoded_preds,decoded_labels
    def train_step_bart(self,input_text, target_text, model, tokenizer, device='cuda'):
        """
        A single function to tokenize, compute loss, and get predictions for BART.
        """
        model.train()

        # Tokenize inputs
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True
        ).to(device)

        # Tokenize targets (labels)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                target_text,
                return_tensors="pt",
                max_length=64,
                padding="max_length",
                truncation=True
            )["input_ids"].to(device)

        # Mask pad tokens in labels
        labels[labels == tokenizer.pad_token_id] = -100

        # Forward pass through BART
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )

        loss = outputs.loss

        # Generate predictions (optional: during training/validation)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        predicted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return loss, predicted_text 
        
    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed_seg(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask
    def forward(self, samples: NestedTensor,  prompt, targets:List=None,):
        
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        device=features[0].device
        # bs = features[0].mask.shape[0]
        # cls_tok = self.cls_token.expand(bs, -1, -1)  # [bs, 1, C]
        # cls_pos = self.cls_pos.expand(bs, -1, -1)    # [bs, 1, C]
        
        srcs = []
        masks = []
        # poss = []
        # for l, feat in enumerate(features):
        #     src, mask = feat.decompose()
        #     srcs.append(self.input_proj[l](src))
        #     masks.append(mask)
        #     assert mask is not None
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        # for l, feat in enumerate(features):
        #     src, mask = feat.decompose()  # src: [B, C, H, W]
        #     src_proj = self.input_proj[l](src)  # [B, C, H, W]

        #     B, C, H, W = src_proj.shape

        #     # Prepare class token
        #     class_token = self.class_tokens[l].expand(B, -1, 1, 1)  # [B, C, 1, 1]

        #     # Pad to get [B, C, H+1, W+1]: pad top=1, left=1
        #     src_with_cls = F.pad(src_proj, (1, 0, 1, 0))  # pad (left, right, top, bottom)
        #     src_with_cls[:, :, 0, 0] = class_token.squeeze(-1).squeeze(-1)  # place class token at top-left

        #     srcs.append(src_with_cls)

        #     # Update mask (1 for masked, 0 for visible): pad same way
        #     mask = F.pad(mask, (1, 0, 1, 0), value=False)  # [B, H+1, W+1]
        #     masks.append(mask)

        #     # Positional encoding for padded input
        #     # compute pos embedding for original src_proj, then pad on top & left
        #     pos = self.backbone[1](NestedTensor(src_proj, mask[:, 1:, 1:]))  # mask without the new top & left row/col
        #     pos = F.pad(pos, (1, 0, 1, 0))  # pad left=1, right=0, top=1, bottom=0
        #     poss.append(pos)
        
        
        # if self.num_feature_levels > len(srcs):
        #     _len_srcs = len(srcs)
        #     for l in range(_len_srcs, self.num_feature_levels):
        #         if l == _len_srcs:
        #             src = self.input_proj[l](features[-1].tensors)
        #         else:
        #             src = self.input_proj[l](srcs[-1])
        #         m = samples.mask
        #         mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
        #         pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                
        #         srcs.append(src)
        #         masks.append(mask)
        #         poss.append(pos_l)
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)  # [B, C, H, W]
                else:
                    # Remove previously added class token patch before reuse
                    # srcs[-1] shape: [B, C, H+1, W+1]
                    prev_src = srcs[-1]
                    src = prev_src[:, :, 1:, 1:]  # crop off the first row and column (top-left)

                    src = self.input_proj[l](src)  # project again if needed

                B, C, H, W = src.shape

                # Prepare class token spatial patch
                class_token = self.class_tokens[l].expand(B, -1, 1, 1)  # [B, C, 1, 1]

                # Pad src spatially (H+1, W+1): pad bottom and right by 1
                src_with_cls = F.pad(src, (1, 0, 1, 0))  # pad left=1, right=0, top=1, bottom=0
                src_with_cls[:, :, 0, 0] = class_token.squeeze(-1).squeeze(-1)  # place class token at top-left

                # Prepare mask: interpolate to src spatial shape then pad
                m = samples.mask  # [B, H_orig, W_orig]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:], mode="nearest").to(torch.bool)[0]  # [B, H, W]
                mask = F.pad(mask, (1, 0, 1, 0), value=False)  # pad left=1, right=0, top=1, bottom=0

                srcs.append(src_with_cls)
                masks.append(mask)

                # Positional encoding: compute on unpadded src, then pad
                pos_l = self.backbone[1](NestedTensor(src, mask[:, 1:, 1:])).to(src.dtype)  # mask before pad
                pos_l = F.pad(pos_l, (1, 0, 1, 0))  # pad left=1, right=0, top=1, bottom=0
                poss.append(pos_l)

                
                
        if self.dn_number > 0 and (targets and len(targets) > 0):
            
            has_segmentation = any(
                t.get("segmentation") is not None and t["segmentation"].item() != 1
                for t in targets
            )

            if has_segmentation:
                input_query_label, input_query_bbox, attn_mask, dn_meta = \
                    prepare_for_cdn(
                        dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                        training=self.training,
                        num_queries=self.num_queries,
                        num_classes=self.num_classes,
                        hidden_dim=self.hidden_dim,
                        label_enc=self.label_enc
                    )
            else:
            
              input_query_bbox = input_query_label = attn_mask = dn_meta = None
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None
            
        
        # just bert embeding token for encoder 
        
        
        # embedding_file_path = f"prompts/{prompt}_embeddings.pt"  # e.g., "leukemia_embeddings.pt" or "malaria_embeddings.pt"

        # # Check if the embeddings file exists
        # if os.path.exists(embedding_file_path):
          
        #     text_embeddings = torch.load(embedding_file_path)
      
        # else:
        #     text_embeddings  = self.get_prompt_embeddings(prompt, device='cuda')
        #     torch.save(text_embeddings, embedding_file_path)  # Save as tensor
        completed_texts = (
                tuple(t['completed_text'] for t in targets if 'completed_text' in t)
                if targets else ()
            )
        self.bart_model.train
        

        text_embeddings, attention_mask, decoder_input_ids,labels_encoder = self.encode_prompts(prompt,completed_texts, self.bart_model, self.bart_tokenizer)
        # valid_token_indices = attention_mask[0].nonzero(as_tuple=True)[0]
        # meaningful_embeddings = text_embeddings[0, valid_token_indices, :] 
        # Step 2: Decode
        # 
        # text_embeddings= None
        
       
        
        hs, reference, hs_enc, ref_enc, init_box_proposal,predictions_class, predictions_mask,mask_features,enc_class_tokens_encder,encoder_output,decoder_output,_,cls_feature, pooled_feature= self.transformer(srcs,text_embeddings,prompt, masks, input_query_bbox, poss,input_query_label,attn_mask)
        # In case num object=0
        

        logits_image = self.classifier_image(enc_class_tokens_encder.squeeze(1))
        # logits_image, enc_class_tokens = self.classification_decoder(cls_feature, pooled_feature)
        
        logits_image_featuers= self.classifier_feat (enc_class_tokens_encder.squeeze(1))

        hs[0] += self.label_enc.weight[0,0]*0.0
        text_embeddings_attented = self.dformer(enc_class_tokens_encder, decoder_output, text_embeddings)
        
        # enc_class_tokens_encder = torch.cat([enc_class_tokens_encder, cls_feature], dim=-1) 
        # text_loss,decoded_text,decoded_labels= self.decode_from_encoder(text_embeddings_attented, attention_mask,decoder_input_ids,labels_encoder, self.bart_model, self.bart_tokenizer, device)
        if decoder_input_ids is not None:
            text_loss, decoded_text, decoded_labels = self.decode_from_encoder(
                text_embeddings_attented,
                attention_mask,
                decoder_input_ids,
                labels_encoder,
                self.bart_model,
                self.bart_tokenizer,
                device
            )
        else:
            # Set safe fallback values
            labels_encoder= None
            text_loss = torch.tensor(0.0, device=device)
            decoded_text = [""] #* text_embeddings_attented.size(0)  # empty text for each batch
            decoded_labels = torch.zeros_like(labels_encoder) if labels_encoder is not None else None

        # text_loss,decoded_text= self.train_step_bart(prompt,completed_texts, self.bart_model, self.bart_tokenizer)
        # deformable-detr-like anchor text_embeddings_attented
        # print(decoded_text)
        # IMAGE CLASSIFICATION
        
        
        # logits_image = self.image_classifier(memory, memory_class)
        # memory_pooled = self.pool(memory.transpose(1, 2)).squeeze(-1)  # → [B, D]

        # # logits: [B, num_classes]
        
        # combined_memory = torch.cat([memory_pooled, memory_class], dim=1)
        
        # logits_image = self.classifier_image(combined_memory)
        
        
        

        # output_image_classifies = {"logits": logits_image}

        # if targets is not None:
        #     # extract category_id from target dicts
        #     labels_image_classifier = torch.stack([t["category_id"] for t in targets])  # [B]
        #     loss_image_classifier = F.cross_entropy(output_image_classifies, labels_image_classifier)
        #     # output["loss"] = loss_image_classifier
        
        
        
        
        
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)        

        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        # projected_text=self.text_proj_dec(text_embeddings)  
        # outputs_class = torch.stack([
        #     layer_cls_embed(
        #         torch.matmul(
        #             F.normalize(layer_hs, dim=-1),          # (B, Q, D)
        #             projected_text.transpose(1, 2)           # (B, D, C)
        #         )                                           # (B, Q, C)
        #     )
        #     for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
        # ])
        
        
        
        
        # outputs_morphology = []

        # # Collect outputs for each morphology feature
        # for feature_idx in range(self.num_morphology_features):
        #     # Pass the hidden states from the last decoder layer to the morphology head for each feature
        #     # hs[-1] represents the last decoder layer's hidden states
        #     feature_morphology_outputs = self.morphology_embed[feature_idx](hs[-1])  # Shape: [batch_size, num_queries, num_classes_per_feature
        #     # Append the output for this feature
        #     outputs_morphology.append(feature_morphology_outputs)
            
            
        
        # outputs_morphology_list = []
        # # Loop through each decoder layer to predict morphology at every step (similar to class and bbox)
        # for dec_lid, layer_hs in enumerate(hs):
        #     # Collect outputs for each morphology feature at this decoder layer
        #     outputs_morphology_layer = []
        #     for feature_idx in range(self.num_morphology_features):
        #         feature_morphology_output = self.morphology_embed[feature_idx](layer_hs)  # Predict morphology for this feature
        #         outputs_morphology_layer.append(feature_morphology_output)
        #     # Stack the morphology predictions for all features at this layer
        #     outputs_morphology_layer = torch.stack(outputs_morphology_layer, dim=-1)  # Shape: [batch_size, num_queries, num_classes_per_feature, num_features]
        #     outputs_morphology_list.append(outputs_morphology_layer)
        # # Stack predictions across layers (same as bbox and class)
        # outputs_morphology = torch.stack(outputs_morphology_list, dim=0)
        
        # outputs_morphology = []
        # for layer_morphology_embed, layer_hs in zip(self.morphology_embed, hs):
        #     morphology_preds = [morph_layer(layer_hs) for morph_layer in layer_morphology_embed]
        #     outputs_morphology.append(morphology_preds)
        # predictions_class = []
        predictions_masks = []
        # tgt_mask = None
        # mask_dict = None
        # if self.dn != "no" and self.training:
        #     assert targets is not None
        #     input_query_label, input_query_bbox, tgt_mask, mask_dict = \
        #         self.prepare_for_dn(targets, None, None, srcs[0].shape[0])
        #     if mask_dict is not None:
        #         tgt=torch.cat([input_query_label, tgt],dim=1)

        # direct prediction from the matching and denoising part in the begining
        segem= False 
        
        
        if segem:
            for i, output in enumerate(hs):
                outputs_classes, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), mask_features, self.training or (i == len(hs)-1))
                # predictions_class.append(outputs_class)
                predictions_masks.append(outputs_mask)

        # iteratively box prediction
        # if self.initial_pred:
        #     out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
        #     assert len(predictions_class) == self.num_layers + 1
        # else:
        #     out_boxes = self.pred_box(references, hs)
        # if mask_dict is not None:
        #     predictions_mask=torch.stack(predictions_mask)
        #     predictions_class=torch.stack(predictions_class)
        #     predictions_class, out_boxes,predictions_mask=\
        #         self.dn_post_process(predictions_class,outputs_coord_list,mask_dict,predictions_mask)
        #     predictions_class,predictions_mask=list(predictions_class),list(predictions_mask)
        
        
        
        outputs_morphology = torch.stack([
        layer_morph_embed(layer_hs)  # Shape: [batch_size, num_queries, total_morphology_classes]
        for layer_morph_embed, layer_hs in zip(self.morphology_embed, hs)
         ])
            
        
        # predictions_masks=torch.stack(predictions_masks)
        # predictions_class=torch.stack(predictions_class)
        # predictions_class, out_boxes,predictions_mask=\
        #         self.dn_post_process_seg(predictions_class,outputs_coord_list,mask_dict,predictions_mask)
        # predictions_class,predictions_mask=list(predictions_class),list(predictions_mask)
        
        
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list, outputs_morphology= \
                dn_post_process(outputs_class, outputs_coord_list,outputs_morphology,
                                dn_meta,self.aux_loss,self._set_aux_loss)
        text_embeddings, attention_mask, decoder_input_ids,labels_encoder = self.encode_prompts(prompt,completed_texts, self.bart_model, self.bart_tokenizer)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1], 'pred_morphology': outputs_morphology[-1], 'pred_mask':predictions_mask[-1],'pred_image_class':logits_image,'pred_image_feat':logits_image_featuers,'loss_text':text_loss,'pred_text':decoded_text,'completed_text':decoded_labels,'encoder_class_feat':enc_class_tokens_encder,'text_embeddings':text_embeddings}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord_list, outputs_morphology
            )



        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_morphology):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # return [{'pred_logits': a, 'pred_boxes': b}
        #         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        # return [{'pred_logits': a, 'pred_boxes': b, 'pred_morphology': torch.stack(c, dim=-1)}
        #     for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_morphology)]
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_morphology': c}
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_morphology[:-1])]


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip  #  prevents nan for large gamma_neg numbers
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """"
        Parameters:
        x: input logits (before sigmoid), shape [N, num_classes]
        y: targets (multi-label binarized vector), shape [N, num_classes]
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss = y * torch.log(xs_pos.clamp(min=self.eps)) + (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)
        loss *= one_sided_w

        return -loss.sum()
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule

def dice_loss2(pred, target, smooth=1.):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))
def focal_loss(pred, target, alpha=0.25, gamma=2.):
        pred = pred.view(-1)
        target = target.view(-1)
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule
def calculate_uncertainty(logits):
        """
        We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
            foreground class in `classes`.
        Args:
            logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
                class-agnostic, where R is the total number of predicted masks in all images and C is
                the number of foreground classes. The values are logits.
        Returns:
            scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
                the most uncertain locations having the highest uncertainty score.
        """
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -(torch.abs(gt_class_logits))
    
class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    def loss_morphology(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the morphology predictions, ignoring labels equal to 4."""
        assert 'pred_morphology' in outputs

        idx = self._get_src_permutation_idx(indices)

        src_morphology = outputs['pred_morphology'][idx]  # Shape: [num_matched, total_morphology_classes]

        # Define number of attributes and classes per attribute
        num_attributes = 6
        num_classes_per_attribute = 2  # Valid labels are 0 and 1
        src_morphology = src_morphology.view(-1, num_attributes, num_classes_per_attribute)

        # Get target morphology attributes
        if targets[0]['morphology'].numel() > 0:

            target_morphology = torch.cat([
                t['morphology'][i] for t, (_, i) in zip(targets, indices)
            ], dim=0)  # Shape: [num_matched, num_attributes]

            # Create mask for valid labels (labels not equal to 4)
            
            
            mask = target_morphology != 4  # Mask for the none class 

            # Flatten the tensors
            src_morphology = src_morphology.reshape(-1, num_classes_per_attribute)  # [num_matched * num_attributes, num_classes_per_attribute]
            target_morphology = target_morphology.reshape(-1)  # [num_matched * num_attributes]
            mask = mask.reshape(-1)  # [num_matched * num_attributes], dtype=bool

            # Select only valid elements
            src_morphology = src_morphology[mask]
            target_morphology = target_morphology[mask]

            # Check if there are valid elements to compute loss
            if src_morphology.numel() == 0:
                # No valid elements, return zero loss
                losses = {'loss_morphology': src_morphology.sum()*0}
                return losses

            # Perform one-hot encoding on the target morphology
            target_morphology_onehot = F.one_hot(target_morphology, num_classes=num_classes_per_attribute).float()

            # Compute Asymmetric Loss
            criterion = AsymmetricLoss(
                gamma_neg=4, gamma_pos=1, clip=0.08, disable_torch_grad_focal_loss=True
            ).to(src_morphology.device)
            object_batch_loss = criterion(src_morphology, target_morphology_onehot)

            # Normalize the loss by the number of boxes
            losses = {'loss_morphology': object_batch_loss  / num_boxes} 
            return losses
        else:
            # Handle the case where 't['morphology']' is empty for any of the 'targets'
            losses = {'loss_morphology': torch.tensor(0.0, device=src_morphology.device)}
            return losses

    
    # def loss_morphology(self, outputs, targets, indices, num_boxes):
    #     """Compute the losses related to the morphology predictions using Asymmetric Loss."""
    #     assert 'pred_morphology' in outputs

    #     idx = self._get_src_permutation_idx(indices)
      
    #     src_morphology = outputs['pred_morphology'][idx]  # Shape: [num_matched, 18]
       
    #     # Reshape src_morphology to [num_matched, 6, 3]
    #     src_morphology = src_morphology.view(-1, 6, 3)

    #     # Get target morphology attributes
    #     target_morphology = torch.cat([
    #         t['morphology'][i] for t, (_, i) in zip(targets, indices)
    #     ], dim=0)  # Shape: [num_matched, 6]

    #     # Perform one-hot encoding for each attributeget_loss
    #     num_classes = 3
    #     one_hot_encoded_tensors = []
    #     for i in range(target_morphology.size(1)):
    #         # Extract the current attribute column
    #         column_values = target_morphology[:, i].long()  # Shape: [num_matched]

    #         # Generate one-hot encoded tensor for the current attribute
    #         one_hot_encoded_col = torch.eye(num_classes, device=target_morphology.device)[column_values]
    #         one_hot_encoded_col = one_hot_encoded_col.unsqueeze(1)  # Shape: [num_matched, 1, num_classes]

    #         one_hot_encoded_tensors.append(one_hot_encoded_col)

    #     # Concatenate the one-hot encoded tensors along the attribute dimension
    #     target_morphology_onehot = torch.cat(one_hot_encoded_tensors, dim=1)  # Shape: [num_matched, 6, 3]

    #     # Flatten the predictions and targets for loss computation
    #     src_morphology = src_morphology.reshape(-1, num_classes)       # Shape: [num_matched * 6, 3]
    #     target_morphology_onehot = target_morphology_onehot.reshape(-1, num_classes)  # Shape: [num_matched * 6, 3]

    #     # Compute AsymmetricLoss
    #     criterion = AsymmetricLoss(
    #         gamma_neg=4, gamma_pos=1, clip=0.08, disable_torch_grad_focal_loss=True
    #     ).to(src_morphology.device)
    #     object_batch_loss = criterion(src_morphology, target_morphology_onehot)

    #     # Normalize the loss by the number of boxes
    #     losses = {'loss_morphology': object_batch_loss / num_boxes}
    #     return losses

    # mask loss 
    
    
    # def loss_masks(self, outputs, targets, indices, num_masks):
    #     """Compute the losses related to the masks: the focal loss and the dice loss.
    #     targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    #     """
    #     assert "pred_masks" in outputs

    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)
    #     src_masks = outputs["pred_masks"]
    #     src_masks = src_masks[src_idx]
    #     masks = [t["masks"] for t in targets]
    #     # TODO use valid to mask invalid areas due to padding in loss
    #     target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    #     target_masks = target_masks.to(src_masks)
    #     target_masks = target_masks[tgt_idx]

    #     # No need to upsample predictions as we are using normalized coordinates :)
    #     # N x 1 x H x W
    #     src_masks = src_masks[:, None]
    #     target_masks = target_masks[:, None]

    #     with torch.no_grad():
    #         # sample point_coords
    #         point_coords = get_uncertain_point_coords_with_randomness(
    #             src_masks,
    #             lambda logits: calculate_uncertainty(logits),
    #             self.num_points,
    #             self.oversample_ratio,
    #             self.importance_sample_ratio,
    #         )
    #         # get gt labels
    #         point_labels = point_sample(
    #             target_masks,
    #             point_coords,
    #             align_corners=False,
    #         ).squeeze(1)

    #     point_logits = point_sample(
    #         src_masks,
    #         point_coords,
    #         align_corners=False,
    #     ).squeeze(1)

    #     losses = {
    #         "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
    #         "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
    #     }

    #     del src_masks
    #     del target_masks
    #     return losses
    
    

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'morphology': self.loss_morphology,
            'masks': self.loss_masks,
            
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs, targets,args, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        if any(t.get("masked_traning", torch.tensor(0)).item() == 1 for t in targets):
            # extract category_id from target dicts
            losses = {}
            
            
            l_dict = dict()
            text_loss= outputs['loss_text']
            # Add segmentation losses
            # l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['image_classification_loss']  = torch.as_tensor(0.).to('cuda')

            # Add segmentation losses
            # l_dict['loss_dice_seg'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_focal_seg'] = torch.as_tensor(0.).to('cuda')

            # Apply index suffix
            # l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
            l_dict['text_loss'] = text_loss
            # Update main loss dict
            
            losses.update(l_dict)
            return losses
            # output["loss"] = loss_image_classifier
        
        
        
        if any(t.get("classification", torch.tensor(0)).item() == 2 for t in targets):
            # extract category_id from target dicts
            losses = {}
            
            labels_image_classifier = torch.stack([t["category_id"] for t in targets])  
            
            invalid_mask = (labels_image_classifier < 0) | (labels_image_classifier >= 26)

            if invalid_mask.any():
                invalid_values = labels_image_classifier[invalid_mask]
                # raise ValueError(f"❌ Invalid target values found: {invalid_values.tolist()} — Valid range is [0, {5 - 1}]")# [B]
            loss_image_classifier = F.cross_entropy(outputs['pred_image_class'], labels_image_classifier)
            loss_image_featuers = F.cross_entropy(F.normalize(outputs['pred_image_feat']), F.normalize(outputs['text_embeddings'].mean(dim=1)).detach() )  
            l_dict = dict()

            # Add segmentation losses
            # l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['image_classification_loss']  = torch.as_tensor(0.).to('cuda')

            # Add segmentation losses
            # l_dict['loss_dice_seg'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_focal_seg'] = torch.as_tensor(0.).to('cuda')

            # Apply index suffix
            # l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
            l_dict['image_classification_loss'] = loss_image_classifier #+loss_image_featuers
            # Update main loss dict
            
            losses.update(l_dict)
            return losses
            # output["loss"] = loss_image_classifier
        
        
        
        if any(t.get("segmentation", torch.tensor(0)).item() == 1 for t in targets):
            losses = {}
            pred_masks = outputs['pred_mask'][:, 0]  # shape: (B, 64, 64)

            # Upsample to match ground truth
            pred_masks = F.interpolate(pred_masks.unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
            # shape: (B, 1, 512, 512)
            gt_masks = torch.stack([t['binary_mask'] for t in targets]) 
            gt_masks = gt_masks.to(pred_masks.device)
            pred_probs = torch.sigmoid(pred_masks)
            dice = dice_loss2(pred_probs, gt_masks)
            focal = focal_loss(pred_probs, gt_masks)

            l_dict = dict()
            # l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['image_classification_loss']  = torch.as_tensor(0.).to('cuda')

            # Add segmentation losses
            l_dict['loss_dice_seg'] = dice.to('cuda')
            l_dict['loss_focal_seg'] = focal.to('cuda')

            # Apply index suffix
            # l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}

            # Update main loss dict
            losses.update(l_dict)
            del pred_probs
            del gt_masks

            # mask losses 
            # pred_masks: shape (B, 1, 512, 512), after sigmoid and upsampling
# gt_masks: shape (B, 1, 512, 512)

  # scalar tensor

            return losses
        
             
        else:
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
            device=next(iter(outputs.values())).device
            indices = self.matcher(outputs_without_aux, targets)

            if return_indices:
                indices0_copy = indices
                indices_list = []

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            losses = {}

            # prepare for dn loss
            dn_meta = outputs['dn_meta']

            if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

                dn_pos_idx = []
                dn_neg_idx = []
                for i in range(len(targets)):
                    if len(targets[i]['labels']) > 0:
                        t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                        t = t.unsqueeze(0).repeat(scalar, 1)
                        tgt_idx = t.flatten()
                        output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                        output_idx = output_idx.flatten()
                    else:
                        output_idx = tgt_idx = torch.tensor([]).long().cuda()

                    dn_pos_idx.append((output_idx, tgt_idx))
                    dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

                output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
                l_dict = {}
                for loss in self.losses:
                    kwargs = {}
                    if 'labels' in loss:
                        kwargs = {'log': False}
                    l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

                l_dict = {k + f'_dn': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                losses.update(l_dict)

            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    if return_indices:
                        indices_list.append(indices)
                    for loss in self.losses:
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

                    if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                        aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                        l_dict={}
                        for loss in self.losses:
                            kwargs = {}
                            if loss == 'morphology' and any('morphology_attributes' not in t for t in targets):
                                continue  # Skip morphology loss if attributes are missing
                            if 'labels' in loss:
                                kwargs = {'log': False}

                            l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                    **kwargs))

                        l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    else:
                        l_dict = dict()
                        l_dict['loss_bbox_dn']=torch.as_tensor(0.).to('cuda')
                        l_dict['loss_giou_dn']=torch.as_tensor(0.).to('cuda')
                        l_dict['loss_ce_dn']=torch.as_tensor(0.).to('cuda')
                        l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            # interm_outputs loss
            if 'interm_outputs' in outputs:
                interm_outputs = outputs['interm_outputs']
                indices = self.matcher(interm_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'morphology':
                        if 'pred_morphology' not in interm_outputs:
                            # Skip morphology loss if not available
                            continue
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # enc output loss
            if 'enc_outputs' in outputs:
                for i, enc_outputs in enumerate(outputs['enc_outputs']):
                    indices = self.matcher(enc_outputs, targets)
                    if return_indices:
                        indices_list.append(indices)
                    for loss in self.losses:
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            if return_indices:
                indices_list.append(indices0_copy)
                return losses, indices_list

            return losses

        def prep_for_dn(self,dn_meta):
            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
            assert pad_size % num_dn_groups==0
            single_pad=pad_size//num_dn_groups

            return output_known_lbs_bboxes,single_pad,num_dn_groups


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        out_morphology = outputs['pred_morphology']
        batch_size, num_queries, total_morph_classes = out_morphology.shape
        num_attributes = 6 
        num_classes_per_attribute = 2  #
        # Reshape to [batch_size, num_queries, num_attributes, num_classes_per_attribute]
        out_morphology = out_morphology.view(batch_size, num_queries, num_attributes, num_classes_per_attribute)
        # Apply softmax to get probabilities per attribute
        morphology_probs = F.softmax(out_morphology, dim=-1)  # [batch_size, num_queries, num_attributes, num_classes_per_attribute]
        # Get predicted morphology labels per attribute
        morphology_labels = morphology_probs.argmax(-1) 
        

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        morphology_labels = torch.gather(morphology_labels, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, num_attributes))

        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i], 'morphology_labels':morph_labels[i]} for s, l, b, i, morph_labels in zip(scores, labels, boxes, item_indices, morphology_labels)]
        else:
            # Corrected else clause
            results = [{'scores': s, 'labels': l, 'boxes': b, 'morphology_labels': m} for s, l, b, m in zip(scores, labels, boxes, morphology_labels)]

        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name='dino')
def build_dino(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = DINO(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number = args.dn_number if args.use_dn else 0,
        dn_box_noise_scale = args.dn_box_noise_scale,
        dn_label_noise_ratio = args.dn_label_noise_ratio,
        dn_labelbook_size = dn_labelbook_size,
        classification_head=args.classification_head
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef,  'loss_morphology': args.morphology_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    
    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef
        weight_dict['loss_morphology_dn'] = args.morphology_loss_coef
        weight_dict['loss_focal_seg'] = args.dice_loss_coef
        weight_dict['loss_dice_seg'] = args.focal_loss_coef
        weight_dict['image_classification_loss'] = args.image_loss_coef
        

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_morphology': 1.0 , 
            'loss_focal_seg': 1.0,
            'loss_dice_seg': 1.0,
             'image_classification_loss': 1.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'morphology']
    # losses += ['focal_seg', 'dice_seg', 'image_classification']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

=======
# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]MemoryClassifier
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from torch.nn.utils.rnn import pad_sequence
import math
from typing import List
import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .deformable_transformer import build_deformable_transformer
from .utils import sigmoid_focal_loss, MLP
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
from transformers import BartTokenizer, BartForConditionalGeneration
from ..registry import MODULE_BUILD_FUNCS
from .dn_components import prepare_for_cdn,dn_post_process
# from deformable_transformer import forward_prediction_heads
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
class ImageClassifier2(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=45, dropout=0.1):
        super(ImageClassifier2, self).__init__()
        self.classifier2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, output_dim)
        )

    def forward(self, x):
        return self.classifier2(x)
# class classificationdecoder(nn.Module):
#     def __init__(self, input_dim=256, feature_dim=368, output_dim=45, num_heads=4, dropout=0.1):
#         super().__init__()

#         # Cross-attention
#         self.learnable_query = nn.Parameter(torch.zeros(1, 1, input_dim))
#         nn.init.trunc_normal_(self.learnable_query, std=0.02)
#         self.cross_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)

#         # Layer norms
#         self.norm1 = nn.LayerNorm(input_dim)

#         # === MLP 1: Feature extraction ===
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_dim, feature_dim),
#             # nn.LeakyReLU(negative_slope=0.01),
#             # nn.Dropout(dropout)
#         )

#         # === MLP 2: Classification ===
#         self.classifier = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim // 2),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(dropout),
#             nn.Linear(feature_dim // 2, output_dim)
#         )

#     def forward(self, cls_feature, encoder_features):
#         """
#         cls_feature: [B, 1, C] (learnable)
#         pooled_feature: [B, 1, C] (pooled DINO encoder features)
#         """
#         # Cross-attention
#         B = encoder_features.size(0)

#         # Repeat learnable query for batch
#         cls_query = self.learnable_query.expand(B, -1, -1)  # [B, 1, C]

#         # Cross-attention
        
#         attn_output, _ = self.cross_attn(
#             query=cls_query,
#             key=encoder_features,
#             value=encoder_features
#         )
#         # cls_feature = cls_query + attn_output  # residual
#         # cls_feature = torch.cat([cls_query, attn_output], dim=-1) 
#         cls_feature = self.norm1(attn_output)

#         # Flatten [B, 1, C] → [B, C]
#         cls_feature = cls_feature.squeeze(1)

#         # Feature extraction
#         features = self.feature_extractor(cls_feature)

#         # Classification
#         logits = self.classifier(features)
#         return logits, features
class ImageClassifier_feat(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=768, dropout=0.1):
        super(ImageClassifier_feat, self).__init__()
        self.classifier_feat = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.classifier_feat(x)
# class ImageClassifier2(nn.Module):
#     def __init__(self, input_dim=256, hidden_dim=256, output_dim=45, dropout=0.1):
#         super(ImageClassifier2, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LeakyReLU(negative_slope=0.01),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.classifier(x)

class MultimodalQFormerWithTextCrossAttn(nn.Module):
    def __init__(self, vision_dim=256, text_dim=768, num_queries=64, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, vision_dim)).cuda()  # Learnable queries

        # Cross-attention: queries attend to vision features (encoder + decoder)
        self.vision_attn = nn.MultiheadAttention(embed_dim=vision_dim, num_heads=num_heads, batch_first=True)

        # Project vision attended output from vision_dim → text_dim
        self.vision_proj = nn.Linear(vision_dim, text_dim)

        # Cross-attention: queries attend to text embeddings
        self.text_attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, batch_first=True)

        # Optional final projection or layer norm (add as needed)
        self.final_proj = nn.Linear(text_dim, text_dim)
        self.norm = nn.LayerNorm(text_dim)

    def forward(self, encoder_output, decoder_output, text_embeddings):
        """
        Args:
            encoder_output: [B, N_enc, vision_dim] (e.g. [B, 1154, 256])
            decoder_output: [B, N_dec, vision_dim] (e.g. [B, 900, 256])
            text_embeddings: [B, num_queries, text_dim] (e.g. [B, 64, 768])

        Returns:
            Tensor of shape [B, num_queries, text_dim] — fused multimodal output
        """
        B = encoder_output.size(0)

        # Combine vision tokens
        # vision_feats
        vision_feats = encoder_output.unsqueeze(1).detach()#torch.cat([encoder_output, decoder_output], dim=1)  # [B, 2054, vision_dim]

        # Expand queries for batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, 64, vision_dim]

        # # Step 1: Cross-attention queries -> vision features
        vision_attended, _ = self.vision_attn(query=queries, key=vision_feats, value=vision_feats)  # [B, 64, vision_dim]

        # Project vision attended output to text embedding dim
        vision_proj = self.vision_proj(vision_attended)  # [B, 64, text_dim]

        # Step 2: Cross-attention queries (vision_proj) -> text embeddings
        text_attended, _ = self.text_attn(query=text_embeddings, key=vision_proj, value=vision_proj)  # [B, 64, text_dim]

        # Optional: residual + norm + final projection
        combined = self.norm(text_attended + vision_proj)
        output = self.final_proj(combined)  # [B, 64, text_dim]

        return output
    
class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, iter_update=False,
                    query_dim=2, 
                    random_refpoints_xy=False,
                    fix_refpoints_hw=-1,
                    num_feature_levels=1,
                    nheads=8,
                    # two stage
                    two_stage_type='no', # ['no', 'standard']
                    two_stage_add_query_num=0,
                    dec_pred_class_embed_share=True,
                    dec_pred_bbox_embed_share=True,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    decoder_sa_type = 'sa',
                    num_patterns = 0,
                    dn_number = 100,
                    dn_box_noise_scale = 0.4,
                    dn_label_noise_ratio = 0.5,
                    dn_labelbook_size = 100,
                    dn="seg",
                    noise_scale=0.4,
                    dn_num=100,
                    initial_pred=True,
                    classification_head=False
                    
        
                    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size
        # self.learn_tgt = learn_tgt
        self.dn=dn
        self.noise_scale=noise_scale
        self.dn_num=dn_num
        self.initial_pred = initial_pred
        self.class_embed_seg = nn.Linear(hidden_dim, num_classes+1)
        self.text_proj_dec = nn.Linear(768, 256).cuda()  # 768 -> 256
        self.mask_embed = MLP(hidden_dim, hidden_dim, 256, 3)
        self.dformer = MultimodalQFormerWithTextCrossAttn()
        for param in self.dformer.parameters():
            param.requires_grad = False
            print(f"Frozen: requires_grad = {param.requires_grad}")
        self.classifier_image = ImageClassifier2().cuda()
        # self.classification_decoder = classificationdecoder(
        #                     input_dim=256, 
        #                     feature_dim=368, 
        #                     output_dim=45, 
        #                     num_heads=4
        #                 ).cuda()
        self.classifier_feat =  ImageClassifier_feat().cuda()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.classification_head=classification_head
        
        # self.pool = nn.AdaptiveAvgPool1d(1)  # global pooling over tokens
        # self.classifier_image = nn.Sequential(
        #                         nn.Linear(512, 256),
        #                         nn.LeakyReLU(negative_slope=0.01),
        #                         nn.Dropout(0.25),
        #                         nn.Linear(256, 26)
        #                     ) # final linear layer
        # prepare input projection layers
        # self.image_classifier = MemoryClassifier(hidden_dim=512, num_classes=26)
        # self.bart_model = AutoModelForSeq2SeqLM.from_pretrained('GanjinZero/biobart-v2-base').cuda()
        # self.bart_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/biobart-v2-base')
        self.bart_tokenizer  = BartTokenizer.from_pretrained("/home/iml/DINO/coco_data/bart_models/finetuned_bart_manual")
        self.bart_model = BartForConditionalGeneration.from_pretrained("/home/iml/DINO/coco_data/bart_models/finetuned_bart_manual").cuda()
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.class_tokens = nn.ParameterList([
    nn.Parameter(torch.randn(1, 256, 1, 1)) for _ in range(self.num_feature_levels)
])
        self.class_pos_tokens = nn.ParameterList([
    nn.Parameter(torch.randn(1, 256, 1, 1)) for _ in range(self.num_feature_levels)
])
        
        
        # self.num_morphology_features = 6
        # self.num_classes_per_feature = 3
        # self.morphology_embed = nn.ModuleList([
        # nn.Linear(self.hidden_dim, self.num_classes_per_feature)
        # for _ in range(self.num_morphology_features)
        # ])
        
        num_morph_attributes = 6
        num_morph_classes = 2
        
              
        total_morphology_classes = num_morph_attributes * num_morph_classes  # 6 * 3 = 18

        _morphology_embed = nn.Linear(hidden_dim, total_morphology_classes)

        if dec_pred_class_embed_share:
            morphology_embed_layerlist = [_morphology_embed for _ in range(transformer.num_decoder_layers)]
        else:
            morphology_embed_layerlist = [copy.deepcopy(_morphology_embed) for _ in range(transformer.num_decoder_layers)]

        self.morphology_embed = nn.ModuleList(morphology_embed_layerlist)
                    
                    
            
        
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value  # did not add for the morphology 
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        # self.class_embed = nn.ModuleList([
        #     nn.Linear(64, num_classes)
        #     for _ in range(transformer.num_decoder_layers)
        # ])
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
    
            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
    
            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
         # Initialize morphology embedding layers
        # for morphology_layer_list in self.morphology_embed:
        #     for morphology_layer in morphology_layer_list:
        #         nn.init.normal_(morphology_layer.weight, std=0.01)
        #         nn.init.constant_(morphology_layer.bias, 0)
    def dn_post_process_seg(self,outputs_class,outputs_coord,mask_dict,outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1],'pred_masks': output_known_mask[-1]}

        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask,output_known_coord)
        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class, outputs_coord, outputs_mask
    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:

            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))
    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num,self.noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num)>0:
                scalar = scalar//(int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_bbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_bbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_bboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of bbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_bbox_expand)
                diff[:, :2] = known_bbox_expand[:, 2:] / 2
                diff[:, 2:] = known_bbox_expand[:, 2:]
                known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

            m = known_labels_expaned.long().to('cuda')
            input_label_embed = self.label_enc(m)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
            padding_bbox = torch.zeros(pad_size, 4).cuda()

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label=padding_label.repeat(batch_size, 1, 1)
                input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label=None
                input_query_bbox=None
            attn_mask = None
            mask_dict=None

        # 100*batch*256
        if not input_query_bbox is None:
            input_query_label = input_query_label
            input_query_bbox = input_query_bbox

        return input_query_label,input_query_bbox,attn_mask,mask_dict
    
    def get_prompt_embeddings(self,prompts, device='cpu'):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

        # Tokenize the list of prompts
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            # Get the [CLS] token embeddings for each prompt
            text_embeddings = bert_model(**tokenized_prompts).last_hidden_state[:, 0, :]  # shape: [N, 768]

        return text_embeddings  # [num_prompts, 768]
    def shift_right(self,input_ids, bos_token_id):
        # Shift tokens one step to the right and insert BOS token at the front
        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[:, 0] = bos_token_id
        shifted[:, 1:] = input_ids[:, :-1]
        return shifted
    
    def repeat_to_fill(self, text, target_len=64, tokenizer=None):
        if not isinstance(text, str):
            text = ", ".join(text)

        # Get token count (excluding special tokens)
        base_tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        token_len = len(base_tokens)

        if token_len >= target_len:
            return text

        # Repeat the string enough times and join with commas
        repeat_count = (target_len // token_len) + 1
        repeated_text = ", ".join([text] * repeat_count)

        return repeated_text
    def get_meaningful_embeddings(self,text_embeddings, attention_mask):
        batch = []
        for i in range(text_embeddings.size(0)):
            valid_embeddings = text_embeddings[i][attention_mask[i].bool()]  # (valid_len, hidden_dim)
            batch.append(valid_embeddings)
        return pad_sequence(batch, batch_first=True)
    def encode_prompts(self, input_text,target_text, bart_model, bart_tokenizer, device='cuda'):
        # bart_model.eval()  # No gradients for encoder
        # processed_text = [
        #     text if isinstance(text, str) else ", ".join(text)
        #     for text in input_text
        # ]
        processed_text = [
                            f"This image is for the diagnosis of {text if isinstance(text, str) else ', '.join(text)}"
                            for text in input_text
]
#         processed_text = [
#             self.repeat_to_fill(text, target_len=64, tokenizer=bart_tokenizer)
#             for text in input_text
# ]

        input_enc = bart_tokenizer(
            processed_text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        if target_text:
            # Join target_text entries if they are not strings
            target_text = [
                text if isinstance(text, str) else ", ".join(text)
                for text in target_text
            ]
            with bart_tokenizer.as_target_tokenizer():
                target_enc = bart_tokenizer(
                    target_text,
                    max_length=64,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

            labels = target_enc["input_ids"]
            labels[labels == bart_tokenizer.pad_token_id] = -100
        else:
            labels = None

        input_ids = input_enc["input_ids"].to(device)
        attention_mask = input_enc["attention_mask"].to(device)
        if labels is not None:
            labels = labels.to(device)

        # Step 1: Encoder forward
        with torch.no_grad():
            encoder_outputs = bart_model.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        # Step 2: Prepare decoder inputs (only if labels exist)
        if labels is not None:
            decoder_input_ids = bart_model.prepare_decoder_input_ids_from_labels(labels)
        else:
            decoder_input_ids = None  # or handle as needed

        # meaningful_embeddings = self.get_meaningful_embeddings(encoder_hidden_states, attention_mask)

        return encoder_hidden_states, attention_mask, decoder_input_ids,labels
    def decode_from_encoder(
                self, 
            encoder_hidden_states, 
            attention_mask, 
            decoder_input_ids,  # decoder inputs for teacher forcing
            labels,             # target output tokens for loss
            model, 
            tokenizer, 
            device='cuda'
        ):
            batch_size = encoder_hidden_states.size(0)
            max_length=64
            # Wrap encoder outputs
            decoder_outputs = model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask
        )

            # Step 4: Compute logits and loss
            logits = model.lm_head(decoder_outputs.last_hidden_state)
            #labels_for_loss=labels.clone()
            #labels_for_loss[labels_for_loss == -100]=tokenizer.pad_token_id
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            generated_ids = model.generate(
                input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            # labels = labels.clone()
            # labels[labels == -100] = tokenizer.pad_token_id
# decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_labels= tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            return  loss,decoded_preds,decoded_labels
    def train_step_bart(self,input_text, target_text, model, tokenizer, device='cuda'):
        """
        A single function to tokenize, compute loss, and get predictions for BART.
        """
        model.train()

        # Tokenize inputs
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True
        ).to(device)

        # Tokenize targets (labels)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                target_text,
                return_tensors="pt",
                max_length=64,
                padding="max_length",
                truncation=True
            )["input_ids"].to(device)

        # Mask pad tokens in labels
        labels[labels == tokenizer.pad_token_id] = -100

        # Forward pass through BART
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels
        )

        loss = outputs.loss

        # Generate predictions (optional: during training/validation)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        predicted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return loss, predicted_text 
        
    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed_seg(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask
    def forward(self, samples: NestedTensor,  prompt, targets:List=None,):
        
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        device=features[0].device
        # bs = features[0].mask.shape[0]
        # cls_tok = self.cls_token.expand(bs, -1, -1)  # [bs, 1, C]
        # cls_pos = self.cls_pos.expand(bs, -1, -1)    # [bs, 1, C]
        
        srcs = []
        masks = []
        # poss = []
        # for l, feat in enumerate(features):
        #     src, mask = feat.decompose()
        #     srcs.append(self.input_proj[l](src))
        #     masks.append(mask)
        #     assert mask is not None
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        # for l, feat in enumerate(features):
        #     src, mask = feat.decompose()  # src: [B, C, H, W]
        #     src_proj = self.input_proj[l](src)  # [B, C, H, W]

        #     B, C, H, W = src_proj.shape

        #     # Prepare class token
        #     class_token = self.class_tokens[l].expand(B, -1, 1, 1)  # [B, C, 1, 1]

        #     # Pad to get [B, C, H+1, W+1]: pad top=1, left=1
        #     src_with_cls = F.pad(src_proj, (1, 0, 1, 0))  # pad (left, right, top, bottom)
        #     src_with_cls[:, :, 0, 0] = class_token.squeeze(-1).squeeze(-1)  # place class token at top-left

        #     srcs.append(src_with_cls)

        #     # Update mask (1 for masked, 0 for visible): pad same way
        #     mask = F.pad(mask, (1, 0, 1, 0), value=False)  # [B, H+1, W+1]
        #     masks.append(mask)

        #     # Positional encoding for padded input
        #     # compute pos embedding for original src_proj, then pad on top & left
        #     pos = self.backbone[1](NestedTensor(src_proj, mask[:, 1:, 1:]))  # mask without the new top & left row/col
        #     pos = F.pad(pos, (1, 0, 1, 0))  # pad left=1, right=0, top=1, bottom=0
        #     poss.append(pos)
        
        
        # if self.num_feature_levels > len(srcs):
        #     _len_srcs = len(srcs)
        #     for l in range(_len_srcs, self.num_feature_levels):
        #         if l == _len_srcs:
        #             src = self.input_proj[l](features[-1].tensors)
        #         else:
        #             src = self.input_proj[l](srcs[-1])
        #         m = samples.mask
        #         mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
        #         pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                
        #         srcs.append(src)
        #         masks.append(mask)
        #         poss.append(pos_l)
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)  # [B, C, H, W]
                else:
                    # Remove previously added class token patch before reuse
                    # srcs[-1] shape: [B, C, H+1, W+1]
                    prev_src = srcs[-1]
                    src = prev_src[:, :, 1:, 1:]  # crop off the first row and column (top-left)

                    src = self.input_proj[l](src)  # project again if needed

                B, C, H, W = src.shape

                # Prepare class token spatial patch
                class_token = self.class_tokens[l].expand(B, -1, 1, 1)  # [B, C, 1, 1]

                # Pad src spatially (H+1, W+1): pad bottom and right by 1
                src_with_cls = F.pad(src, (1, 0, 1, 0))  # pad left=1, right=0, top=1, bottom=0
                src_with_cls[:, :, 0, 0] = class_token.squeeze(-1).squeeze(-1)  # place class token at top-left

                # Prepare mask: interpolate to src spatial shape then pad
                m = samples.mask  # [B, H_orig, W_orig]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:], mode="nearest").to(torch.bool)[0]  # [B, H, W]
                mask = F.pad(mask, (1, 0, 1, 0), value=False)  # pad left=1, right=0, top=1, bottom=0

                srcs.append(src_with_cls)
                masks.append(mask)

                # Positional encoding: compute on unpadded src, then pad
                pos_l = self.backbone[1](NestedTensor(src, mask[:, 1:, 1:])).to(src.dtype)  # mask before pad
                pos_l = F.pad(pos_l, (1, 0, 1, 0))  # pad left=1, right=0, top=1, bottom=0
                poss.append(pos_l)

                
                
        if self.dn_number > 0 and (targets and len(targets) > 0):
            
            has_segmentation = any(
                t.get("segmentation") is not None and t["segmentation"].item() != 1
                for t in targets
            )

            if has_segmentation:
                input_query_label, input_query_bbox, attn_mask, dn_meta = \
                    prepare_for_cdn(
                        dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                        training=self.training,
                        num_queries=self.num_queries,
                        num_classes=self.num_classes,
                        hidden_dim=self.hidden_dim,
                        label_enc=self.label_enc
                    )
            else:
            
              input_query_bbox = input_query_label = attn_mask = dn_meta = None
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None
            
        
        # just bert embeding token for encoder 
        
        
        # embedding_file_path = f"prompts/{prompt}_embeddings.pt"  # e.g., "leukemia_embeddings.pt" or "malaria_embeddings.pt"

        # # Check if the embeddings file exists
        # if os.path.exists(embedding_file_path):
          
        #     text_embeddings = torch.load(embedding_file_path)
      
        # else:
        #     text_embeddings  = self.get_prompt_embeddings(prompt, device='cuda')
        #     torch.save(text_embeddings, embedding_file_path)  # Save as tensor
        completed_texts = (
                tuple(t['completed_text'] for t in targets if 'completed_text' in t)
                if targets else ()
            )
        self.bart_model.train
        

        text_embeddings, attention_mask, decoder_input_ids,labels_encoder = self.encode_prompts(prompt,completed_texts, self.bart_model, self.bart_tokenizer)
        # valid_token_indices = attention_mask[0].nonzero(as_tuple=True)[0]
        # meaningful_embeddings = text_embeddings[0, valid_token_indices, :] 
        # Step 2: Decode
        # 
        # text_embeddings= None
        
       
        
        hs, reference, hs_enc, ref_enc, init_box_proposal,predictions_class, predictions_mask,mask_features,enc_class_tokens_encder,encoder_output,decoder_output,_,cls_feature, pooled_feature= self.transformer(srcs,text_embeddings,prompt, masks, input_query_bbox, poss,input_query_label,attn_mask)
        # In case num object=0
        

        logits_image = self.classifier_image(enc_class_tokens_encder.squeeze(1))
        # logits_image, enc_class_tokens = self.classification_decoder(cls_feature, pooled_feature)
        
        logits_image_featuers= self.classifier_feat (enc_class_tokens_encder.squeeze(1))

        hs[0] += self.label_enc.weight[0,0]*0.0
        text_embeddings_attented = self.dformer(enc_class_tokens_encder, decoder_output, text_embeddings)
        
        # enc_class_tokens_encder = torch.cat([enc_class_tokens_encder, cls_feature], dim=-1) 
        # text_loss,decoded_text,decoded_labels= self.decode_from_encoder(text_embeddings_attented, attention_mask,decoder_input_ids,labels_encoder, self.bart_model, self.bart_tokenizer, device)
        if decoder_input_ids is not None:
            text_loss, decoded_text, decoded_labels = self.decode_from_encoder(
                text_embeddings_attented,
                attention_mask,
                decoder_input_ids,
                labels_encoder,
                self.bart_model,
                self.bart_tokenizer,
                device
            )
        else:
            # Set safe fallback values
            labels_encoder= None
            text_loss = torch.tensor(0.0, device=device)
            decoded_text = [""] #* text_embeddings_attented.size(0)  # empty text for each batch
            decoded_labels = torch.zeros_like(labels_encoder) if labels_encoder is not None else None

        # text_loss,decoded_text= self.train_step_bart(prompt,completed_texts, self.bart_model, self.bart_tokenizer)
        # deformable-detr-like anchor text_embeddings_attented
        # print(decoded_text)
        # IMAGE CLASSIFICATION
        
        
        # logits_image = self.image_classifier(memory, memory_class)
        # memory_pooled = self.pool(memory.transpose(1, 2)).squeeze(-1)  # → [B, D]

        # # logits: [B, num_classes]
        
        # combined_memory = torch.cat([memory_pooled, memory_class], dim=1)
        
        # logits_image = self.classifier_image(combined_memory)
        
        
        

        # output_image_classifies = {"logits": logits_image}

        # if targets is not None:
        #     # extract category_id from target dicts
        #     labels_image_classifier = torch.stack([t["category_id"] for t in targets])  # [B]
        #     loss_image_classifier = F.cross_entropy(output_image_classifies, labels_image_classifier)
        #     # output["loss"] = loss_image_classifier
        
        
        
        
        
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)        

        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        # projected_text=self.text_proj_dec(text_embeddings)  
        # outputs_class = torch.stack([
        #     layer_cls_embed(
        #         torch.matmul(
        #             F.normalize(layer_hs, dim=-1),          # (B, Q, D)
        #             projected_text.transpose(1, 2)           # (B, D, C)
        #         )                                           # (B, Q, C)
        #     )
        #     for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
        # ])
        
        
        
        
        # outputs_morphology = []

        # # Collect outputs for each morphology feature
        # for feature_idx in range(self.num_morphology_features):
        #     # Pass the hidden states from the last decoder layer to the morphology head for each feature
        #     # hs[-1] represents the last decoder layer's hidden states
        #     feature_morphology_outputs = self.morphology_embed[feature_idx](hs[-1])  # Shape: [batch_size, num_queries, num_classes_per_feature
        #     # Append the output for this feature
        #     outputs_morphology.append(feature_morphology_outputs)
            
            
        
        # outputs_morphology_list = []
        # # Loop through each decoder layer to predict morphology at every step (similar to class and bbox)
        # for dec_lid, layer_hs in enumerate(hs):
        #     # Collect outputs for each morphology feature at this decoder layer
        #     outputs_morphology_layer = []
        #     for feature_idx in range(self.num_morphology_features):
        #         feature_morphology_output = self.morphology_embed[feature_idx](layer_hs)  # Predict morphology for this feature
        #         outputs_morphology_layer.append(feature_morphology_output)
        #     # Stack the morphology predictions for all features at this layer
        #     outputs_morphology_layer = torch.stack(outputs_morphology_layer, dim=-1)  # Shape: [batch_size, num_queries, num_classes_per_feature, num_features]
        #     outputs_morphology_list.append(outputs_morphology_layer)
        # # Stack predictions across layers (same as bbox and class)
        # outputs_morphology = torch.stack(outputs_morphology_list, dim=0)
        
        # outputs_morphology = []
        # for layer_morphology_embed, layer_hs in zip(self.morphology_embed, hs):
        #     morphology_preds = [morph_layer(layer_hs) for morph_layer in layer_morphology_embed]
        #     outputs_morphology.append(morphology_preds)
        # predictions_class = []
        predictions_masks = []
        # tgt_mask = None
        # mask_dict = None
        # if self.dn != "no" and self.training:
        #     assert targets is not None
        #     input_query_label, input_query_bbox, tgt_mask, mask_dict = \
        #         self.prepare_for_dn(targets, None, None, srcs[0].shape[0])
        #     if mask_dict is not None:
        #         tgt=torch.cat([input_query_label, tgt],dim=1)

        # direct prediction from the matching and denoising part in the begining
        segem= False 
        
        
        if segem:
            for i, output in enumerate(hs):
                outputs_classes, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), mask_features, self.training or (i == len(hs)-1))
                # predictions_class.append(outputs_class)
                predictions_masks.append(outputs_mask)

        # iteratively box prediction
        # if self.initial_pred:
        #     out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
        #     assert len(predictions_class) == self.num_layers + 1
        # else:
        #     out_boxes = self.pred_box(references, hs)
        # if mask_dict is not None:
        #     predictions_mask=torch.stack(predictions_mask)
        #     predictions_class=torch.stack(predictions_class)
        #     predictions_class, out_boxes,predictions_mask=\
        #         self.dn_post_process(predictions_class,outputs_coord_list,mask_dict,predictions_mask)
        #     predictions_class,predictions_mask=list(predictions_class),list(predictions_mask)
        
        
        
        outputs_morphology = torch.stack([
        layer_morph_embed(layer_hs)  # Shape: [batch_size, num_queries, total_morphology_classes]
        for layer_morph_embed, layer_hs in zip(self.morphology_embed, hs)
         ])
            
        
        # predictions_masks=torch.stack(predictions_masks)
        # predictions_class=torch.stack(predictions_class)
        # predictions_class, out_boxes,predictions_mask=\
        #         self.dn_post_process_seg(predictions_class,outputs_coord_list,mask_dict,predictions_mask)
        # predictions_class,predictions_mask=list(predictions_class),list(predictions_mask)
        
        
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list, outputs_morphology= \
                dn_post_process(outputs_class, outputs_coord_list,outputs_morphology,
                                dn_meta,self.aux_loss,self._set_aux_loss)
        text_embeddings, attention_mask, decoder_input_ids,labels_encoder = self.encode_prompts(prompt,completed_texts, self.bart_model, self.bart_tokenizer)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1], 'pred_morphology': outputs_morphology[-1], 'pred_mask':predictions_mask[-1],'pred_image_class':logits_image,'pred_image_feat':logits_image_featuers,'loss_text':text_loss,'pred_text':decoded_text,'completed_text':decoded_labels,'encoder_class_feat':enc_class_tokens_encder,'text_embeddings':text_embeddings}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord_list, outputs_morphology
            )



        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_morphology):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # return [{'pred_logits': a, 'pred_boxes': b}
        #         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        # return [{'pred_logits': a, 'pred_boxes': b, 'pred_morphology': torch.stack(c, dim=-1)}
        #     for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_morphology)]
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_morphology': c}
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_morphology[:-1])]


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip  #  prevents nan for large gamma_neg numbers
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """"
        Parameters:
        x: input logits (before sigmoid), shape [N, num_classes]
        y: targets (multi-label binarized vector), shape [N, num_classes]
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        loss = y * torch.log(xs_pos.clamp(min=self.eps)) + (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)
        loss *= one_sided_w

        return -loss.sum()
def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule

def dice_loss2(pred, target, smooth=1.):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))
def focal_loss(pred, target, alpha=0.25, gamma=2.):
        pred = pred.view(-1)
        target = target.view(-1)
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule
def calculate_uncertainty(logits):
        """
        We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
            foreground class in `classes`.
        Args:
            logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
                class-agnostic, where R is the total number of predicted masks in all images and C is
                the number of foreground classes. The values are logits.
        Returns:
            scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
                the most uncertain locations having the highest uncertainty score.
        """
        assert logits.shape[1] == 1
        gt_class_logits = logits.clone()
        return -(torch.abs(gt_class_logits))
    
class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    def loss_morphology(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the morphology predictions, ignoring labels equal to 4."""
        assert 'pred_morphology' in outputs

        idx = self._get_src_permutation_idx(indices)

        src_morphology = outputs['pred_morphology'][idx]  # Shape: [num_matched, total_morphology_classes]

        # Define number of attributes and classes per attribute
        num_attributes = 6
        num_classes_per_attribute = 2  # Valid labels are 0 and 1
        src_morphology = src_morphology.view(-1, num_attributes, num_classes_per_attribute)

        # Get target morphology attributes
        if targets[0]['morphology'].numel() > 0:

            target_morphology = torch.cat([
                t['morphology'][i] for t, (_, i) in zip(targets, indices)
            ], dim=0)  # Shape: [num_matched, num_attributes]

            # Create mask for valid labels (labels not equal to 4)
            
            
            mask = target_morphology != 4  # Mask for the none class 

            # Flatten the tensors
            src_morphology = src_morphology.reshape(-1, num_classes_per_attribute)  # [num_matched * num_attributes, num_classes_per_attribute]
            target_morphology = target_morphology.reshape(-1)  # [num_matched * num_attributes]
            mask = mask.reshape(-1)  # [num_matched * num_attributes], dtype=bool

            # Select only valid elements
            src_morphology = src_morphology[mask]
            target_morphology = target_morphology[mask]

            # Check if there are valid elements to compute loss
            if src_morphology.numel() == 0:
                # No valid elements, return zero loss
                losses = {'loss_morphology': src_morphology.sum()*0}
                return losses

            # Perform one-hot encoding on the target morphology
            target_morphology_onehot = F.one_hot(target_morphology, num_classes=num_classes_per_attribute).float()

            # Compute Asymmetric Loss
            criterion = AsymmetricLoss(
                gamma_neg=4, gamma_pos=1, clip=0.08, disable_torch_grad_focal_loss=True
            ).to(src_morphology.device)
            object_batch_loss = criterion(src_morphology, target_morphology_onehot)

            # Normalize the loss by the number of boxes
            losses = {'loss_morphology': object_batch_loss  / num_boxes} 
            return losses
        else:
            # Handle the case where 't['morphology']' is empty for any of the 'targets'
            losses = {'loss_morphology': torch.tensor(0.0, device=src_morphology.device)}
            return losses

    
    # def loss_morphology(self, outputs, targets, indices, num_boxes):
    #     """Compute the losses related to the morphology predictions using Asymmetric Loss."""
    #     assert 'pred_morphology' in outputs

    #     idx = self._get_src_permutation_idx(indices)
      
    #     src_morphology = outputs['pred_morphology'][idx]  # Shape: [num_matched, 18]
       
    #     # Reshape src_morphology to [num_matched, 6, 3]
    #     src_morphology = src_morphology.view(-1, 6, 3)

    #     # Get target morphology attributes
    #     target_morphology = torch.cat([
    #         t['morphology'][i] for t, (_, i) in zip(targets, indices)
    #     ], dim=0)  # Shape: [num_matched, 6]

    #     # Perform one-hot encoding for each attributeget_loss
    #     num_classes = 3
    #     one_hot_encoded_tensors = []
    #     for i in range(target_morphology.size(1)):
    #         # Extract the current attribute column
    #         column_values = target_morphology[:, i].long()  # Shape: [num_matched]

    #         # Generate one-hot encoded tensor for the current attribute
    #         one_hot_encoded_col = torch.eye(num_classes, device=target_morphology.device)[column_values]
    #         one_hot_encoded_col = one_hot_encoded_col.unsqueeze(1)  # Shape: [num_matched, 1, num_classes]

    #         one_hot_encoded_tensors.append(one_hot_encoded_col)

    #     # Concatenate the one-hot encoded tensors along the attribute dimension
    #     target_morphology_onehot = torch.cat(one_hot_encoded_tensors, dim=1)  # Shape: [num_matched, 6, 3]

    #     # Flatten the predictions and targets for loss computation
    #     src_morphology = src_morphology.reshape(-1, num_classes)       # Shape: [num_matched * 6, 3]
    #     target_morphology_onehot = target_morphology_onehot.reshape(-1, num_classes)  # Shape: [num_matched * 6, 3]

    #     # Compute AsymmetricLoss
    #     criterion = AsymmetricLoss(
    #         gamma_neg=4, gamma_pos=1, clip=0.08, disable_torch_grad_focal_loss=True
    #     ).to(src_morphology.device)
    #     object_batch_loss = criterion(src_morphology, target_morphology_onehot)

    #     # Normalize the loss by the number of boxes
    #     losses = {'loss_morphology': object_batch_loss / num_boxes}
    #     return losses

    # mask loss 
    
    
    # def loss_masks(self, outputs, targets, indices, num_masks):
    #     """Compute the losses related to the masks: the focal loss and the dice loss.
    #     targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    #     """
    #     assert "pred_masks" in outputs

    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)
    #     src_masks = outputs["pred_masks"]
    #     src_masks = src_masks[src_idx]
    #     masks = [t["masks"] for t in targets]
    #     # TODO use valid to mask invalid areas due to padding in loss
    #     target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    #     target_masks = target_masks.to(src_masks)
    #     target_masks = target_masks[tgt_idx]

    #     # No need to upsample predictions as we are using normalized coordinates :)
    #     # N x 1 x H x W
    #     src_masks = src_masks[:, None]
    #     target_masks = target_masks[:, None]

    #     with torch.no_grad():
    #         # sample point_coords
    #         point_coords = get_uncertain_point_coords_with_randomness(
    #             src_masks,
    #             lambda logits: calculate_uncertainty(logits),
    #             self.num_points,
    #             self.oversample_ratio,
    #             self.importance_sample_ratio,
    #         )
    #         # get gt labels
    #         point_labels = point_sample(
    #             target_masks,
    #             point_coords,
    #             align_corners=False,
    #         ).squeeze(1)

    #     point_logits = point_sample(
    #         src_masks,
    #         point_coords,
    #         align_corners=False,
    #     ).squeeze(1)

    #     losses = {
    #         "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
    #         "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
    #     }

    #     del src_masks
    #     del target_masks
    #     return losses
    
    

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes


        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'morphology': self.loss_morphology,
            'masks': self.loss_masks,
            
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def forward(self, outputs, targets,args, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        if any(t.get("masked_traning", torch.tensor(0)).item() == 1 for t in targets):
            # extract category_id from target dicts
            losses = {}
            
            
            l_dict = dict()
            text_loss= outputs['loss_text']
            # Add segmentation losses
            # l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['image_classification_loss']  = torch.as_tensor(0.).to('cuda')

            # Add segmentation losses
            # l_dict['loss_dice_seg'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_focal_seg'] = torch.as_tensor(0.).to('cuda')

            # Apply index suffix
            # l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
            l_dict['text_loss'] = text_loss
            # Update main loss dict
            
            losses.update(l_dict)
            return losses
            # output["loss"] = loss_image_classifier
        
        
        
        if any(t.get("classification", torch.tensor(0)).item() == 2 for t in targets):
            # extract category_id from target dicts
            losses = {}
            
            labels_image_classifier = torch.stack([t["category_id"] for t in targets])  
            
            invalid_mask = (labels_image_classifier < 0) | (labels_image_classifier >= 26)

            if invalid_mask.any():
                invalid_values = labels_image_classifier[invalid_mask]
                # raise ValueError(f"❌ Invalid target values found: {invalid_values.tolist()} — Valid range is [0, {5 - 1}]")# [B]
            loss_image_classifier = F.cross_entropy(outputs['pred_image_class'], labels_image_classifier)
            loss_image_featuers = F.cross_entropy(F.normalize(outputs['pred_image_feat']), F.normalize(outputs['text_embeddings'].mean(dim=1)).detach() )  
            l_dict = dict()

            # Add segmentation losses
            # l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['image_classification_loss']  = torch.as_tensor(0.).to('cuda')

            # Add segmentation losses
            # l_dict['loss_dice_seg'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_focal_seg'] = torch.as_tensor(0.).to('cuda')

            # Apply index suffix
            # l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
            l_dict['image_classification_loss'] = loss_image_classifier #+loss_image_featuers
            # Update main loss dict
            
            losses.update(l_dict)
            return losses
            # output["loss"] = loss_image_classifier
        
        
        
        if any(t.get("segmentation", torch.tensor(0)).item() == 1 for t in targets):
            losses = {}
            pred_masks = outputs['pred_mask'][:, 0]  # shape: (B, 64, 64)

            # Upsample to match ground truth
            pred_masks = F.interpolate(pred_masks.unsqueeze(1), size=(512, 512), mode='bilinear', align_corners=False)
            # shape: (B, 1, 512, 512)
            gt_masks = torch.stack([t['binary_mask'] for t in targets]) 
            gt_masks = gt_masks.to(pred_masks.device)
            pred_probs = torch.sigmoid(pred_masks)
            dice = dice_loss2(pred_probs, gt_masks)
            focal = focal_loss(pred_probs, gt_masks)

            l_dict = dict()
            # l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
            # l_dict['image_classification_loss']  = torch.as_tensor(0.).to('cuda')

            # Add segmentation losses
            l_dict['loss_dice_seg'] = dice.to('cuda')
            l_dict['loss_focal_seg'] = focal.to('cuda')

            # Apply index suffix
            # l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}

            # Update main loss dict
            losses.update(l_dict)
            del pred_probs
            del gt_masks

            # mask losses 
            # pred_masks: shape (B, 1, 512, 512), after sigmoid and upsampling
# gt_masks: shape (B, 1, 512, 512)

  # scalar tensor

            return losses
        
             
        else:
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
            device=next(iter(outputs.values())).device
            indices = self.matcher(outputs_without_aux, targets)

            if return_indices:
                indices0_copy = indices
                indices_list = []

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            losses = {}

            # prepare for dn loss
            dn_meta = outputs['dn_meta']

            if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                output_known_lbs_bboxes,single_pad, scalar = self.prep_for_dn(dn_meta)

                dn_pos_idx = []
                dn_neg_idx = []
                for i in range(len(targets)):
                    if len(targets[i]['labels']) > 0:
                        t = torch.range(0, len(targets[i]['labels']) - 1).long().cuda()
                        t = t.unsqueeze(0).repeat(scalar, 1)
                        tgt_idx = t.flatten()
                        output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                        output_idx = output_idx.flatten()
                    else:
                        output_idx = tgt_idx = torch.tensor([]).long().cuda()

                    dn_pos_idx.append((output_idx, tgt_idx))
                    dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

                output_known_lbs_bboxes=dn_meta['output_known_lbs_bboxes']
                l_dict = {}
                for loss in self.losses:
                    kwargs = {}
                    if 'labels' in loss:
                        kwargs = {'log': False}
                    l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar,**kwargs))

                l_dict = {k + f'_dn': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                losses.update(l_dict)

            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                    indices = self.matcher(aux_outputs, targets)
                    if return_indices:
                        indices_list.append(indices)
                    for loss in self.losses:
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

                    if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                        aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                        l_dict={}
                        for loss in self.losses:
                            kwargs = {}
                            if loss == 'morphology' and any('morphology_attributes' not in t for t in targets):
                                continue  # Skip morphology loss if attributes are missing
                            if 'labels' in loss:
                                kwargs = {'log': False}

                            l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                                    **kwargs))

                        l_dict = {k + f'_dn_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    else:
                        l_dict = dict()
                        l_dict['loss_bbox_dn']=torch.as_tensor(0.).to('cuda')
                        l_dict['loss_giou_dn']=torch.as_tensor(0.).to('cuda')
                        l_dict['loss_ce_dn']=torch.as_tensor(0.).to('cuda')
                        l_dict['loss_xy_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict['loss_hw_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict['cardinality_error_dn'] = torch.as_tensor(0.).to('cuda')
                        l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            # interm_outputs loss
            if 'interm_outputs' in outputs:
                interm_outputs = outputs['interm_outputs']
                indices = self.matcher(interm_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'morphology':
                        if 'pred_morphology' not in interm_outputs:
                            # Skip morphology loss if not available
                            continue
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # enc output loss
            if 'enc_outputs' in outputs:
                for i, enc_outputs in enumerate(outputs['enc_outputs']):
                    indices = self.matcher(enc_outputs, targets)
                    if return_indices:
                        indices_list.append(indices)
                    for loss in self.losses:
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                        l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            if return_indices:
                indices_list.append(indices0_copy)
                return losses, indices_list

            return losses

        def prep_for_dn(self,dn_meta):
            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            num_dn_groups,pad_size=dn_meta['num_dn_group'],dn_meta['pad_size']
            assert pad_size % num_dn_groups==0
            single_pad=pad_size//num_dn_groups

            return output_known_lbs_bboxes,single_pad,num_dn_groups


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        out_morphology = outputs['pred_morphology']
        batch_size, num_queries, total_morph_classes = out_morphology.shape
        num_attributes = 6 
        num_classes_per_attribute = 2  #
        # Reshape to [batch_size, num_queries, num_attributes, num_classes_per_attribute]
        out_morphology = out_morphology.view(batch_size, num_queries, num_attributes, num_classes_per_attribute)
        # Apply softmax to get probabilities per attribute
        morphology_probs = F.softmax(out_morphology, dim=-1)  # [batch_size, num_queries, num_attributes, num_classes_per_attribute]
        # Get predicted morphology labels per attribute
        morphology_labels = morphology_probs.argmax(-1) 
        

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        if test:
            assert not not_to_xyxy
            boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        morphology_labels = torch.gather(morphology_labels, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, num_attributes))

        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i], 'morphology_labels':morph_labels[i]} for s, l, b, i, morph_labels in zip(scores, labels, boxes, item_indices, morphology_labels)]
        else:
            # Corrected else clause
            results = [{'scores': s, 'labels': l, 'boxes': b, 'morphology_labels': m} for s, l, b, m in zip(scores, labels, boxes, morphology_labels)]

        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name='dino')
def build_dino(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = DINO(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number = args.dn_number if args.use_dn else 0,
        dn_box_noise_scale = args.dn_box_noise_scale,
        dn_label_noise_ratio = args.dn_label_noise_ratio,
        dn_labelbook_size = dn_labelbook_size,
        classification_head=args.classification_head
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef,  'loss_morphology': args.morphology_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    
    # for DN training
    if args.use_dn:
        weight_dict['loss_ce_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef
        weight_dict['loss_morphology_dn'] = args.morphology_loss_coef
        weight_dict['loss_focal_seg'] = args.dice_loss_coef
        weight_dict['loss_dice_seg'] = args.focal_loss_coef
        weight_dict['image_classification_loss'] = args.image_loss_coef
        

    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_morphology': 1.0 , 
            'loss_focal_seg': 1.0,
            'loss_dice_seg': 1.0,
             'image_classification_loss': 1.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'morphology']
    # losses += ['focal_seg', 'dice_seg', 'image_classification']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses,
                             )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

>>>>>>> e8a8ec0028059ae8e36eaba4f8a1954505fd2f66
    return model, criterion, postprocessors