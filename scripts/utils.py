import argparse, os, sys, glob

import cv2
from omegaconf import OmegaConf
from PIL import Image, ImageFilter
from tqdm import tqdm
import numpy as np
import torch
from torchvision.models.resnet import resnet50
from torchvision import transforms
from transformers import AutoImageProcessor, ViTMAEForPreTraining
from transformers import CLIPProcessor, CLIPModel
from transformers import ViTImageProcessor, ViTModel

class Diff_Calculater:
    def __init__(self, backbone_type):
        # self.model = resnet50(pretrained=True)
        # self.model.eval()
        # self.tf = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225])])
        if backbone_type == 'vitmae':
            self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
            self.model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")
            self.model.config.mask_ratio = 0
        elif backbone_type == 'vit':
            self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        elif backbone_type == 'vitclip':
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        self.accumulated_mask = np.zeros((512, 512))
        self.accumulated_diff = np.zeros((512, 512))
        self.backbone_type = backbone_type

    def extract_feat_vitclip(self, x):
        with torch.no_grad():
            inputs = self.image_processor(images=x, return_tensors="pt")
            outputs = self.model.vision_model(**inputs, output_hidden_states=True)
            feat = outputs.last_hidden_state
            print(feat.shape)
            feat = torch.reshape(feat[:, 1:, :].permute((0, 2, 1)), (1, -1, 224 // 16,
                                                                     224 // 16))
            return feat

    def extract_feat_vitmae(self, x):
        with torch.no_grad():
            inputs = self.image_processor(images=x, return_tensors="pt")
            outputs = self.model.vit(**inputs, output_hidden_states=True)
            latent = outputs.last_hidden_state
            ids_restore = outputs.ids_restore
            decoder_outputs = self.model.decoder(latent, ids_restore, output_hidden_states=True)
            feat = decoder_outputs.hidden_states[0]
            feat = torch.reshape(feat[:, 1:, :].permute((0, 2, 1)), (1, -1, 224//self.model.config.patch_size,
                                                                     224//self.model.config.patch_size))
        return feat

    def extract_feat_vit(self, x):
        inputs = self.feature_extractor(x, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
        feat = outputs.last_hidden_state
        feat = torch.reshape(feat[:, 1:, :].permute((0, 2, 1)), (1, -1, 14, 14))
        return feat

    def extract_feat(self, x):
        x = self.tf(x).unsqueeze(0)
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
        return x

    def __call__(self, image, patch_image, box_image, mask, box_loc, ord=1):
        # image =
        # image_feat = self.extract_feat(image)
        x, y, xx, yy = box_loc
        mask = np.array(mask)
        if self.backbone_type == 'vit':
            extract_feat = self.extract_feat_vit
        elif self.backbone_type == 'vitmae':
            extract_feat = self.extract_feat_vitmae
        elif self.backbone_type == 'vitclip':
            extract_feat = self.extract_feat_vitclip
        else:
            raise NotImplementedError
        image_feat = extract_feat(image)
        patch_feat = extract_feat(patch_image)
        diff_pat_ori = torch.sum((patch_feat - image_feat) ** 2, dim=1, keepdim=True)
        # torch.linalg.norm((patch_feat - image_feat), ord=ord, dim=1, keepdim=True)
        diff_pat_ori = torch.nn.functional.interpolate(diff_pat_ori, (yy-y, xx-x))[0, 0]

        box_feat = extract_feat(box_image)
        diff_pat_box = torch.sum((patch_feat - box_feat) ** 2, dim=1, keepdim=True)
        # torch.linalg.norm((patch_feat - box_feat), ord=ord, dim=1, keepdim=True)
        diff_pat_box = torch.nn.functional.interpolate(diff_pat_box, (yy-y, xx-x))[0, 0]

        diff_ori_box = torch.sum((image_feat - box_feat) ** 2, dim=1, keepdim=True)
        # torch.linalg.norm((patch_feat - box_feat), ord=ord, dim=1, keepdim=True)
        diff_ori_box = torch.nn.functional.interpolate(diff_ori_box, (yy - y, xx - x))[0, 0]

        # diff_pat_ori = cv2.cvtColor(np.float32(diff_pat_ori), cv2.COLOR_GRAY2BGR)
        # diff_pat_box = cv2.cvtColor(np.float32(diff_pat_box), cv2.COLOR_GRAY2BGR)
        # return diff_pat_ori, diff_pat_box

        # segm = np.zeros_like(diff_pat_box)
        # segm[diff_pat_ori < diff_pat_box] = 255
        # # mask out uninpainted area
        # segm = segm * mask
        # self.accumulated_mask += mask
        # self.accumulated_diff += segm
        # segm = cv2.cvtColor(segm, cv2.COLOR_GRAY2BGR)
        # accumulated = cv2.cvtColor(np.float32(self.accumulated_diff / self.accumulated_mask), cv2.COLOR_GRAY2BGR)

        segm = (diff_ori_box).detach().numpy()  # diff_pat_box-diff_pat_ori

        # mask out uninpainted area
        segm_ = segm
        segm = np.zeros((512, 512))
        segm[y:yy, x:xx] = segm_
        mask_ = mask
        mask = np.zeros((512, 512))
        mask[y:yy, x:xx] = mask_
        mask = np.float32(mask)
        segm = np.float32(segm) * mask

        self.accumulated_diff += + segm
        self.accumulated_mask += mask

        segm = (segm - np.min(segm)) / (np.max(segm) - np.min(segm))
        segm = cv2.cvtColor(255 * segm, cv2.COLOR_GRAY2BGR)
        acc = np.float32(self.accumulated_diff / (1 + self.accumulated_mask))
        # acc = np.float32(self.accumulated_diff)
        acc = (acc - np.min(acc)) / (np.max(acc) - np.min(acc))
        accumulated = cv2.cvtColor(255 * np.float32(acc), cv2.COLOR_GRAY2BGR)

        return segm, accumulated