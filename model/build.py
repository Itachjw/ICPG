from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import pdb
import torch.nn.functional as F


class ICPG(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self._set_task()
        self.e_l = args.e_l
        self.margin = args.margin

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']  

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')] 
        print(f'Training Model with {self.current_task} tasks')
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND  
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x
    
    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch, flag, image_pseudo_labels=None, n_iter=None, epoch=None):
        ret = dict()
        images = batch['images']    # 
        caption_ids = batch['caption_ids']   
        image_feats, text_feats = self.base_model(images, caption_ids)  
       
        i_feats = image_feats[:, 0, :].float()  # for CLIP ViT-B/16 model   
        # i_feats = image_feats.float()         # for CLIP ResNet visual model

        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float() 

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if flag == True:     # calculate loss
            if 'cdm' in self.current_task:
                ret.update({'cdm_loss':objectives.compute_cdm(i_feats, t_feats, image_pseudo_labels, n_iter, logit_scale)})

            if epoch > self.e_l:
                if 'chm' in self.current_task:
                    ret.update({'chm_loss':objectives.compute_chm(i_feats, t_feats, image_pseudo_labels, self.margin, n_iter)})    

            if 'itc' in self.current_task:
                ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
                        
            if 'id' in self.current_task:
                image_logits = self.classifier(i_feats.half()).float()  
                text_logits = self.classifier(t_feats.half()).float()
                ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

                image_pred = torch.argmax(image_logits, dim=1)
                text_pred = torch.argmax(text_logits, dim=1)

                image_precision = (image_pred == batch['pids']).float().mean()
                text_precision = (text_pred == batch['pids']).float().mean()
                ret.update({'img_acc': image_precision})
                ret.update({'txt_acc': text_precision})
            return ret
        
        else :  # for clustering
            return i_feats
   

def build_model(args, num_classes=11003):
    model = ICPG(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
