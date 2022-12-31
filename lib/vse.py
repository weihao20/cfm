"""VSE model"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import ContrastiveLoss
from lib.generator import NegativeGenerator

import logging

logger = logging.getLogger(__name__)

class VSEModel(object):
    """
        The standard VSE model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = get_image_encoder(opt.data_name, opt.img_dim, opt.embed_size,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)

        # Counterfactual generator
        self.generator = NegativeGenerator(self.img_enc, self.txt_enc, opt)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params
        self.opt = opt

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                ],
                    lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))
        else:
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.backbone.top.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.img_enc.backbone.base.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD([
                    {'params': self.txt_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.img_enc.backbone.parameters(), 'lr': opt.learning_rate * opt.backbone_lr_factor,
                     'weight_decay': decay_factor},
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, momentum=0.9, nesterov=True)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def freeze_backbone(self):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.freeze_backbone()
            else:
                self.img_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.unfreeze_backbone(fixed_blocks)
            else:
                self.img_enc.unfreeze_backbone(fixed_blocks)

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def store_cuda(self, images, captions, lengths, image_lengths=None):
        lengths = torch.tensor(lengths).cuda()
        if self.opt.precomp_enc_type == 'basic':
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                image_lengths = image_lengths.cuda()
            return images, captions, lengths, image_lengths
        else:
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            return images, captions, lengths, image_lengths

    def forward_emb(self, images, captions, lengths, image_lengths=None):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if self.opt.precomp_enc_type == 'basic':
            img_emb = self.img_enc(images, image_lengths)
        else:
            img_emb = self.img_enc(images)

        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb

    def forward_tri_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, scores, mask_i2t, mask_t2i = self.criterion.forward_triplet(img_emb, cap_emb, self.logger)
        self.logger.update('Tri', loss.data.item(), img_emb.size(0))
        return loss, scores, mask_i2t, mask_t2i

    def forward_sa_loss(self, img_anc, cap_pos, cap_syn, cap_anc, img_pos, img_syn):
        loss = self.criterion.forward_sa(img_anc, cap_pos, cap_syn, cap_anc, img_pos, img_syn, self.logger)
        self.logger.update('Self-Adjustment', loss.data.item(), img_anc.size(0))
        return loss

    def forward_se_loss(self, img_anc, cap_pos, cap_neg, cap_anc, img_pos, img_neg):
        loss = self.criterion.forward_se(img_anc, cap_pos, cap_neg, cap_anc, img_pos, img_neg, self.logger)
        self.logger.update('Self-Exploration', loss.data.item(), img_anc.size(0))
        return loss

    def train_emb(self, images, captions, lengths, image_lengths=None, warmup_alpha=None):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        images, captions, lengths, image_lengths = self.store_cuda(images, captions, lengths, image_lengths)
        img_emb, cap_emb = self.forward_emb(images, captions, lengths, image_lengths=image_lengths)

        # CFM
        tri_loss, scores, mask_i2t, mask_t2i = self.forward_tri_loss(img_emb, cap_emb)
        n, cfm_data = self.generator(images, image_lengths, captions, lengths,
                                     scores, mask_i2t, mask_t2i, logger)
        if n > 0:
            img_emb_pack, cap_emb_pack = self.forward_emb(*cfm_data)
        else:
            img_emb_pack, cap_emb_pack = None, None

        # measure accuracy and record loss
        self.optimizer.zero_grad()

        if n == 1:
            img_syn = img_emb_pack
            cap_syn = cap_emb_pack
            cfm_loss = self.forward_sa_loss(img_emb[mask_i2t], cap_emb[mask_i2t], cap_syn,
                                                 cap_emb[mask_t2i], img_emb[mask_t2i], img_syn)

        elif n == 2:
            img_pos_mask, img_neg_mask = img_emb_pack[:len(img_emb_pack) // 2], img_emb_pack[len(img_emb_pack) // 2:]
            cap_pos_mask, cap_neg_mask = cap_emb_pack[:len(cap_emb_pack) // 2], cap_emb_pack[len(cap_emb_pack) // 2:]
            cfm_loss = self.forward_se_loss(img_emb[mask_i2t], cap_pos_mask, cap_neg_mask,
                                                   cap_emb[mask_t2i], img_pos_mask, img_neg_mask)

        elif n == 3:
            img_syn, img_pos_mask, img_neg_mask = img_emb_pack[:len(img_emb_pack) // 3], \
                                                  img_emb_pack[len(img_emb_pack) // 3: len(img_emb_pack) * 2 // 3], \
                                                  img_emb_pack[len(img_emb_pack) * 2 // 3:]
            cap_syn, cap_pos_mask, cap_neg_mask = cap_emb_pack[:len(cap_emb_pack) // 3], \
                                                  cap_emb_pack[len(cap_emb_pack) // 3: len(cap_emb_pack) * 2 // 3], \
                                                  cap_emb_pack[len(cap_emb_pack) * 2 // 3:]
            cfm_loss = self.forward_sa_loss(img_emb[mask_i2t], cap_emb[mask_i2t], cap_syn,
                                            cap_emb[mask_t2i], img_emb[mask_t2i], img_syn)
            cfm_loss += self.forward_se_loss(img_emb[mask_i2t], cap_pos_mask, cap_neg_mask,
                                            cap_emb[mask_t2i], img_pos_mask, img_neg_mask)

        else:
            cfm_loss = 0

        loss = tri_loss + cfm_loss

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        # compute gradient and update
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

