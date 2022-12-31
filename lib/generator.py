import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import time


class NegativeGenerator(nn.Module):
    def __init__(self, img_enc: nn.Module, txt_enc: nn.Module, opt):

        super(NegativeGenerator, self).__init__()

        self.img_enc = img_enc
        self.txt_enc = txt_enc
        self.opt = opt
        self.mask_token = opt.mask_token

        self.rate_img_drop = opt.rate_img_drop
        self.rate_txt_drop = opt.rate_txt_drop
        self.rate_img_exchange = opt.rate_img_exchange
        self.rate_txt_exchange = opt.rate_txt_exchange
        self.rate_se = opt.rate_se
        self.rate_sa = opt.rate_sa

        self.optm = torch.optim.SGD(
            list(self.img_enc.parameters()) + list(self.txt_enc.parameters()), lr=0.)

    def pnanchor_collapse(self, anc, anc_len, pos, pos_len, neg, neg_len, modal):
        """
        (a, p->), (a, <-n)
        """
        paired = torch.cat([pos, neg])
        paired_len = torch.cat([pos_len, neg_len])

        anc = anc.clone().detach()
        anc_len = anc_len.clone().detach()
        paired = paired.clone().detach()
        paired_len = paired_len.clone().detach()

        if modal == 'i2t':
            # anc -> img
            # pos, neg -> txt
            anc_emb = self.img_enc(anc, anc_len)
            paired_emb = self.txt_enc(paired, paired_len)

            pos_emb = paired_emb[:len(paired_emb) // 2]
            neg_emb = paired_emb[len(paired_emb) // 2:]

            loss_ap = 1 - F.cosine_similarity(anc_emb.detach(), pos_emb)
            loss_an = 1 - F.cosine_similarity(anc_emb.detach(), neg_emb)
            loss = (loss_an - loss_ap).mean()

            self.optm.zero_grad()
            loss.backward()

            # grad = self.txt_enc.embed.weight.grad.data.clone()
            grad = self.txt_enc.bert.embeddings.word_embeddings.weight.grad.data.clone()
            grad = torch.cat([grad[pos], grad[neg]])

        elif modal == 't2i': 
            # anc -> txt
            # pos, neg -> img
            paired.requires_grad = True

            anc_emb = self.txt_enc(anc, anc_len)
            paired_emb = self.img_enc(paired, paired_len)

            pos_emb = paired_emb[:len(paired_emb) // 2]
            neg_emb = paired_emb[len(paired_emb) // 2:]

            loss_ap = 1 - F.cosine_similarity(anc_emb.detach(), pos_emb)
            loss_an = 1 - F.cosine_similarity(anc_emb.detach(), neg_emb)
            loss = (loss_an - loss_ap).mean()

            optx = torch.optim.SGD([paired], lr=1.)
            self.optm.zero_grad()
            optx.zero_grad()
            loss.backward()

            grad = paired.grad.data.clone()

            paired.requires_grad = False

        else:
            raise ValueError("modal with a wrong value (i2t | t2i)")

        return grad

    @staticmethod
    def get_img_ind(grad, rate):
        grad_norm = torch.norm(grad, p=2, dim=-1)
        ind = grad_norm.sort(1)[1]  # (128, length)

        index0 = torch.arange(ind.size(0)).reshape(-1, 1).repeat(1, int(rate * ind.size(1))).reshape(-1)
        index1 = ind[:, -int(rate * ind.size(1)):].reshape(-1)

        return index0, index1

    @staticmethod
    def get_cap_ind(grad, lengths, rate, max_lengths=None):
        grad_norm = torch.norm(grad, p=2, dim=-1)
        ind = grad_norm.sort(1)[1]  # (128, length)

        index0 = []
        index1 = []
        for i in range(len(lengths)):
            if max_lengths is None:
                n = max(int(lengths[i] * rate), 1)
            else:
                n = min(max(int(lengths[i] * rate), 1), max_lengths[i])

            index0 += [i] * n
            index1 += [-j for j in range(n, 0, -1)]

        index0 = torch.tensor(index0, device=ind.device)
        index2 = ind[index0, index1]

        return index0, index2

    @staticmethod
    def img_mask(pos, index0, index1):
        # pos (128, 28, 2048)
        # [index0, index1] (128, 28 * rate, 2048)
        pos = pos.clone()
        mask = torch.zeros_like(pos) == 0    # True
        mask[index0, index1] = False
        remain = pos[mask].reshape(pos.size(0), -1, pos.size(2))    # (128, *, 2048)
        remain_mean = torch.mean(remain, dim=1, keepdim=True).repeat(1, 28 - remain.size(1), 1)
        pos[~mask] = remain_mean.reshape(-1)
        return pos

    def cap_mask(self, pos, index0, index1):
        # pos (128, n)
        # [index0, index1] (128, n * rate)
        pos = pos.clone()
        pos[index0, index1] = self.mask_token
        return pos

    def exchange(self, pos, pos_index0, pos_index1, neg, neg_index0, neg_index1):
        syn = pos.clone()
        syn[pos_index0, pos_index1] = neg[neg_index0, neg_index1]
        return syn

    def explore(self, pos, neg, grad, rate, pos_len=None, neg_len=None):
        if pos_len is None:
            index0, index1 = self.get_img_ind(grad[:len(grad) // 2], rate)
            pos = self.img_mask(pos, index0, index1)
            index0, index1 = self.get_img_ind(grad[len(grad) // 2:], rate)
            neg = self.img_mask(neg, index0, index1)
        else:
            index0, index1 = self.get_cap_ind(grad[:len(grad) // 2], pos_len, rate)
            pos = self.cap_mask(pos, index0, index1)
            index0, index1 = self.get_cap_ind(grad[len(grad) // 2:], neg_len, rate)
            neg = self.cap_mask(neg, index0, index1)
        return pos, neg

    def adjust(self, pos, neg, grad, rate, pos_len=None, neg_len=None):
        if pos_len is None:
            pos_index0, pos_index1 = self.get_img_ind(grad[:len(grad) // 2], rate)
            neg_index0, neg_index1 = self.get_img_ind(grad[len(grad) // 2:], rate)
        else:
            pos_index0, pos_index1 = self.get_cap_ind(grad[:len(grad) // 2], pos_len, rate, neg_len)
            neg_index0, neg_index1 = self.get_cap_ind(grad[len(grad) // 2:], pos_len, rate, neg_len)
        syn = self.exchange(pos, pos_index0, pos_index1, neg, neg_index0, neg_index1)
        return syn

    @staticmethod
    def select_triplet(scores):
        scores2 = scores - 10 * torch.eye(len(scores)).cuda()

        cap_neg_ind = scores2.max(1)[1]
        img_neg_ind = scores2.max(0)[1]

        return cap_neg_ind, img_neg_ind

    def forward(self, img, img_len, cap, cap_len, scores, mask_i2t, mask_t2i, logger=None):

        self.logger = logger

        cap_neg_ind, img_neg_ind = self.select_triplet(scores)

        # i2t triplet
        img_anc = img[mask_i2t]
        img_anc_len = img_len[mask_i2t]
        cap_pos = cap[mask_i2t]
        cap_pos_len = cap_len[mask_i2t]
        cap_neg = cap[cap_neg_ind][mask_i2t]
        cap_neg_len = cap_len[cap_neg_ind][mask_i2t]

        # t2i triplet
        cap_anc = cap[mask_t2i]
        cap_anc_len = cap_len[mask_t2i]
        img_pos = img[mask_t2i]
        img_pos_len = img_len[mask_t2i]
        img_neg = img[img_neg_ind][mask_t2i]
        img_neg_len = img_len[img_neg_ind][mask_t2i]

        img_grad = self.pnanchor_collapse(anc=cap_anc, anc_len=cap_anc_len,
                                          pos=img_pos, pos_len=img_pos_len,
                                          neg=img_neg, neg_len=img_neg_len,
                                          modal='t2i')
        cap_grad = self.pnanchor_collapse(anc=img_anc, anc_len=img_anc_len,
                                          pos=cap_pos, pos_len=cap_pos_len,
                                          neg=cap_neg, neg_len=cap_neg_len,
                                          modal='i2t')

        img_list = []
        img_len_list = []
        cap_list = []
        cap_len_list = []
        if random.random() < self.rate_sa:
            img_syn = self.adjust(img_pos, img_neg, img_grad, self.rate_img_exchange)
            cap_syn = self.adjust(cap_pos, cap_neg, cap_grad, self.rate_txt_exchange, cap_pos_len, cap_neg_len)
            img_list.append(img_syn)
            img_len_list.append(img_pos_len)
            cap_list.append(cap_syn)
            cap_len_list.append(cap_pos_len)

        if random.random() < self.rate_se:
            img_pos_mask, img_neg_mask = self.explore(img_pos, img_neg, img_grad, self.rate_img_drop)
            cap_pos_mask, cap_neg_mask = self.explore(cap_pos, cap_neg, cap_grad, self.rate_txt_drop, cap_pos_len, cap_neg_len)
            img_list.extend([img_pos_mask, img_neg_mask])
            img_len_list.extend([img_pos_len, img_neg_len])
            cap_list.extend([cap_pos_mask, cap_neg_mask])
            cap_len_list.extend([cap_pos_len, cap_neg_len])

        n = len(img_list)
        if n > 0:
            img_pack = torch.cat(img_list)
            img_len_pack = torch.cat(img_len_list)
            cap_pack = torch.cat(cap_list)
            cap_len_pack = torch.cat(cap_len_list)
            data = (img_pack, cap_pack, cap_len_pack, img_len_pack)
        else:
            data = None

        return n, data