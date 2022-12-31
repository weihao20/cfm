import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.margin2 = opt.margin2
        self.max_violation = max_violation
        self.cfm = True

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def cfm_on(self):
        self.cfm = True
        print('Use CFM loss')

    def forward_triplet(self, im, s, logger):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        scores2 = scores - 10 * torch.eye(len(scores)).cuda()
        i2t_neg = scores2.max(1)[0]
        t2i_neg = scores2.max(0)[0]

        cost_s_max = cost_s.max(1)[0]
        cost_im_max = cost_im.max(0)[0]
        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        mask_i2t = cost_s_max > 0
        mask_t2i = cost_im_max > 0

        return cost_s.sum() + cost_im.sum(), scores, mask_i2t, mask_t2i

    def forward_sa(self, img_anc, cap_pos, cap_syn, cap_anc, img_pos, img_syn, logger):
        if not self.cfm:
            return (img_anc * 0).sum()

        i2t_pos = (img_anc * cap_pos).sum(1)
        i2t_neg = (img_anc * cap_syn).sum(1)
        t2i_pos = (cap_anc * img_pos).sum(1)
        t2i_neg = (cap_anc * img_syn).sum(1)

        cost_cap = (self.margin2 + i2t_neg - i2t_pos).clamp(min=0)
        cost_img = (self.margin2 + t2i_neg - t2i_pos).clamp(min=0)

        loss = cost_cap.sum() + cost_img.sum()

        return loss

    def forward_se(self, img_anc, cap_pos, cap_neg, cap_anc, img_pos, img_neg, logger):
        if not self.cfm:
            return (img_anc * 0).sum()

        i2t_pos = (img_anc * cap_pos).sum(1)
        i2t_neg = (img_anc * cap_neg).sum(1)
        t2i_pos = (cap_anc * img_pos).sum(1)
        t2i_neg = (cap_anc * img_neg).sum(1)

        cost_cap = (self.margin2 + i2t_neg - i2t_pos).clamp(min=0)
        cost_img = (self.margin2 + t2i_neg - t2i_pos).clamp(min=0)

        loss = cost_cap.sum() + cost_img.sum()

        return loss


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities