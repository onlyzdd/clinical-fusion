import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np


def hard_mining(neg_output, neg_labels, num_hard, largest=True):
    num_hard = min(max(num_hard, 10), len(neg_output))
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)), largest=largest)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels


class Loss(nn.Module):
    def __init__(self, hard_mining):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.hard_mining = hard_mining
        self.sigmoid = nn.Sigmoid()

    def forward(self, prob, labels, train=True):

        prob = self.sigmoid(prob)

        pos_ind = labels > 0.5
        neg_ind = labels < 0.5
        pos_label = labels[pos_ind]
        neg_label = labels[neg_ind]
        pos_prob = prob[pos_ind]
        neg_prob = prob[neg_ind]
        pos_loss, neg_loss = 0, 0

        # hard mining
        num_hard_pos = 10
        num_hard_neg = 18
        if self.hard_mining:
            pos_prob, pos_label= hard_mining(pos_prob, pos_label, num_hard_pos, largest=False)
            neg_prob, neg_label= hard_mining(neg_prob, neg_label, num_hard_neg, largest=True)

        if len(pos_prob):
            pos_loss = 0.5 * self.classify_loss(pos_prob, pos_label) 

        if len(neg_prob):
            neg_loss = 0.5 * self.classify_loss(neg_prob, neg_label)
        classify_loss = pos_loss + neg_loss
        
        prob = prob.data.cpu().numpy() > 0.5
        labels = labels.data.cpu().numpy()
        pos_l = (labels==1).sum()
        neg_l = (labels==0).sum()
        pos_p = (prob + labels == 2).sum()
        neg_p = (prob + labels == 0).sum()

        return [classify_loss, pos_p, pos_l, neg_p, neg_l]


class MultiClassLoss(nn.Module):
    def __init__(self, hard_mining):
        super(MultiClassLoss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.hard_mining = hard_mining
        self.sigmoid = nn.Sigmoid()


    def forward(self, prob, labels, train=True):
        prob = self.sigmoid(prob)
        classify_loss, pos_p, pos_l, neg_p, neg_l = 0, 0, 0, 0, 0

        prob_list = prob
        labels_list = labels
        for i in range(prob.size(1)):
            prob = prob_list[:, i]
            labels = labels_list[:, i]

            pos_ind = labels > 0.5
            neg_ind = labels < 0.5
            pos_label = labels[pos_ind]
            neg_label = labels[neg_ind]
            pos_prob = prob[pos_ind]
            neg_prob = prob[neg_ind]
            pos_loss, neg_loss = 0, 0

            # hard mining
            num_hard_pos = 10
            num_hard_neg = 18
            if self.hard_mining:
                pos_prob, pos_label= hard_mining(pos_prob, pos_label, num_hard_pos, largest=False)
                neg_prob, neg_label= hard_mining(neg_prob, neg_label, num_hard_neg, largest=True)

            if len(pos_prob):
                pos_loss = 0.5 * self.classify_loss(pos_prob, pos_label) 

            if len(neg_prob):
                neg_loss = 0.5 * self.classify_loss(neg_prob, neg_label)

            classify_loss = classify_loss + pos_loss + neg_loss

            # stati number
            prob = prob.data.cpu().numpy() > 0.5
            labels = labels.data.cpu().numpy()
            pos_l += (labels==1).sum()
            neg_l += (labels==0).sum()
            pos_p += (prob + labels == 2).sum()
            neg_p += (prob + labels == 0).sum()

        return [classify_loss, pos_p, pos_l, neg_p, neg_l]
