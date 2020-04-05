#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *

import numpy as np

import sys
sys.path.append('../tools')
import parse, py_op

output_size = 1

def value_embedding_data(d = 200, split = 200):
    vec = np.array([np.arange(split) * i for i in range(int(d/2))], dtype=np.float32).transpose()
    vec = vec / vec.max() 
    embedding = np.concatenate((np.sin(vec), np.cos(vec)), 1)
    embedding[0, :d] = 0
    embedding = torch.from_numpy(embedding)
    return embedding


def conv3(in_channels, out_channels, stride=1, kernel_size=3):
    return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args


        if args.value_embedding == 'no':
            self.embedding = nn.Linear(args.input_size, args.embed_size)
        else:
            self.embedding = nn.Embedding (args.vocab_size, args.embed_size )
        self.dd_embedding = nn.Embedding (args.n_ehr, args.embed_size )
        self.value_embedding = nn.Embedding.from_pretrained(value_embedding_data(args.embed_size, args.split_num + 1))
        self.value_mapping = nn.Sequential(
                nn.Linear ( args.embed_size * 2, args.embed_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                )
        self.dd_mapping = nn.Sequential(
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                nn.Dropout(0.1),
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                nn.Dropout(0.1),
                )
        self.dx_mapping = nn.Sequential(
                nn.Linear ( args.embed_size * 2, args.embed_size),
                nn.ReLU ( ),
                nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
                )

        self.tv_mapping = nn.Sequential (
            nn.Linear ( args.embed_size * 2, args.embed_size),
            nn.ReLU ( ),
            nn.Linear ( args.embed_size, args.embed_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.1),
        )
        self.relu = nn.ReLU ( )

        lstm_size = args.rnn_size 

        lstm_size *= 2
        self.output_mapping = nn.Sequential (
            nn.Linear (lstm_size, args.rnn_size),
            nn.ReLU ( ),
            nn.Linear (args.rnn_size, args.rnn_size),
            nn.ReLU ( )
        )

        self.cat_output = nn.Sequential (
            nn.Linear (args.rnn_size * 2, args.rnn_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.1),
            nn.Linear ( args.rnn_size, output_size),
        )
        self.output = nn.Sequential (
            nn.Linear (args.rnn_size, args.rnn_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.1),
            nn.Linear ( args.rnn_size, output_size),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)


        layers = [1, 2, 2]
        embed_size = args.embed_size
        block = ResidualBlock
        self.in_channels = embed_size
        self.bn1 = nn.BatchNorm1d(embed_size)
        self.bn2 = nn.BatchNorm1d(embed_size)
        self.bn3 = nn.BatchNorm1d(embed_size)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, embed_size, layers[0], 2)
        self.layer2 = self.make_layer(block, embed_size, layers[1], 2)
        self.layer3 = self.make_layer(block, embed_size, layers[2], 2)



        # unstructure
        if args.use_unstructure:
            self.vocab_embedding = nn.Embedding (args.unstructure_size+10, args.embed_size )
            # self.vocab_layer = self.make_layer(block, embed_size, layers[0], 2)
            self.vocab_layer = nn.Sequential(
                    nn.Dropout(0.2),
                    conv3(embed_size, embed_size, 2, 2),
                    nn.BatchNorm1d(embed_size),
                    nn.Dropout(0.2),
                    nn.ReLU(),
                    # conv3(embed_size, embed_size, 2, 3),
                    # nn.BatchNorm1d(embed_size),
                    # nn.Dropout(0.1),
                    # nn.ReLU(),
                    # conv3(embed_size, embed_size, 2, 3),
                    # nn.BatchNorm1d(embed_size),
                    # nn.Dropout(0.1),
                    # nn.ReLU(),
                    )
            self.vocab_output = nn.Sequential (
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                nn.Linear (args.embed_size * 3, 3 * args.embed_size),
                nn.ReLU ( ),
                nn.Dropout ( 0.1),
                nn.Linear ( 3 * args.embed_size, output_size),
            )
            self.one_output = nn.Sequential (
                # nn.Linear (args.embed_size * 3, args.embed_size),
                # nn.ReLU ( ),
                nn.Dropout ( 0.1),
                nn.Linear ( args.embed_size, output_size),
            )


    def visit_pooling(self, x):
        output = x
        size = output.size()
        output = output.view(size[0] * size[1], size[2], output.size(3))    # (64*30, 13, 200)
        output = torch.transpose(output, 1,2).contiguous()                  # (64*30, 200, 13)
        output = self.pooling(output)                                       # (64*30, 200, 1)
        output = output.view(size[0], size[1], size[3])                     # (64, 30, 200)
        return output

    def value_order_embedding(self, x):
        size = list(x[0].size())               # (64, 30, 13)
        index, value = x
        xi = self.embedding(index.view(-1))          # (64*30*13, 200)
        # xi = xi * (value.view(-1).float() + 1.0 / self.args.split_num)
        xv = self.value_embedding(value.view(-1))    # (64*30*13, 200)
        x = torch.cat((xi, xv), 1)                   # (64*30*13, 1024)
        x = self.value_mapping(x)                    # (64*30*13, 200)   
        size.append(-1)
        x = x.view(size)                    # (64, 30, 13, 200)
        return x

        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)



    def forward(self, x, t, dd, content=None):

        if content is not None:
            # content = self.vocab_embedding(content).transpose(1,2)
            content = self.vocab_layer(content.transpose(1,2))
            content = self.pooling(content)                                       # (64*30, 200, 1)
            content = content.view((content.size(0), -1))
            return self.one_output(content)

        # value embedding
        x = self.value_order_embedding(x)
        x = self.visit_pooling(x)


        # time embedding
        # t = self.value_embedding(t)
        # x = self.tv_mapping(torch.cat((x, t), 2))

        # cnn
        x = torch.transpose(x, 1, 2,).contiguous()
        out = self.bn1(x)
        out = self.relu(out)
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)

        output = self.pooling(out)                                       # (64*30, 200, 1)
        output = output.view((output.size(0), -1))


        if len(dd.size()) > 1:
            # demo embedding
            dsize = list(dd.size()) + [-1]
            d = self.dd_embedding(dd.view(-1)).view(dsize)
            d = self.dd_mapping(d)
            d = torch.transpose(d, 1,2).contiguous()                  # (64*30, 200, 100)
            d = self.pooling(d)
            d = d.view((d.size(0), -1))
            output = torch.cat((output, d), 1)
            out = self.cat_output(output)
        # else:
        #     out = self.output(output)

        if content is not None:
            # content = self.vocab_embedding(content)
            content = self.vocab_layer(content.transpose(1,2))
            content = self.pooling(content)                                       # (64*30, 200, 1)
            content = content.view((content.size(0), -1))
            # content = self.one_output(content) + 0.3 * out
            # out = content + out
            # out = content 
            output = torch.cat((output, content), 1)
            out = self.vocab_output(output)

        return out

