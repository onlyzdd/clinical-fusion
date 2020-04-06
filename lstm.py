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

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        # unstructure
        if args.use_unstructure:
            self.vocab_embedding = nn.Embedding (args.unstructure_size, args.embed_size )
            self.vocab_lstm = nn.LSTM ( input_size=args.embed_size,
                              # hidden_size=args.hidden_size,
                              hidden_size=1,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=True)
            self.vocab_mapping = nn.Sequential(
                    nn.Linear(args.embed_size * 2, args.embed_size),
                    nn.ReLU ( ),
                    nn.Dropout ( 0.1),
                    nn.Linear(args.embed_size, args.embed_size),
                    )
            self.cat_output = nn.Sequential (
                    nn.Linear (args.rnn_size * 3, args.rnn_size),
                    nn.ReLU ( ),
                    nn.Dropout ( 0.1),
                    nn.Linear ( args.rnn_size, output_size),
                    )
            self.cat_output = nn.Sequential (
                    nn.ReLU ( ),
                    nn.Dropout ( 0.1),
                    nn.Linear (args.rnn_size * 3, output_size),
                    )

        if args.value_embedding == 'no':
            self.embedding = nn.Linear(args.input_size, args.embed_size)
        else:
            self.embedding = nn.Embedding (args.vocab_size, args.embed_size )
        self.lstm1 = nn.LSTM (input_size=args.embed_size,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=True)
        self.lstm2 = nn.LSTM (input_size=args.embed_size,
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=True)
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

        self.output = nn.Sequential (
            nn.Linear (args.rnn_size * 2, args.rnn_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.1),
            nn.Linear ( args.rnn_size, output_size),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)

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


    def forward(self, x, t, dd, content=None):

        if 0 and content is not None:
            content, _ = self.lstm1(content)
            content = self.vocab_mapping(content)
            content = torch.transpose(content, 1, 2).contiguous()
            content = self.pooling(content)
            content = content.view((content.size(0), -1))
            return self.one_output(content)

        # value embedding
        x = self.value_order_embedding(x)
        x = self.visit_pooling(x)

        # demo embedding
        dsize = list(dd.size()) + [-1]
        d = self.dd_embedding(dd.view(-1)).view(dsize)
        d = self.dd_mapping(d)
        d = torch.transpose(d, 1,2).contiguous()                  # (64*30, 200, 100)
        d = self.pooling(d)
        d = d.view((d.size(0), -1))

        # x = torch.cat((x, d), 2)
        # x = self.dx_mapping(x)

        # time embedding
        # t = self.value_embedding(t)
        # x = self.tv_mapping(torch.cat((x, t), 2))

        # lstm
        lstm_out, _ = self.lstm2( x )            # (64, 30, 1024)
        output = self.output_mapping(lstm_out)
        output = torch.transpose(output, 1,2).contiguous()                  # (64*30, 200, 100)
        # print('ouput.size', output.size())
        output = self.pooling(output)                                       # (64*30, 200, 1)
        output = output.view((output.size(0), -1))
        out = self.output(torch.cat((output, d), 1))

        # unstructure
        if content is not None:
            # print(content.size())   # [64, 1000]
            content, _ = self.lstm1(content)
            content = self.vocab_mapping(content)
            content = torch.transpose(content, 1, 2).contiguous()
            content = self.pooling(content)
            content = content.view((content.size(0), -1))
            out = self.cat_output(torch.cat((output, content, d), 1))
        

        return out

