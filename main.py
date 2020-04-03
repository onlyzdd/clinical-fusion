import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import argparse
import os
import json

from load_data import FusionDataset
from utils import cal_metric, get_ids, load_model, save_model


class Net(nn.Module):
    def __init__(self, args, n_classes, input_size, hidden_size, n_layers=1):
        super(Net, self).__init__()
        self.args = args
        self.n_classes = n_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_size / 2), n_classes)
        )

    def forward(self, x_demo, x_notes, x_temporal):
        out, (hn, cn) = self.lstm(x_temporal, None)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)


def run(model, args, device, data_loader, optimizer, epoch, best_metric, phase="train"):
    total_loss = 0
    probs_list, labels_list = [], []
    if phase == "train":
        model.train()
        print("Training epoch %d:" % epoch)
    else:
        model.eval()
        print("%s:" % phase.capitalize())
    for batch_idx, (x_demo, x_notes, x_temporal, target) in enumerate(tqdm(data_loader)):
        x_demo, x_notes, x_temporal, target = x_demo.to(device), x_notes.to(device), x_temporal.to(device), target.to(device)
        output = model(x_demo, x_notes, x_temporal)
        loss = F.binary_cross_entropy(output, target)
        total_loss += loss.item()
        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        probs = output.cpu().detach().numpy()
        labels = target.cpu().detach().numpy()
        probs_list.append(probs)
        labels_list.append(labels)
    all_probs = np.row_stack(probs_list)
    all_labels = np.row_stack(labels_list)
    aucs = []
    for i in range(all_labels.shape[1]):
        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        aucs.append(auc)
    avg_metric = np.mean(aucs)
    if phase == "train":
        print("Loss: %.4f, average metric: %.4f" % (total_loss, avg_metric))
    elif phase == "val":
        if avg_metric > best_metric:
            best_metric = avg_metric
            save_model(
                {
                    "args": args,
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
            )
        print(
            "Loss: %.4f, best metric: %.4f, AUCs: %s"
            % (total_loss, best_metric, ", ".join(["%.4f" % auc for auc in aucs]))
        )
    else:
        metrics = []
        for i in range(all_labels.shape[1]):
            metric = cal_metric(all_labels[:, i], all_probs[:, i])
            print(i + 1, metric)
            metrics.append(metric)
        metrics = np.matrix(metrics)
        print('Avg', np.mean(metrics, axis=0).tolist())


def parse_args():
    parser = argparse.ArgumentParser(description='Help')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--seed", type=int, default=42, help="torchs seed")
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument(
        "--model-dir", type=str, default="models", help="path to data directory"
    )
    parser.add_argument("--phase", type=str,
                        default="train", help="train/test")
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume checkpoint"
    )
    parser.add_argument('--doc2vec-path', type=str, default='models/doc2vec.model',
                        help='path to doc2vec model')
    parser.add_argument('--root-dir', type=str, default='data/processed',
                        help='path to root dir')
    parser.add_argument('--n-classes', type=int, default=1,
                        help='number of classes')
    parser.add_argument('--n-features', type=int, default=26,
                        help='number of temporal features')
    parser.add_argument("--no-cuda", default=False, action="store_true", 
                        help="disable CUDA")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    train_ids, val_ids, test_ids = get_ids(os.path.join(args.root_dir, 'files', 'splits.json'))    

    train_data = FusionDataset(train_ids, args)
    val_data = FusionDataset(val_ids, args)
    test_data = FusionDataset(test_ids, args)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = Net(args, n_classes=args.n_classes, input_size=args.n_features, hidden_size=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_metric = 0
    start_epoch = 1

    if args.resume:
        model_dict = {"args": args, "model": model}
        model, best_metric, start_epoch = load_model(model_dict)

    if args.phase == "test":
        print("Testing...")
        run(model, args, device, test_loader, optimizer, 0, best_metric, "test")
    else:
        print("Training start...")
        for epoch in range(start_epoch, args.epochs + 1):
            run(model, args, device, train_loader, optimizer, epoch, best_metric, "train",
            )
            run(model, args, device, val_loader, optimizer, epoch, best_metric, "val")

