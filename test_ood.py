import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models import *
from utils import CIFAR10C


device = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(config):
    if config.Train.Train_Method == "LFCM":
        net = WRN34_10_LFCM(Num_class=config.DATA.num_class)
    else:
        net = WRN34_10_F(Num_class=config.DATA.num_class)
    net.Num_class = config.DATA.num_class
    norm_mean = torch.tensor(config.DATA.mean).to(device)
    norm_std = torch.tensor(config.DATA.std).to(device)

    if config.Train.Train_Method in {"AT", "HFDR", "TRADES", "LFCM"}:
        net.Norm = True
        net.norm_mean = norm_mean
        net.norm_std = norm_std
        data_norm = False
    else:
        net.Norm = False
        data_norm = True

    net = net.to(device)
    net = torch.nn.DataParallel(net)
    return net, data_norm


def load_checkpoint(net, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net


def get_cifar10c_corruptions() -> List[str]:
    root = os.path.join("/data/xujiazhao", "CIFAR-10-C")
    if os.path.isdir(root):
        names = []
        for fname in sorted(os.listdir(root)):
            if fname.endswith(".npy") and fname != "labels.npy":
                names.append(os.path.splitext(fname)[0])
        return names
    return []


def build_cifar10c_loader(corruption: str, norm: bool, mean, std, batch_size: int = 100):
    if norm:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    dataset = CIFAR10C(root=os.path.join("/data/xujiazhao", "CIFAR-10-C"), name=corruption, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader


@torch.no_grad()
def evaluate_accuracy(net, loader) -> Tuple[float, float]:
    net.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    bar = tqdm(loader, desc="Eval")
    for inputs, targets in bar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss_sum += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        bar.set_postfix(acc=round(100.0 * correct / total, 2))

    return 100.0 * correct / total, loss_sum / total


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10-C evaluation for HFDR models")
    parser.add_argument("--config", default="configs_train.yml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-name", default="model_best.pth.tar")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--corruptions", nargs="*", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    file_name = config.Operation.Prefix
    data_set = config.Train.Data
    check_path = args.checkpoint_dir or os.path.join("./checkpoint", data_set, file_name)
    checkpoint_path = os.path.join(check_path, args.checkpoint_name)

    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger("cifar10c")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net, data_norm = build_model(config)
    net = load_checkpoint(net, checkpoint_path)

    corruptions = args.corruptions if args.corruptions else get_cifar10c_corruptions()
    if not corruptions:
        raise FileNotFoundError("No CIFAR-10-C corruption files found under ./data/CIFAR-10-C")

    logger.info("Loaded checkpoint: %s", checkpoint_path)
    logger.info("Corruptions: %s", ", ".join(corruptions))

    results = []
    for corruption in corruptions:
        loader = build_cifar10c_loader(
            corruption=corruption,
            norm=data_norm,
            mean=config.DATA.mean,
            std=config.DATA.std,
            batch_size=args.batch_size,
        )
        acc, loss = evaluate_accuracy(net, loader)
        results.append((corruption, acc, loss))
        logger.info("%s | Acc: %.2f | Loss: %.4f", corruption, acc, loss)

    mean_acc = float(np.mean([x[1] for x in results]))
    mean_loss = float(np.mean([x[2] for x in results]))
    logger.info("Mean Acc: %.2f | Mean Loss: %.4f", mean_acc, mean_loss)


if __name__ == "__main__":
    cudnn.benchmark = True
    main()
