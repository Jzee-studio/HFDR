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
from utils import CIFAR10C, CIFAR100C, TinyImageNetC
from wandb_utils import WandBLogger


device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Dataset registry: maps config Train.Data values to corruption-dataset metadata
# ---------------------------------------------------------------------------
DATA_ROOT = "/data/xujiazhao"

DATASET_CONFIG = {
    "CIFAR10": {
        "cls": CIFAR10C,
        "data_subdir": "CIFAR-10-C",
        "num_class": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2471, 0.2435, 0.2616),
    },
    "CIFAR100": {
        "cls": CIFAR100C,
        "data_subdir": "CIFAR-100-C",
        "num_class": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "TinyImageNet": {
        "cls": TinyImageNetC,
        "data_subdir": "Tiny-ImageNet-C",
        "num_class": 200,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


def build_model(config):
    if config.Train.Train_Method == "LFCM":
        lfcm_cfg = config.get('LFCM', {})
        net = WRN34_10_LFCM(
            Num_class=config.DATA.num_class,
            codebook_size=lfcm_cfg.get('codebook_size', 64),
            code_dim=lfcm_cfg.get('code_dim', 32),
            hidden_dim=lfcm_cfg.get('hidden_dim', 64),
        )
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


def get_corruptions(dataset_name: str) -> List[str]:
    """Discover available corruption .npy files for a given dataset."""
    subdir = DATASET_CONFIG[dataset_name]["data_subdir"]
    root = os.path.join(DATA_ROOT, subdir)
    if os.path.isdir(root):
        names = []
        for fname in sorted(os.listdir(root)):
            if fname.endswith(".npy") and fname != "labels.npy":
                names.append(os.path.splitext(fname)[0])
        return names
    return []


def build_corruption_loader(dataset_name: str, corruption: str, norm: bool, batch_size: int = 100):
    """Build a DataLoader for one corruption of a given dataset."""
    cfg = DATASET_CONFIG[dataset_name]
    if norm:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    dataset_cls = cfg["cls"]
    root = os.path.join(DATA_ROOT, cfg["data_subdir"])
    dataset = dataset_cls(root=root, name=corruption, transform=transform)
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
    parser = argparse.ArgumentParser(description="CIFAR-10/100-C & TinyImageNet-C OOD evaluation for HFDR models")
    parser.add_argument("--config", default="configs_train.yml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-name", default="model_best.pth.tar")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--corruptions", nargs="*", default=None)
    parser.add_argument("--dataset", default=None,
                        choices=["CIFAR10", "CIFAR100", "TinyImageNet"],
                        help="Override dataset (default: derived from config Train.Data)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    # Resolve dataset name: CLI arg takes priority, otherwise use config
    dataset_name = args.dataset or config.Train.Data
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported: {list(DATASET_CONFIG.keys())}"
        )

    file_name = config.Operation.Prefix
    data_set = config.Train.Data
    check_path = args.checkpoint_dir or os.path.join("./checkpoint", data_set, file_name)
    checkpoint_path = os.path.join(check_path, args.checkpoint_name)
    os.makedirs(check_path, exist_ok=True)

    logger = logging.getLogger("ood_eval")
    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(check_path, file_name + "_ood_test.log")),
            logging.StreamHandler(),
        ],
    )

    wandb_logger = WandBLogger(enabled=bool(config.Operation.get('Use_WandB', False)), config={
        'operation_prefix': config.Operation.Prefix,
        'train_method': config.Train.Train_Method,
        'dataset': dataset_name,
    })
    wandb_logger.init(
        project=config.Operation.get('WandB_Project', 'HFDR'),
        entity=config.Operation.get('WandB_Entity', None) or None,
        name=config.Operation.get('WandB_RunName') or file_name,
        group=config.Operation.get('WandB_Group', None) or None,
        tags=config.Operation.get('WandB_Tags', []),
        mode=config.Operation.get('WandB_Mode', 'online'),
        job_type='ood_eval',
        reinit=True,
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    net, data_norm = build_model(config)
    net = load_checkpoint(net, checkpoint_path)

    corruptions = args.corruptions if args.corruptions else get_corruptions(dataset_name)
    if not corruptions:
        subdir = DATASET_CONFIG[dataset_name]["data_subdir"]
        raise FileNotFoundError(
            f"No corruption files found under {os.path.join(DATA_ROOT, subdir)}"
        )

    logger.info("Dataset: %s  |  Checkpoint: %s", dataset_name, checkpoint_path)
    logger.info("Corruptions (%d): %s", len(corruptions), ", ".join(corruptions))

    results = []
    for corruption in corruptions:
        loader = build_corruption_loader(
            dataset_name=dataset_name,
            corruption=corruption,
            norm=data_norm,
            batch_size=args.batch_size,
        )
        acc, loss = evaluate_accuracy(net, loader)
        results.append((corruption, acc, loss))
        logger.info("%s | Acc: %.2f | Loss: %.4f", corruption, acc, loss)
        wandb_logger.log({
            f'ood/{dataset_name}/{corruption}/acc': float(acc),
            f'ood/{dataset_name}/{corruption}/loss': float(loss),
        })

    mean_acc = float(np.mean([x[1] for x in results]))
    mean_loss = float(np.mean([x[2] for x in results]))
    logger.info("Dataset: %s | Mean Acc: %.2f | Mean Loss: %.4f", dataset_name, mean_acc, mean_loss)
    wandb_logger.log({
        f'ood/{dataset_name}/mean_acc': mean_acc,
        f'ood/{dataset_name}/mean_loss': mean_loss,
    })
    wandb_logger.finish()


if __name__ == "__main__":
    cudnn.benchmark = True
    main()
