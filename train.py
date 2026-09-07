import os
import logging

import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict
import yaml

from models import *
from utils_train import *
from utils import *
from wandb_utils import WandBLogger


device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('configs_train.yml') as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

use_wandb = bool(config.Operation.get('Use_WandB', False))
wandb_logger = WandBLogger(enabled=use_wandb, config={
    'operation_prefix': config.Operation.Prefix,
    'train_method': config.Train.Train_Method,
    'dataset': config.Train.Data,
    'epochs': config.Train.Epoch,
    'learning_rate': config.Train.Lr,
    'resume': config.Operation.Resume,
    'factor': config.Train.Factor,
    'clip_eps': config.Train.clip_eps,
    'fgsm_step': config.Train.fgsm_step,
    'pgd_train': config.Train.pgd_train,
})

# modify the load model
net = build_model(
    backbone=config.Train.get('Backbone', 'WRN34'),
    method=config.Train.Train_Method,
    num_class=config.DATA.num_class,
    lfcm_cfg=config.get('LFCM', {}),
)

file_name = config.Operation.Prefix
data_set = config.Train.Data
check_path = os.path.join('./checkpoint', data_set, file_name)
os.makedirs(check_path, exist_ok=True)
learning_rate = config.Train.Lr

wandb_logger.init(
    project=config.Operation.get('WandB_Project', 'HFDR'),
    entity=config.Operation.get('WandB_Entity', None) or None,
    name=config.Operation.get('WandB_RunName') or file_name,
    group=config.Operation.get('WandB_Group', None) or None,
    tags=config.Operation.get('WandB_Tags', []),
    mode=config.Operation.get('WandB_Mode', 'online'),
    job_type='train',
    reinit=True,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(os.path.join(check_path, file_name + '_record.log')),
        logging.StreamHandler()
    ])

net.Num_class = config.DATA.num_class
norm_mean = torch.tensor(config.DATA.mean).to(device)
norm_std = torch.tensor(config.DATA.std).to(device)
if config.Train.Train_Method in ('AT', 'TRADES', 'HFDR', 'LFCM'):
    net.Norm = True
    net.norm_mean = norm_mean
    net.norm_std = norm_std
    Data_norm = False
    logger.info('Adversarial Training || net: '+config.Operation.Prefix + ' || '+config.Train.Train_Method)
else:
    net.Norm = False
    Data_norm = True
    logger.info('Natural Training || net: '+config.Operation.Prefix)

train_loader, test_loader = create_dataloader(data_set, Norm=Data_norm)

net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

if config.Operation.Resume is True:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(check_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(check_path, 'checkpoint.pth.tar'))
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
else:
    start_epoch = 0
    best_prec1 = 0
    logger.info(config.Operation.record_words)
    logger.info('%-5s\t%-10s\t%-9s\t%-9s\t%-8s\t%-15s', 'Epoch', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc', 'Test Robust Acc')

wandb_logger.log({
    'meta/resume': int(bool(config.Operation.Resume)),
    'meta/start_epoch': start_epoch,
    'meta/best_prec1': best_prec1,
})

for epoch in range(start_epoch + 1, config.Train.Epoch + 1):
    learning_rate = adjust_learning_rate(learning_rate, optimizer, epoch)
    if config.Train.Train_Method == 'AT':
        acc_train, train_loss = train_adversarial(net, epoch, train_loader, optimizer, config)
    elif config.Train.Train_Method == 'HFDR':
        acc_train, train_loss = train_adversarial_HF_1(net, epoch, train_loader, optimizer, config)
    elif config.Train.Train_Method == 'LFCM':
        acc_train, train_loss = train_LFCM(net, epoch, train_loader, optimizer, config)
    else:
        acc_train, train_loss = train(net, epoch, train_loader, optimizer, config)

    acc_test, pgd_acc, loss_test, best_prec1 = test_net_robust(net, test_loader, epoch, optimizer, best_prec1, config, save_path=check_path)
    logger.info('%-5d\t%-10.2f\t%-9.2f\t%-9.2f\t%-8.2f\t%.2f', epoch, train_loss, acc_train, loss_test, acc_test, pgd_acc)
    wandb_logger.log({
        'train/loss': float(train_loss),
        'train/acc': float(acc_train),
        'test/loss': float(loss_test),
        'test/acc': float(acc_test),
        'test/robust_acc': float(pgd_acc),
        'lr': float(learning_rate),
        'epoch': int(epoch),
        'best_prec1': float(best_prec1),
    }, step=epoch)

wandb_logger.finish()
