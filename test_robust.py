import logging
import os

import torch
import torch.backends.cudnn as cudnn
from easydict import EasyDict
import yaml

from models import *
from utils_test import evaluate_normal, evaluate_pgd, evaluate_autoattack, evaluate_cw
from utils import *
from wandb_utils import WandBLogger


device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('configs_test.yml') as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

net = build_model(
    backbone=config.Operation.get('Backbone', 'WRN34'),
    method=config.Operation.Method,
    num_class=config.DATA.num_class,
    lfcm_cfg=config.get('LFCM', {}),
)

file_name = config.Operation.Prefix
data_set = config.DATA.Data
check_path = os.path.join('./checkpoint', data_set, file_name)
os.makedirs(check_path, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(os.path.join(check_path, file_name + '_test.log')),
        logging.StreamHandler()
    ])

wandb_logger = WandBLogger(enabled=bool(config.Operation.get('Use_WandB', False)), config={
    'operation_prefix': config.Operation.Prefix,
    'method': config.Operation.Method,
    'dataset': config.DATA.Data,
})
wandb_logger.init(
    project=config.Operation.get('WandB_Project', 'HFDR'),
    entity=config.Operation.get('WandB_Entity', None) or None,
    name=config.Operation.get('WandB_RunName') or file_name,
    group=config.Operation.get('WandB_Group', None) or None,
    tags=config.Operation.get('WandB_Tags', []),
    mode=config.Operation.get('WandB_Mode', 'online'),
    job_type='robust_eval',
    reinit=True,
)

net.Num_class = config.DATA.num_class
norm_mean = torch.tensor(config.DATA.mean).to(device)
norm_std = torch.tensor(config.DATA.std).to(device)
if config.Operation.Method in ('AT', 'LFCM', 'HFDR', 'TRADES'):
    net.Norm = True
    net.norm_mean = norm_mean
    net.norm_std = norm_std
    Data_norm = False
    logger.info("Adversarial Training Model Robustness")
else:
    net.Norm = False
    Data_norm = True
    logger.info("Natural Training Model Robustness")

_, test_loader = create_dataloader(data_set, Norm=Data_norm)

net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
net.eval()

print("==> Loading best model:" + file_name + "\n")
assert os.path.isdir(check_path), 'Error: no checkpoint directory found!'
checkpoint_best = torch.load(os.path.join(check_path, 'model_best.pth.tar'))
checkpoint_last = torch.load(os.path.join(check_path, 'checkpoint.pth.tar'))

auto_attacks_methods = ['apgd-ce', 'apgd-t', 'fab-t', 'square']

if config.Operation.Validate_Best is True:
    logger.info("=======Best_trained_model Performance=======")
    net.load_state_dict(checkpoint_best['state_dict'])
    if config.Operation.Validate_Natural:
        clean_acc = evaluate_normal(net, test_loader)
        logger.info(f"Normal Acc: {clean_acc:.2f}")
        wandb_logger.log({'test_best/clean_acc': float(clean_acc)})
    if config.Operation.Validate_PGD:
        fgsm_acc = evaluate_pgd(net, test_loader, config.ADV.clip_eps, config.ADV.fgsm_step, 1)
        logger.info(f"PGD_attack:[nb_iter:1,eps:{config.ADV.clip_eps},step_size:{config.ADV.fgsm_step}]->pgd_acc: {fgsm_acc: .2f}")
        wandb_logger.log({'test_best/fgsm_acc': float(fgsm_acc)})
        for pgd_param in config.ADV.pgd_test:
            pgd_acc = evaluate_pgd(net, test_loader, pgd_param[1], pgd_param[2], pgd_param[0])
            logger.info(f"PGD_attack:[nb_iter:{pgd_param[0]},eps:{pgd_param[1]},step_size:{pgd_param[2]}]->pgd_acc: {pgd_acc: .2f}")
            wandb_logger.log({f'test_best/pgd_{pgd_param[0]}': float(pgd_acc)})
    if config.Operation.Validate_CW:
        cw_acc = evaluate_cw(net, test_loader, config.ADV.clip_eps, config.ADV.fgsm_step, 20)
        logger.info(f"CW_attack:[nb_iter:20,eps:{config.ADV.clip_eps},step_size:{config.ADV.fgsm_step}]->CW_acc: {cw_acc: .2f}")
        wandb_logger.log({'test_best/cw_acc': float(cw_acc)})
    if config.Operation.Validate_Autoattack:
        auto_acc = evaluate_autoattack(net, test_loader, config.ADV.clip_eps, auto_attacks_methods)
        logger.info(f"Auto_attack:[eps:{config.ADV.clip_eps}]->AA_acc: {auto_acc: .2f}")
        wandb_logger.log({'test_best/autoattack_acc': float(auto_acc)})

if config.Operation.Validate_Last is True:
    print("==> Loading last model:" + file_name + "\n")
    logger.info("=======Last_trained_model Performance=======")
    net.load_state_dict(checkpoint_last['state_dict'])
    if config.Operation.Validate_Natural:
        clean_acc = evaluate_normal(net, test_loader)
        logger.info(f"Normal Acc: {clean_acc:.2f}")
        wandb_logger.log({'test_last/clean_acc': float(clean_acc)})
    if config.Operation.Validate_PGD:
        fgsm_acc = evaluate_pgd(net, test_loader, config.ADV.clip_eps, config.ADV.fgsm_step, 1)
        logger.info(f"PGD_attack:[nb_iter:1,eps:{config.ADV.clip_eps},step_size:{config.ADV.fgsm_step}]->pgd_acc: {fgsm_acc: .2f}")
        wandb_logger.log({'test_last/fgsm_acc': float(fgsm_acc)})
        for pgd_param in config.ADV.pgd_test:
            pgd_acc = evaluate_pgd(net, test_loader, pgd_param[1], pgd_param[2], pgd_param[0])
            logger.info(f"PGD_attack:[nb_iter:{pgd_param[0]},eps:{pgd_param[1]},step_size:{pgd_param[2]}]->pgd_acc: {pgd_acc: .2f}")
            wandb_logger.log({f'test_last/pgd_{pgd_param[0]}': float(pgd_acc)})
    if config.Operation.Validate_CW:
        cw_acc = evaluate_cw(net, test_loader, config.ADV.clip_eps, config.ADV.fgsm_step, 20)
        logger.info(f"CW_attack:[nb_iter:20,eps:{config.ADV.clip_eps},step_size:{config.ADV.fgsm_step}]->CW_acc: {cw_acc: .2f}")
        wandb_logger.log({'test_last/cw_acc': float(cw_acc)})
    if config.Operation.Validate_Autoattack:
        auto_acc = evaluate_autoattack(net, test_loader, config.ADV.clip_eps, auto_attacks_methods)
        logger.info(f"Auto_attack:[eps:{config.ADV.clip_eps}]->AA_acc: {auto_acc: .2f}")
        wandb_logger.log({'test_last/autoattack_acc': float(auto_acc)})

wandb_logger.finish()
