import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from typing import Any, Tuple
import numpy as np

from tqdm import tqdm
import os
import shutil
from typing import Tuple
from torch import Tensor
from torch.autograd import Variable
from models.utils import Normalization

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def adjust_learning_rate(learning_rate, optimizer, epoch):
    lr = learning_rate
    if epoch >= 100:
        lr /= 10
    if epoch >= 105:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, filepath):
    filename = os.path.join(filepath, 'checkpoint.pth.tar')
    # Save model
    torch.save(state, filename)
    # Save best model
    if is_best:
        shutil.copyfile(filename, os.path.join(filepath, 'model_best.pth.tar'))

def train_adversarial(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
          config: Any) -> Tuple[float, float]:
    print('\n[ Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)

        optimizer.zero_grad()

        benign_outputs = net(adv_inputs)
        if config.Train.Factor > 0.0001:
            label_smoothing = Variable(torch.tensor(_label_smoothing(targets, config.DATA.num_class, config.Train.Factor)).to(device))
            loss = LabelSmoothLoss(benign_outputs, label_smoothing.float())
        else:
            loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(train_acc=round(100. * correct / total, 2))
        train_bar.update()
    train_bar.close()
    print('Total benign train accuarcy:', 100. * correct / total)
    print('Total benign train loss:', train_loss)

    return 100. * correct / total, train_loss

def _label_smoothing(label, num_class=10, factor=0.1):
    one_hot = np.eye(num_class)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_class - 1))

    return result

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

def train(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer, config: Any) -> Tuple[float, float]:
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc='>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        benign_outputs = net(inputs)

        if config.Train.Factor > 0.0001:
            label_smoothing = Variable(torch.tensor(_label_smoothing(targets, config.DATA.num_class, config.Train.Factor)).to(device))
            c_ls = LabelSmoothLoss(benign_outputs, label_smoothing.float())
        else:
            c_ls = criterion(benign_outputs, targets)
        c_ls.backward()

        optimizer.step()
        train_loss += c_ls.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(train_acc=round(100. * correct / total, 2), loss=train_loss)
        train_bar.update(1)
    train_bar.close()

    return 100. * correct / total, train_loss

def test_net_normal(net: nn.Module, test_loader: DataLoader, epoch: int, optimizer: Optimizer, 
         best_prec: float, config: Any,save_path='./checkpoint',) -> Tuple[float, float, float, float]:
    net.eval()
    benign_loss_test = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_bar = tqdm(total=len(test_loader), desc='Test>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, config.ADV.clip_eps/255.,
                        config.ADV.fgsm_step/255., config.ADV.pgd_attack_test)
        with torch.no_grad():
            adv_outputs = net(adv)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss_test += loss.item()

        _, predicted = outputs.max(1)
        _, predicted_adv = adv_outputs.max(1)
        adv_correct += predicted_adv.eq(targets).sum().item()
        benign_correct += predicted.eq(targets).sum().item()
        if total % 100 == 0:
            test_acc = 100. * benign_correct / total
            adv_acc = 100. * adv_correct / total
            test_bar.set_postfix(acc=round(test_acc, 2), adv_acc = round(adv_acc, 2) )
        test_bar.update(1)
    test_bar.close()
    test_acc = 100. * benign_correct / total
    adv_acc = 100. * adv_correct / total
    is_best = test_acc > best_prec
    best_prec_robust = max(test_acc, best_prec)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'best_prec1': best_prec_robust,
        'optimizer': optimizer.state_dict(),
    }, is_best, os.path.join(save_path))
    print('Model Saved!')
    return test_acc, adv_acc, benign_loss_test, best_prec_robust

def test_net_robust(net: nn.Module, test_loader: DataLoader, epoch: int, optimizer: Optimizer, 
         best_prec: float, config: Any,save_path='./checkpoint',) -> Tuple[float, float, float, float]:
    net.eval()
    benign_loss_test = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_bar = tqdm(total=len(test_loader), desc='Test>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, config.ADV.clip_eps/255.,
                        config.ADV.fgsm_step/255., config.ADV.pgd_attack_test)
        with torch.no_grad():
            adv_outputs = net(adv)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss_test += loss.item()

        _, predicted = outputs.max(1)
        _, predicted_adv = adv_outputs.max(1)
        adv_correct += predicted_adv.eq(targets).sum().item()
        benign_correct += predicted.eq(targets).sum().item()
        if total % 100 == 0:
            test_acc = 100. * benign_correct / total
            adv_acc = 100. * adv_correct / total
            test_bar.set_postfix(acc=round(test_acc, 2), adv_acc = round(adv_acc, 2) )
        test_bar.update(1)
    test_bar.close()
    test_acc = 100. * benign_correct / total
    adv_acc = 100. * adv_correct / total
    is_best = adv_acc > best_prec
    best_prec_robust = max(adv_acc, best_prec)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'best_prec1': best_prec_robust,
        'optimizer': optimizer.state_dict(),
    }, is_best, os.path.join(save_path))
    print('Model Saved!')
    return test_acc, adv_acc, benign_loss_test, best_prec_robust

# PGD attack
def pgd_attack(model: nn.Module, x: Tensor, y: Tensor, epsilon: float, alpha: float, iters: int) -> Tensor:
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    for _ in range(iters):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def test_pgd(net: nn.Module, test_loader: DataLoader, config: Any) -> float:
    net.eval()
    adv_correct = 0
    total = 0
    progress_bar = tqdm(total=len(test_loader), desc='Testing-PGD>>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, config.ADV.clip_eps/255.,
                         config.ADV.fgsm_step/255., config.ADV.pgd_attack_test)
        with torch.no_grad():
            adv_outputs = net(adv)
        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()
        progress_bar.set_postfix(test_pgd_acc=round(100. * adv_correct / total, 2))
        progress_bar.update(1)  # update bar
    progress_bar.close()  # close bar
    adv_acc = 100. * adv_correct / total
    print('\n---->PGD attack test accuarcy:', adv_acc)
    return adv_acc

def test_net(net: nn.Module, test_loader: DataLoader, config: Any) -> Tuple[float, float, float]:
    net.eval()
    benign_loss_test = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_bar = tqdm(total=len(test_loader), desc='Test>')
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, config.ADV.clip_eps/255.,
                        config.ADV.fgsm_step/255., config.ADV.pgd_attack_test)
        with torch.no_grad():
            adv_outputs = net(adv)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss_test += loss.item()

        _, predicted = outputs.max(1)
        _, predicted_adv = adv_outputs.max(1)
        adv_correct += predicted_adv.eq(targets).sum().item()
        benign_correct += predicted.eq(targets).sum().item()
        if total % 100 == 0:
            test_acc = 100. * benign_correct / total
            adv_acc = 100. * adv_correct / total
            test_bar.set_postfix(acc=round(test_acc, 2), adv_acc = round(adv_acc, 2) )
        test_bar.update(1)
    test_bar.close()
    test_acc = 100. * benign_correct / total
    adv_acc = 100. * adv_correct / total
    return test_acc, adv_acc, benign_loss_test

def val_net(net: nn.Module, epoch: int, val_loader: DataLoader, optimizer: Optimizer, 
         best_val_robust_acc: float, config: Any, check_path='./checkpoint') -> Tuple[float, float, float]:
    benign_loss_val = 0
    val_benign_correct = 0
    val_adv_correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = pgd_attack(net, inputs, targets, config.ADV.clip_eps/255.,
                        config.ADV.fgsm_step/255., 10)
        with torch.no_grad():
            adv_outputs = net(adv)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss_val += loss.item()
        _, predicted = outputs.max(1)
        _, predicted_adv = adv_outputs.max(1)
        val_adv_correct += predicted_adv.eq(targets).sum().item()
        val_benign_correct += predicted.eq(targets).sum().item()
    val_test_acc = 100. * val_benign_correct / total
    val_adv_acc = 100. * val_adv_correct / total
    is_best = (val_adv_acc > best_val_robust_acc)
    best_val_robust_acc = max(val_adv_acc, best_val_robust_acc)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'best_prec1': best_val_robust_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, os.path.join(check_path))
    return val_test_acc, val_adv_acc, best_val_robust_acc

def record_path_words(record_path, record_words):
    print(record_words)
    with open(record_path, "a+") as f:
        f.write(record_words)
    f.close()
    return

def DFT_diff_L1(HF, LF, Lambda=0.1):

    # Perform 2D FFT on each feature map
    feature_maps_fft_HF = torch.fft.fftn(HF, dim=[2, 3], norm="ortho")
    feature_maps_fft_LF = torch.fft.fftn(LF, dim=[2, 3], norm="ortho")

    # Shift the zero-frequency component to the center of the spectrum
    feature_maps_fft_shifted_HF = torch.fft.fftshift(feature_maps_fft_HF, dim=[2, 3])
    feature_maps_fft_shifted_LF = torch.fft.fftshift(feature_maps_fft_LF, dim=[2, 3])

    diff = feature_maps_fft_shifted_HF- feature_maps_fft_shifted_LF
    l1_norm = torch.norm(diff, p=1, dim=[2,3])

    return Lambda*torch.sum(l1_norm)/HF.shape[1]

def train_adversarial_HF(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
          config: Any) -> Tuple[float, float]:
    print('\n[ Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)

        optimizer.zero_grad()
        benign_outputs= net(adv_inputs)
        if config.Train.Factor > 0.0001:
            label_smoothing = Variable(torch.tensor(_label_smoothing(targets, config.DATA.num_class, config.Train.Factor)).to(device))
            loss = LabelSmoothLoss(benign_outputs, label_smoothing.float())
        else:
            loss = criterion(benign_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(train_acc=round(100. * correct / total, 2))
        train_bar.update()
    train_bar.close()

    return 100. * correct / total, train_loss

def train_adversarial_HF_1(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
          config: Any) -> Tuple[float, float]:
    print('\n[ Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)

        optimizer.zero_grad()
        benign_outputs, mask = net(adv_inputs, True)
        if config.Train.Factor > 0.0001:
            label_smoothing = Variable(torch.tensor(_label_smoothing(targets, config.DATA.num_class, config.Train.Factor)).to(device))
            loss = LabelSmoothLoss(benign_outputs, label_smoothing.float()) + 0.1*mask_constrain_loss(mask,0.1)
        else:
            loss = criterion(benign_outputs, targets) + 0.1*mask_constrain_loss(mask,0.1)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(acc=round(100. * correct / total, 2), loss=loss.item())
        train_bar.update()
    train_bar.close()

    return 100. * correct / total, train_loss

def train_adversarial_TRADES(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
          config: Any, beta = 6.0) -> Tuple[float, float]:
    print('\n[ Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    train_bar = tqdm(total=len(train_loader), desc=f'>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inputs = pgd_attack(net, inputs, targets, config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255., config.Train.pgd_train)

        optimizer.zero_grad()
        benign_outputs,mask = net(adv_inputs, True)
        natural_outputs = net(inputs)
        loss_natural = criterion(natural_outputs, targets)
        loss_1 = F.kl_div(F.log_softmax(benign_outputs, dim=1),
                               F.softmax(net(inputs), dim=1),
                               reduction='batchmean')
        loss = loss_natural + beta*loss_1 + 0.1*mask_constrain_loss(mask,0.1)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = benign_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_bar.set_postfix(acc=round(100. * correct / total, 2), loss=loss.item())
        train_bar.update()
    train_bar.close()

    return 100. * correct / total, train_loss

def mask_constrain(mask, ratio):
    k = (1-ratio)/ratio
    return torch.abs(k*torch.sum(mask==1)-torch.sum(mask==0))/((mask.shape[2]**2)*mask.shape[1]*mask.shape[0])

def mask_constrain_loss(mask, ratio_HF):
    ratio = ratio_HF/(1-ratio_HF)
    freq_ratio = torch.sum(mask)/torch.sum(1-mask)
    return torch.pow(freq_ratio-ratio,2)/(mask.shape[1]*mask.shape[0])


# ==============================================================================
# LFCM Loss Functions
# ==============================================================================

def codebook_perplexity_loss(w):
    """
    Encourage uniform codebook usage via perplexity maximization.

    Penalizes low effective codebook usage: loss = ((K - perplexity) / K)^2

    Args:
        w: (B, K) soft assignment weights from LFCM.forward()

    Returns:
        scalar loss in [0, 1]. 0 = all K codes used uniformly.
    """
    K = w.shape[1]
    # Average assignment probability per code across batch
    p_k = w.mean(dim=0)  # (K,)
    # Perplexity = exp(entropy), measures effective number of codes used
    entropy = -(p_k * torch.log(p_k + 1e-8)).sum()
    perplexity = torch.exp(entropy)
    # Loss: squared relative deviation from perfect uniformity
    loss = ((K - perplexity) / K) ** 2
    return loss


def commitment_loss(z, z_hat):
    """
    VQ-VAE style commitment loss: encoder should commit to the codebook.

    Stop-gradient on z_hat so only the encoder moves toward the codebook.

    Args:
        z:     (B, code_dim) encoded latent
        z_hat: (B, code_dim) soft-quantized code

    Returns:
        scalar MSE loss
    """
    return F.mse_loss(z, z_hat.detach())


def canonicalization_loss(z_hat_clean, z_hat_adv, targets):
    """
    Pull clean and adversarial canonical codes together for same-class pairs.

    Class-conditional: only penalizes distance between clean/adv codes of the
    SAME class, preserving between-class LF diversity.

    Args:
        z_hat_clean: (B, code_dim) canonical codes from clean inputs
        z_hat_adv:   (B, code_dim) canonical codes from adversarial inputs
        targets:     (B,) class labels

    Returns:
        scalar loss
    """
    B = z_hat_clean.shape[0]
    # Pairwise L2 distance matrix
    diff = torch.cdist(z_hat_clean, z_hat_adv, p=2)  # (B, B)
    # Same-class mask
    same_class_mask = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()  # (B, B)
    # Mean distance for same-class pairs only
    n_pairs = same_class_mask.sum()
    if n_pairs < 1:
        return torch.tensor(0.0, device=z_hat_clean.device)
    loss = (diff * same_class_mask).sum() / n_pairs
    return loss


# ==============================================================================
# LFCM Training Loop
# ==============================================================================

def train_LFCM(net: nn.Module, epoch: int, train_loader: DataLoader, optimizer: Optimizer,
               config: Any) -> Tuple[float, float]:
    """
    LFCM training with staged loss schedule.

    Stage 1 (epochs 1-30):  CE + mask_constrain (standard HFDR warmup)
    Stage 2 (epochs 31-60): Add commitment + diversity (ramp up)
    Stage 3 (epochs 61+):   Add canonicalization (ramp up)

    Loss weights:
        L_total = L_ce + w_mask * L_mask + w_commit * L_commit
                  + w_div * L_div + w_canon * L_canon

    Temperature annealing: tau starts at 2.0, anneals to 0.5
    """
    total_epochs = config.Train.Epoch
    print('\n[ LFCM Epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    # ---- Read LFCM config with defaults ----
    lfcm_cfg = config.get('LFCM', {})
    warmup_epochs = lfcm_cfg.get('warmup_epochs', 30)
    commit_ramp_end = lfcm_cfg.get('commit_ramp_end', 60)
    canon_ramp_start = lfcm_cfg.get('canon_ramp_start', 60)
    canon_ramp_end = lfcm_cfg.get('canon_ramp_end', 70)
    w_commit_full = lfcm_cfg.get('w_commit', 0.25)
    w_div_full = lfcm_cfg.get('w_div', 0.1)
    w_canon_full = lfcm_cfg.get('w_canon', 0.5)
    tau_init = lfcm_cfg.get('tau_init', 2.0)
    tau_final = lfcm_cfg.get('tau_final', 0.5)
    tau_anneal_start = lfcm_cfg.get('tau_anneal_start', 30)
    tau_anneal_end = lfcm_cfg.get('tau_anneal_end', 100)
    dead_code_interval = lfcm_cfg.get('dead_code_reset_interval', 5)

    # ---- Stage-dependent loss weights ----
    # Stage 1: CE + mask only (warmup)
    w_mask = 0.1  # always active (same as HFDR)
    if epoch <= warmup_epochs:
        w_commit = 0.0
        w_div = 0.0
        w_canon = 0.0
    elif epoch <= commit_ramp_end:
        # Stage 2: ramp commitment and diversity
        ramp = (epoch - warmup_epochs) / float(commit_ramp_end - warmup_epochs)
        w_commit = w_commit_full * ramp
        w_div = w_div_full * ramp
        w_canon = 0.0
    else:
        # Stage 3: full objective with canonicalization ramp
        w_commit = w_commit_full
        w_div = w_div_full
        if epoch <= canon_ramp_end:
            ramp_canon = (epoch - canon_ramp_start) / float(canon_ramp_end - canon_ramp_start)
            w_canon = w_canon_full * ramp_canon
        else:
            w_canon = w_canon_full

    # ---- Temperature annealing ----
    if hasattr(net, 'module'):
        lfcm = net.module.LFCM
    else:
        lfcm = net.LFCM
    if epoch <= tau_anneal_start:
        lfcm.tau = tau_init
    elif epoch >= tau_anneal_end:
        lfcm.tau = tau_final
    else:
        progress = (epoch - tau_anneal_start) / (tau_anneal_end - tau_anneal_start)
        lfcm.tau = tau_init + (tau_final - tau_init) * progress

    # ---- Dead code reset ----
    if epoch % dead_code_interval == 0 and epoch > 0:
        # Collect a batch of encoder outputs for reset
        z_samples = []
        lfcm.eval()  # temporarily disable EMA during collection
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(train_loader):
                if batch_idx >= 1:
                    break
                inputs = inputs.to(device)
                if hasattr(net, 'module'):
                    module = net.module
                else:
                    module = net
                if hasattr(module, 'norm') and module.norm:
                    inputs = Normalization(inputs, module.mean, module.std)
                out = module.conv1(inputs)
                _, LF, _ = module.Filter(out)
                F_pool = LF.mean(dim=[2, 3])
                z_samples.append(lfcm.encoder(F_pool))
        lfcm.train()
        if z_samples:
            z_batch = torch.cat(z_samples, dim=0)
            n_reset = lfcm.reset_dead_codes(z_batch)
            if n_reset > 0:
                print(f'  [LFCM] Reset {n_reset} dead codebook entries')

    # ---- Training loop ----
    train_bar = tqdm(total=len(train_loader), desc='>>')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Generate adversarial examples
        adv_inputs = pgd_attack(net, inputs, targets,
                                config.Train.clip_eps / 255.,
                                config.Train.fgsm_step / 255.,
                                config.Train.pgd_train)

        optimizer.zero_grad()

        # Forward pass on clean AND adversarial inputs
        clean_logits, aux_clean = net(inputs, return_aux=True)
        adv_logits, aux_adv = net(adv_inputs, return_aux=True)

        # 1. Classification loss (adversarial)
        if config.Train.Factor > 0.0001:
            label_smoothing = Variable(
                torch.tensor(_label_smoothing(targets, config.DATA.num_class,
                                              config.Train.Factor)).to(device))
            loss_ce = LabelSmoothLoss(adv_logits, label_smoothing.float())
        else:
            loss_ce = criterion(adv_logits, targets)

        # 2. Mask constraint loss
        loss_mask = mask_constrain_loss(aux_adv['mask'], 0.1)

        # 3. LFCM commitment loss
        loss_commit = commitment_loss(aux_adv['z'], aux_adv['z_hat'])

        # 4. Codebook diversity (perplexity) loss
        loss_div = codebook_perplexity_loss(aux_adv['w'])

        # 5. Canonicalization loss (clean ↔ adv, same class)
        loss_canon = canonicalization_loss(
            aux_clean['z_hat'], aux_adv['z_hat'], targets
        )

        # ---- Total loss with stage-dependent weights ----
        loss = loss_ce
        loss = loss + w_mask * loss_mask
        loss = loss + w_commit * loss_commit
        loss = loss + w_div * loss_div
        loss = loss + w_canon * loss_canon

        loss.backward()

        # Gradient clipping for LFCM parameters
        lfcm_params = list(lfcm.encoder.parameters()) + \
                      list(lfcm.decoder.parameters()) + \
                      [lfcm.lambda_raw]
        torch.nn.utils.clip_grad_norm_(lfcm_params, max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        _, predicted = adv_logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Detailed loss logging in progress bar
        if batch_idx % 50 == 0:
            train_bar.set_postfix(
                acc=round(100. * correct / total, 2),
                loss=loss.item(),
                ce=loss_ce.item(),
                commit=loss_commit.item(),
                div=loss_div.item(),
                canon=loss_canon.item(),
                tau=round(lfcm.tau, 2),
            )
        train_bar.update()
    train_bar.close()

    # ---- Epoch-level LFCM metrics logging ----
    log_lfcm_metrics(net, test_loader=None, epoch=epoch, aux_adv=aux_adv,
                     aux_clean=aux_clean, targets=targets, config=config)

    print(f'  LFCM tau={lfcm.tau:.3f} | w_commit={w_commit:.3f} w_div={w_div:.3f} w_canon={w_canon:.3f}')
    print(f'  Total train accuracy: {100. * correct / total:.2f}%')
    print(f'  Total train loss: {train_loss:.4f}')

    return 100. * correct / total, train_loss


# ==============================================================================
# LFCM Metrics Logging
# ==============================================================================

def log_lfcm_metrics(net, test_loader, epoch, aux_adv, aux_clean, targets, config):
    """
    Log LFCM-specific mechanism metrics for monitoring training health.

    Metrics logged:
    - Codebook perplexity (effective number of codes used)
    - Clean-adv canonical code distance
    - Per-code usage distribution (top-5 / bottom-5 codes)
    - Lambda gate statistics
    """
    if hasattr(net, 'module'):
        lfcm = net.module.LFCM
    else:
        lfcm = net.LFCM

    # 1. Codebook perplexity
    with torch.no_grad():
        w = aux_adv['w']  # (B, K)
        K = w.shape[1]
        p_k = w.mean(dim=0)  # average usage per code
        entropy = -(p_k * torch.log(p_k + 1e-8)).sum()
        perplexity = torch.exp(entropy).item()
        # Dead codes count
        n_dead = (lfcm.ema_cluster_size < lfcm.dead_threshold).sum().item()
        # Top-5 and bottom-5 code usage
        usage_sorted = p_k.sort().values
        top5_usage = usage_sorted[-5:].sum().item()
        bottom5_usage = usage_sorted[:5].sum().item()

    # 2. Clean-adv code distance
    with torch.no_grad():
        z_clean = aux_clean['z_hat']
        z_adv = aux_adv['z_hat']
        # Mean pairwise L2 distance between clean and adv codes (same class)
        canon_dist = 0.0
        n_pairs = 0
        for c in range(config.DATA.num_class):
            mask_c = (targets == c)
            if mask_c.sum() >= 2:
                z_c_clean = z_clean[mask_c]
                z_c_adv = z_adv[mask_c]
                dist = torch.cdist(z_c_clean, z_c_adv, p=2).mean().item()
                canon_dist += dist * mask_c.sum().item()
                n_pairs += mask_c.sum().item()
        canon_dist = canon_dist / n_pairs if n_pairs > 0 else 0.0

    # 3. Lambda gate statistics
    with torch.no_grad():
        lambda_gate = torch.sigmoid(lfcm.lambda_raw)
        lambda_mean = lambda_gate.mean().item()
        lambda_std = lambda_gate.std().item()
        lambda_min = lambda_gate.min().item()
        lambda_max = lambda_gate.max().item()

    print(f'  [LFCM Metrics] perplexity={perplexity:.1f}/{K} | dead_codes={n_dead} | '
          f'top5_usage={top5_usage:.3f} bottom5_usage={bottom5_usage:.3f}')
    print(f'  [LFCM Metrics] clean-adv canon_dist={canon_dist:.4f} | '
          f'lambda: mean={lambda_mean:.3f} std={lambda_std:.3f} [{lambda_min:.3f}, {lambda_max:.3f}]')