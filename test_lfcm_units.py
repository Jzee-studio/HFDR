"""Unit verification tests for LFCM implementation."""
import torch
import sys
sys.path.insert(0, '.')

print('=== Test 1: LFCM Module ===')
from models.utils import LFCM

lfcm = LFCM(in_channel=16, codebook_size=64, code_dim=32, hidden_dim=64, tau=1.0)
x = torch.randn(4, 16, 32, 32)
lfcm.train()
LF_out, z, z_hat, w = lfcm(x)

print(f'  Input shape:  {x.shape}')
print(f'  LF_out shape: {LF_out.shape}')
assert LF_out.shape == x.shape, "LF_out shape mismatch!"
print(f'  z shape:      {z.shape}')
assert z.shape == (4, 32), "z shape mismatch!"
print(f'  z_hat shape:  {z_hat.shape}')
assert z_hat.shape == (4, 32), "z_hat shape mismatch!"
print(f'  w shape:      {w.shape}')
assert w.shape == (4, 64), "w shape mismatch!"

w_sums = w.sum(dim=1)
assert torch.allclose(w_sums, torch.ones_like(w_sums), atol=1e-5), f"w sums not 1: {w_sums}"
print(f'  w sums OK (~1.0)')

assert (LF_out != x).any(), "LF_out should differ from input!"
print(f'  LF_out != LF: True')

lambda_gate = torch.sigmoid(lfcm.lambda_raw)
assert 0 <= lambda_gate.min() <= 1, "lambda out of bounds!"
print(f'  lambda gate OK in [0,1]: [{lambda_gate.min().item():.4f}, {lambda_gate.max().item():.4f}]')

print(f'  EMA cluster_size sum: {lfcm.ema_cluster_size.sum().item():.2f}')

lfcm.eval()
with torch.no_grad():
    LF_out_eval, _, _, _ = lfcm(x)
assert LF_out_eval.shape == x.shape, "eval mode shape mismatch!"
print(f'  Eval mode output OK')

print()
print('=== Test 2: Loss Functions ===')
from utils_train import codebook_perplexity_loss, commitment_loss, canonicalization_loss

w_uniform = torch.ones(4, 64) / 64.0
loss_uniform = codebook_perplexity_loss(w_uniform)
print(f'  Perplexity loss (uniform): {loss_uniform.item():.6f}')
assert loss_uniform.item() < 0.01, "Uniform weights should give near-zero loss!"

w_collapsed = torch.zeros(4, 64)
w_collapsed[:, 0] = 1.0
loss_collapsed = codebook_perplexity_loss(w_collapsed)
print(f'  Perplexity loss (collapsed): {loss_collapsed.item():.4f}')
assert loss_collapsed.item() > 0.5, "Collapsed weights should give large loss!"

loss_commit = commitment_loss(z, z_hat)
print(f'  Commitment loss: {loss_commit.item():.4f}')
assert loss_commit.item() > 0, "Commitment loss should be > 0!"

z_clean = torch.randn(8, 32)
z_adv = z_clean + 0.1 * torch.randn(8, 32)
targets = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
loss_canon = canonicalization_loss(z_clean, z_adv, targets)
print(f'  Canonicalization loss (close): {loss_canon.item():.4f}')

z_adv_far = z_clean + 5.0 * torch.randn(8, 32)
loss_canon_far = canonicalization_loss(z_clean, z_adv_far, targets)
print(f'  Canonicalization loss (far): {loss_canon_far.item():.4f}')
assert loss_canon_far > loss_canon, "Far codes should give larger canon loss!"

print()
print('=== Test 3: WideResNet_LFCM ===')
from models.wrnnet import WRN34_10_LFCM

net = WRN34_10_LFCM(Num_class=10)
print(f'  Model: {type(net).__name__}')
print(f'  Filter: {type(net.Filter).__name__}')
print(f'  Recon:  {type(net.Recon).__name__}')
print(f'  LFCM:   {type(net.LFCM).__name__}')
assert net.filter_size == net.conv1.out_channels, "Channel assertion failed!"
print(f'  Channel assertion OK: {net.filter_size} == {net.conv1.out_channels}')

img = torch.randn(2, 3, 32, 32)
net.eval()
with torch.no_grad():
    logits = net(img)
assert logits.shape == (2, 10), f"Wrong inference shape: {logits.shape}"
print(f'  Inference shape OK: {logits.shape}')

net.train()
logits, aux = net(img, return_aux=True)
assert logits.shape == (2, 10), f"Wrong training shape: {logits.shape}"
assert 'mask' in aux and 'z' in aux and 'z_hat' in aux and 'w' in aux, f"Missing aux keys: {list(aux.keys())}"
print(f'  Training shape OK, aux keys: {list(aux.keys())}')
print(f'  aux shapes: mask={aux["mask"].shape}, z={aux["z"].shape}, z_hat={aux["z_hat"].shape}, w={aux["w"].shape}')

# Verify PGD-compatible (no-aux path)
logits_pgd = net(img)
assert logits_pgd.shape == (2, 10), "PGD path failed!"
print(f'  PGD-compatible (no aux) OK')

print()
print('=== Test 4: ResNet_LFCM ===')
from models.resnet import ResNet18_LFCM

net_rn = ResNet18_LFCM(Num_class=10)
img_rn = torch.randn(2, 3, 32, 32)
net_rn.train()
logits_rn, aux_rn = net_rn(img_rn, return_aux=True)
assert logits_rn.shape == (2, 10), f"Wrong RN shape: {logits_rn.shape}"
print(f'  ResNet_LFCM OK: logits={logits_rn.shape}, aux_keys={list(aux_rn.keys())}')

print()
print('=== Test 5: Dead Code Reset ===')
lfcm2 = LFCM(in_channel=16, codebook_size=64, code_dim=32)
x2 = torch.randn(16, 16, 32, 32)
lfcm2.train()
_, z2, _, _ = lfcm2(x2)
lfcm2.ema_cluster_size[:] = 0.0
lfcm2.ema_cluster_size[:10] = 10.0
n_reset = lfcm2.reset_dead_codes(z2)
assert n_reset == 54, f"Expected 54 dead codes, got {n_reset}"
print(f'  Dead codes reset: {n_reset} OK')

print()
print('=== Test 6: Training Mode vs Eval Mode ===')
lfcm3 = LFCM(in_channel=16, codebook_size=64, code_dim=32)
x3 = torch.randn(4, 16, 32, 32)

# Training: EMA should update
lfcm3.train()
ema_before = lfcm3.ema_cluster_size.clone()
_, _, _, _ = lfcm3(x3)
ema_after = lfcm3.ema_cluster_size.clone()
assert not torch.equal(ema_before, ema_after), "EMA should update in training mode!"
print(f'  EMA updates in train mode: OK')

# Eval: EMA should NOT update
lfcm3.eval()
ema_before_eval = lfcm3.ema_cluster_size.clone()
with torch.no_grad():
    _, _, _, _ = lfcm3(x3)
ema_after_eval = lfcm3.ema_cluster_size.clone()
assert torch.equal(ema_before_eval, ema_after_eval), "EMA should NOT update in eval mode!"
print(f'  EMA frozen in eval mode: OK')

print()
print('=== ALL TESTS PASSED ===')
