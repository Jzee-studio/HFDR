import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Normalization(data, mean, std):
    mean = mean.view(1,-1, 1, 1)
    std = std.view(1,-1, 1, 1)
    return (data.to(device)-mean.to(device))/std.to(device)

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.size()
        global_feature = self.global_pool(x)
        attention_weights = self.fc(global_feature).view(n, c, 1, 1)
        attended_feature = x * attention_weights
        return attended_feature


class GumbelSigmoid(nn.Module):
    def __init__(self, tau=1.0):
        super(GumbelSigmoid, self).__init__()

        self.tau = tau
        self.softmax = nn.Softmax(dim=1)
        self.p_value = 1e-8

    def forward(self, x, is_eval=False):
        r = 1 - x

        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        if not is_eval:
            x_N = torch.rand_like(x)
            r_N = torch.rand_like(r)
        else:
            x_N = 0.5 * torch.ones_like(x)
            r_N = 0.5 * torch.ones_like(r)

        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        x = x + x_N
        x = x / (self.tau + self.p_value)
        r = r + r_N
        r = r / (self.tau + self.p_value)

        x = torch.cat((x, r), dim=1)
        x = self.softmax(x)

        return x


class SRMFilter(nn.Module):
    def __init__(self, in_channel=64):
        super(SRMFilter, self).__init__()

        q = torch.tensor([4.0, 12.0, 2.0])

        filter1 = torch.tensor([[0, 0, 0, 0, 0],
                                [0, -1, 2, -1, 0],
                                [0, 2, -4, 2, 0],
                                [0, -1, 2, -1, 0],
                                [0, 0, 0, 0, 0]], dtype=torch.float32) / q[0]
        filter2 = torch.tensor([[-1, 2, -2, 2, -1],
                                [2, -6, 8, -6, 2],
                                [-2, 8, -12, 8, -2],
                                [2, -6, 8, -6, 2],
                                [-1, 2, -2, 2, -1]], dtype=torch.float32) / q[1]
        filter3 = torch.tensor([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 1, -2, 1, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]], dtype=torch.float32) / q[2]

        filters = torch.stack([filter1, filter2, filter3])
        filters = filters.repeat(in_channel, 1, 1).view(in_channel * 3, 1, 5, 5)
        self.conv = nn.Conv2d(in_channel, in_channel * 3, kernel_size=5, padding=2, groups=in_channel, bias=False)
        self.conv.weight = nn.Parameter(filters, requires_grad=False)
        self.conv1 = nn.Conv2d(in_channel * 3, in_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channel * 3)

    def forward(self, x):
        feature = F.relu(self.bn1(self.conv(x)))
        feature = self.conv1(feature)
        mask = feature.reshape(feature.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelSigmoid(tau=0.1)(mask)
        mask = mask[:, 0].reshape(mask.shape[0], feature.shape[1], feature.shape[2], feature.shape[3])

        r_feat = x * mask
        nr_feat = x * (1 - mask)
        return r_feat, nr_feat, mask

class Separation(nn.Module):
    def __init__(self, in_channel=64):
        super(Separation, self).__init__()
        C = in_channel
        num_channel = 64
        self.sep_net = nn.Sequential(
            nn.Conv2d(C, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, C, kernel_size=3, stride=1, padding=1, bias=False)
        )
    def forward(self,x):
        feature = self.sep_net(x)
        mask = feature.reshape(feature.shape[0], 1, -1)
        mask = torch.nn.Sigmoid()(mask)
        mask = GumbelSigmoid(tau=0.1)(mask)
        mask = mask[:, 0].reshape(mask.shape[0], feature.shape[1], feature.shape[2], feature.shape[3])

        HF_feat = feature * mask
        LF_feat = feature * (1 - mask)
        return HF_feat, LF_feat, mask


class Recalibration(nn.Module):
    def __init__(self, size, num_channel=64):
        super(Recalibration, self).__init__()
        self.rec_net = nn.Sequential(
            nn.Conv2d(size, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channel),
            nn.ReLU(),
            nn.Conv2d(num_channel, size, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, feat, mask):
        rec_units = self.rec_net(feat)
        rec_units = rec_units * mask

        return rec_units


class LFCM(nn.Module):
    """
    Low-Frequency Canonicalization Module.

    Extracts global low-frequency statistics via average pooling, encodes them
    to a latent code, soft-quantizes via a learned codebook, and applies
    channel-wise FiLM to canonicalize LF features.

    Key mechanisms:
    - EMA codebook updates (VQ-VAE v2 style) for stable training
    - Dead code reset to prevent codebook collapse
    - Channel-wise λ gating for adaptive canonicalization strength
    - Temperature annealing support for curriculum learning
    """

    def __init__(self, in_channel=16, codebook_size=64, code_dim=32,
                 hidden_dim=64, tau=1.0, ema_decay=0.99, dead_threshold=2):
        """
        Args:
            in_channel: Number of input feature channels (C)
            codebook_size: Number of canonical codes (K)
            code_dim: Dimension of each code vector (d)
            hidden_dim: Hidden dimension of encoder/decoder MLPs
            tau: Temperature for soft quantization (higher = softer)
            ema_decay: EMA decay rate for codebook updates
            dead_threshold: Min assignments to avoid reset (in batch count units)
        """
        super(LFCM, self).__init__()
        self.in_channel = in_channel
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.tau = tau
        self.ema_decay = ema_decay
        self.dead_threshold = dead_threshold

        # Encoder: C -> hidden -> code_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_channel, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, code_dim),
        )

        # Learnable codebook: (K, code_dim)
        self.register_buffer('codebook', torch.randn(codebook_size, code_dim) * 0.01)
        # EMA cluster sizes for dead code detection
        self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
        # EMA accumulated encoder outputs per code
        self.register_buffer('ema_embed_sum', torch.randn(codebook_size, code_dim) * 0.01)

        # Decoder: code_dim -> hidden -> 2*C (γ and β)
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * in_channel),
        )

        # Channel-wise lambda (residual strength), init small via sigmoid(-4) ≈ 0.018
        self.lambda_raw = nn.Parameter(torch.full((in_channel, 1, 1), -4.0))

    def forward(self, LF):
        """
        Args:
            LF: (B, C, H, W) low-frequency feature map

        Returns:
            LF_out: (B, C, H, W) canonicalized LF features
            z:      (B, code_dim) encoded latent code (for commitment loss)
            z_hat:  (B, code_dim) soft-quantized canonical code (for canon loss)
            w:      (B, K) soft assignment weights (for diversity/perplexity loss)
        """
        B, C, H, W = LF.shape

        # 1. Global Average Pool -> (B, C)
        F_pool = LF.mean(dim=[2, 3])  # (B, C)

        # 2. Encode -> (B, code_dim)
        z = self.encoder(F_pool)  # (B, code_dim)

        # 3. Soft quantization via cosine similarity
        z_norm = F.normalize(z, dim=1)                     # (B, code_dim)
        cb_norm = F.normalize(self.codebook, dim=1)        # (K, code_dim)
        sim = torch.matmul(z_norm, cb_norm.t())            # (B, K)
        sim = sim / self.tau
        w = F.softmax(sim, dim=1)                          # (B, K)

        # 4. Weighted sum of codebook entries -> (B, code_dim)
        z_hat = torch.matmul(w, self.codebook)             # (B, code_dim)

        # 5. Decode -> (B, 2*C)
        decoded = self.decoder(z_hat)                       # (B, 2*C)
        gamma = decoded[:, :C].view(B, C, 1, 1)            # (B, C, 1, 1)
        beta = decoded[:, C:].view(B, C, 1, 1)             # (B, C, 1, 1)

        # 6. FiLM transformation
        LF_prime = gamma * LF + beta                        # (B, C, H, W)

        # 7. Channel-wise gated residual
        lambda_gate = torch.sigmoid(self.lambda_raw)       # (C, 1, 1), in [0, 1]
        LF_out = LF + lambda_gate * LF_prime               # (B, C, H, W)

        # 8. Update EMA statistics (only when not in a gradient-computing context,
        #    e.g., skip during PGD attack inner loop where inputs require grad)
        if self.training and not LF.requires_grad:
            self._ema_update(z, w)

        return LF_out, z, z_hat, w

    @torch.no_grad()
    def _ema_update(self, z, w):
        """
        EMA update of codebook entries (VQ-VAE v2 style).
        Called during training to track cluster sizes and update embeddings.

        Args:
            z:  (B, code_dim) encoded latent codes
            w:  (B, K) soft assignment weights
        """
        B = z.shape[0]
        # Hard assignment for EMA: argmax of soft weights
        hard_w = F.one_hot(w.argmax(dim=1), num_classes=self.codebook_size).float()  # (B, K)

        # Update cluster sizes
        cluster_size = hard_w.sum(dim=0)  # (K,)
        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )

        # Update embedding sums
        embed_sum = torch.matmul(hard_w.t(), z)  # (K, code_dim)
        self.ema_embed_sum.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        # Normalize to update codebook entries
        n = self.ema_cluster_size.sum()
        cluster_size_norm = (
            (self.ema_cluster_size + 1e-8) / (n + self.codebook_size * 1e-8) * n
        )
        embed_norm = self.ema_embed_sum / cluster_size_norm.unsqueeze(1)
        self.codebook.copy_(embed_norm)

    @torch.no_grad()
    def reset_dead_codes(self, z_batch):
        """
        Reset codebook entries that have near-zero usage.
        Re-initialize them to random encoder outputs from the current batch.

        Args:
            z_batch: (B, code_dim) encoder outputs from a recent batch
        """
        batch_size = z_batch.shape[0]
        dead_codes = self.ema_cluster_size < self.dead_threshold
        n_dead = dead_codes.sum().item()

        if n_dead > 0 and batch_size > 0:
            # Sample random encoder outputs to re-initialize dead codes
            indices = torch.randint(0, batch_size, (n_dead,), device=z_batch.device)
            self.codebook[dead_codes] = z_batch[indices].clone()
            # Reset EMA stats for re-initialized codes
            self.ema_cluster_size[dead_codes] = self.dead_threshold * torch.ones(n_dead, device=self.codebook.device)
            self.ema_embed_sum[dead_codes] = z_batch[indices].clone() * self.dead_threshold

            return n_dead
        return 0