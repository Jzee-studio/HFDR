"""Shared model factory: backbone x training-method -> network.

Usage (train.py / test_robust.py / test_ood.py):
    net = build_model(
        backbone=config.Train.get('Backbone', 'WRN34'),   # or Operation.get(...)
        method=config.Train.Train_Method,                  # or Operation.Method
        num_class=config.DATA.num_class,
        lfcm_cfg=config.get('LFCM', {}),
    )

Method -> variant mapping:
    Natural / AT / HFDR / TRADES  ->  *_F      (forward(x, is_eval))
    LFCM                          ->  *_LFCM   (forward(x, return_aux))
"""
from .resnet import ResNet18_F, ResNet18_LFCM, ResNet50_F, ResNet50_LFCM
from .wrnnet import WRN34_10_F, WRN34_10_LFCM

__all__ = ['build_model']

_F_FACTORIES = {
    'WRN34': WRN34_10_F,
    'ResNet18': ResNet18_F,
    'ResNet50': ResNet50_F,
}
_LFCM_FACTORIES = {
    'WRN34': WRN34_10_LFCM,
    'ResNet18': ResNet18_LFCM,
    'ResNet50': ResNet50_LFCM,
}


def build_model(backbone='WRN34', method='Natural', num_class=10, lfcm_cfg=None):
    """Build a model for the given backbone and training method.

    Args:
        backbone: 'WRN34' | 'ResNet18' | 'ResNet50'
        method:   training-method string, e.g. 'Natural'/'AT'/'HFDR'/'LFCM'
        num_class: number of output classes
        lfcm_cfg:  dict of LFCM hyper-parameters; only consulted when
                   method == 'LFCM' (missing keys fall back to module defaults)

    Returns:
        A bare nn.Module. Caller wraps it in DataParallel and sets
        Norm / norm_mean / norm_std / Data_norm.
    """
    lfcm_cfg = lfcm_cfg or {}

    if method == 'LFCM':
        if backbone not in _LFCM_FACTORIES:
            raise ValueError(
                f"Unknown backbone {backbone!r} for LFCM. "
                f"Choose from {sorted(_LFCM_FACTORIES)}")
        net = _LFCM_FACTORIES[backbone](
            Num_class=num_class,
            codebook_size=lfcm_cfg.get('codebook_size', 64),
            code_dim=lfcm_cfg.get('code_dim', 32),
            hidden_dim=lfcm_cfg.get('hidden_dim', 64),
            tau=lfcm_cfg.get('tau_init', 1.0),
            ema_decay=lfcm_cfg.get('ema_decay', 0.99),
            dead_threshold=lfcm_cfg.get('dead_threshold', 2),
        )
        # 仅 LFCM 网络挂 lfcm_arch（复刻 train.py 原 L45-49；save_checkpoint
        # utils_train.py L212-215 的 hasattr 分支必须对非 LFCM 网络不触发）
        net.lfcm_arch = {
            'codebook_size': lfcm_cfg.get('codebook_size', 64),
            'code_dim': lfcm_cfg.get('code_dim', 32),
            'hidden_dim': lfcm_cfg.get('hidden_dim', 64),
        }
    else:
        if backbone not in _F_FACTORIES:
            raise ValueError(
                f"Unknown backbone {backbone!r} for method {method!r}. "
                f"Choose from {sorted(_F_FACTORIES)}")
        net = _F_FACTORIES[backbone](Num_class=num_class)

    return net
