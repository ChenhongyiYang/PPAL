from mmcv.utils import Registry, build_from_cfg
from torch import nn
import warnings

SAMPLER = Registry('sampler')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def builder_al_sampler(cfg):
    """Build distill loss."""
    return build(cfg, SAMPLER)
