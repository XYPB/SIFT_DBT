import copy

import torch
from torch import nn

from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad


class MoCo(nn.Module):
    def __init__(self, backbone, feat_chs=2048, output_dim=128):
        super().__init__()

        self.encoder_q = nn.Sequential(
            backbone,
            MoCoProjectionHead(feat_chs, feat_chs, output_dim)
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)

        deactivate_requires_grad(self.encoder_k)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, m):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda(device=x.device)

        # broadcast to all gpus
        gpu_idx = torch.distributed.get_rank()
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, input, momentum_val=0.99):
        query, key = input
        q = self.encoder_q(query)
        with torch.no_grad():
            self._momentum_update_key_encoder(momentum_val)
            key, idx_unshuffle = self._batch_shuffle_ddp(key)
            k = self.encoder_k(key)
            torch.cuda.synchronize()
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return q, k


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
