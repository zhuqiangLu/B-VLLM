# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import torch.nn.functional as F
import math
from typing import Callable, Tuple

import torch
from torch_geometric.utils import to_dense_adj
from einops import einsum
import math
def do_nothing(x, mode=None):
    return x

def multi_layer_merge(metrics, target_num_tokens):
    # print(num_tokens, target_num_tokens, math.log(num_tokens/target_num_tokens),math.ceil(math.log(num_tokens/target_num_tokens)))
    # step = max(0, math.ceil(math.log(num_tokens/target_num_tokens)))
    # print(step)
    
    while True:
        if metrics.shape[1] // 2 <= target_num_tokens:
            break
        # if metrics.shape[1] // 2 < 1:
        #     break
        b, _ = bipartite_soft_matching(metrics, metrics.shape[1]//2)
        metrics, _ = merge_wavg(b, metrics)

    offset= metrics.shape[1] - target_num_tokens
    if offset <= 0:
        return metrics
    b, _ = bipartite_soft_matching(metrics, offset)
    metrics, _ = merge_wavg(b, metrics) 
    return metrics
def threshold_soft_matching(metric: torch.Tensor, threshold=0.75):
    # print(threshold)
    # if threshold < 0:
    #     return do_nothing, do_nothing 

    
    
    with torch.no_grad():
        # metric = metric / metric.norm(dim=-1, keepdim=True)
        metric = metric.detach().clone() 
        sims = F.cosine_similarity(metric[None], metric[:, None], dim=-1)
        sims = (sims + 1) / 2
        # print(sims.shape, metric.shape)
        # print(sims)


        # preserve temporal order via upper triangle 
        # diagonal = 1 is to remove the diagonal similarity( whihc is one)
        # print(metric.shape, sims.shape, sims)

        sims =  torch.tril(sims.to(torch.float), diagonal=-1)
        # print(sims)

        node_max, node_idx = sims.max(dim=-1)
        # print(node_max, node_idx)
        src_idx = (node_max > threshold).nonzero()#.squeeze(dim=-1)
        # print(src_idx)
        unm_idx = (node_max <= threshold).nonzero()# .squeeze(dim=-1)
        # print(unm_idx)
        # print(node_max)
        # print(node_idx) 
        # print(src_idx)
        # print(unm_idx)

        # print(sims.shape, src_idx.shape, unm_idx.shape)
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        # print(src_idx, dst_idx,  unm_idx)
        # print(sims.shape, src_idx.shape, unm_idx.shape, dst_idx.shape)

        argmax_idxs = metric.argmax(-1)[unm_idx]
        # print(argmax_idxs.shape)
        # print(argmax_idxs)
        # print(argmax_idxs.squeeze(dim=-1).sort())

       
        



    def merge(x: torch.Tensor, mode="mean", sort=False) -> torch.Tensor:

        edge = torch.cat([dst_idx, src_idx], dim=-1).transpose(-1, -2)
        # print(edge)
        adj = to_dense_adj(edge, max_num_nodes=x.shape[-2]).squeeze(dim=0) + torch.eye(x.shape[-2]).to(x.device)

        adj = adj.to(x.dtype)
        # print(adj, src_idx, dst_idx)
        # adj = torch.eye(x.shape[-2])

        # adj = adj.to(x.device)

        aggre_x = einsum(adj, x, 'l j, j c -> l c') / adj.sum(dim=-1, keepdim=True)
        # print(aggre_x.shape)

        merged_x = aggre_x[unm_idx.squeeze(dim=-1)]
        if sort:
            merged_x = merged_x[argmax_idxs.squeeze(dim=-1).sort().indices]





        return merged_x


    return merge

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    # r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

  

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        # print(x.shape, src.shape, dst.shape, node_idx.shape, unm_idx.shape, src_idx.shape, dst_idx.shape)
        # raise
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge



def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    size = size.to(x.dtype)
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


if __name__ == '__main__':
    # a = torch.randn(2, 256, 768)
    # b, _ = bipartite_soft_matching(a, 100) 
    # c, s = merge_wavg(b, a)
    a = torch.randn(5, 768).cuda()
    # a[1] = a[0]
    # a[2] = a[-1]
    # a[3] = a[0]
    b = threshold_soft_matching(a, 1) 
    c, _ = merge_wavg(b, a)
    print(c.shape)
    # print((c[0] - a[1]).sum())
    # print(c.shape,)
    # c  = torch.randn(32, 256)

    # c = multi_layer_merge(c.unsqueeze(dim=0), c.shape[0], 32)
    # print(c.shape)
