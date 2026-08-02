import torch
import torch.distributed as dist
from torch import Tensor
from llmfullcycle.common import COMPUTE_DTYPE

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p:Tensor, 
    grad:Tensor, 
    exp_avg:Tensor, 
    exp_avg_sqr:Tensor, 
    step_t:Tensor, 
    lr:Tensor, 
    beta_1:Tensor, 
    beta_2:Tensor, 
    eps_t:Tensor, 
    wd_t:Tensor
    ):
    p.mul_(1 - lr * wd_t)
    exp_avg.lerp(grad, 1 - beta_1)
    

