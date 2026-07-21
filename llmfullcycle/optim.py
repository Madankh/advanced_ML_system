import torch
import torch.distributed as dist
from torch import Tensor

@torch.compile(dynamic=False,  fullgraph=True)
def adamw_step_fused(
    p:Tensor,
    grad:Tensor,
    exp_avg:Tensor,
    exp_avg_sqr:Tensor,
    step_t:Tensor,
    lr_t:Tensor,
    beta1_t:Tensor,
    beta2_t:Tensor,
    eps_t:Tensor,
    wd_t:Tensor
)-> None:
   p.mul_(1 - lr_t * wd_t)
   exp_avg.lerp_(grad, 1 - beta1_t)
   exp_avg_sqr.lerp_(grad.square(), 1 - beta2_t)
   
