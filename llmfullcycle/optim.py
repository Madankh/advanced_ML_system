import torch
import torch.distributed as dist
from torch import Tensor
from llmfullcycle.helper import COMPUTE_DTYPE

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
    exp_avg_sqr.lerp(grad.square(), 1 - beta_2)
    bias_correction1 = 1 - beta_1 ** step_t
    bias_correction2 = 1 - beta_2 ** step_t
    denom = (exp_avg_sqr / bias_correction2).sqrt() + eps_t
    step_size = lr / bias_correction1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
        stack_grad:Tensor,
        stack_params:Tensor,
        momentum_buffer:Tensor,
        second_moment_buffer:Tensor,
        momentum_t:Tensor,
        lr_t:Tensor,
        wd_t:Tensor,
        beta_2:Tensor,
        ns_steps:int,
        red_dim:int
)->None:
    momentum = momentum_t.to(stack_grad.dtype)
    momentum_buffer.lerp(stack_grad, 1 - momentum)
    g = stack_grad.lerp(momentum_buffer, momentum)

    # Polar express 
    x = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)




