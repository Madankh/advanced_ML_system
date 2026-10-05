import os 
# "Use the expandable memory allocator."
# This helps reduce GPU memory fragmentation.
# You usually set this before importing PyTorch, because PyTorch reads this setting when it initializes.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import argparse
import time 
import math
import json
from dataclasses import asdist
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from llmfullcycle.gpt import GPT, Linear,GPTConfig
from llmfullcycle.helper import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
print_banner()

# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
# runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|mps|cpu")
# fp8 traning
parser.add_argument("--fp8", action="store_true", help="Enable FP8 training")
parser.add_argument("--fp8_recipe", type=str, default="tensorwise", choices=["tensorwise", "rowwise"], help="FP8 recipe use tensorwise faster and rowwise more accurate but slower")
# model architecture
parser.add_argument("--depth", type=int, default=20, help="number of transformer blocks")
parser.add_argument("--aspect_ratio", type=float, default=64, help="")
parser.add_argument("--head_dim", type=int, default=128, help="")
parser.add_argument("--max_seq_len", type=int, default=2048, help="")
parser.add_argument("--window_pattern", type=str, default="SSSL", help="window pattern")

# tranining horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit of optimization steps")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate thge num_iterations to reach target_flops")
parser.add_argument("--target-param-data-ratio", type=float, default=12,help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")

# optimization 
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")

# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")

# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()  # for logging

# Compute init and wandb logging
device_type = autodetect_device_type(args.device_type) if args.device_type == "" else args.device_type
ddp,ddp_rank,ddp_local_rank,ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda", else lambda: None


# wandb logging init

# Flash attention status 

# Build tokenizer

# Initialize the model

# Build the model 
 
# Resume traning

# Optional Precision conversion

# Build evaluation model

# compile model

# Analyze model

# Compute scaling law targets

# Auto batchsize

# Weight decay

# Optimizer

# Dataloder

# Decide tranining horizon

# Create schedulers

# Tranining loops start with restore loop state if resuming

   # Compute gradient accumulation


   # Tranining loops 
        
        # Evaluate  model

        # Once in while evaluate the val bpb
        
        # 
        
        # Once in while sample from the model (only master process)
        
        # Save checkpoint at the end of the run 
        
        # Tranining loops using grad accumulation steps
        
        # step optimizer 
        
        # logging (CPU action only) 







