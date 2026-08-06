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
parser.add_argument("--")
parser.add_argument("--")

# optimization 

# evaluation

# Output

# Compute init and wandb logging

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







