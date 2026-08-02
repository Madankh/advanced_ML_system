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

# model architecture

# tranining horizon

# optimization 

# evaluation

# Output


# Compute init and wandb logging

# wandb logging init

# Flash attention status 

# Build tokenizer

# Initialize the model

# Build the model , move to device , init the weights 


