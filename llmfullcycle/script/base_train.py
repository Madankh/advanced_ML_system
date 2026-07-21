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
