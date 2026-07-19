import os
import io
import sys
import hashlib
import random
import numpy as np
import torch
import psutil
import GPUtil
import logging
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset, IterableDataset


def compact_path_part(value):
    return str(value).replace("/", "-").replace("\\", "-").replace(":", "-")


def compact_config_name(args, config_name_keys):
    full_name = "_".join(f"{key}_{getattr(args, key)}" for key in config_name_keys)
    short_name = (
        f"z{args.z_dim}_tok{args.total_tokens}_buf{args.buffer_size}_"
        f"L{args.layer}_tau{args.tau}_topk{args.topk}_"
        f"obr{args.out_batch_ratio}_lr{args.lr}_seed{args.seed}"
    )
    digest = hashlib.sha1(full_name.encode("utf-8")).hexdigest()[:10]
    return f"{short_name}_h{digest}", full_name


def set_seed(seed):
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)              # CPU
    torch.cuda.manual_seed(seed)         # Current GPU
    torch.cuda.manual_seed_all(seed)     # All GPUs (if using DataParallel or multi-GPU)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Turn off autotune to keep it deterministic

def setup_logging(args, config_name_keys):
    # Prepare model saving directory
    log_model_save_prefix = os.path.join(
        args.results_dir,
        compact_path_part(args.model_name),
        compact_path_part(args.context),
        compact_path_part(args.text),
    )
    config_name, full_config_name = compact_config_name(args, config_name_keys)
    
    log_model_save_dir = os.path.join(log_model_save_prefix, config_name)
    os.makedirs(log_model_save_dir, exist_ok=True)
    assert os.path.exists(log_model_save_dir) 

    model_save_dir = os.path.join(log_model_save_dir, 'ckps')
    os.makedirs(model_save_dir, exist_ok=True)
    assert os.path.exists(model_save_dir) 

    log_path = os.path.join(log_model_save_dir, "train_log.output")
    loss_path = os.path.join(log_model_save_dir, "loss.json")

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

    # Console handler (optional, for real-time monitoring)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    # Clear existing handlers (if rerunning in notebook or script)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Redirect stdout and stderr to logging
    sys.stdout = open(log_path, 'a')
    sys.stderr = sys.stdout

    # First log
    logger.info("Logging to: {:s}".format(log_path))
    logger.info("Compact config dir: %s", config_name)
    logger.info("Full config name: %s", full_config_name)

    return logger, log_path, loss_path, model_save_dir

def logging_mem_usage(logger):
    # CPU usage
    process = psutil.Process()
    cpu_mem_mb = process.memory_info().rss / 1024**2  # in MB
    cpu_percent = process.cpu_percent(interval=0.1)
    logger.info(f"CPU Mem Used: {cpu_mem_mb} MB ({cpu_percent} %)")

    # MPS usage on Apple Silicon
    if torch.backends.mps.is_available():
        current_mb = torch.mps.current_allocated_memory() / 1024**2
        driver_mb = torch.mps.driver_allocated_memory() / 1024**2
        max_mb = torch.mps.recommended_max_memory() / 1024**2
        logger.info(f"MPS Mem Used: current={current_mb:.2f} MB, driver={driver_mb:.2f} MB, recommended max={max_mb:.2f} MB")

    # CUDA usage from PyTorch. This is more reliable than GPUtil inside some envs.
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        allocated_mb = torch.cuda.memory_allocated(device_idx) / 1024**2
        reserved_mb = torch.cuda.memory_reserved(device_idx) / 1024**2
        total_mb = torch.cuda.get_device_properties(device_idx).total_memory / 1024**2
        logger.info(
            f"CUDA Mem Used: allocated={allocated_mb:.2f} MB, reserved={reserved_mb:.2f} MB, total={total_mb:.2f} MB"
        )
        return

    # Fallback GPU usage from GPUtil, mostly useful when PyTorch CUDA is unavailable.
    gpus = GPUtil.getGPUs()
    if not gpus:
        logger.info("GPU Mem Used: no GPU detected by GPUtil")
        return

    gpu = gpus[0]
    logger.info(f"GPU Mem Used: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB")

# Tqdm to Logger class
class TqdmToLogger(io.StringIO):
    def __init__(self, logger, level=logging.INFO):
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, buf):
        self.logger.log(self.level, buf.strip())

    def flush(self):
        pass  # tqdm doesn't need flush here

def gen_window_slicing_batch(batch, window_size, stride=1):
    # Slide over the sequence length dimension
    seq_len = batch.shape[0]
    window_slicing_batch = []
    for i in range(0, seq_len - window_size + 1, stride):
        window_slicing_batch.append(batch[i:i + window_size, :].T)
   
    return torch.stack(window_slicing_batch, dim=0)

def draw_loss(loss_dict, loss_path, total_tokens):
    f, axs = plt.subplots(nrows=2, ncols=3, figsize=(3*3, 2*2))

    for idx, (key, vals) in enumerate(loss_dict.items()):
        n_points = 5
        token_per_points = total_tokens / (n_points-1)

        xticks = [i * token_per_points for i in range(n_points)]
        xtick_labels = [f"{x :.1f}" for x in xticks]

        ax = axs[idx//3, idx%3]
        ax.plot(vals)
        ax.set_title(key)
        ax.set_xlabel('Training tokens (M)')
        # ax.set_xticks(range(len(vals)))
        # ax.set_xticklabels(xtick_labels, rotation=45)

    fig_path = loss_path.replace('.json', '.png')
    f.tight_layout()
    f.savefig(fig_path)

def hugging_face_login(token):
    from huggingface_hub import login
    login(token)
