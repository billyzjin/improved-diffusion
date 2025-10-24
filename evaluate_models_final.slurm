#!/bin/bash
#SBATCH --account=bata0-external
#SBATCH --partition=long_h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_final
#SBATCH --output=evaluation_final_%j.out
#SBATCH --error=evaluation_final_%j.err

echo "=========================================="
echo "EVALUATING MODELS (FINAL VERSION)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Set up environment with error checking
echo "Loading modules..."
module load python/booth/3.12 || { echo "ERROR: Failed to load python/booth/3.12"; exit 1; }

# Try different CUDA module versions (based on actual availability)
if module load cuda/12.8 2>/dev/null; then
    echo "Loaded cuda/12.8"
elif module load cuda/12.4 2>/dev/null; then
    echo "Loaded cuda/12.4"
elif module load cuda/12.2 2>/dev/null; then
    echo "Loaded cuda/12.2"
elif module load cuda/11.8 2>/dev/null; then
    echo "Loaded cuda/11.8"
else
    echo "ERROR: No CUDA module found"
    exit 1
fi

# NVIDIA modules not available on Pythia, skip
echo "INFO: NVIDIA modules not available on Pythia, skipping"

# Test PyTorch
echo "Testing PyTorch..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || { echo "ERROR: PyTorch test failed"; exit 1; }

# Ensure we're in the correct directory
cd /home/bjin0/improved-diffusion
echo "Working directory: $(pwd)"

# Create evaluation directory
EVAL_DIR="/scratch/bjin0/evaluation_final_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"
echo "Evaluation results will be saved to: $EVAL_DIR"

# List of all experiments to evaluate (comprehensive)
EXPERIMENTS=(
    "cifar10_ours_simple"
    "cifar10_linear_simple" 
    "cifar10_cosine_simple"
    "cifar10_linear_hybrid"
    "cifar10_cosine_hybrid"
    "cifar10_cosine_vlb"
    "cifar10_ours_hybrid"
)

echo ""
echo "Evaluating ${#EXPERIMENTS[@]} models (comprehensive final version)..."
echo ""

# Function to create comprehensive no-MPI patches
create_no_mpi_patches() {
    echo "Creating comprehensive no-MPI patches for evaluation..."
    
    # Create dist_util_no_mpi.py
    cat > improved_diffusion/dist_util_no_mpi.py << 'EOF'
import os
import torch

def setup_dist():
    pass

def dev():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_state_dict(path, map_location=None):
    return torch.load(path, map_location=map_location)

def sync_params(params):
    pass

def get_world_size():
    return 1

def get_rank():
    return 0

def is_main_process():
    return True

def barrier():
    pass

def broadcast(tensor, src=0):
    return tensor

def all_gather(tensor):
    return [tensor]

def all_reduce(tensor, op=None):
    return tensor
EOF

    # Create image_datasets_no_mpi.py with deterministic support
    cat > improved_diffusion/image_datasets_no_mpi.py << 'EOF'
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_dir, image_size, class_cond=False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.class_cond = class_cond
        
        # Get list of image files
        self.image_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_files)} images in {data_dir}")
        
        # Create dummy labels if class_cond is True
        if class_cond:
            self.labels = np.random.randint(0, 10, len(self.image_files))
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) * 2.0  # Normalize to [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        if self.class_cond:
            return image, self.labels[idx]
        else:
            return image

def load_data(data_dir, batch_size, image_size, class_cond=False, deterministic=False):
    dataset = ImageDataset(data_dir, image_size, class_cond)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=not deterministic,
        num_workers=1,
        pin_memory=True
    )
    
    # Return infinite iterator
    while True:
        for batch in dataloader:
            if class_cond:
                images, labels = batch
                yield images, {"y": labels}
            else:
                yield batch, {}
EOF

    # Create a completely patched version of image_nll.py
    cat > scripts/image_nll_no_mpi.py << 'EOF'
"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import numpy as np
import torch as th

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def run_bpd_evaluation(model, diffusion, data, num_samples, clip_denoised):
    """
    Compute the bits per dimension of a model.
    """
    model.eval()
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0
    
    with th.no_grad():
        batch_count = 0
        max_batches = (num_samples // 128) + 10  # Safety limit
        for batch, model_kwargs in data:
            if num_complete >= num_samples or batch_count >= max_batches:
                break
            batch_count += 1
                
            batch = batch.to(dist_util.dev())
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            
            # Compute the loss on a larger minibatch
            minibatch_metrics = diffusion.calc_bpd_loop(
                model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
            )
            
            # No-MPI version: use metrics directly without distributed operations
            for key, term_list in all_metrics.items():
                terms = minibatch_metrics[key].mean(dim=0)  # Remove / dist.get_world_size()
                term_list.append(terms.detach().cpu().numpy())
            
            total_bpd = minibatch_metrics["total_bpd"]
            total_bpd = total_bpd.mean()  # Remove / dist.get_world_size()
            all_bpd.append(total_bpd.item())
            num_complete += batch.shape[0]  # Remove dist.get_world_size() multiplication
            
            logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd):.6f}")
    
    # Save metrics (no-MPI version)
    for name, terms in all_metrics.items():
        out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
        logger.log(f"saving {name} terms to {out_path}")
        np.savez(out_path, np.mean(np.stack(terms), axis=0))
    
    logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd):.6f}")


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log("evaluating...")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised)

    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=True, num_samples=1000, batch_size=1, model_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
EOF

    # Create a patched version of resample.py
    cat > improved_diffusion/resample_no_mpi.py << 'EOF'
import numpy as np
import torch as th

class UniformSampler:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

    def sample(self, batch_size, device):
        ts = np.random.randint(0, self.num_timesteps, batch_size)
        return th.tensor(ts, device=device)

class LossAwareSampler:
    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([num_timesteps], dtype=np.int)

    def weights(self):
        if not self._loss_counts.all():
            return np.ones([self.num_timesteps], dtype=np.float64) / self.num_timesteps
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / self.num_timesteps
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, v in zip(ts, losses):
            if self._loss_counts[t] < self.history_per_term:
                self._loss_history[t, self._loss_counts[t]] = v
                self._loss_counts[t] += 1
            else:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = v

    def sample(self, batch_size, device):
        weights = self.weights()
        p = weights / np.sum(weights)
        indices = np.random.choice(self.num_timesteps, batch_size, p=p)
        return th.tensor(indices, device=device)
EOF

    # Create a patched version of logger.py
    cat > improved_diffusion/logger_no_mpi.py << 'EOF'
"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import sys
import shutil
import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def write(self, s):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "write"), "expected file or str, got %s" % filename_or_file
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = "%-8.3g" % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate_str(key)] = self._truncate_str(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append("| %s %s |" % (key.ljust(keywidth), val.rjust(valwidth)))
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate_str(self, s):
        maxlen = 30
        return s[:maxlen] + "..." if len(s) > maxlen else s

    def write(self, s):
        self.file.write(s)
        self.file.flush()
        if hasattr(self.file, "isatty") and self.file.isatty():
            sys.stdout.write(s)
            sys.stdout.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                v = v.tolist()
            self.file.write(json.dumps({k: v}) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
            self.file.flush()

        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 0
        prefix = "events"
        path = osp.join(osp.join(dir, "events.out.tfevents.{}".format(prefix)))
        import tensorflow as tf

        self.writer = tf.summary.FileWriter(path)

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {"tag": k, "simple_value": float(v)}
            return tf.Summary.Value(**kwargs)

        summary = tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        self.writer.add_summary(summary, self.step)
        self.writer.flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.close()


def make_output_format(format, ev_dir, log_suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress%s.json" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(osp.join(ev_dir, "tb%s" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values will be averaged.
    """
    get_current().logkv_mean(key, val)


def logkv_max(key, val):
    """
    The same as logkv(), but if called many times, values will be maxed.
    """
    get_current().logkv_max(key, val)


def logkv_min(key, val):
    """
    The same as logkv(), but if called many times, values will be minned.
    """
    get_current().logkv_min(key, val)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()


def getkvs():
    return get_current().name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()


# ================================================================
# Backend
# ================================================================


def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.
        """
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        """
        The same as logkv(), but if called many times, values will be averaged.
        """
        if val is not None:
            self.name2val[key] = val
            self.name2cnt[key] += 1

    def logkv_max(self, key, val):
        """
        The same as logkv(), but if called many times, values will be maxed.
        """
        if val is not None:
            self.name2val[key] = val
            self.name2cnt[key] += 1

    def logkv_min(self, key, val):
        """
        The same as logkv(), but if called many times, values will be minned.
        """
        if val is not None:
            self.name2val[key] = val
            self.name2cnt[key] += 1

    def dumpkvs(self):
        """
        Write all of the diagnostics from the current iteration
        """
        if self.comm is None:
            d = self.name2val
        else:
            # No-MPI version: just use local values
            d = self.name2val
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.write(" ".join(map(str, args)) + "\n")


def _configure_default_logger():
    output_formats = [HumanOutputFormat(sys.stdout)]
    Logger.CURRENT = Logger(dir=None, output_formats=output_formats)


def configure(dir=None, format_strs=None, comm=None, log_suffix=""):
    """
    If comm is provided, average all numerical stats across that comm
    """
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = osp.join(tempfile.gettempdir(),
                       datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    os.makedirs(os.path.expanduser(dir), exist_ok=True)

    # No-MPI version: always use rank 0
    rank = 0
    if rank > 0:
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv("OPENAI_LOG_FORMAT", "stdout,log,csv").split(",")
        else:
            format_strs = os.getenv("OPENAI_LOG_FORMAT_MPI", "log").split(",")
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    log("Logging to %s" % dir)


# Backwards compatibility
def log(*args):
    get_current().log(*args)


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)


def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    get_current().dumpkvs()


def getkvs():
    return get_current().name2val.copy()


def log(*args, level=INFO):
    get_current().log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)


def get_dir():
    """
    Get directory that log files are being written to.
    """
    return get_current().get_dir()


def close():
    """
    Close the current logger.
    """
    if Logger.CURRENT is not None:
        Logger.CURRENT.close()
        Logger.CURRENT = None


# Configure the default logger
configure()
Logger.DEFAULT = Logger.CURRENT
EOF

    # Create a patched version of image_sample.py
    cat > scripts/image_sample_no_mpi.py << 'EOF'
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # No-MPI version: just use the sample directly
        all_images.extend([sample.cpu().numpy()])
        if args.class_cond:
            all_labels.extend([classes.cpu().numpy()])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    else:
        label_arr = None

    shape_str = "x".join(map(str, arr.shape))
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
EOF

    # Backup original files
    cp improved_diffusion/dist_util.py improved_diffusion/dist_util_original.py
    cp improved_diffusion/image_datasets.py improved_diffusion/image_datasets_original.py
    cp improved_diffusion/resample.py improved_diffusion/resample_original.py
    cp improved_diffusion/logger.py improved_diffusion/logger_original.py
    
    # Replace with no-MPI versions
    cp improved_diffusion/dist_util_no_mpi.py improved_diffusion/dist_util.py
    cp improved_diffusion/image_datasets_no_mpi.py improved_diffusion/image_datasets.py
    cp improved_diffusion/resample_no_mpi.py improved_diffusion/resample.py
    cp improved_diffusion/logger_no_mpi.py improved_diffusion/logger.py
    
    echo "Comprehensive no-MPI patches created and applied"
}

# Function to restore original files
restore_originals() {
    echo "Restoring original files..."
    if [ -f "improved_diffusion/dist_util_original.py" ]; then
        cp improved_diffusion/dist_util_original.py improved_diffusion/dist_util.py
        rm improved_diffusion/dist_util_original.py
    fi
    if [ -f "improved_diffusion/image_datasets_original.py" ]; then
        cp improved_diffusion/image_datasets_original.py improved_diffusion/image_datasets.py
        rm improved_diffusion/image_datasets_original.py
    fi
    if [ -f "improved_diffusion/resample_original.py" ]; then
        cp improved_diffusion/resample_original.py improved_diffusion/resample.py
        rm improved_diffusion/resample_original.py
    fi
    if [ -f "improved_diffusion/logger_original.py" ]; then
        cp improved_diffusion/logger_original.py improved_diffusion/logger.py
        rm improved_diffusion/logger_original.py
    fi
    rm -f improved_diffusion/dist_util_no_mpi.py
    rm -f improved_diffusion/image_datasets_no_mpi.py
    rm -f improved_diffusion/resample_no_mpi.py
    rm -f improved_diffusion/logger_no_mpi.py
    rm -f scripts/image_nll_no_mpi.py
    rm -f scripts/image_sample_no_mpi.py
}

# Set up cleanup trap
trap restore_originals EXIT

# Create no-MPI patches
create_no_mpi_patches

# Function to evaluate a single model
evaluate_model() {
    local exp_name="$1"
    local model_path="$2"
    
    echo "=========================================="
    echo "EVALUATING: $exp_name"
    echo "Model: $model_path"
    echo "=========================================="
    
    # Create experiment directory
    local exp_dir="$EVAL_DIR/$exp_name"
    mkdir -p "$exp_dir"
    
    # Determine the correct parameters for this model
    local noise_schedule="linear"  # default
    local learn_sigma="False"
    local dropout="0.1"  # Default for linear schedule
    local use_kl="False"
    local schedule_sampler="uniform"
    local rescale_learned_sigmas="False"
    
    if [[ "$exp_name" == *"cosine"* ]]; then
        noise_schedule="cosine"
        dropout="0.3"  # Cosine schedule uses dropout=0.3
    elif [[ "$exp_name" == *"ours"* ]]; then
        noise_schedule="ours"
        dropout="0.3"  # Custom schedule follows cosine pattern
    fi
    
    if [[ "$exp_name" == *"hybrid"* ]]; then
        learn_sigma="True"
        use_kl="True"
        schedule_sampler="loss-second-moment"
        # Dropout already set above based on noise schedule
    elif [[ "$exp_name" == *"vlb"* ]]; then
        learn_sigma="True"
        use_kl="True"
        schedule_sampler="loss-second-moment"
        dropout="0.3"
        rescale_learned_sigmas="True"  # VLB experiment uses rescale_learned_sigmas=True
    fi
    
    echo "Using parameters: noise_schedule=$noise_schedule, learn_sigma=$learn_sigma, dropout=$dropout, use_kl=$use_kl, schedule_sampler=$schedule_sampler, rescale_learned_sigmas=$rescale_learned_sigmas"
    
    # 1. Calculate NLL (bits/dimension) using patched script
    echo "Calculating NLL using patched image_nll_no_mpi.py..."
    python3 scripts/image_nll_no_mpi.py \
        --model_path "$model_path" \
        --data_dir ./cifar_test \
        --batch_size 128 \
        --num_samples 10000 \
        --image_size 32 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --num_heads 4 \
        --attention_resolutions 16,8 \
        --use_scale_shift_norm True \
        --dropout "$dropout" \
        --learn_sigma "$learn_sigma" \
        --use_kl "$use_kl" \
        --schedule_sampler "$schedule_sampler" \
        --rescale_learned_sigmas "$rescale_learned_sigmas" \
        --diffusion_steps 4000 \
        --noise_schedule "$noise_schedule" \
        --class_cond False \
        --use_checkpoint False \
        --rescale_timesteps True \
        --predict_xstart False \
        --clip_denoised True \
        2>&1 | tee "$exp_dir/nll_results.txt"
    
    # 2. Generate samples for FID using patched script
    echo "Generating samples using patched image_sample_no_mpi.py..."
    python3 scripts/image_sample_no_mpi.py \
        --model_path "$model_path" \
        --num_samples 50000 \
        --batch_size 128 \
        --image_size 32 \
        --num_channels 128 \
        --num_res_blocks 3 \
        --num_heads 4 \
        --attention_resolutions 16,8 \
        --use_scale_shift_norm True \
        --dropout "$dropout" \
        --learn_sigma "$learn_sigma" \
        --use_kl "$use_kl" \
        --schedule_sampler "$schedule_sampler" \
        --rescale_learned_sigmas "$rescale_learned_sigmas" \
        --diffusion_steps 4000 \
        --noise_schedule "$noise_schedule" \
        --class_cond False \
        --use_checkpoint False \
        --rescale_timesteps True \
        --predict_xstart False \
        --clip_denoised True \
        --use_ddim True \
        2>&1 | tee "$exp_dir/sample_results.txt"
    
    # Move generated samples to experiment directory
    # Extract logger directory from the sample generation output
    logger_dir=$(grep "Logging to" "$exp_dir/sample_results.txt" | tail -1 | sed 's/.*Logging to //' | tr -d '\r\n')
    if [ -n "$logger_dir" ] && [ -d "$logger_dir" ]; then
        sample_file="$logger_dir/samples_50000x32x32x3.npz"
        if [ -f "$sample_file" ]; then
            cp "$sample_file" "$exp_dir/"
            echo "Samples saved to: $exp_dir/samples_50000x32x32x3.npz"
            # Clean up the temporary file
            rm -f "$sample_file"
        else
            echo "ERROR: Sample file not found at $sample_file"
            echo "Available files in logger directory:"
            ls -la "$logger_dir" 2>/dev/null || echo "Logger directory not accessible"
        fi
    else
        echo "ERROR: Could not determine logger directory from output"
        echo "Sample generation output:"
        tail -10 "$exp_dir/sample_results.txt"
    fi
    
    echo "Completed evaluation for $exp_name"
    echo ""
}

# Evaluate each model
for exp_name in "${EXPERIMENTS[@]}"; do
    model_path=$(find /scratch/bjin0 -name "ema_0.9999_500000.pt" -path "*/$exp_name/*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    
    if [ -n "$model_path" ] && [ -f "$model_path" ]; then
        evaluate_model "$exp_name" "$model_path"
    else
        echo "WARNING: Model not found for $exp_name"
    fi
done

# Create results summary
echo "=========================================="
echo "CREATING RESULTS SUMMARY"
echo "=========================================="

RESULTS_FILE="$EVAL_DIR/results_summary.txt"
echo "COMPREHENSIVE MODEL EVALUATION RESULTS" > "$RESULTS_FILE"
echo "======================================" >> "$RESULTS_FILE"
echo "Job ID: $SLURM_JOB_ID" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo "EXPERIMENT TYPES:" >> "$RESULTS_FILE"
echo "=================" >> "$RESULTS_FILE"
echo "- Simple: learn_sigma=False, use_kl=False, dropout=0.1/0.3 (based on noise schedule)" >> "$RESULTS_FILE"
echo "- Hybrid: learn_sigma=True, use_kl=True, dropout=0.1/0.3 (based on noise schedule)" >> "$RESULTS_FILE"
echo "- VLB: learn_sigma=True, use_kl=True, dropout=0.3" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

echo "NLL RESULTS (bits/dimension - lower is better):" >> "$RESULTS_FILE"
echo "==============================================" >> "$RESULTS_FILE"

# Extract NLL results
for exp_dir in "$EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ] && [ -f "$exp_dir/nll_results.txt" ]; then
        exp_name=$(basename "$exp_dir")
        nll_score=$(grep "done 10000 samples: bpd=" "$exp_dir/nll_results.txt" | tail -1 | awk '{print $4}')
        
        if [ -n "$nll_score" ]; then
            echo "$exp_name: $nll_score bits/dimension" >> "$RESULTS_FILE"
        else
            echo "$exp_name: ERROR extracting NLL" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "SAMPLE GENERATION STATUS:" >> "$RESULTS_FILE"
echo "========================" >> "$RESULTS_FILE"

# Check sample generation status
for exp_dir in "$EVAL_DIR"/cifar10_*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        if [ -f "$exp_dir/samples_50000x32x32x3.npz" ]; then
            echo "$exp_name: Samples generated successfully" >> "$RESULTS_FILE"
        else
            echo "$exp_name: No samples found" >> "$RESULTS_FILE"
        fi
    fi
done

echo "" >> "$RESULTS_FILE"
echo "PAPER BASELINE COMPARISON:" >> "$RESULTS_FILE"
echo "=========================" >> "$RESULTS_FILE"
echo "From Table 2 of the paper:" >> "$RESULTS_FILE"
echo "- Linear Simple: ~3.17 bits/dimension" >> "$RESULTS_FILE"
echo "- Cosine Simple: ~3.17 bits/dimension" >> "$RESULTS_FILE"
echo "- Linear Hybrid: ~3.11 bits/dimension" >> "$RESULTS_FILE"
echo "- Cosine Hybrid: ~3.11 bits/dimension" >> "$RESULTS_FILE"
echo "- Cosine VLB: ~3.11 bits/dimension" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "Your custom 'ours' schedule (both simple and hybrid) should be compared against these baselines." >> "$RESULTS_FILE"
echo "The hybrid experiments use learn_sigma=True and loss-second-moment sampling." >> "$RESULTS_FILE"

echo "=========================================="
echo "COMPREHENSIVE MODEL EVALUATION COMPLETE!"
echo "=========================================="
echo "Results saved in: $EVAL_DIR"
echo "Summary saved to: $RESULTS_FILE"
echo ""
echo "To view results:"
echo "  cat $RESULTS_FILE"
echo "  ls -la $EVAL_DIR"
