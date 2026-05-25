import os
import sys
import json
import torch
import argparse
import datetime
import traceback
from tqdm import tqdm
from nnsight import LanguageModel
from linear_idol_model import LinearIDOL

from utils import *

# import disctionary_learning 

import submodule_dl.dictionary_learning.utils as utils
from submodule_dl.dictionary_learning.buffer import ActivationBuffer
from submodule_dl.dictionary_learning.utils import hf_dataset_to_generator
from submodule_dl.dictionary_learning.training import get_norm_factor

def _shape_of(value):
    if hasattr(value, "shape"):
        return tuple(value.shape)
    return None

def log_training_error(logger, exc, stage, refresh_idx, n_tokens, args, activation_buffer=None, act=None):
    logger.exception(
        "Training failed during %s at refresh_idx=%s, n_tokens=%s/%s",
        stage,
        refresh_idx,
        n_tokens,
        args.total_tokens,
    )

    if activation_buffer is not None:
        try:
            logger.error(
                "ActivationBuffer state: activations_shape=%s, read_shape=%s, unread=%s, "
                "out_batch_size=%s, activation_buffer_size=%s, ctx_len=%s, refresh_batch_size=%s, device=%s",
                _shape_of(activation_buffer.activations),
                _shape_of(activation_buffer.read),
                int((~activation_buffer.read).sum().item()) if len(activation_buffer.read) > 0 else 0,
                activation_buffer.out_batch_size,
                activation_buffer.activation_buffer_size,
                activation_buffer.ctx_len,
                activation_buffer.refresh_batch_size,
                activation_buffer.device,
            )
        except Exception:
            logger.exception("Failed to log ActivationBuffer state")

    if act is not None:
        logger.error(
            "Current activation batch: shape=%s, dtype=%s, device=%s",
            _shape_of(act),
            getattr(act, "dtype", None),
            getattr(act, "device", None),
        )

    print(
        f"\nERROR: training failed during {stage} at refresh_idx={refresh_idx}, "
        f"n_tokens={n_tokens}/{args.total_tokens}. See log: {getattr(args, 'log_path', '(not set)')}",
        file=sys.__stderr__,
    )
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.__stderr__)

def get_torch_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32

def estimate_trainable_memory_gb(model, optimizer_name):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_bytes = param_bytes
    optimizer_bytes = 0
    if optimizer_name == "adam":
        optimizer_bytes = param_bytes * 2

    return {
        "params": param_bytes / 1024**3,
        "grads": grad_bytes / 1024**3,
        "optimizer": optimizer_bytes / 1024**3,
        "total": (param_bytes + grad_bytes + optimizer_bytes) / 1024**3,
    }

def build_optimizer(args, model):
    if args.optimizer == "sgd":
        return torch.optim.SGD(lr=args.lr, weight_decay=args.wd, params=model.parameters())
    return torch.optim.Adam(lr=args.lr, weight_decay=args.wd, params=model.parameters())

def save_model_checkpoint(model, model_save_path):
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(state_dict, model_save_path)
    del state_dict
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_acts_buffer(model_name,
                    text,
                    layer,
                    buffer_size, 
                    out_batch_ratio,
                    dtype, 
                    device,
                    ctx_len=64,
                    refresh_batch_size=2,
                    data_source="hf",
                    local_text_path="mini_pile.jsonl",
                    llm_device="cpu"):

    n_ctx = buffer_size // ctx_len

    model = LanguageModel(model_name, device_map=llm_device)
    submodule = utils.get_submodule(model, layer)
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    act_dim = model.config.hidden_size

    generator = hf_dataset_to_generator(
        text,
        data_source=data_source,
        local_text_path=local_text_path,
    )

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=n_ctx,
        ctx_len=ctx_len,
        refresh_batch_size=refresh_batch_size,
        out_batch_size=int(buffer_size * out_batch_ratio),
        io=io,
        d_submodule=act_dim,
        device=device,
    )
    
    return activation_buffer, act_dim

def train(model, 
          activation_buffer, 
          optimizer, 
          args,
          logger,
          device, 
          normalize_activations=False):
    
    # Prepare pbar log 
    n_refreshes = int(args.total_tokens // int(args.buffer_size * args.out_batch_ratio))
    if args.total_tokens % int(args.buffer_size * args.out_batch_ratio) > 0:
        n_refreshes = n_refreshes + 1
    
    # Use this in place of regular tqdm
    tqdm_logger = TqdmToLogger(logger)
    total_steps = n_refreshes
    progress_bar = tqdm(total=total_steps, file=tqdm_logger, desc="Training", dynamic_ncols=True)
    # progress_bar = tqdm(total=total_steps, desc="Training", dynamic_ncols=True)

    list_loss_mse_Xt = []
    list_loss_mse_Zt = []
    list_loss_indep = []
    list_loss_sparse_Bs = []
    list_loss_sparse_M = []
    list_loss_sparse_Zt = []

    n_tokens = 0
    model_saving_flags = [0,0,0]
    refresh_idx = 0
    while n_tokens < args.total_tokens:
        stage = "refresh"
        act = None
        try:
            logger.info("Before refresh")
            activation_buffer.refresh()
            logger.info("After refresh")

            stage = "read_activation_batch"
            unread = (~activation_buffer.read).sum()
            logger.info("Unread activations after refresh: %s / out_batch_size=%s", int(unread.item()), activation_buffer.out_batch_size)
            if unread >= activation_buffer.out_batch_size:
                out_batch = next(activation_buffer) # output from activation buffer

                stage = "window_slicing"
                act_batch = gen_window_slicing_batch(out_batch, window_size=args.tau+1)
                # Forward and training 
                model.train() 
                model_dtype = next(model.parameters()).dtype
                act = act_batch.to(device=device, dtype=model_dtype)

                # Compute tokens
                batch_size, act_dim, p = act.shape
                this_batch_tokens = batch_size + p - 1 # p = tau + 1
                n_tokens = n_tokens + this_batch_tokens

                # Log the mem usage status
                stage = "memory_logging"
                logging_mem_usage(logger=logger)

                # Whether normalize activations
                stage = "normalize_activations"
                if normalize_activations:
                    norm_factor = get_norm_factor(act, steps=100)
                    act = act / norm_factor

                progress_bar.set_description(f"Refresh {refresh_idx+1}/{n_refreshes} | token {n_tokens/1_000_000}/{args.total_tokens/1_000_000}M")
                
                stage = "model_forward"
                loss_mse_Xt, loss_mse_Zt, loss_indep, loss_sparse_Bs, loss_sparse_M, loss_sparse_Zt = model(act)
                
                l_mse_Zt = 0.
                # Enable loss_mse_Zt or not
                if args.mse_Zt:
                    l_mse_Zt = 1.
                    
                loss = loss_mse_Xt + l_mse_Zt * loss_mse_Zt + args.l_ind * loss_indep + args.l_spB * loss_sparse_Bs + args.l_spM * loss_sparse_M + args.l_spZ * loss_sparse_Zt

                # pbar log
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "loss_mse_Xt": f"{loss_mse_Xt:.4f}",
                    "loss_mse_Zt": f"{loss_mse_Zt:.4f}",
                    "loss_indep": f"{loss_indep:.4f}",
                    "loss_sp_B": f"{loss_sparse_Bs:.4f}",
                    "loss_sp_M": f"{loss_sparse_M:.4f}",
                    "loss_sp_Zt": f"{loss_sparse_Zt:.4f}",
                })

                list_loss_mse_Xt.append(loss_mse_Xt.item())
                list_loss_mse_Zt.append(loss_mse_Zt.item())
                list_loss_indep.append(loss_indep.item())
                list_loss_sparse_Bs.append(loss_sparse_Bs.item())
                list_loss_sparse_M.append(loss_sparse_M.item())
                list_loss_sparse_Zt.append(loss_sparse_Zt.item())

                # Step
                stage = "optimizer_step"
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.update(1)

                # Save ckps
                stage = "checkpoint_save"
                if n_tokens > int(args.total_tokens * 0.25) and n_tokens <= int(args.total_tokens * 0.5):
                    if model_saving_flags[0] == 0:
                        model_name = os.path.basename(args.model_save_dir) + 'Token_{:.2f}M.ckp'.format(n_tokens / 1_000_000)
                        model_save_path = os.path.join(args.model_save_dir, model_name)
                        logger.info('Saving ckp at training tokens of {:.2f}M at location {:s}...'.format(n_tokens/1_000_000, model_save_path))
                        model.eval()
                        save_model_checkpoint(model, model_save_path)
                        model_saving_flags[0] = 1
                elif n_tokens > int(args.total_tokens * 0.5) and n_tokens <= int(args.total_tokens * 0.75):
                    if model_saving_flags[1] == 0:
                        model_name = os.path.basename(args.model_save_dir) + 'Token_{:.2f}M.ckp'.format(n_tokens / 1_000_000)
                        model_save_path = os.path.join(args.model_save_dir, model_name)
                        logger.info('Saving ckp at training tokens of {:.2f}M at location {:s}...'.format(n_tokens/1_000_000, model_save_path))
                        model.eval()
                        save_model_checkpoint(model, model_save_path)
                        model_saving_flags[1] = 1
                elif n_tokens > int(args.total_tokens * 0.75):
                    if model_saving_flags[2] == 0:
                        model_name = os.path.basename(args.model_save_dir) + 'Token_{:.2f}M.ckp'.format(n_tokens / 1_000_000)
                        model_save_path = os.path.join(args.model_save_dir, model_name)
                        logger.info('Saving ckp at training tokens of {:.2f}M at location {:s}...'.format(n_tokens/1_000_000, model_save_path))
                        model.eval()
                        save_model_checkpoint(model, model_save_path)
                        model_saving_flags[2] = 1
        except StopIteration as exc:
            progress_bar.close()
            logger.warning(
                "Training data exhausted during %s at refresh_idx=%s, n_tokens=%s/%s. "
                "Ending training early and saving current artifacts.",
                stage,
                refresh_idx,
                n_tokens,
                args.total_tokens,
            )
            print(
                f"\nINFO: training data exhausted during {stage} at refresh_idx={refresh_idx}, "
                f"n_tokens={n_tokens}/{args.total_tokens}. Ending training early.",
                file=sys.__stderr__,
            )
            break
        except Exception as exc:
            progress_bar.close()
            log_training_error(logger, exc, stage, refresh_idx, n_tokens, args, activation_buffer=activation_buffer, act=act)
            raise
        finally:
            # refresh_idx ++1
            refresh_idx += 1

    # Save the final ckp, log and plot the losses
    model_name = os.path.basename(args.model_save_dir) + 'Token_{:.2f}M.ckp'.format(n_tokens / 1_000_000)
    model_save_path = os.path.join(args.model_save_dir, model_name)
    logger.info('Saving ckp at training tokens of {:.2f}M at location {:s}...'.format(n_tokens/1_000_000, model_save_path))
    model.eval()
    save_model_checkpoint(model, model_save_path)

    logger.info('Saving loss to loss.json and plotting to loss.png')
    loss_dict = {
        'loss_mse_Xt': list_loss_mse_Xt,
        'loss_mse_Zt': list_loss_mse_Zt,
        'loss_indep': list_loss_indep,
        'loss_sparse_Bs': list_loss_sparse_Bs,
        'loss_sparse_M': list_loss_sparse_M,
        'loss_sparse_Zt': list_loss_sparse_Zt
    }
    with open(args.loss_path, 'w') as f:
        json.dump(loss_dict, f, indent=4)

    draw_loss(loss_dict=loss_dict, loss_path=args.loss_path, total_tokens=args.total_tokens)
    
                                                                                                                                                               
def main():
    parser = argparse.ArgumentParser()
    # Training configs in general
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd'])
    parser.add_argument('--device', default='mps', type=str, choices=['mps', 'cpu', 'auto'])
    parser.add_argument('--buffer-device', default='cpu', type=str, choices=['cpu', 'mps'], help='device used to store ActivationBuffer activations')
    parser.add_argument('--llm-device', default='cpu', type=str, choices=['cpu', 'mps'], help='device used by nnsight/LanguageModel when extracting activations')
    parser.add_argument('--model-dtype', default='float32', type=str, choices=['float32', 'float16', 'bfloat16'])
    
    # Model configs
    parser.add_argument('--z-dim', default=8192, type=int)
    parser.add_argument('--tau', default=20, type=int)
    parser.add_argument('--w', default=0.5, type=float) # disabled for now
    # use gaussian or laplacian to constrain the noise to be independent from each other
    # notice that the loss function is slightly different regarding the two different modes
    parser.add_argument('--noise-mode', default='lap', type=str, choices=['gau', 'lap'])
    parser.add_argument('--mse-Zt', default=False, action='store_true', help='enable to use mse on Zt when training B and M')
    parser.add_argument('--l-ind', default=0.1, type=float)
    parser.add_argument('--l-spB', default=0.1, type=float)
    parser.add_argument('--l-spM', default=0.1, type=float)
    parser.add_argument('--l-spZ', default=0.1, type=float)
    parser.add_argument('--normalize-activations', default=False, action='store_true')
    parser.add_argument('--topk', default=25, type=int, help='the topk spartisy; if it is set to 0 (by default), then only use l1 sparsity', choices=[0, 25, 50, 100])
    parser.add_argument('--results-dir', required=True, help='The dir to saving results, including logs, losses (plots), and ckps')
    
    # Hugging face login
    parser.add_argument('--hgf-token', default='', type=str, help='The hugging face login tokens, for avoiding request limited when the number of total tokens is big or many conccurnet tasks.')
    
    # General LLM activations datasets configs
    parser.add_argument('--total-tokens', default='1.0M', type=str, help='(Unit M) the total number of tokens to train on')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-7B')
    parser.add_argument('--layer', type=int, default=16)
    parser.add_argument('--out-batch-ratio', type=float, default=0.1, help='Specifies the fraction of the activation buffer to use as the output batch size. The out_batch_size is computed as: int(out_batch_ratio * buffer_size)')
    parser.add_argument('--buffer-size', default='0.1M', type=str, help='(Unit M) the size of ActivationBuffer; note that the actual batch_size during training is (buffer_size - tau)')
    parser.add_argument('--text', default='monology/pile-uncopyrighted', type=str, help='the name of the text ot use, by default is the unspecifed corpus monology/pile-uncopyrighted')    
    parser.add_argument('--data-source', default='hf', type=str, choices=['hf', 'local', 'local-cycle'])
    parser.add_argument('--local-text-path', default='mini_pile.jsonl', type=str)
    parser.add_argument('--refresh-batch-size', default=2, type=int)
    
    # Contexual LLM datasets configs (synthetic real-world)
    parser.add_argument('--context', type=str, default='unspecified', help='the context of the tokens', choices=['unspecified'])

    # Synthetic datasets configs
    # TODO

    # Read args
    args = parser.parse_args()

    assert args.total_tokens[-1].lower() == 'm'
    assert args.buffer_size[-1].lower() == 'm'
    args.total_tokens = str(float(args.total_tokens[:-1])) + 'M'
    args.buffer_size = str(float(args.buffer_size[:-1])) + 'M'

    # Pick up params to store the ckps and log
    config_name_keys = ['z_dim', 'total_tokens', 'noise_mode', 'topk', 'buffer_size', 'layer', 'tau', 'mse_Zt', 'l_ind', 'l_spB', 'l_spB', 'l_spZ', 'normalize_activations', 'out_batch_ratio', 'lr', 'wd', 'seed'] 

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    config_str = f"L{args.layer}_Z{args.z_dim}_tau{args.tau}_topk{args.topk}"
    unique_run_name = f"{timestamp}_{config_str}"
    args.results_dir = os.path.join(args.results_dir, unique_run_name)
    os.makedirs(args.results_dir, exist_ok=True)

    # Staring logging
    logger, log_path, loss_path, model_save_dir = setup_logging(args, config_name_keys=config_name_keys)
    args.log_path = log_path
    args.loss_path = loss_path
    args.model_save_dir = model_save_dir

    args.total_tokens = int(float(args.total_tokens[:-1]) * 1_000_000)
    args.buffer_size = int(float(args.buffer_size[:-1]) * 1_000_000)

    # Hugging face login
    if args.hgf_token != '':
        hugging_face_login(token=args.hgf_token)

    # Set device and random seed
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS was requested but is not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    buffer_device = torch.device(args.buffer_device)
    model_dtype = get_torch_dtype(args.model_dtype)
    logger.info('Using device for training: %s', device)
    logger.info('Using device for activation buffer: %s', buffer_device)
    logger.info('Using device for LLM activation extraction: %s', args.llm_device)
    logger.info('Using model dtype: %s', model_dtype)
    logger.info('Using data source: %s', args.data_source)
    
    logger.info('Setting global random seed: {:d}'.format(args.seed))
    set_seed(args.seed)

    # Init data loader
    logger.info('Initiating data loader...')
    activation_buffer, act_dim = get_acts_buffer(model_name=args.model_name,
                                                text=args.text,
                                                layer=args.layer,
                                                buffer_size=args.buffer_size,
                                                out_batch_ratio=args.out_batch_ratio,
                                                device=buffer_device,
                                                dtype=torch.float16,
                                                refresh_batch_size=args.refresh_batch_size,
                                                data_source=args.data_source,
                                                local_text_path=args.local_text_path,
                                                llm_device=args.llm_device)
    logger.info('Total n_tokens to train on: {:d}M, n_refresh: {:d} of the activation buffer with each buffer of {:.2f}M tokens'.format(
        (args.total_tokens // 1_000_000), 
        (args.total_tokens // args.buffer_size),
        (args.buffer_size / 1_000_000)))
    logger.info('Please check the esimated mem usage below to adjust your params carefully:')
    logger.info('Esimated CPU usage for activation buffer: {:.2f}GB, estimated GPU usage of one batch when a full buffer is processed: {:.2f}GB'.format(
        (args.buffer_size * 4 * act_dim / 1024 **3),
        ((int(args.out_batch_ratio * args.buffer_size) - args.tau) * (args.tau + 1) * 4 * act_dim / (1024 **3))))
    
    # Init LinearIDOL
    logger.info('Initiating LinearIDOL model...')
    model = LinearIDOL(x_dim=act_dim, 
                       z_dim=args.z_dim, 
                       w=args.w,
                       tau=args.tau,
                       noise_mode=args.noise_mode,
                       topk_sparsity=args.topk).to(device=device, dtype=model_dtype)
    
    # Optimization
    memory_estimate = estimate_trainable_memory_gb(model, args.optimizer)
    logger.info(
        'Estimated trainable memory for %s: params=%.2fGB, grads=%.2fGB, optimizer_state=%.2fGB, total=%.2fGB',
        args.optimizer,
        memory_estimate["params"],
        memory_estimate["grads"],
        memory_estimate["optimizer"],
        memory_estimate["total"],
    )
    if device.type == "mps" and args.optimizer == "adam":
        logger.warning(
            "Adam keeps two optimizer-state tensors per parameter. If MPS runs out of memory, "
            "try --optimizer sgd, lower --z-dim/--tau, or use --device cpu."
        )
    optimizer = build_optimizer(args, model)
    
    logger.info('Training...')
    train(model=model, 
          activation_buffer=activation_buffer, 
          logger=logger, 
          optimizer=optimizer, 
          args=args, 
          device=device,
          normalize_activations=args.normalize_activations)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nFATAL: main.py exited with an exception.", file=sys.__stderr__)
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.__stderr__)
        raise
