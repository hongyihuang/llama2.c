"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m finetune_fsdp.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 finetune_fsdp.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 finetune_fsdp.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 finetune_fsdp.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from torch.distributed import destroy_process_group, init_process_group, all_reduce, ReduceOp
import torch.nn.functional as F

from stackoverflow import Task
from export import model_export

from tokenizer import Tokenizer
from tinystories import get_tokenizer_model_path

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import BackwardPrefetch

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

# -----------------------------------------------------------------------------
# I/O
out_dir = "SparseLlama7B_1"
eval_interval = 500
log_interval = 1
eval_iters = 10
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
# init_from = "scratch"  # 'scratch' or 'resume'
model_dir = 'ReluLLaMA-7B/'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
batch_size = 1  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 2048
vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens
# adamw optimizer
gradient_accumulation_steps = 32*6  # used to simulate larger batch sizes
learning_rate = 5e-9  # max learning rate
max_iters = 5000  # total number of training iterations, approx 10k for 40GB of data
# 36*1024^3/2/2048 = 960,000 batches
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# Clipping gradient would result in problematic behavior with fp16
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 10  # how many steps to warm up for
# system
distributed = False
device = "cuda:2"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "float32"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
sparse = False
enable_flash = False  # enable flash kernels for faster fp16 training, but TITAX RTX (sm75) doesn't support kernel, requiring sm80+
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
print("Torch Verison: ", torch.__version__)
print(torch.__config__.show().replace("\n", "\n\t"))
# -----------------------------------------------------------------------------

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# validating checks
assert vocab_source in ["llama2", "custom"]
assert vocab_source == "custom" or vocab_size == 32000, "The vocab from Meta has 32K tokens"

# import lora_model that has meta device which does not materialize the weights in RAM
from lora_model import Transformer, ModelArgs

# model
model_args = ModelArgs

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
print("Is this a ddp run?", int(os.environ.get("RANK", -1)))
if ddp:
    print("RANK, LOCAL, WORLD", int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]))

    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    rank_map = [1, 3, 4, 5, 6, 7] #[1, 3] 
    device = f"cuda:{rank_map[ddp_local_rank]}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert (gradient_accumulation_steps % ddp_world_size) == 0
    gradient_accumulation_steps //= ddp_world_size
    out_dir = f"{out_dir}_Cluster"
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")
    os.makedirs(out_dir, exist_ok=True)


torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    #if device_type == "cpu"
    #else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers=0,
)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

'''
Name Map:

original        | new
================|================
attention_norm  | input_layernorm
ffn_norm        | post_attention_layernorm
feed_forward.w1 | mlp.gate_proj
feed_forward.w2 | mlp.down_proj
feed_forward.w3 | mlp.up_proj
attention.wq    | self_attn.q_proj
attention.wk    | self_attn.k_proj
attention.wv    | self_attn.v_proj
attention.wo    | self_attn.o_proj
norm            | norm
output          | lm_head
tok_embeddings  | embed_tokens
'''

nameMap = {"attention_norm": "input_layernorm",
           "ffn_norm": "post_attention_layernorm",
           "feed_forward": "mlp",
           "w1": "gate_proj",
           "w2": "down_proj",
           "w3": "up_proj",
           "attention": "self_attn",
           "wq": "q_proj",
           "wk": "k_proj",
           "wv": "v_proj",
           "wo": "o_proj",
           "norm": "norm",
           "output": "lm_head",
           "tok_embeddings": "embed_tokens"}

nameMap_reverse = {v: k for k, v in nameMap.items()}

# init from a model saved in a specific directory
def remap_names(file):
    model_dict = torch.load(file, map_location='cpu')
    unwanted_prefix = 'model.'

    for k,v in list(model_dict.items()):
        if k.startswith(unwanted_prefix):
            model_dict[k[len(unwanted_prefix):]] = model_dict.pop(k)

    for k,v in list(model_dict.items()):
        split_keys = k.split(".")
        for i in range(len(split_keys)):
            if split_keys[i] in nameMap_reverse:
                split_keys[i] = nameMap_reverse[split_keys[i]]
        model_dict[".".join(split_keys)] = model_dict.pop(k)
    
    for k,v in list(model_dict.items()):
        model_dict[k] = v.to(torch.float16)
    
    return model_dict

dir = os.listdir(model_dir)
# access all {x}of6_fp16.bin files
print(dir)
model_dict = {}
for file in dir:
    if file.startswith("pytorch_model-"):
        print("Loading file: ", file)
        model_dict.update(remap_names(model_dir + file))

# for k,v in list(model_dict.items()):
#    print(k, v.shape if isinstance(v, torch.Tensor) else v)

# 'data/ReluLLaMA7B/6of6_fp16.bin'
    
# see trick of using meta device and assign a mmap tensor https://huggingface.co/blog/accelerate-large-models
# materialize the transformer model
model = Transformer(ModelArgs) #default is llama7B
model.load_state_dict(model_dict, strict=False, assign=True)

model.eval()
model.to(device)

for param in model.parameters():
    param.requires_grad = True

# Steps
# 1. Load model from llama7b after mapping to our implementation
# 2. Convert the model to lora and only enable gradient for for low rank adapter
# 3. Train the model with lora

print(f"Resuming training from {out_dir}")
# resume training from a checkpoint.
ckpt_path = os.path.join(out_dir, "ckpt.pt")

# initialize a GradScaler. If enabled=False scaler is a no-op
#scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    print("wrapping model into FSDP...")
    #model = DDP(model)
    my_auto_wrap_policy = partial(
        size_based_auto_wrap_policy, min_num_params=10000
    )
    torch.cuda.set_device(device)

    model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy,
                sharding_strategy = ShardingStrategy.FULL_SHARD,
                #backward_prefetch = BackwardPrefetch.BACKWARD_POST
                )

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    ddp_loss = torch.zeros(2).to(device=device)  # keep on GPU
    print("Estimating loss on device: ", device)
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx, torch.backends.cuda.sdp_kernel(enable_flash=enable_flash):
                logits = model(X, Y)
                loss = raw_model.last_loss
            if (not torch.isnan(loss)): # occasionally nan happens, don't let it propagate
                ddp_loss[0] += loss
                ddp_loss[1] += 1
        
        all_reduce(ddp_loss, op=ReduceOp.SUM)

        out[split] = ddp_loss[0]/ddp_loss[1] 
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def sample(prompt="Once upon a time, ", max_new_tokens=128, temperature=0.9, top_k=32, num_samples=1):
    vocab_size = ModelArgs.vocab_size

    # let's try to find the tokenizer model automatically. bit gross here...
    query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
    tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
    enc = Tokenizer(tokenizer_model=tokenizer_model)

    prompt_ids = enc.encode(prompt, bos=True, eos=False)
    idx = (torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    model.eval()
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                for _ in range(max_new_tokens):
                    # if the sequence context is growing too long we must crop it at block_size
                    idx_cond = idx if idx.size(1) <= ModelArgs.max_seq_len else idx[:, -ModelArgs.params.max_seq_len:]
                    # forward the model to get the logits for the index in the sequence
                    logits = model(idx_cond) # dummy Y, doesn't matter
                    logits = logits[:, -1, :] # crop to just the final time step
                    if temperature == 0.0:
                        # "sample" the single most likely index
                        _, idx_next = torch.topk(logits, k=1, dim=-1)
                    else:
                        # pluck the logits at the final step and scale by desired temperature
                        logits = logits / temperature
                        # optionally crop the logits to only the top k options
                        if top_k is not None:
                            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = -float('Inf')
                        # apply softmax to convert logits to (normalized) probabilities
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                    # append sampled index to the running sequence and continue
                    y = torch.cat((idx, idx_next), dim=1)
                print(enc.decode(y[0].tolist()))
                print(y)
                print('---------------')
    model.train()

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
print("Starting training loop:")
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

#if master_process:
#    print("Attempting to sample: ")
#    sample()

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            #sample()
            if wandb_log:
                try:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "tokens": iter_num * tokens_per_iter,
                            "loss/train": losses["train"],
                            "loss/val": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        }, step = iter_num
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                    model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    print("Starting forward backward update")
    ddp_loss = torch.zeros(3).to(device=device)  # keep on GPU
    for micro_step in range(gradient_accumulation_steps):
        # don't delay syncing gradients to avoid running out of memory
        print("Forward pass", device)
        with ctx, torch.backends.cuda.sdp_kernel(enable_flash=enable_flash):
            logits = model(X, Y)
            loss = raw_model.last_loss
            print("Loss: ", device)
            ddp_loss[0] += loss / gradient_accumulation_steps
            ddp_loss[1] += 1
            ddp_loss[2] = loss / gradient_accumulation_steps
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # backward pass, with gradient scaling if training in fp16
        print("Backward pass", device)
        #scaler.scale(ddp_loss[2]).backward()
        loss.backward()
        
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
    
    print("All reduce", device)
    all_reduce(ddp_loss, op=ReduceOp.SUM)
    loss = ddp_loss[0] / ddp_loss[1]

    print("Clipping gradient and stepping optimizer")
    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    optimizer.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item()
        
        up = 0
        attn = 0
        ffn = 0
        coverage = 0
        count = 0
        n_layers = 0

        for layer in raw_model.layers:
            up += layer.feed_forward.stats.item()
            attn += layer.attn_stats.item()
            ffn += layer.ffn_stats.item()
            coverage += layer.feed_forward.coverage.item()
            n_layers += 1
        
        layer.feed_forward.stats = 0
        layer.feed_forward.count = 0
        layer.attn_stats.stats = 0
        layer.attn_stats.count = 0
        layer.ffn_stats.stats = 0
        layer.ffn_stats.count = 0
        up /= n_layers
        attn /= n_layers
        ffn /= n_layers
        coverage /= n_layers
        
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}% | up {up:.2f}, {coverage:.2f} | attn {attn:.2f} | ffn {ffn:.2f}"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
