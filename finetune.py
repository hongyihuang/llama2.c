"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m finetune.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 finetune.py

nohup torchrun --standalone --nproc_per_node=2 finetune.py > GPU_2xcluster_4.out &

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 finetune.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 finetune.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from tqdm import tqdm

os.environ["NCCL_P2P_DISABLE"] = "1"

import torch
from torch.distributed import destroy_process_group, init_process_group, all_reduce, ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from stackoverflow import Task
from export import model_export

from tokenizer import Tokenizer
from tinystories import get_tokenizer_model_path

# -----------------------------------------------------------------------------
# I/O
out_dir = "SparseLlama7B_1"
eval_interval = 100
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
batch_size = 2 # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 2048
vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens
# adamw optimizer
gradient_accumulation_steps = 8*6  # used to simulate larger batch sizes
learning_rate = 1.5e-5  # max learning rate
max_iters = 70000  # total number of training iterations, approx 28k for 28GB of data
# 28*1024^3/2/2048 = 960,000 batches 7,340,000 batches
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# Clipping gradient would result in problematic behavior with fp16
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 10  # how many steps to warm up for
# system
distributed = False
device = "cuda:2"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
enable_flash = True  # enable flash kernels for faster fp16 training, but TITAX RTX (sm75) doesn't support kernel, requiring sm80+
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
    rank_map = [0, 1, 2, 3, 5, 7] #[1, 3, 4, 5, 6, 7] #[1, 3] 
    device = f"cuda:{rank_map[5-ddp_local_rank]}"
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
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
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

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)

    return int8val, scale #, maxerr

def dequantize_q80(w, scale, group_size, shape):
    """
    takes a Q8_0 tensor and returns the float32 version
    """
    w = w.view(-1, group_size)

    # dequantize by rescaling
    fpval = (w.type(ptdtype) * scale[:,None]).view(-1)
    return fpval.reshape(shape)

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, gpu_num, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())

        self.A = torch.nn.Parameter(torch.randn(in_dim, rank, dtype = ptdtype, device=device) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim, dtype = ptdtype, device=device))
        self.A.requires_grad_(True)
        self.B.requires_grad_(True)
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        x = self.alpha/self.rank * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, gpu_num, rank, alpha):
        super().__init__()
        self.linear_w, self.linear_s = quantize_q80(linear.weight, 64)
        self.linear_w.to(device)
        self.linear_s.to(device)
        self.linear_shape = linear.weight.shape
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, gpu_num, rank, alpha
        )
        del linear

    def forward(self, x):
        return self.lora(x) + F.linear(x, dequantize_q80(self.linear_w, self.linear_s, 64, self.linear_shape))

for param in model.parameters():
    param.requires_grad = False

print("Before: ", model)

# default hyperparameter choices
lora_r = 16
lora_alpha = lora_r
lora_dropout = 0.00

assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_r)

gpu_num = None
for i, layer in enumerate(model.layers):
    if distributed:
        gpu_num = rank_map[int(math.floor((i/ModelArgs.n_layers) * ddp_world_size))]

    print("Layer: ", i, " GPU: ", gpu_num, "N_Layers", ModelArgs.n_layers)
    layer.attention.wq = LinearWithLoRA(layer.attention.wq, gpu_num, lora_r, lora_alpha)
    layer.attention.wk = LinearWithLoRA(layer.attention.wk, gpu_num, lora_r, lora_alpha)
    layer.attention.wv = LinearWithLoRA(layer.attention.wv, gpu_num, lora_r, lora_alpha)
    layer.attention.wo = LinearWithLoRA(layer.attention.wo, gpu_num, lora_r, lora_alpha)
    layer.feed_forward.w1 = LinearWithLoRA(layer.feed_forward.w1, gpu_num, lora_r*4, lora_alpha*4)
    layer.feed_forward.w2 = LinearWithLoRA(layer.feed_forward.w2, gpu_num, lora_r*4, lora_alpha*4)
    layer.feed_forward.w3 = LinearWithLoRA(layer.feed_forward.w3, gpu_num, lora_r*4, lora_alpha*4)

model.output = LinearWithLoRA(model.output, gpu_num, lora_r, lora_alpha)

torch.cuda.empty_cache()

print("After: ", model)

# Steps
# 1. Load model from llama7b after mapping to our implementation
# 2. Convert the model to lora and only enable gradient for for low rank adapter
# 3. Train the model with lora

print(f"Resuming training from {out_dir}")
# resume training from a checkpoint.
ckpt_path = os.path.join(out_dir, "ckpt.pt")

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

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
    print("wrapping model into DDP...")
    model = DDP(model)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    ddp_loss = torch.zeros(2).to(device=device)  # keep on GPU
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx, torch.backends.cuda.sdp_kernel(enable_flash=enable_flash):
                logits = model(X, Y)
                loss = raw_model.last_loss
                if (not torch.isnan(loss)): # occasionally nan happens, don't let it propagate
                    ddp_loss[1] += 1
                    ddp_loss[0] += loss
        if ddp:
            all_reduce(ddp_loss, op=ReduceOp.SUM)
        out[split] = ddp_loss[0]/ddp_loss[1] 
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it+1) / warmup_iters
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
if master_process:
    sample()
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
            sample()
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
    ddp_loss = torch.zeros(2).to(device=device)  # keep on GPU
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx, torch.backends.cuda.sdp_kernel(enable_flash=enable_flash):
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        if (not torch.isnan(loss)): 
            ddp_loss[0] += loss
            ddp_loss[1] += 1

    #clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # all-reduce the loss if training with DDP
    if ddp:
        all_reduce(ddp_loss, op=ReduceOp.SUM)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = (ddp_loss[0]/ddp_loss[1]).item() * gradient_accumulation_steps
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
