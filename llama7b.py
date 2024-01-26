import os
import pickle
import time
from contextlib import nullcontext
import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from numpy import inf
from lora_model import ModelArgs, Transformer
from export import serialize_fp32, serialize_int8, quantize_q80, quantize_int8
import struct
from functools import partial

from tokenizer import Tokenizer
from export import model_export

from tinystories import get_tokenizer_model_path

# -----------------------------------------------------------------------------
checkpoint = 'ckpt.pt' # 'data/ReluLLaMA7B/5of6_fp16.bin'
model_dir = 'data/ReluLLaMA7B_bf16/'
start = "Once upon a time," # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 128 # number of tokens generated in each sample
temperature = 0.9 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 32 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "" # override the tokenizer model path
seed = int(time.time())
device = 'cpu'
dtype = "float16"
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
vocab_source = "llama2"
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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
    model_dict = torch.load(file, map_location=device, mmap=True)
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

    return model_dict
    
dir = os.listdir(model_dir)
# access all {x}of6_fp16.bin files
print(dir)
model_dict = {}
for file in dir:
    print("Loading file: ", file)
    model_dict.update(remap_names(model_dir + file))

# for k,v in list(model_dict.items()):
#    print(k, v.shape if isinstance(v, torch.Tensor) else v)

# 'data/ReluLLaMA7B/6of6_fp16.bin'
    
# see trick of using meta device and assign a mmap tensor https://huggingface.co/blog/accelerate-large-models
model = Transformer(ModelArgs) #default is llama7B
model.load_state_dict(model_dict, strict=False, assign=True)

model.eval()
# DON'T DO THIS UNLESS YOU WANT TO MATERIALIZE WEIGHTS IN RAM
#model.to('cpu')

def sample(prompt=""):
    vocab_size = ModelArgs.vocab_size
    if tokenizer:
        # a specific tokenizer is provided, use it
        tokenizer_model = tokenizer
    else:
        # let's try to find the tokenizer model automatically. bit gross here...
        query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
        tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
    enc = Tokenizer(tokenizer_model=tokenizer_model)

    # encode the beginning of the prompt
    if prompt.startswith('FILE:'):
        with open(prompt[5:], 'r', encoding='utf-8') as f:
            prompt = f.read()
    prompt_ids = enc.encode(prompt, bos=True, eos=False)
    x = (torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, enc=enc)
                print(enc.decode(y[0].tolist()))
                print('---------------')

#sample(start)

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

for param in model.parameters():
    param.requires_grad = False

print("Before: ", model)

# default hyperparameter choices
lora_r = 256
lora_alpha = 16
lora_dropout = 0.05

assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

for layer in model.layers:
    layer.attention.wq = assign_lora(layer.attention.wq)
    layer.attention.wk = assign_lora(layer.attention.wk)
    layer.attention.wv = assign_lora(layer.attention.wv)
    layer.attention.wo = assign_lora(layer.attention.wo)
    layer.feed_forward.w1 = assign_lora(layer.feed_forward.w1)
    layer.feed_forward.w2 = assign_lora(layer.feed_forward.w2)
    layer.feed_forward.w3 = assign_lora(layer.feed_forward.w3)

model.output = assign_lora(model.output)

print("After: ", model)

def version2_export(model, filepath, group_size=64):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    version = 2

    # let's first do some validation for this export type
    while model.params.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        assert w.numel() % group_size == 0, f"weight {i} has numel {w.numel()}, not a multiple of group_size {group_size}"

    # write
    out_file = open(filepath, 'wb')
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack('I', 0x616b3432))
    # 2) write version, which will be int
    out_file.write(struct.pack('i', version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len)
    out_file.write(header)
    # 4) write some other flags
    out_file.write(struct.pack('B', int(shared_classifier)))
    out_file.write(struct.pack('i', group_size)) # group size used for quantization
    pad = 256 - out_file.tell() # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b'\0' * pad)
    # now that the header is done, let's write out the model

    # first let's write out all the params that we are keeping in fp32: the norms
    for layer in model.layers: # attention norms
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers: # MLP norms
        serialize_fp32(out_file, layer.ffn_norm.weight)
    serialize_fp32(out_file, model.norm.weight) # final pre-classifier norm

    # now let's write out all the params that we are quantizing to Q8_0
    # note we skip classifier weights, which are shared with the embedding
    ew = []
    for i, w in enumerate(weights):
        # quantize this weight
        q, s, err = quantize_q80(w, group_size)
        # save the int8 weights to file
        serialize_int8(out_file, q) # save the tensor in int8
        serialize_fp32(out_file, s) # save scale factors
        # logging
        ew.append((err, w.shape))
        print(f"{i+1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")

    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")

    '''
    # test no group quantization at all
    ew = []
    bins = None
    for i, w in enumerate(weights):
        # quantize this weight
        q, s, err = quantize_int8(w)
        # logging
        ew.append((err, w.shape))
        print(f"{i+1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}")
        if (bins is None):
            bins = torch.bincount(q.flatten().long()+128, minlength=256)
        else:
            bins += torch.bincount(q.flatten().long()+128, minlength=256)
    
    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")
    print(bins)
    '''

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")
