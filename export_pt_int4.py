import os
import torch
print(torch.__version__)
in_dir = os.path.abspath("data/ReluLLaMA7B/1of6_fp16.bin")
out_dir = os.path.abspath("data/relullama7B.pt")

in_dict = torch.load(in_dir, map_location="cpu", mmap=True)
out_dict = {}

for k, v in in_dict.items():
    print(k, v.size(), v.dtype, v.numel() % 64 == 0)
# xxd -i image.jpg > image.h
# https://github.com/ucb-bar/Baremetal-llama/tree/main
'''
assert w.numel() % group_size == 0
ori_shape = w.shape
w = w.reshape(-1, group_size)
# find the max in each group
wmax = torch.abs(w).max(dim=1).values

# calculate the scaling factor such that float = quant * scale
scale = wmax / 127.0
# scale into range [-127, 127]
quant = w / scale[:,None]
# round to nearest integer
int8val = torch.round(quant).to(torch.int8)
# dequantize by rescaling
fp32val = (int8val.float() * scale[:,None]).view(-1)
fp32valr = fp32val.reshape(-1, group_size)
# calculate the max error in each group
err = torch.abs(fp32valr - w).max(dim=1).values
# find the max error across all groups
maxerr = err.max().item()
return int8val, scale, maxerr
'''

a = torch.Tensor([-8, 7, 1, 4, 2, 3, 5, 6, 0]).to(torch.int8)
b = torch.flip(a, [0])
print(a, b)

left = a<<4
right = torch.bitwise_and(b, 0x0f)
print(left, right)

merged = torch.bitwise_or(left, right)
print(merged)

post_a = merged >> 4
post_b = (merged << 4) >> 4

print(post_a, post_b)