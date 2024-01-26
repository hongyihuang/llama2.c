import torch
import os

device = 'cpu'

def cast_tensors(dir, new_dir, file):
    model_dict = torch.load(dir + file, map_location=device, mmap=True)

    for k,v in list(model_dict.items()):
        model_dict[k] = v.to(torch.bfloat16)

    torch.save(model_dict, new_dir + file)
    
dir = os.listdir('data/ReluLLaMA7B')
# access all {x}of6_fp16.bin files
print(dir)
for file in dir:
    print("Casting file: ", file)
    cast_tensors('data/ReluLLaMA7B/', 'data/ReluLLaMA7B_bf16/', file)