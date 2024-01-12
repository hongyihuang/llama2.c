import torch
import numpy as np
from spmodel import ModelArgs, Transformer
from export import model_export


# -----------------------------------------------------------------------------
checkpoint = 'out2Msp/ckpt.pt'

checkpoint_dict = torch.load(checkpoint, map_location='cpu')
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = Transformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to('cpu')

model_export(model, 'out2Msp/model_qint80.bin', version=2)
model_export(model, 'out2Msp/model_mcu.bin', version=3)