"""
Download, preprocess and serve the stackoverflow dataset as a DataLoader.
"""

import glob
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
DATA_PATH = "../RedPajama-Data-1T/data/stackexchange.jsonl"

def pretokenize(filename):
    enc = Tokenizer(None)

    SIZE = 0
    TOTAL_SIZE = 0
    # if we're using Llama 2, just save the tokenized file in the same dir
    tokenized_filename = filename.replace(".jsonl", ".bin")

    with open(filename, encoding="utf-8") as in_file:
        with open(tokenized_filename, "wb") as out_file:
            for row in tqdm(in_file):
                data = json.loads(row)
                text = data["text"]
                text = text.strip()  # get rid of leading/trailing whitespace
                tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
                # convert to uint16 nparray
                tokens = np.array(tokens, dtype=np.uint16)
                out_file.write(tokens.tobytes())
                TOTAL_SIZE += len(tokens)
                SIZE += 1
                del tokens

    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = TOTAL_SIZE / SIZE
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")

class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")

        # the .bin files are right along the .json files
        filename = DATA_PATH.replace(".jsonl", ".bin")

        while True:
            # open the dataset for reading but keep it on disk with memmap
            m = np.memmap(filename, dtype=np.uint16, mode="r")
            num_batches = len(m) // self.max_seq_len
            num_batches -= 1  # drop the last partial batch
            assert num_batches > 0, "this shard is way too small? investigate."
            ixs = list(range(num_batches))
            rng.shuffle(ixs)
            # train/test split, use 1% of the data for validation
            ixs = ixs[int(num_batches * 0.01):] if self.split == "train" else ixs[:int(num_batches * 0.01)]
            for ix in ixs:
                start = ix * self.max_seq_len
                end = start + self.max_seq_len + 1
                # calling .astype will copy the data into a new numpy array, now in RAM
                chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y

# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, f"tok{vocab_size}.model")

class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset
def stats():
    filename = DATA_PATH.replace(".jsonl", ".bin")
    m = np.memmap(filename, dtype=np.uint16, mode="r")
    max_seq_len = 2048
    num_batches = len(m) // max_seq_len
    num_batches -= 1  # drop the last partial batch
    assert num_batches > 0, "this shard is way too small? investigate."
    ixs = list(range(num_batches))

    # train/test split
    for ix in ixs:
        start = ix * max_seq_len
        end = start + max_seq_len + 1
        # calling .astype will copy the data into a new numpy array, now in RAM
        chunk = torch.from_numpy((m[start:end]).astype(np.int16))
        print(f"Number of BOS tokens in a chunk: {torch.sum(chunk==1)}")

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """

    # depending on the stage call the appropriate function
    pretokenize(DATA_PATH)
    #stats()
