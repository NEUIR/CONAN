# This file is get vectors for build faiss index
# Author: Hanbin Wang
# Date: 2023/4/15
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script is used to generate the embedding vectors for the given dataset.

import argparse
import logging
import os
import random
import re
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from itertools import cycle
from functools import partial
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

def load_data(file_path=None):
    data_format = file_path.split(".")[-1]
    idx = []
    data = []
    if "jsonl" in data_format:
        logger.info("   jsonl file")
        logger.info(f"   load data from file:{file_path}")
        logger.info("   infer concode code")
        with open(file_path) as f:
            for i,line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                idx.append(i)
                # data.append(' '.join(js['docstring_tokens']))
                data.append(' '.join(js['code_tokens']))
                # data.append(js['code'])
                if i <= 5:
                    print(data)
    elif "pkl"in data_format:
        logger.info("   pkl file")
        logger.info(f"   load data from file:{file_path}")

        with open(file_path, 'rb') as f:
            python_dedupe_definitions_v2 = pickle.load(f)

        for i,line in enumerate(python_dedupe_definitions_v2):
            # line = line.strip()
            # js = json.loads(line)
            idx.append(i)
            data.append(' '.join(line['function_tokens']))
            if i <=5:
                print(data)
    elif "json" in data_format:  # json
        logger.info("   json file")
        logger.info(f"   load data from file:{file_path}")
        with open(file_path) as f:
            for i,line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                idx.append(i)
                data.append(js['code'])
                if i <= 5:
                    print(data)

    else:  # txt  包括deduplicated.summaries.txt
        logger.info("   txt file")
        logger.info(f"   load data from file:{file_path}")
        with open(file_path, "r") as f:
            for i,line in enumerate(f):
                line = line.strip()
                idx.append(i)
                data.append(line)
                if i <= 5:
                    print(data)
    return idx,data


class InferDataset(Dataset):
    def __init__(self, idx,data):
          self.idx = idx
          self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):

        return {
                "idx": self.idx[i],
                "line":self.data[i]
            }

def encode_batch(batch_text, tokenizer, max_length):
    outputs = tokenizer.batch_encode_plus(
        batch_text,
        max_length=max_length,
        pad_to_max_length=True,
        return_tensors='pt',
        truncation=True,

    )
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"].bool()
    return input_ids, attention_mask


class Collator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    # 我们在创建一个PyTorch DataLoader实例时，可以将一个名为collate_fn的参数指定为一个Collator类的实例。
    # 此时，在DataLoader内部调用collate_fn函数时，Python解释器就会自动调用Collator类的__call__方法
    def __call__(self, batch):
        line = [ex["line"] for ex in batch]
        line_ids, line_masks = encode_batch(line, self.tokenizer, self.args.block_size)

        idx = [ex["idx"] for ex in batch]
        return {
            "idx":torch.tensor(idx),
            "line_ids": line_ids,
            "line_masks": line_masks,
        }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)





def inference(args, tokenizer, model, save_name, inferdataset,collator):
    # build dataloader
    sampler = SequentialSampler(inferdataset)
    dataloader = DataLoader(inferdataset, sampler=sampler, batch_size=args.eval_batch_size,drop_last=False,
                            num_workers=4,collate_fn=collator)

    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank % args.gpu_per_node],
                                                          output_device=args.local_rank % args.gpu_per_node,
                                                          find_unused_parameters=True)

    # Eval!
    logger.info("***** Running Inference *****")
    logger.info("  Num examples = %d", len(inferdataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    steps = 0
    n_vec = max(0, args.num_vec)
    saved = {}

    for batch in dataloader:
        with torch.no_grad():
            idx = batch["idx"].to(args.device)
            line_inputs = batch["line_ids"].to(args.device)
            line_masks = batch["line_masks"].to(args.device)

            decoder_input_ids = torch.zeros((line_inputs.shape[0], 1), dtype=torch.long).to(args.device)
            outputs = model(
                input_ids=line_inputs,
                attention_mask=line_masks,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,  # 其中的 output_hidden_states=True 表示要保存所有层的隐藏状态，以便后续使用。
            )
            # 获取解码器的最后一层的隐藏状态
            hidden = outputs.decoder_hidden_states[-1]
            vec = hidden[:, 0, :]
            vec = vec.detach().to("cpu").numpy()
            idxs = idx.cpu().numpy()
        for i in range(vec.shape[0]):
            saved[idxs[i]] = vec[i]
        steps += 1
        if steps % args.logging_steps == 0:
            logger.info(f"Inferenced {steps} steps")

    if args.local_rank != -1:
        pickle.dump(saved, open(save_name + f"_{args.local_rank}.pkl", "wb"))
    else:
        pickle.dump(saved, open(save_name + ".pkl", "wb"))


def merge(args, num, save_name):
    saved = {}
    for i in range(num):
        saved.update(pickle.load(open(save_name + f"_{i}.pkl", "rb")))
        os.remove(save_name + f"_{i}.pkl")
    pickle.dump(saved, open(save_name + ".pkl", "wb"))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--save_name", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--lang", default=None, type=str, required=True,
                        help="Language of the dataset.")
    parser.add_argument("--pretrained_dir", default=None, type=str,
                        help="The directory where the trained model and tokenizer are saved.")

    parser.add_argument("--cut_ratio", type=float, default=0.5,
                        help="Ratio of replaced variables")
    parser.add_argument('--num_vec', type=int, default=-1,
                        help="number of vectors")

    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--node_index", type=int, default=0,
                        help="node index if multi-node running")
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="num of gpus per node")

    args = parser.parse_args()

    logger.warning(
        "local_rank: %d, node_index: %d, gpu_per_node: %d" % (args.local_rank, args.node_index, args.gpu_per_node))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device

    world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, world size: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), world_size)

    # Set seed
    set_seed(args)

    args.start_epoch = 0
    args.start_step = 0

    # 加载tokenizer和model
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.pretrained_dir)


    collator = Collator(tokenizer, args)
    idx,inferdata =load_data(args.data_path)
    inferdataset = InferDataset(idx,inferdata)
    inference(args, tokenizer, model, args.save_name,inferdataset,collator)
    logger.info(f"device {args.local_rank} finished")

    if args.local_rank != -1:
        torch.distributed.barrier()
        if args.local_rank in [-1, 0]:
            import time
            time.sleep(10)
            merge(args, world_size, save_name=args.save_name)


if __name__ == "__main__":
    main()


