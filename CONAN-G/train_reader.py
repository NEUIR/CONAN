# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys
import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer
import random
from tqdm import tqdm
import os
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu


import src.slurm
import src.util
import src.evaluation
import src.data
import src.model


def train(model, optimizer, scheduler, step, train_dataset, eval_dataset,eval_examples, opt, collator, best_dev_bleu, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=1,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while step < opt.total_steps:
        epoch += 1
        for i, batch in enumerate(tqdm(train_dataloader)):
            step += 1
            (idx, labels, label_mask, context_ids, context_mask),batch_example = batch
            if step <= 2:
                write = {"idx": idx, "labels": labels, "label_mask": label_mask, "context_ids": context_ids,"context_mask": context_mask, "batch_example": batch_example}
                # 用pickle dumpy保存pkl文件
                import pickle
                with open("/data1/wanghanbin/fidgen/batch_example.pkl", "wb") as f:
                    pickle.dump(write, f)

            train_loss = model(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                labels=labels.cuda()
            )[0]

            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                # 修改evaluate函数
                # dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                eval_sampler = SequentialSampler(eval_dataset)
                eval_dataloader = DataLoader(eval_dataset,
                                             sampler=eval_sampler,
                                             batch_size=opt.per_gpu_batch_size,
                                             drop_last=False,
                                             num_workers=10,
                                             collate_fn=collator
                                             )
                result = eval_bleu(opt,model,tokenizer, eval_examples,eval_dataset, eval_dataloader)
                test_bleu, test_em = result['bleu'], result['em']
                test_codebleu = result['codebleu'] if 'codebleu' in result else 0

                model.train()
                if opt.is_main:
                    if test_bleu > best_dev_bleu:
                        best_dev_bleu = test_bleu
                        src.util.save(model, optimizer, scheduler, step, best_dev_bleu,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += "evaluation: [%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f |" % ("best-bleu", test_bleu, test_em, test_codebleu)
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", test_bleu, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_bleu,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break


def eval_bleu(opt,model,tokenizer, eval_examples,eval_dataset, eval_dataloader):

    logger.info("***** Running evaluation on test set *****")

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0

    for batch in tqdm(eval_dataloader):
        (idx, _, _, context_ids, context_mask),batch_example = batch
        with torch.no_grad():
            preds = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=256,
                use_cache=True,
                early_stopping=False,
                num_beams=10
                )
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]
    # 获取当前时间
    time_str = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # 保存预测结果
    output_fn = os.path.join("/data1/wanghanbin/fidgen/prediction/", "test_{}.output_{}".format("best-bleu", time_str))
    gold_fn = os.path.join("/data1/wanghanbin/fidgen/prediction/", "test_{}.gold_{}".format("best-bleu", time_str))
    src_fn = os.path.join("/data1/wanghanbin/fidgen/prediction/", "test_{}.src_{}".format("best-bleu", time_str))

    dev_accs, predictions = [], []
    #将source target predictions 写入文件保存下来

    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
        for pred_nl, gold in zip(pred_nls, eval_examples):
            dev_accs.append(pred_nl.strip() == gold['target'].strip())
            f.write(pred_nl.strip() + '\n')
            f1.write(gold['target'].strip() + '\n')
            f2.write(gold['question'].strip() + '\n')

    bleu = round(_bleu(gold_fn, output_fn), 2)
    codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, "java")

    result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
    result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


if __name__ == "__main__":

    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    # 创建文件夹
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    # model_name = 't5-' + opt.model_size
    model_class = src.model.FiDT5


    tokenizer = RobertaTokenizer.from_pretrained(opt.model_path)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    # train_examples = train_examples[30:]
    logger.info("   Loaded train examples from: {}".format(opt.train_data))
    logger.info("   Loaded {} train examples.".format(len(train_examples)))
    logger.info("   Example 0")
    logger.info(f"   id:{train_examples[0]['id']}")
    logger.info(f"   question:{train_examples[0]['question']}")
    logger.info(f"   target:{train_examples[0]['target']}")
    logger.info(f"   ctx0_text:{train_examples[0]['ctxs'][0]['text']}")
    logger.info(f"   ctx0_linked:{train_examples[0]['ctxs'][0]['linked']}")
    logger.info(f"   ctx1_text:{train_examples[0]['ctxs'][1]['text']}")
    logger.info(f"   ctx1_linked:{train_examples[0]['ctxs'][1]['linked']}")

    logger.info("   Example 1")
    logger.info(f"   id:{train_examples[1]['id']}")
    logger.info(f"   question:{train_examples[1]['question']}")
    logger.info(f"   target:{train_examples[1]['target']}")
    logger.info(f"   ctx0_text:{train_examples[1]['ctxs'][0]['text']}")
    logger.info(f"   ctx0_linked:{train_examples[1]['ctxs'][0]['linked']}")
    logger.info(f"   ctx1_text:{train_examples[1]['ctxs'][1]['text']}")
    logger.info(f"   ctx1_linked:{train_examples[1]['ctxs'][1]['linked']}")


    train_dataset = src.data.CodeDataset(opt,train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    logger.info(f"   original eval {len(eval_examples)} examples from dev set")
    # 训练时在valid set上评估太慢，故抽取valid的subset进行评估
    eval_nums = 200
    if opt.eval_part:
        logger.info(f"   eval {eval_nums} examples from dev set")
        eval_examples = random.sample(eval_examples, eval_nums)
    else:
        logger.info(f"   eval {len(eval_examples)} examples from dev set")

    logger.info("   Loaded valid examples from: {}".format(opt.eval_data))
    logger.info("   Loaded {} valid examples.".format(len(eval_examples)))
    logger.info("   Example 0")
    logger.info(f"   id:{eval_examples[0]['id']}")
    logger.info(f"   question:{eval_examples[0]['question']}")
    logger.info(f"   target:{eval_examples[0]['target']}")
    logger.info(f"   ctx0_text:{eval_examples[0]['ctxs'][0]['text']}")
    logger.info(f"   ctx0_linked:{eval_examples[0]['ctxs'][0]['linked']}")
    logger.info(f"   ctx1_text:{eval_examples[0]['ctxs'][1]['text']}")
    logger.info(f"   ctx1_linked:{eval_examples[0]['ctxs'][1]['linked']}")

    logger.info("   Example 1")
    logger.info(f"   id:{eval_examples[1]['id']}")
    logger.info(f"   question:{eval_examples[1]['question']}")
    logger.info(f"   target:{eval_examples[1]['target']}")
    logger.info(f"   ctx0_text:{eval_examples[1]['ctxs'][0]['text']}")
    logger.info(f"   ctx0_linked:{eval_examples[1]['ctxs'][0]['linked']}")
    logger.info(f"   ctx1_text:{eval_examples[1]['ctxs'][1]['text']}")
    logger.info(f"   ctx1_linked:{eval_examples[1]['ctxs'][1]['linked']}")

    eval_dataset = src.data.CodeDataset(opt,eval_examples, opt.n_context)

    if not checkpoint_exists:
        t5 = T5ForConditionalGeneration.from_pretrained(opt.model_path)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_bleu = 0, 0.0
    elif opt.model_path == "none":
        logger.info("   删除checkpoint文件夹")
        exit()
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        logger.info("   删除checkpoint文件夹")
        exit()
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        eval_examples,
        opt,
        collator,
        best_dev_bleu,
        checkpoint_path
    )
