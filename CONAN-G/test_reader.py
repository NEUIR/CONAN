# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
import os
from tqdm import tqdm
import re
import pickle
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model


def eval_bleu(opt,model,tokenizer, eval_examples,eval_dataset, eval_dataloader):

    logger.info("***** Running evaluation on test set *****")

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0

    for batch in tqdm(eval_dataloader):
        (idx, _, _, context_ids, context_mask),bach_example = batch
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


def test_eval_bleu(opt,model,tokenizer,  eval_examples, eval_dataset,eval_dataloader):

    output_fn = os.path.join(
        "/data1/wanghanbin/CodeT5-sourcecode/sh/saved_models/generate/python/codet5_base_all_lr5_bs48_src128_trg256_pat3_e30/prediction/test_e1.output")
    gold_fn = os.path.join(
        "/data1/wanghanbin/CodeT5-sourcecode/sh/saved_models/generate/python/codet5_base_all_lr5_bs48_src128_trg256_pat3_e30/prediction/test_e1.gold")
    src_fn = os.path.join(
        "/data1/wanghanbin/CodeT5-sourcecode/sh/saved_models/generate/python/codet5_base_all_lr5_bs48_src128_trg256_pat3_e30/prediction/test_e1.src")

    dev_accs, predictions = [], []

    pred_nls = []
    with open(output_fn, 'r') as f:
        for line in f:
            pred_nls.append(line.strip())



    #将source target predictions 写入文件保存下来
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
        pass
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

def get_data(path):

    with open(path, 'rb') as f:
        gen_python_csn_test_100 = pickle.load(f)
    return gen_python_csn_test_100


def test_gen_python(gen_python_csn_test_100):

    output_fn_python = os.path.join("/data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/",
                                    "test_{}.output_{}".format("retrieve", "python"))
    gold_fn_python = os.path.join("/data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/",
                                  "test_{}.gold_{}".format("retrieve", "python"))
    dev_accs = []

    with open(output_fn_python, 'w') as f, open(gold_fn_python, 'w') as f1:
        for i, data in enumerate(gen_python_csn_test_100):
            dev_accs.append(data["ctxs"][0]["text"].strip() == data["target"].strip())
            data["ctxs"][0]["text"] = re.sub("[\n\r\t ]+", " ", data["ctxs"][0]["text"])
            data["target"] = re.sub("[\n\r\t ]+", " ", data["target"])
            f.write(data["ctxs"][0]["text"] + '\n')
            f1.write(data["target"] + '\n')

    bleu = round(_bleu(gold_fn_python, output_fn_python), 2)
    codebleu = calc_code_bleu.get_codebleu(gold_fn_python, output_fn_python, "python")

    result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
    result['codebleu'] = codebleu * 100

    print("***** Eval Python results *****")
    print('em:',np.mean(dev_accs) * 100)
    print("bleu:",bleu)
    print("codebleu:",codebleu * 100)


def test_gen_java(gen_java_csn_test_100):
    output_fn_java = os.path.join("/data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/",
                                  "test_{}.output_{}".format("retrieve", "java"))
    gold_fn_java = os.path.join("/data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/",
                                "test_{}.gold_{}".format("retrieve", "java"))
    dev_accs = []
    with open(output_fn_java, 'w') as f, open(gold_fn_java, 'w') as f1:
        for i, data in enumerate(gen_java_csn_test_100):
            dev_accs.append(data["ctxs"][0]["text"].strip() == data["target"].strip())
            data["ctxs"][0]["text"] = re.sub("[\n\r\t ]+", " ", data["ctxs"][0]["text"])
            data["target"] = re.sub("[\n\r\t ]+", " ", data["target"])
            f.write(data["ctxs"][0]["text"] + '\n')
            f1.write(data["target"] + '\n')

    bleu = round(_bleu(gold_fn_java, output_fn_java), 2)
    codebleu = calc_code_bleu.get_codebleu(gold_fn_java, output_fn_java, "java")

    result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
    result['codebleu'] = codebleu * 100

    print("***** Eval Java results *****")
    print('em:', np.mean(dev_accs) * 100)
    print("bleu:", bleu)
    print("codebleu:", codebleu * 100)

if __name__ == "__main__":

    # 测试检索结果
    # gen_python_csn_test_100 = get_data(
    #     "/data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_python_csn_test_100.pkl")
    # gen_java_csn_test_100 = get_data(
    #     "/data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_java_csn_test_100.pkl")
    #
    # test_gen_python(gen_python_csn_test_100)
    # test_gen_java(gen_java_csn_test_100)

    # 加载参数
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()

    # 初始化
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()



    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()

    if opt.is_distributed:
        torch.distributed.barrier()

    dir_path.mkdir(parents=True, exist_ok=True)

    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')

    if not directory_exists and opt.is_main:
        options.print_options(opt)

    tokenizer = RobertaTokenizer.from_pretrained("/data1/wanghanbin/fidgen/Salesforce/codet5-base_csn_gen_python/")

    collator_function = src.data.Collator(text_maxlength=opt.text_maxlength, tokenizer=tokenizer, answer_maxlength=opt.answer_maxlength)

    # 数据处理流程：examples ---> dataset ---> sampler ---> dataloader
    opt.wrold_size = 1

    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )

    # eval_examples = eval_examples[:100]

    logger.info("   Loaded examples from: {}".format(opt.eval_data))
    logger.info("   Loaded {} eval examples.".format(len(eval_examples)))
    logger.info("   Example0")
    logger.info(f"   id:{eval_examples[0]['id']}")
    logger.info(f"   question:{eval_examples[0]['question']}")
    logger.info(f"   target:{eval_examples[0]['target']}")
    logger.info(f"   ctx0_text:{eval_examples[0]['ctxs'][0]['text']}")
    logger.info(f"   ctx0_linked:{eval_examples[0]['ctxs'][0]['linked']}")
    logger.info(f"   ctx1_text:{eval_examples[0]['ctxs'][1]['text']}")
    logger.info(f"   ctx1_linked:{eval_examples[0]['ctxs'][1]['linked']}")

    logger.info("   Example1")
    logger.info(f"   id:{eval_examples[1]['id']}")
    logger.info(f"   question:{eval_examples[1]['question']}")
    logger.info(f"   target:{eval_examples[1]['target']}")
    logger.info(f"   ctx0_text:{eval_examples[1]['ctxs'][0]['text']}")
    logger.info(f"   ctx0_linked:{eval_examples[1]['ctxs'][0]['linked']}")
    logger.info(f"   ctx1_text:{eval_examples[1]['ctxs'][1]['text']}")
    logger.info(f"   ctx1_linked:{eval_examples[1]['ctxs'][1]['linked']}")

    logger.info("   Creating eval dataset.(CodeDataset)")
    logger.info("   opt.n_context:{}".format(opt.n_context))
    logger.info("   opt.with_target:{}".format(opt.with_target))
    eval_dataset = src.data.CodeDataset(
        opt,
        eval_examples,
        opt.n_context,
    )

    logger.info("   Creating eval sampler.(SequentialSampler)")
    eval_sampler = SequentialSampler(eval_dataset)

    logger.info("   Creating eval dataloader.(DataLoader)")
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=20, 
        collate_fn=collator_function
    )
    
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("   Start eval")


    # result = test_eval_bleu(opt,model,tokenizer,  eval_examples, eval_dataset,eval_dataloader)
    result = eval_bleu(opt,model,tokenizer,  eval_examples, eval_dataset,eval_dataloader)
    test_bleu, test_em = result['bleu'], result['em']
    test_codebleu = result['codebleu'] if 'codebleu' in result else 0
    result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % ("best-bleu", test_bleu, test_em, test_codebleu)
    logger.info(result_str)

