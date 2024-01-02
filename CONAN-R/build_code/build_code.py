# This file is .....
# Author: Hanbin Wang
# Date: 2023/4/6
import logging
from transformers import RobertaTokenizer
import json
from tqdm import tqdm
import argparse
logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def print_options(args,parser):
    message = 'Arguments:\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default_value = parser.get_default(k)
        if v != default_value:
            comment = f'\t(default: {default_value})'
        message += f'{str(k):>30}: {str(v):<40}{comment}\n'

    print(message)

def build_gen_cgcsn(tokenizer,trainfile,savefile):
    logger.info("   start!")
    logger.info("   build_gen_cgcsn")
    with open(savefile,"w") as fout:
        with open(trainfile) as fin:
            for i,line in enumerate(tqdm(fin)):
                line = line.strip()
                line = json.loads(line)
                code = ' '.join(line['code_tokens'])
                nl = ' '.join(line['docstring_tokens'])
                group = {}
                positives = []
                positives.append(code)
                positives = tokenizer(positives, add_special_tokens=False, max_length=240, truncation=True, padding=False)[
                    'input_ids']
                query = tokenizer.encode(nl, add_special_tokens=False, max_length=50, truncation=True)
                group['query'] = query
                group['positives'] = positives
                group['negatives'] = []

                fout.write(json.dumps(group) + '\n')
                if i% 10000 ==0:
                    logger.info(f'   {i} done!')

    logger.info("   done!")

def build_gen_concode(tokenizer,trainfile,savefile):
    logger.info("   start!")
    logger.info("   build_gen_concode")
    with open(savefile,"w") as fout:
        with open(trainfile) as fin:
            for i,line in enumerate(tqdm(fin)):
                line = line.strip()
                line = json.loads(line)
                code = line['code']
                nl = line['nl']
                group = {}
                positives = []
                positives.append(code)
                positives = tokenizer(positives, add_special_tokens=False, max_length=100, truncation=True, padding=False)[
                    'input_ids']
                query = tokenizer.encode(nl, add_special_tokens=False, max_length=256, truncation=True)
                group['query'] = query
                group['positives'] = positives
                group['negatives'] = []

                fout.write(json.dumps(group) + '\n')
                if i% 10000 ==0:
                    logger.info(f'   {i} done!')

    logger.info("   done!")


def build_sum_cscsn(tokenizer,trainfile,savefile):
    logger.info("   start!")
    logger.info("   build_sum_cscsn")
    with open(savefile,"w") as fout:
        with open(trainfile) as fin:
            for i,line in enumerate(tqdm(fin)):
                line = line.strip()
                line = json.loads(line)
                code = ' '.join(line['code_tokens'])
                nl = ' '.join(line['docstring_tokens'])
                group = {}
                positives = []
                positives.append(nl)
                positives = tokenizer(positives, add_special_tokens=False, max_length=50, truncation=True, padding=False)[
                    'input_ids']
                query = tokenizer.encode(code, add_special_tokens=False, max_length=240, truncation=True)
                group['query'] = query
                group['positives'] = positives
                group['negatives'] = []

                fout.write(json.dumps(group) + '\n')
                if i% 10000 ==0:
                    logger.info(f'   {i} done!')

    logger.info("   done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("build data for OpenMatch.")
    parser.add_argument(
        "--task",
        type=str,
        default="gen",
        choices=["gen","concode","sum"],
        help="The tokenizer to use.(path)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        default="",
        help="The tokenizer to use.(path)",
    )
    parser.add_argument(
        "--trainfile",
        type=str,
        required=True,
        default="",
        help="The trainfile to use.(e.g. xxx/train.jsonl)",
    )
    parser.add_argument(
        "--savefile",
        type=str,
        required=True,
        default="",
        help="The path to save the ids file.(e.g. xxx/train_ids.jsonl)",
    )
    args = parser.parse_args()
    print_options(args, parser)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    trainfile = args.trainfile
    savefile = args.savefile

    if args.task == "gen":
        build_gen_cgcsn(tokenizer,trainfile,savefile)
    elif args.task == "concode":
        build_gen_concode(tokenizer,trainfile,savefile)
    elif args.task == "sum":
        build_sum_cscsn(tokenizer,trainfile,savefile)

