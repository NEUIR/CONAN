# This file is .....
# Author: Hanbin Wang
# Date: 2023/4/6
import logging
from transformers import RobertaTokenizer
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def merge_file():
    data = []
    for i in range(0,16):
        datafile = f"/data1/wanghanbin/CodeSearchNet/java/final/jsonl/train/java_train_{i}.jsonl"
        logger.info(f'   processing file:{datafile}')
        with open(datafile) as f:
            for line in f:
                data.append(line.strip())
    logger.info("   merge files done!")

    savefile = "/data1/wanghanbin/CodeSearchNet/java/final/jsonl/train/train.jsonl"
    with open(savefile, "w") as fout:
        for i,line in enumerate(data):
            fout.write(line + "\n")

    logger.info(f"   write to:{savefile}")

def build():
    tokenizer = RobertaTokenizer.from_pretrained(
        "/data1/wanghanbin/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl/checkpoint-35000/")

    savefile = "/data1/wanghanbin/CONAN/dataset_ids_codet5/gen/concode/java/train_ids.jsonl"
    trainfile = "/data1/wanghanbin/CONAN/dataset/gen/concode/java/train.json"
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



if __name__ == '__main__':
    # merge_file()
    build()