# CONAN-R

CONAN-R使用[**SANTA**](https://github.com/OpenMatch/SANTA)初始化，然后在每个生成任务的训练集上进行微调。

### 1.Requirements

(1)创建虚拟环境

```bash
conda create -n conanr python=3.8
```

(2) Install the following packages using Pip or Conda under this environment:

```
transformers==4.22.2    # 其他版本可能会出现错误。
datasets
Pillow
torch
tensorboard
```

(3) install openmatch. To download OpenMatch as a library and obtain openmatch-thunlp-0.0.1.

```
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install .
```



### 2.获取SANTA Checkpoint

The checkpoint of the pretrained SANTA model on `Python` data is [here](https://huggingface.co/OpenMatch/santa-code-python-adv). 



### 3.Finetune

1. **download dataset**

   你可以从Google Drive上下载实验所需要的数据。

   ```
   数据目录结构
   ```

   

2. **build code**

   将数据集转化成OpenMatch所需要的格式。cd到CONAN-R/build_code目录，运行build_code.sh。build_code.sh文件内容如下：

   ```bash
   export CUDA_VISIBLE_DEVICES=3
   
   # build gen_cgcsn_python_train.jsonl
   nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
       --task gen \
       --tokenizer_path /data4/codebert/base/ \
       --trainfile /data1/wanghanbin/CONAN/dataset/gen/cgcsn/python/train.jsonl \
       --savefile /data1/wanghanbin/CONAN/dataset_ids_codebert/gen/cgcsn/python/train_ids.jsonl > ../build_code_log/build_gen_cgcsn_python_train.log 2>&1 &
   
   
   ... build other dataset
   ```

   

3. **finetune**

   你可以参考CONAN-R/OpenMatch/train.sh脚本去在每个任务的数据集上微调检索器，得到CONAN-R。train.sh.sh文件内容如下：

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   nohup python -m openmatch.driver.train_dr  \
       --output_dir /data1/wanghanbin/CONAN/code_bert_hanbin  \
       --model_name_or_path /data4/codebert/base \
       --do_train  \
       --save_steps 1000  \
       --eval_steps 1000  \
       --train_path /data1/wanghanbin/CONAN/dataset_ids_codebert/gen/cgcsn/python/train_ids.jsonl  \
       --eval_path /data1/wanghanbin/CONAN/dataset_ids_codebert/gen/cgcsn/python/dev_ids.jsonl  \
       --per_device_train_batch_size 16  \
       --train_n_passages 1  \
       --learning_rate 2e-5  \
       --evaluation_strategy steps  \
       --q_max_len 50  \
       --p_max_len 256  \
       --num_train_epochs 10  \
       --logging_dir /data1/wanghanbin/CONAN/code_bert_log_hanbin > /data1/wanghanbin/CONAN/code_bert_hanbin.log 2>&1 &
   ```

   

4. infer (get embedding)

   

5. search (get search results)

   ```bash
   export CUDA_VISIBLE_DEVICES=1    #指定gpu 
   nohup python -m openmatch.driver.train_dr  \
       --output_dir /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl_xinze/  \	#输出目录
       --model_name_or_path /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_xinze/best_dev/ \	#模型路径
       --do_train  \
       --save_steps 1000  \
       --eval_steps 1000  \
       --train_path /data1/wanghanbin/train_retriever/dataset/python/train_ids.jsonl  \    # train_ids目录
       --eval_path /data1/wanghanbin/train_retriever/dataset/python/valid_ids.jsonl  \		# dev_ids目录
       --per_device_train_batch_size 16  \	
       --train_n_passages 1  \
       --learning_rate 1e-5  \
       --evaluation_strategy steps  \
       --q_max_len 50  \
       --p_max_len 240  \
       --num_train_epochs 10  \
       --logging_dir /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl_xinze_log/ > /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl_xinze.log 2>&1 &
   ```

