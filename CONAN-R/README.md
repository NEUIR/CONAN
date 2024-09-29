# CONAN-R

CONAN -r use [SANTA](https://github.com/OpenMatch/SANTA) initialization, and then fine-tune on each task of training set.

### 1.Requirements

(1) Creating a virtual Environment

```bash
conda create -n conanr python=3.8
```

(2) Install the following packages using Pip or Conda under this environment:

```
transformers==4.22.2    # Other versions may have errors.
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



### 2.Get SANTA Checkpoint

The checkpoint of the pretrained SANTA model on `Python` data is [here](https://huggingface.co/OpenMatch/santa-code-python-adv). 



### 3.Finetune

1. **download dataset**

   You can download the data for the experiment from [REDCODER](https://arxiv.org/abs/2305.19912)

   

2. **build code**

   Convert the dataset to the format required by OpenMatch. cd to CONAN-R/build_code and run build_code.sh. The build_code.sh file has the following contents:

   ```bash
   export CUDA_VISIBLE_DEVICES=3
   
   # build gen_cgcsn_python_train.jsonl
   nohup python /data1/xxx/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
       --task gen \
       --tokenizer_path /data4/codebert/base/ \
       --trainfile /data1/xxx/CONAN/dataset/gen/cgcsn/python/train.jsonl \
       --savefile /data1/xxx/CONAN/dataset_ids_codebert/gen/cgcsn/python/train_ids.jsonl > ../build_code_log/build_gen_cgcsn_python_train.log 2>&1 &
   
   
   ... build other dataset
   ```

   

3. **finetune**

   You can use the CONAN-R/OpenMatch/train.sh script to fine-tune the retriever on each task's dataset to get CONAN-R. The train.sh.sh file is as follows:

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   nohup python -m openmatch.driver.train_dr  \
       --output_dir /data1/xxx/CONAN/code_bert_hanbin  \
       --model_name_or_path /data4/codebert/base \
       --do_train  \
       --save_steps 1000  \
       --eval_steps 1000  \
       --train_path /data1/xxx/CONAN/dataset_ids_codebert/gen/cgcsn/python/train_ids.jsonl  \
       --eval_path /data1/xxx/CONAN/dataset_ids_codebert/gen/cgcsn/python/dev_ids.jsonl  \
       --per_device_train_batch_size 16  \
       --train_n_passages 1  \
       --learning_rate 2e-5  \
       --evaluation_strategy steps  \
       --q_max_len 50  \
       --p_max_len 256  \
       --num_train_epochs 10  \
       --logging_dir /data1/xxx/CONAN/code_bert_log_hanbin > /data1/xxx/CONAN/code_bert_hanbin.log 2>&1 &
   ```

   

4. infer (get embedding)

   You can use the infer bash scripts: /infer/infer.sh

   ```bash
   export CUDA_VISIBLE_DEVICES=3
   nohup python infer.py \
          --data_path /data1/xxx/CONAN/dataset/gen/concode/java/train.jsonl \
          --save_name save_vec_concode_java_train \
          --lang java \
          --pretrained_dir /data3/xxx/CONAN/codebert/gen/concode/code_bert/checkpoint-60000/ \
          --num_vec -1 \
          --eval_batch_size 64 \
          --block_size 512 \
          --logging_steps 100 > /data1/xxx/CONAN/search_dense_results_codebert/gen/concode/java/infer_save_vec_concode_java_train.log 2>&1 &
   
   ```

   

5. search (get search results)

   ```bash
   export CUDA_VISIBLE_DEVICES=1     
   nohup python -m openmatch.driver.train_dr  \
       --output_dir /data1/xxx/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl_xinze/  \	#output dir
       --model_name_or_path /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_xinze/best_dev/ \	#model dir
       --do_train  \
       --save_steps 1000  \
       --eval_steps 1000  \
       --train_path /data1/xxx/train_retriever/dataset/python/train_ids.jsonl  \    # train_ids dir
       --eval_path /data1/xxx/train_retriever/dataset/python/valid_ids.jsonl  \		# dev_ids dir
       --per_device_train_batch_size 16  \	
       --train_n_passages 1  \
       --learning_rate 1e-5  \
       --evaluation_strategy steps  \
       --q_max_len 50  \
       --p_max_len 240  \
       --num_train_epochs 10  \
       --logging_dir /data1/xxx/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl_xinze_log/ > /data1/xxx/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl_xinze.log 2>&1 &
   ```

