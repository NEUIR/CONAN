# CONAN-G

### 1.Requirements

(1) Creating a virtual Environment

```bash
conda create -n conang python=3.8
```

(2) Install packages 

```sh
cd CONAN-G
pin install -r requirements.txt
```





### 2. Training

Using the script CONAN-G/train_reader.sh to train CodeT5.

```bash
export CUDA_VISIBLE_DEVICES=0
nohup python train_reader.py \
        --train_data /data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_python_csn_train_100.pkl \
        --eval_data /data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_python_csn_valid_100.pkl \
        --model_path /data1/wanghanbin/fidgen/Salesforce/codet5-base_csn_gen_python/ \
        --per_gpu_batch_size 1 \
        --n_context 1 \
        --text_maxlength 512 \
        --answer_maxlength 256 \
        --name gen_ctxs1_no_target_codet5_saleforce_eval5000step_python_v3_test \
        --total_steps 20000 \
        --eval_freq 5000 \
        --save_freq 5000 \
        --with_target no \
        --eval_part \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --warmup_step 1000 \
        --lr 0.00005 > /data1/wanghanbin/fidgen/train_reader_fid_gen_ctxs1_no_target_codet5_saleforce_python_v3.log 2>&1 &
```



### 3. Evaluation

After training the model, test it using the following script:

```bash
export CUDA_VISIBLE_DEVICES=3
nohup python test_reader.py \
       --model_path /data1/wanghanbin/fidgen/checkpoint/gen_ctxs5_no_target_codet5_saleforce_eval5000step_java_v2/checkpoint/step-200000/ \
       --eval_data /data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_java_csn_test_100.pkl \
       --per_gpu_batch_size 1 \
       --n_context 5 \
       --with_target no \
       --text_maxlength 512 \
       --answer_maxlength 256 \
       --name my_test \
       --checkpoint_dir checkpoint > /data1/wanghanbin/fidgen/test_log/test_gen_ctxs5_no_target_codet5_java_200000.log 2>&1 &
```

