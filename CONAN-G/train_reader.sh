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