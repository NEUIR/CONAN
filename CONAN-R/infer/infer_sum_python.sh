
export CUDA_VISIBLE_DEVICES=3
nohup python infer_sum_python.py \
        --data_path /data1/wanghanbin/redcoder_data/retrieval_database/deduplicated.summaries.remove_def.txt \
        --save_name finetune_39000_save_vec_deduplicated_summaries_remove_def_python \
        --lang python \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_sum_python/codet5_2e-5_10_sum_python/checkpoint-39000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 512 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_39000_infer_save_vec_deduplicated_summaries_remove_def_python.log 2>&1 &


export CUDA_VISIBLE_DEVICES=3
nohup python infer_sum_python.py \
        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/summarize/python/test.jsonl \
        --save_name finetune_39000_save_vec_csn_sum_test_code_python \
        --lang python \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_sum_python/codet5_2e-5_10_sum_python/checkpoint-39000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 512 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_39000_infer_save_vec_csn_sum_test_code_python.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python infer_sum_python.py \
        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/summarize/python/valid.jsonl \
        --save_name finetune_39000_save_vec_csn_sum_valid_code_python \
        --lang python \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_sum_python/codet5_2e-5_10_sum_python/checkpoint-39000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 512 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_39000_infer_save_vec_csn_sum_valid_code_python.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python infer_sum_python.py \
        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/summarize/python/train.jsonl \
        --save_name finetune_39000_save_vec_csn_sum_train_code_python \
        --lang python \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_sum_python/codet5_2e-5_10_sum_python/checkpoint-39000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 512 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_39000_infer_save_vec_csn_sum_train_code_python.log 2>&1 &


