
export CUDA_VISIBLE_DEVICES=1
nohup python infer_sum_java.py \
        --data_path /data1/wanghanbin/redcoder_data/retrieval_database/deduplicated.summaries.remove_def.txt \
        --save_name finetune_25000_save_vec_deduplicated_summaries_remove_def_java \
        --lang java \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_sum_java/codet5_2e-5_10_sum_java/checkpoint-25000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 512 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_25000_infer_save_vec_deduplicated_summaries_remove_def_java.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0
nohup python infer_sum_java.py \
        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/summarize/java/test.jsonl \
        --save_name finetune_25000_save_vec_csn_sum_test_code_java \
        --lang java \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_sum_java/codet5_2e-5_10_sum_java/checkpoint-25000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 512 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_25000_infer_save_vec_csn_sum_test_code_java.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python infer_sum_java.py \
        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/summarize/java/valid.jsonl \
        --save_name finetune_25000_save_vec_csn_sum_valid_code_java \
        --lang java \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_sum_java/codet5_2e-5_10_sum_java/checkpoint-25000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 512 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_25000_infer_save_vec_csn_sum_valid_code_java.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python infer_sum_java.py \
        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/summarize/java/train.jsonl \
        --save_name finetune_25000_save_vec_csn_sum_train_code_java \
        --lang java \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_sum_java/codet5_2e-5_10_sum_java/checkpoint-25000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 512 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_25000_infer_save_vec_csn_sum_train_code_java.log 2>&1 &


