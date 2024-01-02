export CUDA_VISIBLE_DEVICES=3
nohup python infer_concode.py \
        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/concode/test.json \
        --save_name finetune_15000_save_vec_concode_test \
        --lang java \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_concode/codet5_2e-5_10_concode/checkpoint-15000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 256 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_15000_infer_save_vec_concode_test.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python infer_concode.py \
        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/concode/dev.json \
        --save_name finetune_15000_save_vec_concode_dev \
        --lang java \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_concode/codet5_2e-5_10_concode/checkpoint-15000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 256 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_15000_infer_save_vec_concode_dev.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python infer_concode.py \
        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/concode/train.json \
        --save_name finetune_15000_save_vec_concode_train \
        --lang java \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_concode/codet5_2e-5_10_concode/checkpoint-15000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 256 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_15000_infer_save_vec_concode_train.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python infer_concode.py \
        --data_path /data1/wanghanbin/redcoder_data/retrieval_database/concode_corpus.jsonl \
        --save_name finetune_15000_save_vec_concode_retrieval_database \
        --lang java \
        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_concode/codet5_2e-5_10_concode/checkpoint-15000/ \
        --num_vec -1 \
        --eval_batch_size 64 \
        --block_size 256 \
        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_15000_infer_save_vec_concode_retrieval_database.log 2>&1 &

