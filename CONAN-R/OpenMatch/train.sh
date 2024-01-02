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