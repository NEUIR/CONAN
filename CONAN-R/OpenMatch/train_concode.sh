export CUDA_VISIBLE_DEVICES=0
nohup python -m openmatch.driver.train_dr  \
    --output_dir /data1/wanghanbin/train_retriever/OpenMatch/save_concode/codet5_2e-5_10_concode/  \
    --model_name_or_path /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
    --do_train  \
    --save_steps 1000  \
    --eval_steps 1000  \
    --train_path /data1/wanghanbin/train_retriever/train_retriever_dataset_concode/train_ids.jsonl  \
    --eval_path /data1/wanghanbin/train_retriever/train_retriever_dataset_concode/dev_ids.jsonl  \
    --per_device_train_batch_size 64  \
    --train_n_passages 1  \
    --learning_rate 2e-5  \
    --evaluation_strategy steps  \
    --q_max_len 256  \
    --p_max_len 100  \
    --num_train_epochs 10  \
    --logging_dir /data1/wanghanbin/train_retriever/OpenMatch/save_concode/codet5_2e-5_10_concode_log/ > /data1/wanghanbin/train_retriever/OpenMatch/save_concode/codet5_2e-5_128_10_concode.log 2>&1 &

