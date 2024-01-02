export CUDA_VISIBLE_DEVICES=1
nohup python -m openmatch.driver.train_dr  \
    --output_dir /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl_xinze/  \
    --model_name_or_path /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_xinze/best_dev/ \
    --do_train  \
    --save_steps 1000  \
    --eval_steps 1000  \
    --train_path /data1/wanghanbin/train_retriever/dataset/python/train_ids.jsonl  \
    --eval_path /data1/wanghanbin/train_retriever/dataset/python/valid_ids.jsonl  \
    --per_device_train_batch_size 16  \
    --train_n_passages 1  \
    --learning_rate 1e-5  \
    --evaluation_strategy steps  \
    --q_max_len 50  \
    --p_max_len 240  \
    --num_train_epochs 10  \
    --logging_dir /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl_xinze_log/ > /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl_xinze.log 2>&1 &

