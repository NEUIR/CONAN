export CUDA_VISIBLE_DEVICES=1
nohup python -m openmatch.driver.train_dr  \
    --output_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_2e-5_128_10_dual_task_xinze/  \
    --model_name_or_path /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
    --do_train  \
    --save_steps 1000  \
    --eval_steps 1000  \
    --train_path /data1/wanghanbin/train_retriever/dataset/java/train_ids.jsonl  \
    --eval_path /data1/wanghanbin/train_retriever/dataset/java/valid_ids.jsonl  \
    --per_device_train_batch_size 64  \
    --train_n_passages 1  \
    --learning_rate 2e-5  \
    --evaluation_strategy steps  \
    --q_max_len 50  \
    --p_max_len 256  \
    --num_train_epochs 10  \
    --logging_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_2e-5_128_10_dual_task_xinze_log/ > /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_2e-5_128_10_dual_task_xinze.log 2>&1 &

