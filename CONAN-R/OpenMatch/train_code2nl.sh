export CUDA_VISIBLE_DEVICES=0
nohup python -m openmatch.driver.train_dr  \
    --output_dir /data/wanghanbin/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl/  \
    --model_name_or_path Salesforce/codet5-base \
    --do_train  \
    --save_steps 1000  \
    --eval_steps 1000  \
    --train_path /data/wanghanbin/train_retriever/dataset/python/train_ids.jsonl  \
    --eval_path /data/wanghanbin/train_retriever/dataset/python/valid_ids.jsonl  \
    --per_device_train_batch_size 16  \
    --train_n_passages 1  \
    --learning_rate 1e-5  \
    --evaluation_strategy steps  \
    --q_max_len 50  \
    --p_max_len 240  \
    --num_train_epochs 10  \
    --logging_dir /data/wanghanbin/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl_log/ > /data/wanghanbin/train_retriever/OpenMatch/save/codet5_1e-5_10_code2nl.log 2>&1 &

