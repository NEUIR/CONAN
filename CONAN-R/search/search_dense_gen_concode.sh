export CUDA_VISIBLE_DEVICES=0
nohup python search_dense_gen_concode.py \
        --index_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_15000_save_vec_concode_retrieval_database.pkl \
        --query_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_15000_save_vec_concode_test.pkl \
        --query_file=/data1/wanghanbin/CodeT5-sourcecode/data/concode/test.json \
        --corpus_file=/data1/wanghanbin/redcoder_data/retrieval_database/concode_corpus.jsonl \
        --save_name=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_gen_concode/finetune_15000_gen_concode_test_100.pkl > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_gen_concode/finetune_15000_search_dense_gen_concode_test_100.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python search_dense_gen_concode.py \
        --index_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_15000_save_vec_concode_retrieval_database.pkl \
        --query_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_15000_save_vec_concode_dev.pkl \
        --query_file=/data1/wanghanbin/CodeT5-sourcecode/data/concode/dev.json \
        --corpus_file=/data1/wanghanbin/redcoder_data/retrieval_database/concode_corpus.jsonl \
        --save_name=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_gen_concode/finetune_15000_gen_concode_dev_100.pkl > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_gen_concode/finetune_15000_search_dense_gen_concode_dev_100.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0
nohup python search_dense_gen_concode.py \
        --index_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_15000_save_vec_concode_retrieval_database.pkl \
        --query_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_15000_save_vec_concode_train.pkl \
        --query_file=/data1/wanghanbin/CodeT5-sourcecode/data/concode/train.json \
        --corpus_file=/data1/wanghanbin/redcoder_data/retrieval_database/concode_corpus.jsonl \
        --save_name=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_gen_concode/finetune_15000_gen_concode_train_100.pkl > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_gen_concode/finetune_15000_search_dense_gen_concode_train_100.log 2>&1 &