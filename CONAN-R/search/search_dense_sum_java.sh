
export CUDA_VISIBLE_DEVICES=0
nohup python search_dense.py \
        --index_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_25000_save_vec_deduplicated_summaries_remove_def_java.pkl \
        --query_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_25000_save_vec_csn_sum_valid_code_java.pkl \
        --query_file=/data1/wanghanbin/CodeT5-sourcecode/data/summarize/java/valid.jsonl \
        --corpus_file=/data1/wanghanbin/redcoder_data/retrieval_database/deduplicated_summaries_remove_def_with_java_code.jsonl \
        --save_name=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_sum_java/finetune_25000_sum_java_csn_valid_100.pkl > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_sum_java/finetune_25000_search_dense_sum_java_csn_valid_100.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0
nohup python search_dense.py \
        --index_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_25000_save_vec_deduplicated_summaries_remove_def_java.pkl \
        --query_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_25000_save_vec_csn_sum_test_code_java.pkl \
        --query_file=/data1/wanghanbin/CodeT5-sourcecode/data/summarize/java/test.jsonl \
        --corpus_file=/data1/wanghanbin/redcoder_data/retrieval_database/deduplicated_summaries_remove_def_with_java_code.jsonl \
        --save_name=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_sum_java/finetune_25000_sum_java_csn_test_100.pkl > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_sum_java/finetune_25000_search_dense_sum_java_csn_test_100.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python search_dense.py \
        --index_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_25000_save_vec_deduplicated_summaries_remove_def_java.pkl \
        --query_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_25000_save_vec_csn_sum_train_code_java.pkl \
        --query_file=/data1/wanghanbin/CodeT5-sourcecode/data/summarize/java/train.jsonl \
        --corpus_file=/data1/wanghanbin/redcoder_data/retrieval_database/deduplicated_summaries_remove_def_with_java_code.jsonl \
        --save_name=/data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_sum_java/finetune_25000_sum_java_csn_train_100.pkl > /data1/wanghanbin/CodeT5-sourcecode/retrieve/finetune_retrieve_result_sum_java/finetune_25000_search_dense_sum_java_csn_train_100.log 2>&1 &
