export CUDA_VISIBLE_DEVICES=1
nohup python search_dense_gen.py \
        --index_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/save_vec_python_dedupe_definitions_v2.pkl \
        --query_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/save_vec_csn_gen_train_docstring.pkl \
        --query_file=/data1/wanghanbin/CodeT5-sourcecode/data/generate/python/train.jsonl \
        --corpus_file=/data1/wanghanbin/redcoder_data/retrieval_database/python_dedupe_definitions_v2.pkl \
        --save_name=/data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_python_csn_train_100.pkl > /data1/wanghanbin/CodeT5-sourcecode/retrieve/search_dense_gen_python_csn_train_100.log 2>&1 &