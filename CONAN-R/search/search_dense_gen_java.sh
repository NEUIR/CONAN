export CUDA_VISIBLE_DEVICES=0
nohup python search_dense_gen.py \
        --index_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/save_vec_java_dedupe_definitions_v2.pkl \
        --query_vector_file=/data1/wanghanbin/CodeT5-sourcecode/retrieve/save_vec_csn_gen_valid_docstring_java.pkl \
        --query_file=/data1/wanghanbin/CodeT5-sourcecode/data/generate/java/valid.jsonl \
        --corpus_file=/data1/wanghanbin/redcoder_data/retrieval_database/java_dedupe_definitions_v2.pkl \
        --save_name=/data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_java_csn_valid_100.pkl > /data1/wanghanbin/CodeT5-sourcecode/retrieve/search_dense_gen_java_csn_valid_100.log 2>&1 &