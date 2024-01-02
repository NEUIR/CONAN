#-------------------------------gen  concode java-----------------------------------------------------------
export CUDA_VISIBLE_DEVICES=3
nohup python infer.py \
       --data_path /data1/wanghanbin/CONAN/dataset/gen/concode/java/train.jsonl \
       --save_name save_vec_concode_java_train \
       --lang java \
       --pretrained_dir /data3/lixinze/CONAN/codebert/gen/concode/code_bert/checkpoint-60000/ \
       --num_vec -1 \
       --eval_batch_size 64 \
       --block_size 512 \
       --logging_steps 100 > /data1/wanghanbin/CONAN/search_dense_results_codebert/gen/concode/java/infer_save_vec_concode_java_train.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python infer.py \
       --data_path /data1/wanghanbin/CONAN/dataset/gen/concode/java/dev.jsonl \
       --save_name save_vec_concode_java_dev \
       --lang java \
       --pretrained_dir /data3/lixinze/CONAN/codebert/gen/concode/code_bert/checkpoint-60000/ \
       --num_vec -1 \
       --eval_batch_size 64 \
       --block_size 512 \
       --logging_steps 100 > /data1/wanghanbin/CONAN/search_dense_results_codebert/gen/concode/java/infer_save_vec_concode_java_dev.log 2>&1 &


export CUDA_VISIBLE_DEVICES=1
nohup python infer.py \
       --data_path /data1/wanghanbin/CONAN/dataset/gen/concode/java/test.jsonl \
       --save_name save_vec_concode_java_test \
       --lang java \
       --pretrained_dir /data3/lixinze/CONAN/codebert/gen/concode/code_bert/checkpoint-60000/ \
       --num_vec -1 \
       --eval_batch_size 64 \
       --block_size 512 \
       --logging_steps 100 > /data1/wanghanbin/CONAN/search_dense_results_codebert/gen/concode/java/infer_save_vec_concode_java_test.log 2>&1 &

#-------------------------------gen  cgcsn java-----------------------------------------------------------
# export CUDA_VISIBLE_DEVICES=0
# nohup python infer.py \
#        --data_path /data1/wanghanbin/CONAN/dataset/gen/cgcsn/java/train.jsonl \
#        --save_name save_vec_cgcsn_java_train \
#        --lang java \
#        --pretrained_dir /data3/lixinze/CONAN/codebert/gen/cgcsn/java/code_bert/checkpoint-70000/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CONAN/search_dense_results_codebert/gen/cgcsn/java/infer_save_vec_cgcsn_java_train.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python infer.py \
#        --data_path /data1/wanghanbin/CONAN/dataset/gen/cgcsn/java/dev.jsonl \
#        --save_name save_vec_cgcsn_java_dev \
#        --lang java \
#        --pretrained_dir /data3/lixinze/CONAN/codebert/gen/cgcsn/java/code_bert/checkpoint-70000/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CONAN/search_dense_results_codebert/gen/cgcsn/java/infer_save_vec_cgcsn_java_dev.log 2>&1 &


# export CUDA_VISIBLE_DEVICES=3
# nohup python infer.py \
#        --data_path /data1/wanghanbin/CONAN/dataset/gen/cgcsn/java/test.jsonl \
#        --save_name save_vec_cgcsn_java_test \
#        --lang java \
#        --pretrained_dir /data3/lixinze/CONAN/codebert/gen/cgcsn/java/code_bert/checkpoint-70000/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CONAN/search_dense_results_codebert/gen/cgcsn/java/infer_save_vec_cgcsn_java_test.log 2>&1 &

#-------------------------------gen  cgcsn python-----------------------------------------------------------
# export CUDA_VISIBLE_DEVICES=0
# nohup python infer.py \
#        --data_path /data1/wanghanbin/CONAN/dataset/gen/cgcsn/python/train.jsonl \
#        --save_name save_vec_cgcsn_python_train \
#        --lang python \
#        --pretrained_dir /data3/lixinze/CONAN/codebert/gen/cgcsn/python/code_bert/checkpoint-70000/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CONAN/search_dense_results_codebert/gen/cgcsn/python/infer_save_vec_cgcsn_python_train.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# nohup python infer.py \
#        --data_path /data1/wanghanbin/CONAN/dataset/gen/cgcsn/python/dev.jsonl \
#        --save_name save_vec_cgcsn_python_dev \
#        --lang python \
#        --pretrained_dir /data3/lixinze/CONAN/codebert/gen/cgcsn/python/code_bert/checkpoint-70000/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CONAN/search_dense_results_codebert/gen/cgcsn/python/infer_save_vec_cgcsn_python_dev.log 2>&1 &


# export CUDA_VISIBLE_DEVICES=3
# nohup python infer.py \
#        --data_path /data1/wanghanbin/CONAN/dataset/gen/cgcsn/python/test.jsonl \
#        --save_name save_vec_cgcsn_python_test \
#        --lang python \
#        --pretrained_dir /data3/lixinze/CONAN/codebert/gen/cgcsn/python/code_bert/checkpoint-70000/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CONAN/search_dense_results_codebert/gen/cgcsn/python/infer_save_vec_cgcsn_python_test.log 2>&1 &


#-------------------------------上面为新跑的-----------------------------------------------------------  
# export CUDA_VISIBLE_DEVICES=0
# nohup python infer.py \
#         --data_path /data1/wanghanbin/CodeT5-sourcecode/data/concode/test.json \
#         --save_name save_vec_concode_test \
#         --lang java \
#         --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
#         --num_vec -1 \
#         --eval_batch_size 64 \
#         --block_size 512 \
#         --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_concode_test.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python infer.py \
#         --data_path /data1/wanghanbin/CodeT5-sourcecode/data/concode/dev.json \
#         --save_name save_vec_concode_dev \
#         --lang java \
#         --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
#         --num_vec -1 \
#         --eval_batch_size 64 \
#         --block_size 512 \
#         --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_concode_dev.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python infer.py \
#         --data_path /data1/wanghanbin/CodeT5-sourcecode/data/concode/train.json \
#         --save_name save_vec_concode_train \
#         --lang java \
#         --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
#         --num_vec -1 \
#         --eval_batch_size 64 \
#         --block_size 512 \
#         --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_concode_train.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python infer.py \
#         --data_path /data1/wanghanbin/redcoder_data/retrieval_database/concode_corpus.jsonl \
#         --save_name save_vec_concode_retrieval_database \
#         --lang java \
#         --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
#         --num_vec -1 \
#         --eval_batch_size 64 \
#         --block_size 512 \
#         --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_concode_retrieval_database.log 2>&1 &


#
#export CUDA_VISIBLE_DEVICES=0
#nohup python infer.py \
#        --data_path /data1/wanghanbin/redcoder_data/retrieval_database/deduplicated.summaries.remove_def.txt \
#        --save_name save_vec_deduplicated_summaries_remove_def_java \
#        --lang java \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_deduplicated_summaries_remove_def_java.log 2>&1 &


#export CUDA_VISIBLE_DEVICES=0
#nohup python infer.py \
#        --data_path /data1/wanghanbin/redcoder_data/retrieval_database/deduplicated.summaries.remove_def.txt \
#        --save_name save_vec_deduplicated_summaries_remove_def \
#        --lang java \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_best_dualtask/checkpoint_best_dualtask/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_deduplicated_summaries_remove_def.log 2>&1 &

#export CUDA_VISIBLE_DEVICES=0
#nohup python infer.py \
#        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/summarize/java/test.jsonl \
#        --save_name save_vec_csn_sum_test_code_java \
#        --lang java \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_csn_sum_test_code_java.log 2>&1 &

#export CUDA_VISIBLE_DEVICES=0
#nohup python infer.py \
#        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/summarize/java/train.jsonl \
#        --save_name save_vec_csn_sum_train_code_java \
#        --lang java \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_csn_sum_train_code_java.log 2>&1 &
#
#export CUDA_VISIBLE_DEVICES=0
#nohup python infer.py \
#        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/summarize/java/valid.jsonl \
#        --save_name save_vec_csn_sum_valid_code_java \
#        --lang java \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_csn_sum_valid_code_java.log 2>&1 &

#infer "/data1/wanghanbin/CodeT5-sourcecode/data/generate/python/train.jsonl"
#export CUDA_VISIBLE_DEVICES=0
#nohup python infer.py \
#        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/generate/java/test.jsonl \
#        --save_name save_vec_csn_gen_test_docstring_java \
#        --lang java \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_csn_gen_test_docstring_java.log 2>&1 &








#infer java_dedupe_definitions_v2.pkl
#export CUDA_VISIBLE_DEVICES=0
#nohup python infer.py \
#        --data_path /data1/wanghanbin/redcoder_data/retrieval_database/java_dedupe_definitions_v2.pkl \
#        --save_name save_vec_java_dedupe_definitions_v2 \
#        --lang java \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save_java/codet5_xinze_finetune/best_dev/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_java_dedupe_definitions_v2.log 2>&1 &




#infer "/data1/wanghanbin/CodeT5-sourcecode/data/generate/python/train.jsonl"
#export CUDA_VISIBLE_DEVICES=2
#nohup python infer.py \
#        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/generate/python/train.jsonl \
#        --save_name save_vec_csn_gen_train_docstring \
#        --lang python \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_best_dualtask/checkpoint_best_dualtask/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_csn_gen_train_docstring.log 2>&1 &

#infer "/data1/wanghanbin/CodeT5-sourcecode/data/generate/python/valid.jsonl"
#export CUDA_VISIBLE_DEVICES=2
#nohup python infer.py \
#        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/generate/python/valid.jsonl \
#        --save_name save_vec_csn_gen_valid_docstring \
#        --lang python \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_best_dualtask/checkpoint_best_dualtask/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_csn_gen_valid_docstring.log 2>&1 &



#infer "/data1/wanghanbin/CodeT5-sourcecode/data/generate/python/test.jsonl"
#export CUDA_VISIBLE_DEVICES=2
#nohup python infer.py \
#        --data_path /data1/wanghanbin/CodeT5-sourcecode/data/generate/python/test.jsonl \
#        --save_name save_vec_csn_gen_test_docstring \
#        --lang python \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_best_dualtask/checkpoint_best_dualtask/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_csn_gen_test_docstring.log 2>&1 &











##infer python_dedupe_definitions_v2.pkl
#export CUDA_VISIBLE_DEVICES=0
#nohup python infer.py \
#        --data_path /data1/wanghanbin/redcoder_data/retrieval_database/python_dedupe_definitions_v2.pkl \
#        --save_name save_vec_python_dedupe_definitions_v2 \
#        --lang python \
#        --pretrained_dir /data1/wanghanbin/train_retriever/OpenMatch/save/codet5_best_dualtask/checkpoint_best_dualtask/ \
#        --num_vec -1 \
#        --eval_batch_size 64 \
#        --block_size 512 \
#        --logging_steps 100 > /data1/wanghanbin/CodeT5-sourcecode/retrieve/infer_save_vec_python_dedupe_definitions_v2.log 2>&1 &





