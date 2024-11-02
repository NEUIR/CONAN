export CUDA_VISIBLE_DEVICES=3

# build gen_cgcsn_python_train.jsonl
nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
    --task gen \
    --tokenizer_path xxx/codet5/base/ \
    --trainfile xxx/CONAN/dataset/gen/cgcsn/python/train.jsonl \
    --savefile xxx/CONAN/dataset_ids_codet5/gen/cgcsn/python/train_ids.jsonl > ../build_code_log/build_gen_cgcsn_python_train.log 2>&1 &


# build gen_cgcsn_python_dev.jsonl
nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
    --task gen \
    --tokenizer_path xxx/codet5/base/ \
    --trainfile xxx/CONAN/dataset/gen/cgcsn/python/dev.jsonl \
    --savefile xxx/CONAN/dataset_ids_codet5/gen/cgcsn/python/dev_ids.jsonl > ../build_code_log/build_gen_cgcsn_python_dev.log 2>&1 &


# build gen_cgcsn_java_train.jsonl
nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
    --task gen \
    --tokenizer_path xxx/codet5/base/ \
    --trainfile xxx/CONAN/dataset/gen/cgcsn/java/train.jsonl \
    --savefile xxx/CONAN/dataset_ids_codet5/gen/cgcsn/java/train_ids.jsonl > ../build_code_log/build_gen_cgcsn_java_train.log 2>&1 &

# build gen_cgcsn_java_dev.jsonl
nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
    --task gen \
    --tokenizer_path xxx/codet5/base/ \
    --trainfile xxx/CONAN/dataset/gen/cgcsn/java/dev.jsonl \
    --savefile xxx/CONAN/dataset_ids_codet5/gen/cgcsn/java/dev_ids.jsonl > ../build_code_log/build_gen_cgcsn_java_dev.log 2>&1 &

# build gen_concode_java_train.jsonl
nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
    --task concode \
    --tokenizer_path xxx/codet5/base/ \
    --trainfile xxx/wanghanbin/CONAN/dataset/gen/concode/java/train.json \
    --savefile xxx/wanghanbin/CONAN/dataset_ids_codet5/gen/concode/java/train_ids.jsonl > ../build_code_log/build_gen_concode_java_train.log 2>&1 &

# build gen_concode_java_dev.jsonl
nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
    --task concode \
    --tokenizer_path xxx/codet5/base/ \
    --trainfile xxx/CONAN/dataset/gen/concode/java/dev.json \
    --savefile xxx/CONAN/dataset_ids_codet5/gen/concode/java/dev_ids.jsonl > ../build_code_log/build_gen_concode_java_dev.log 2>&1 &

# build sum_cscsn_python_train.jsonl
nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
    --task sum \
    --tokenizer_path xxx/codet5/base/ \
    --trainfile xxx/CONAN/dataset/sum/cscsn/python/train.jsonl \
    --savefile xxx/CONAN/dataset_ids_codet5/sum/cscsn/python/train_ids.jsonl > ../build_code_log/build_sum_cscsn_python_train.log 2>&1 &

# build sum_cscsn_python_dev.jsonl
nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
    --task sum \
    --tokenizer_path xxx/codet5/base/ \
    --trainfile xxx/CONAN/dataset/sum/cscsn/python/dev.jsonl \
    --savefile xxx/CONAN/dataset_ids_codet5/sum/cscsn/python/dev_ids.jsonl > ../build_code_log/build_sum_cscsn_python_dev.log 2>&1 &


# build sum_cscsn_java_train.jsonl
nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
    --task sum \
    --tokenizer_path xxx/codet5/base/ \
    --trainfile xxx/CONAN/dataset/sum/cscsn/java/train.jsonl \
    --savefile xxx/CONAN/dataset_ids_codet5/sum/cscsn/java/train_ids.jsonl > ../build_code_log/build_sum_cscsn_java_train.log 2>&1 &


# build sum_cscsn_java_dev.jsonl
nohup python /data1/wanghanbin/CONAN_NEUIR/CONAN-R/build_code/build_code.py  \
    --task sum \
    --tokenizer_path xxx/codet5/base/ \
    --trainfile xxx/CONAN/dataset/sum/cscsn/java/dev.jsonl \
    --savefile xxx/CONAN/dataset_ids_codet5/sum/cscsn/java/dev_ids.jsonl > ../build_code_log/build_sum_cscsn_java_dev.log 2>&1 &