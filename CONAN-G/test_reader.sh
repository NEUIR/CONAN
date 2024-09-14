
export CUDA_VISIBLE_DEVICES=1
nohup python test_reader.py \
        --model_path /data1/wanghanbin/fidgen/checkpoint/gen_ctxs1_no_target_codet5_saleforce_eval5000step_java_v3_test/checkpoint/best_dev/ \
        --eval_data /data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_java_csn_test_100.pkl \
        --per_gpu_batch_size 1 \
        --n_context 1 \
        --with_target no \
        --text_maxlength 512 \
        --answer_maxlength 256 \
        --name my_test_gen_java_1 \
        --checkpoint_dir checkpoint > /data1/wanghanbin/fidgen/test_log/test_gen_ctxs1.log 2>&1 &


export CUDA_VISIBLE_DEVICES=1
nohup python test_reader.py \
        --model_path /data1/wanghanbin/fidgen/checkpoint/gen_ctxs2_no_target_codet5_saleforce_eval5000step_java_v3_test/checkpoint/best_dev/ \
        --eval_data /data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_java_csn_test_100.pkl \
        --per_gpu_batch_size 1 \
        --n_context 2 \
        --with_target no \
        --text_maxlength 512 \
        --answer_maxlength 256 \
        --name my_test_gen_java_2 \
        --checkpoint_dir checkpoint > /data1/wanghanbin/fidgen/test_log/test_gen_ctxs2.log 2>&1 &
        
export CUDA_VISIBLE_DEVICES=1
nohup python test_reader.py \
        --model_path /data1/wanghanbin/fidgen/checkpoint/gen_ctxs3_no_target_codet5_saleforce_eval5000step_java_v3_test/checkpoint/best_dev/ \
        --eval_data /data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_java_csn_test_100.pkl \
        --per_gpu_batch_size 1 \
        --n_context 3 \
        --with_target no \
        --text_maxlength 512 \
        --answer_maxlength 256 \
        --name my_test_gen_java_3 \
        --checkpoint_dir checkpoint > /data1/wanghanbin/fidgen/test_log/test_gen_ctxs3.log 2>&1 &
        
export CUDA_VISIBLE_DEVICES=1
nohup python test_reader.py \
        --model_path /data1/wanghanbin/fidgen/checkpoint/gen_ctxs4_no_target_codet5_saleforce_eval5000step_java_v3_test/checkpoint/best_dev/ \
        --eval_data /data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_java_csn_test_100.pkl \
        --per_gpu_batch_size 1 \
        --n_context 4 \
        --with_target no \
        --text_maxlength 512 \
        --answer_maxlength 256 \
        --name my_test_gen_java_4 \
        --checkpoint_dir checkpoint > /data1/wanghanbin/fidgen/test_log/test_gen_ctxs4.log 2>&1 &
#export CUDA_VISIBLE_DEVICES=2
#nohup python test_reader.py \
#        --model_path /data1/wanghanbin/fidgen/checkpoint/gen_ctxs5_no_target_concode/checkpoint/step-164000/ \
#        --eval_data /data1/wanghanbin/cerag_data/gen/concode/finetune_15000_gen_concode_test_100.pkl \
#        --per_gpu_batch_size 8 \
#        --n_context 5 \
#        --with_target no \
#        --text_maxlength 512 \
#        --answer_maxlength 256 \
#        --name my_test_concode_164000 \
#        --checkpoint_dir checkpoint > /data1/wanghanbin/fidgen/test_log/test_gen_ctxs5_no_target_codet5_concode_164000.log 2>&1 &
#



#export CUDA_VISIBLE_DEVICES=3
#nohup python test_reader.py \
#        --model_path /data1/wanghanbin/fidgen/checkpoint/gen_ctxs5_no_target_codet5_saleforce_eval5000step_java_v2/checkpoint/step-200000/ \
#        --eval_data /data1/wanghanbin/CodeT5-sourcecode/retrieve/retrieve_result_gen/gen_java_csn_test_100.pkl \
#        --per_gpu_batch_size 1 \
#        --n_context 5 \
#        --with_target no \
#        --text_maxlength 512 \
#        --answer_maxlength 256 \
#        --name my_test \
#        --checkpoint_dir checkpoint > /data1/wanghanbin/fidgen/test_log/test_gen_ctxs5_no_target_codet5_java_200000.log 2>&1 &