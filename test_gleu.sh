# /bin/bash

gpu=3
task=mrpc
LINEAR_NUM=(4)

for lr in 1.5e-5 2e-5 2.5e-5 3e-5; do
    for ws in 50 500; do
        # python -u test_gleu.py \
        #     --result_path deberta_${lr}_${ws} \
        #     --glue_task $task \
        #     --gpu $gpu \
        #     --no_add_linear \

        # python -u test_gleu.py \
        #     --result_path deberta_${lr}_${ws} \
        #     --glue_task $task \
        #     --gpu $gpu \
        #     --head_indi \
        #     --init_type unit

        python -u test_gleu.py \
            --result_path deberta_${lr}_${ws} \
            --glue_task $task \
            --gpu $gpu \
            --head_indi \
            --init_type he
    done
done

# for linear_num in ${LINEAR_NUM[@]}; do
#     python -u test_gleu.py \
#         --result_path deberta_befdot_layer_bottom_${linear_num}_${task} \
#         --glue_task $task \
#         --gpu $gpu \
#         --head_indi \
#         --init_type unit \
#         --add_linear_num $linear_num
        
#     python -u test_gleu.py \
#         --result_path deberta_befdot_layer_random_bottom_${linear_num}_${task} \
#         --glue_task $task \
#         --gpu $gpu \
#         --head_indi \
#         --init_type he \
#         --add_linear_num $linear_num
# done
