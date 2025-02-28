export CUDA_VISIBLE_DEVICES=1

bs=8
dropout=0.1
sl=512
psl=50
gpsl=10
epoch=40

A=agnews
B=yelp
C=amazon
D=yahoo
E=dbpedia

lr_A=5e-2
lr_B=5e-2
lr_C=5e-2
lr_D=2e-2
lr_E=2e-2
seed=0

cd /home1/kim03/myubai/kt_til/MoCL-NAACL-2024

# task_list=${C}_${D}_${A}_${B}_${E}
task_list=${D}_${B}_${C}_${A}_${E}
# task_list=${A}_${C}
# lr_list=${lr_C}_${lr_D}_${lr_A}_${lr_B}_${lr_E}
lr_list=${lr_D}_${lr_B}_${lr_C}_${lr_A}_${lr_E}

python3 src/run_continual_mtl5_t5.py \
    --model_name_or_path google-t5/t5-large \
    --task_list $task_list \
    --continual_learning \
    --compose_prompts \
    --matching_loss_v2 \
    --do_train \
    --do_eval \
    --do_predict \
    --max_train_samples 16 \
    --max_eval_samples 200 \
    --early_stop \
    --early_stopping_patience 5 \
    --max_seq_length $sl \
    --per_device_train_batch_size $bs \
    --learning_rate_list $lr_list \
    --num_train_epochs $epoch \
    --pre_seq_len $psl \
    --output_dir checkpoints_continual_mtl5_t5/origin_order3_compose_prompts \
    --overwrite_output_dir \
    --hidden_dropout_prob $dropout \
    --seed $seed \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --prefix