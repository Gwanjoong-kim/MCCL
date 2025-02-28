export CUDA_VISIBLE_DEVICES=2,0

bs=8
lr=2e-4
dropout=0.1
psl=8
epoch=40
seed=2

A=am
B=dz
C=ha
D=ig
E=kr
F=ma
G=pcm
H=pt
I=sw
J=ts
K=twi
L=yo

cd /home1/kim03/myubai/kt_til/MoCL-NAACL-2024_temp

task_order=${A}_${B}_${C}_${D}_${E}_${F}_${G}_${H}_${I}_${J}_${K}_${L}
# task_order=${A}_${B}
# task_order=${A}_${B}_${C}_${D}

python3 src/run_continual_mtl_afrisenti_feature_temp.py \
    --model_name_or_path Davlan/afro-xlmr-large \
    --cl_language_list $task_order \
    --continual_learning \
    --compose_prompts True\
    --task_specific_classifier \
    --matching_loss_v2 \
    --do_train \
    --do_eval \
    --do_predict \
    --early_stop \
    --early_stopping_patience 5 \
    --max_seq_length 128 \
    --per_device_train_batch_size $bs \
    --learning_rate $lr \
    --num_train_epochs $epoch \
    --pre_seq_len $psl \
    --output_dir checkpoints_continual_afrisenti/order1_E2_compose_prompt_feature_seed_2 \
    --overwrite_output_dir \
    --hidden_dropout_prob $dropout \
    --seed $seed \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --prefix