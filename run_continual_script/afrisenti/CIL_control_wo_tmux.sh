TASK_ORDER=$1
TASK_FEATURE_SETTING=$2  # Task Feature를 contrastive learning으로 설정할지 여부
GPU_ID=$3               # GPU ID
SEED=$4                 # SEED 값
DATA_PORTION=$5         # Task별 학습할 데이터 비율
TIMESTAMP=$(date +"%d_%H_%M")

# 기본 하이퍼파라미터 설정
bs=8
lr=2e-4
dropout=0.1
psl=8
epoch=40
# epoch=1

# 언어 리스트
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

# 작업 디렉토리 이동
cd /home1/kim03/myubai/kt_til/MoCL-NAACL-2024_temp

# task order 설정 (TASK_ORDER 값에 따라 순서를 다르게 지정)
if [ "$TASK_ORDER" == 0 ]; then
    task_order=${A}_${B}_${C}_${D}_${E}_${F}_${G}_${H}_${I}_${J}_${K}_${L}
elif [ "$TASK_ORDER" == 1 ]; then
    task_order=${F}_${G}_${E}_${H}_${D}_${I}_${C}_${J}_${B}_${K}_${A}_${L}
else
    task_order=${A}_${B}_${C}_${F}_${D}_${E}_${I}_${J}_${K}_${L}_${G}_${H}
fi

if [ "$TASK_FEATURE_SETTING" == "c" ]; then
    PYTHON_SCRIPT="src/run_continual_mtl_afrisenti_feature_temp.py"
elif [ "$TASK_FEATURE_SETTING" == "o" ]; then
    PYTHON_SCRIPT="src/run_continual_mtl_afrisenti_temp.py"
elif [ "$TASK_FEATURE_SETTING" == "aug" ]; then
    PYTHON_SCRIPT="src/run_continual_mtl_afrisenti_feature_aug.py"
else
    PYTHON_SCRIPT="src/run_continual_mtl_afrisenti_no_kt.py"
    TASK_FEATURE_SETTING="no_kt"
fi

# output_dir을 GPU_ID, SEED, DATA_PORTION 값에 따라 동적으로 변경
OUTPUT_DIR="checkpoints_continual_afrisenti/order_${TASK_ORDER}_${TASK_FEATURE_SETTING}_seed_${SEED}_portion_${DATA_PORTION}_${TIMESTAMP}"

# output_dir이 없으면 자동으로 생성
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created output directory: $OUTPUT_DIR"
else
    echo "Output directory already exists: $OUTPUT_DIR"
fi

echo "Running on GPU: $GPU_ID with seed: $SEED"
echo "Output directory: $OUTPUT_DIR"

# tmux 세션에서 실행할 명령어를 하나의 문자열로 구성
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python3 ${PYTHON_SCRIPT} \
    --model_name_or_path Davlan/afro-xlmr-large \
    --cl_language_list ${task_order} \
    --continual_learning \
    --compose_prompts \
    --task_specific_classifier \
    --matching_loss_v2 \
    --do_train \
    --do_eval \
    --do_predict \
    --early_stop \
    --early_stopping_patience 5 \
    --max_seq_length 128 \
    --data_portion ${DATA_PORTION} \
    --per_device_train_batch_size ${bs} \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --pre_seq_len ${psl} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --hidden_dropout_prob ${dropout} \
    --seed ${SEED} \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --prefix

# tmux 세션 이름 (GPU와 SEED 정보를 포함)
SESSION_NAME=${TASK_ORDER}_${TASK_FEATURE_SETTING}_${SEED}_${DATA_PORTION}
CMD

# 새로운 tmux 세션을 백그라운드에서 생성하며, 그 안에서 CMD를 실행

# tmux new-session -d -s ${SESSION_NAME} bash -c "${CMD}"
# tmux pipe-pane -o -t ${SESSION_NAME} "cat >> ${OUTPUT_DIR}/tmux.log"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach to the session with: tmux attach-session -t ${SESSION_NAME}"