CONTROL="positive"
SEED=1
TRAIN_FILE=./data/SST-2/train.lm
TEST_FILE=./data/SST-2/dev.lm
OUT=./out/mlm_sst/$CONTROL/

python run_salient_mlm.py \
    --output_dir=$OUT \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --block_size 72 \
    --model_type=bert \
    --model_name_or_path=bert-base-uncased \
    --do_eval \
    --data_dir=./data/SST-2 \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$TEST_FILE \
    --num_train_epochs 10\
    --warmup_steps 1256 \
    --learning_rate 1e-5 \
    --logging_steps 100 \
    --eval_steps 1000 \
    --save_steps 4000 \
    --patience -1 \
    --evaluate_during_training \
    --overwrite_output_dir \
    --mlm \
    --control_group $CONTROL \
    --seed $SEED \
    --disable_tqdm
#    --do_train \
