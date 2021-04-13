GLUE_DIR=./data # replace with path to GLUE data
MODELDIR=./pretrained_models/sst-2
DATAPERCLASS=10

for CONTROL in "baseline" "positive" "negative"; do
for SEED in $(seq 1 5); do
  OUTPUTDIR=./out/bert_finetune/sst-2/${CONTROL}/seed_${SEED}
  BOARDDIR=./out/bert_finetune/sst-2/${CONTROL}/seed_${SEED}/tensorboard
  TASK="SST-2"

  python ./run_glue.py \
    --model_name_or_path $MODELDIR/$CONTROL \
    --task_name $TASK \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/$TASK \
    --max_seq_length 72 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-5 \
    --max_steps 400 \
    --warmup_steps 60 \
    --output_dir $OUTPUTDIR \
    --logging_dir $BOARDDIR \
    --logging_steps 20 \
    --eval_steps 40 \
    --save_steps -1 \
    --patience -1 \
    --evaluate_during_training \
    --overwrite_output_dir \
    --data_size_per_class 10 \
    --disable_tqdm \
    --seed $SEED
done
done
