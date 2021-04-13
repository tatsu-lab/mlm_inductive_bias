GLUE_DIR=$SCR/data/glue_data # replace with path to GLUE data
TASK="SST-2"

python ./mlm_toy.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --num_train_epochs 20 \
  --learning_rate 1e-3 \
  --warmup_steps 500 \
  --logging_steps 100 \
  --eval_steps 100 \
  --save_steps 0 \
  --overwrite_output_dir \
  --output_dir ./out/mlm_toy \
  --evaluate_during_training \
  --disable_tqdm
