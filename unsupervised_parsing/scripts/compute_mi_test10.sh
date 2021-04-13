SPLIT="test_10" # Data split to evaluate on
STEP=2000 # Number of steps to run Gibbs Sampling
TRAIN_FILE=./data/raw.${SPLIT}.txt # Path to the evalaute data file
SEED=$1 # Random seed
BATCHSIZE=256 # Batch size of a single Gibbs sampling step
OUTDIR=./out/${SPLIT}/mc${STEP}_seed${SEED} # Path to save mutual information values

python mutual_info_batch_mi.py \
    --output_dir $OUTDIR \
    --model_type=bert \
    --model_name_or_path=bert-base-cased \
    --do_train \
    --data_dir="./data" \
    --output_dir=$OUTDIR \
    --train_data_file=$TRAIN_FILE \
    --overwrite_output_dir \
    --batch_size $BATCHSIZE \
    --mc_samples $STEP \
    --burn_in $(( $STEP / 5 )) \
    --seed ${SEED}
