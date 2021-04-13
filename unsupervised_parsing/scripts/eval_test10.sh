SEED=1
MIPATH="./out/test_10/mc2000_seed${SEED}/all_log_mi.pt" # Path to the saved mutual information values
SPLIT="test_10" # Data split to evaluate on

python mutual_info_eval.py \
   --split ${SPLIT} \
   --model_name bert-base-cased \
   --pmi_clamp_zero \
   --word_piece_agg_space exp \
   --word_piece_agg_type max \
   --word_piece_pmi_path ${MIPATH}
