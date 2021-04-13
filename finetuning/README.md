# Finetuning experiments

This (sub)repo contains code for our experiments in section 3.2 and section 5.1.

## Experiment in Section 3.2

### Data
We use the GLUE version of SST. The data is included in `./data`.

### Command
You can run `bash mlm_toy.py` and the output will be saved under `./out/mlm_toy`. 
The `finetune_results.pt` object in each subdirection in `./out/mlm_toy` contains a dictionary of finetuning results of
the corresponding settings.

`mlm_toy.py` contains the training code for this experiment.
`mlm_toy_utils.py` contains model definition and relevant changes in the dataloader.

## Experiment in Section 5.1

### Data
We provide example scripts with SST-2. You can download the hyperpartisan dataset and the agnews dataset
following the instructions in [this repo](https://github.com/allenai/dont-stop-pretraining).

### Approximating Pretraining
Here are the workflow:
1. For classification datasts like SST-2, we convert the `.tsv` data format to plain text. You 
   can do this by `python pretrain_scripts/convert_lm_dataset.py`.
   
2. Create mask for sentiment word by `python create_sentiment_mask.py`.
3. `bash scrpits/mlm_sst.sh` will continue pretraining BERT models on SST–2 data using different controls.

#### Download pretrained models
Alternatively, you can download our pretrained models at [this link](https://drive.google.com/file/d/11gTu5gDuvRMBsFv6Ix8eYH-geK6dhSoa/view?usp=sharing).
Unzip the downloaded file, and put it under `./pretrained_models`.

### Finetuning
`bash ./scripts/cls_finetune_sst` will perform finetuning for the low-data setting (10 examples per class) 
in our paper for 5 random seeds. The results will be written to `./out/bert_finetune/sst-2`. Note that because of
the noise level, we report with 20 random seeds in our paper.

To aggregate the results over different, you can call a simple script by `python aggregate_results.py`.
Here is the expected output of the released script.
```angular2html
setting: negative, accuracy: 64.36%+-3.39%
setting: baseline, accuracy: 68.17%+-5.25%
setting: positive, accuracy: 73.19%+-3.49%
```

If you change the script to run twenty seeds, here is the expected output
```angular2html
setting: negative, accuracy: 66.97%+-2.60%
setting: baseline, accuracy: 69.04%+-2.69%
setting: positive, accuracy: 74.74%+-2.43%
```
