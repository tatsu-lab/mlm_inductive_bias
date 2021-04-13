# Extracting Mutual Information for Unsupervised Parsing

### Data
We include the two PTB test sets with this repo. They are located in `./data`.

### Experiment Workflow
- First extract the mutual information induced by a pretrained BERT by `bash ./scripts/compute_mi_test10.sh`. 
Change the `SPLIT` variable to compute on different test sets.
- Then form the parse trees and evaluate aginst the ground truth by `batch ./scripts/eval_test10`.
- You can look at the provided scripts for modifiable arguments.

### Expected Results
Here are the expected UUAS numbers with the provided scripts. Note that if you are using a different batch size, 
the results might be slightly different due to randomness in Gibbs Sampling.

| Seed/Dataset     | WSJ10 | WSJ Subsampled |
| ----------- | ----------- | ----------- |
| 1   | 0.5911      | 0.5063       |
| 2   | 0.5897      | 0.5100       |
| 3   | 0.5874      | 0.5023       |

### Hardware Requirements
The provided script uses the following hardware:
- Geforce RTX 3090 (24GB memory)
- 100GB RAM

You can lower `BATCHSIZE` to reduce the GPU memory usage or RAM usage.
