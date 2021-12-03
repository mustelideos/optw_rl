# optw_rl

A PyTorch implementation of the Pointer Network model in:

*[A reinforcement learning approach to the orienteering problem with time windows](https://www.sciencedirect.com/science/article/pii/S0305054821001349) <br/>
[Gama R](https://scholar.google.com/citations?hl=en&user=uHKwsF0AAAAJ&view_op=list_works&sortby=pubdate), 
[Fernandes HL](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=JG7xb2AAAAAJ&sortby=pubdate) - Computers & Operations Research, 2021* 
  ([arXiv](https://arxiv.org/abs/2011.03647), [github](https://github.com/mustelideos/optw_rl))

```
@misc{gama&fernandes2020,
      title={A Reinforcement Learning Approach to the Orienteering Problem with Time Windows},
      author={Ricardo Gama and Hugo L. Fernandes},
      year={2020},
      eprint={2011.03647},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2011.03647}
}
```

![Learning and Inference](https://github.com/mustelideos/optw_rl/blob/main/images/figure_single_subset.png)

## Quick Usage:

This repo includes two already trained models:

* Cordeau's OPTW instance-region pr01, trained with uniform sampling:
```console
$ python inference_optw_rl.py --instance pr01  --model_name article --sample_type uni_samp
route: [0, 9, 24, 47, 12, 38, 30, 2, 32, 37, 10, 45, 11, 28, 1, 16, 36, 31, 35, 34, 22, 7, 0]
total score: 308
inference time: 173 ms
```
using CPU:
```console
$ python inference_optw_rl.py --instance pr01  --model_name article --sample_type uni_samp --device cpu
route: [0, 9, 24, 47, 12, 38, 30, 2, 32, 37, 10, 45, 11, 28, 1, 16, 36, 31, 35, 34, 22, 7, 0]
total score: 308
inference time: 1058 ms
```

* Gavalas' OPTW instance-region t101, trained with correlation sampling of scores:
```console
$ python inference_optw_rl.py --instance t101  --model_name article --sample_type corr_samp
route: [0, 68, 29, 8, 76, 3, 62, 7, 61, 16, 69, 17, 15, 33, 39, 44, 97, 46, 92, 74, 78, 0]
total score: 399
inference time: 343 ms
```

## Setup instructions

0. Install [Anaconda](https://www.anaconda.com/download/) (**Python 3 version**).

1. Clone this repo:
    ```console
    $ git clone https://github.com/mustelideos/optw_rl
    ```

2. Install the environment:
    ```console
    $ cd optw_rl/
    $ conda env create --file environment.yml
    ```

3. Activate the environment:
    ```console
    $ conda activate optw_env
    ```

## Inference
  #### On the Benchmark Instance
1. Using Beam Search (the default option) for Inference ("model")
    ```console
    $ python inference_optw_rl.py --instance t101 --model_name article --sample_type corr_samp
    route: [0, 68, 29, 8, 76, 3, 62, 7, 61, 16, 69, 17, 15, 33, 39, 44, 97, 46, 92, 74, 78, 0]
    total score: 399
    ```

2. Using Active Search followed by Beam Search ("model+as")
    ```console
    $ python inference_optw_rl.py --instance t101 --infe_type as_bs --model_name article --sample_type corr_samp
    route: [0, 68, 29, 8, 76, 3, 62, 7, 61, 24, 46, 97, 47, 16, 69, 17, 15, 21, 95, 92, 74, 0]
    total score: 400
    ```
  #### On the Generated Instances
0. Make sure to generate the validation set of tourist-region-instances first
    ```console
    $ python generate_instances.py --instance t101 --sample_type corr_samp
    ```
1. Using Beam Search for Inference ("model")
    ```console
    $ python inference_optw_rl.py --instance t101 --model_name article --generated --sample_type corr_samp
    average total score: 339.16
    ```
2. Using Active Search followed by Beam Search ("model+as")
    ```console
    $ python inference_optw_rl.py --instance t101 --infe_type as_bs --model_name article --generated --sample_type corr_samp
    average total score: 340.03
    ```

## Train a new Model and Infer

0. Make sure to generate the validation set of tourist-region-instances first. These are used to report performance during training.
      ```console
      $ python generate_instances.py --instance t101 --sample_type corr_samp
      ```
    
1. Choose a name, how many epochs and how often it is saved
      ```console
      $ python train_optw_rl.py --instance t101 --sample_type corr_samp --nepocs 1000 --nsave 1000 --model_name testing_1
      ```

2. Infer (using Beam Search, for instance) specifying the model name and the (saved) number of epochs
      ```console
      $  python inference_optw_rl.py --instance t101  --sample_type corr_samp --model_name testing_1 --saved_model_epoch 1000
      total score: 376
      ```

For optional arguments and default values:
```console
$ python train_optw_rl.py -h
(...)
optional arguments:
    -h, --help            show this help message and exit
  --instance INSTANCE   which instance to train on
  --device DEVICE       device to use (cpu/cuda)
  --use_checkpoint      use checkpoint (see
                        https://pytorch.org/docs/stable/checkpoint.html)
  --sample_type {uni_samp,corr_samp}
                        how to sample the scores of each point of interest:
                        uniformly sampled (uni_samp), score proportional to
                        each point of interest's duration of visit (corr_samp)
  --model_name MODEL_NAME
                        model name
  --debug               debug mode (verbose output and no saving)
  --nsave NSAVE         saves the model weights every <nsave> epochs
  --nprint NPRINT       to log and save the training history (total score in
                        the benchmark and generated instances of the
                        validation set) every <nprint> epochs
  --nepocs NEPOCS       number of training epochs
  --batch_size BATCH_SIZE
                        training batch size
  --max_grad_norm MAX_GRAD_NORM
                        maximum norm value for gradient value clipping
  --lr LR               initial learning rate
  --seed SEED           seed random # generators (for reproducibility)
  --beta BETA           entropy term coefficient
  --rnn_hidden RNN_HIDDEN
                        hidden size of RNN
  --n_layers N_LAYERS   number of attention layers in the encoder
  --n_heads N_HEADS     number heads in attention layers
  --ff_dim FF_DIM       hidden dimension of the encoder's feedforward sublayer
  --nfeatures NFEATURES
                        number of non-dynamic features
  --ndfeatures NDFEATURES
                        number of dynamic features
```
## Directory Structure
```bash
.
├── data
│   ├── benchmark
│   └── generated
├── images
├── results
│   ├── pr01
│   │   └── model_w
│   │       └── model_article_uni_samp
│   └── t101
│       └── model_w
│           └── model_article_corr_samp
└── src

```
