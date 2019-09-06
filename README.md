# NanoDecoder: attention based seq2seq basecaller for nanopore reads

Table of Contents
=================
  * [Requirements](#requirements)
  * [Features](#features)
  * [Quickstart](#quickstart)
  
## Requirements
## Features
- RNN2RNN
- Conv2Conv
- Attention is all you need (Transformer2Transformer)
- NanoDecoder Model (LSTM2Transformer)
## Quickstart
### Step 1: Preprocess the data
```bash
python preprocess.py --prefix demo --src_dir data/ --basecall_group RawGenomeCorrected_000 --save_data data_save/ --src_seq_length 512 --tgt_seq_length 100
```
### Step 2: Train the model
```bash
python train.py -encoder_type transformer -decoder_type transformer -tgt_word_vec_size 256 -enc_rnn_size 256 -dec_rnn_size 256 -audio_enc_pooling 1 -dropout 0 -enc_layers 3 -dec_layers 3 -rnn_type LSTM -data data/ -save_model data_save/ -global_attention mlp -gpu_ranks 0 -batch_size 50 -optim adam -max_grad_norm 100 -learning_rate 0.0003 -learning_rate_decay 0.8 -train_steps 10000
```
### Step 3: Translate (Basecalling)
```bash
python translate.py -model model/demo-step-10000.pt -src_dir data/ -gpu 0 -verbose -save_data data_save/ --src_seq_length 512 --fast --attn_debug
```