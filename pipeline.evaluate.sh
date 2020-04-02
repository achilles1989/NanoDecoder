#!/usr/bin/env bash

model_translate=$1
path_dataset=$2
gpu_id=$3

num_thread=40
path_nanodecoder=/data1/quanc/project/20181227_NanoDecoder/nanodecoder/
path_root=/data1/quanc/project/20181227_NanoDecoder/result/
path_save=$path_root/$path_dataset

## rrwick 10k read
#path_save=$path_root/data_test_rrwick_read10000/
## na12878 chr20 10k read
#path_save=$path_root/data_test_r94_na12878_chr20_read10000/
## ecoli 10k read
#path_save=$path_root/data_test_r94_loman_ecoli_read10000/
## toy data 40 read
#path_save=$path_root/data_test_r94_toy_dataset_read40
## real data CQ091
#path_save=$path_root/data_test_r94_CQ091
## modification
#path_save=/media/quanc/E/data/nanopore_modification/data_test_nanopolish_ecoli_10k_to_12k
#path_read=/media/quanc/E/data/nanopore_basecalling/train_raw/data_10000_12000/pcr_MSssI

# a subset of 500 reads
#path_save=$path_save/test_read500/

path_read=$path_save/01_raw_fast5/

if [ ! -d ${path_save}/02_basecalled_reads ];then
  mkdir ${path_save}/02_basecalled_reads
fi;

if [ $model_translate = 'albacore' ];then
  read_fast5_basecaller.py -c r94_450bps_linear.cfg -i $path_read -t $num_thread -s ${path_save}/result-albacore --disable_filtering -o fast5
  nanopolish extract -r -t template -o ${path_save}/02_basecalled_reads/albacore_v2.3.3 ${path_save}/result-albacore/workspace
  mv ${path_save}/02_basecalled_reads/albacore_v2.3.3.index.readdb ${path_save}/read_id_to_fast5
  rm ${path_save}/02_basecalled_reads/albacore_v2.3.3.index
  rm ${path_save}/02_basecalled_reads/albacore_v2.3.3.index.*
elif [ $model_translate = 'guppy' ];then
  guppy_basecaller -c /home/quanc/software/ont-guppy/data/dna_r9.4.1_450bps_hac.cfg -i $path_read -s ${path_save}/result-guppy --device 'cuda:0' --fast5_out
#  nanopolish extract -r -t template -o ${path_save}/02_basecalled_reads/albacore_v2.3.3 ${path_save}/result-albacore/workspace
#  mv ${path_save}/02_basecalled_reads/albacore_v2.3.3.index.readdb ${path_save}/read_id_to_fast5
#  rm ${path_save}/02_basecalled_reads/albacore_v2.3.3.index
#  rm ${path_save}/02_basecalled_reads/albacore_v2.3.3.index.*
elif [ $model_translate = 'scrappie' ];then
  scrappie raw --threads $num_thread -f FASTA -o ${path_save}/02_basecalled_reads/scrappie_v1.4.0 --model rgrgr_r94 $path_read
elif [ $model_translate = 'chiron' ];then
  chiron call -i $path_read -o ${path_save}/result-chiron -t $num_thread
  cat ${path_save}/result-chiron/result/*.fastq > ${path_save}/02_basecalled_reads/chiron_v0.4.2.fastq
  seqtk seq -a ${path_save}/02_basecalled_reads/chiron_v0.4.2.fastq > ${path_save}/02_basecalled_reads/chiron_v0.4.2
  rm ${path_save}/02_basecalled_reads/chiron_v0.4.2.fastq
elif [ $model_translate = 'nano' ];then

  ## Basecalling Models
  ## 1. PCR ecoli & na12878
  #path_model=/media/quanc/E/data/chiron_train/res_pytorch/data_ecoliANDna12878_nano/model/model-nano-brnn2trans-20190105_step_100000.pt
  ## 2. chiron ecoli & Lamda 512
  #path_model=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/model/model-nano-brnn2trans-20190118_step_100000.pt
  ## 3. chiron ecoli & Lamda 256
  #path_model=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/model/model-nano-brnn2trans-20190123_step_100000.pt

  #BLSTM2Transformer
  path_model=$path_root/data_train_r94_chiron_ecoliANDLamda/model/model-nano-nano2transformer-20190212_step_640000.pt
  #RNN2RNN
  #path_model=$path_root/data_train_r94_chiron_ecoliANDLamda/model/model-nano-nano2rnn-20190212_step_640000.pt
  #Modification
  #path_model=$path_root/data_train_nanopolish_ecoli_50k_to_300k_4000read/model/model-nano-nano2rnn-20190220_step_336000.pt

  ## 4. chiron ecoli & Lamda 256 LR 0.003
  #path_model=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/model/model-nano-brnn2transformer-20190202_step_100000.pt
  #path_model=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/model/model-nano-nano2transformer-20190211_step_100000.pt
  ## Methylate-calling models
  #path_model=/media/quanc/E/data/nanopore_modification/data_train_nanopolish_ecoli_50k_to_300k/model/model-nano-brnn2trans-20190128_step_5000.pt

  #array_max = (50 100)
  array_max=(100)
  #array_stride=(30 60 300)
  array_stride=(60)
  #array_beam=(1 2 5)
  array_beam=(5)
  model_embedding=256
  model_name=rnn2trans
  #
  for model_beam in ${array_beam[@]}; do
      for model_stride in ${array_stride[@]}; do
          for model_max in ${array_max[@]}; do

#          model_batch=2500
          model_batch=1400
  #        model_batch=800

          if [ ${model_beam} -eq 2 ]; then
              model_batch=1000
  #            model_batch=1000
  #            model_batch=500
          elif [ ${model_beam} -eq 5 ]; then
  #            model_batch=800
              model_batch=500
  #            model_batch=200
          fi
          model_prefix=${model_name}_embed${model_embedding}_seq300_stride${model_stride}_beam${model_beam}_max${model_max}_batch${model_batch}
          echo ${model_prefix};
          if [ ${gpu_id} -eq 0 ]; then
              echo "python translate.py -model $path_model -gpu $gpu_id -src_dir $path_read -save_data $path_save --src_seq_length 300 --src_seq_stride $model_stride --fast --beam_size $model_beam --max_length $model_max --batch_size $model_batch --thread $num_thread"
              python $path_nanodecoder/translate.py -model $path_model -gpu $gpu_id -src_dir $path_read -save_data $path_save --src_seq_length 300 --src_seq_stride $model_stride --fast --beam_size $model_beam --max_length $model_max --batch_size $model_batch --thread $num_thread
          elif [ ${gpu_id} -eq 1 ]; then
              echo "CUDA_VISIBLE_DEVICES=$gpu_id python translate.py -model $path_model -src_dir $path_read -gpu $gpu_id -save_data $path_save --src_seq_length 300 --src_seq_stride $model_stride --fast --beam_size $model_beam --max_length $model_max --batch_size $model_batch --thread $num_thread"
              CUDA_VISIBLE_DEVICES=$gpu_id python $path_nanodecoder/translate.py -model $path_model -src_dir $path_read -gpu $gpu_id -save_data $path_save --src_seq_length 300 --src_seq_stride $model_stride --fast --beam_size $model_beam --max_length $model_max --batch_size $model_batch --thread $num_thread
          fi
          mv ${path_save}/result ${path_save}/result-${model_prefix}
          mv ${path_save}/segment ${path_save}/segment-${model_prefix}
          mv ${path_save}/speed.txt ${path_save}/speed-${model_prefix}.txt
          paste --delimiters=\\n --serial ${path_save}/result-${model_prefix}/*.fasta > ${path_save}/02_02_basecalled_reads/${model_prefix}

          done
      done
  done
fi