#!/bin/bash
# pipeline to train

#path_src=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/datasets/data-chiron-update
path_src=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/datasets/data-chiron-mix

embed_model=256
array_encoder=(nano resnet)
array_decoder=(transformer rnn)

for name_encoder in ${array_encoder[@]}; do
    for name_decoder in ${array_decoder[@]}; do
        name_model=model-nano-${name_encoder}2${name_decoder}-20190211
        path_save=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/model/${name_model}
        path_log=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/logs/${name_model}-step-100000.log
        echo ${name_model};
        echo ${path_save};
        echo ${path_log};
#        if [ ${name_decoder} = rnn ]; then
#            python train.py -encoder_type ${name_encoder} -decoder_type ${name_decoder} -tgt_word_vec_size 256 -enc_rnn_size 256 -dec_rnn_size 256 -audio_enc_pooling 1 -dropout 0 -enc_layers 3 -dec_layers 3 -data ${path_src} -save_model ${path_save} -log_file ${path_log} -global_attention mlp -batch_size 50 -optim adam -max_grad_norm 100 -learning_rate 0.0003 -learning_rate_decay 0.8 -train_steps 100000 -gpu_ranks 0 -train_from /media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/model/model-nano-resnet2rnn-20190131_step_40000.pt
#        elif [ ${name_decoder} = transformer ]; then
#            python train.py -encoder_type ${name_encoder} -decoder_type ${name_decoder} -tgt_word_vec_size 256 -enc_rnn_size 256 -dec_rnn_size 256 -audio_enc_pooling 1 -dropout 0 -enc_layers 3 -dec_layers 3 -data ${path_src} -save_model ${path_save} -log_file ${path_log} -global_attention mlp -batch_size 50 -optim adam -max_grad_norm 100 -learning_rate 0.0003 -learning_rate_decay 0.8 -train_steps 100000 -gpu_ranks 0
#        fi

        str_command="python train.py -encoder_type ${name_encoder} -decoder_type ${name_decoder} -tgt_word_vec_size ${embed_model} -enc_rnn_size ${embed_model} -dec_rnn_size ${embed_model} -audio_enc_pooling 1 -dropout 0 -enc_layers 3 -dec_layers 3 -data ${path_src} -save_model ${path_save} -log_file ${path_log} -global_attention mlp -batch_size 50 -optim adam -max_grad_norm 100 -learning_rate 0.003 -start_decay_steps 32000 --decay_steps 32000 -learning_rate_decay 0.5 -train_steps 100000 -gpu_ranks 0"
        echo ${str_command}

        python train.py -encoder_type ${name_encoder} -decoder_type ${name_decoder} -tgt_word_vec_size ${embed_model} -enc_rnn_size ${embed_model} -dec_rnn_size ${embed_model} -audio_enc_pooling 1 -dropout 0 -enc_layers 3 -dec_layers 3 -data ${path_src} -save_model ${path_save} -log_file ${path_log} -global_attention mlp -batch_size 50 -optim adam -max_grad_norm 100 -learning_rate 0.003 -start_decay_steps 32000 --decay_steps 32000 -learning_rate_decay 0.5 -train_steps 100000 -gpu_ranks 0

    done
done

#python train.py -encoder_type nano -decoder_type transformer -tgt_word_vec_size 256 -enc_rnn_size 256 -dec_rnn_size 256 -audio_enc_pooling 1 -dropout 0 -enc_layers 3 -dec_layers 3 -data /media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/datasets/data-chiron-mix -save_model /media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/model/model-nano-brnn2transformer-20190202 -log_file /media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/logs/model-nano-brnn2transformer-20190202-step-100000-to-200000.log -global_attention mlp -batch_size 50 -optim adam -max_grad_norm 100 -learning_rate 0.0003 -learning_rate_decay 0.8 -train_steps 100000 -gpu_ranks 0 -train_from /media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/model/model-nano-brnn2transformer-20190202_step_100000.pt
#python train.py -encoder_type nano -decoder_type rnn -tgt_word_vec_size 256 -enc_rnn_size 256 -dec_rnn_size 256 -audio_enc_pooling 1 -dropout 0 -enc_layers 3 -dec_layers 3 -data /media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/datasets/data-chiron-mix -save_model /media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/model/model-nano-brnn2rnn-20190203 -log_file /media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/logs/model-nano-brnn2rnn-20190203-step-200000.log -global_attention mlp -batch_size 50 -optim adam -max_grad_norm 100 -learning_rate 0.0003 -learning_rate_decay 0.8 -train_steps 100000 -gpu_ranks 0 -train_from /media/quanc/E/data/nanopore_basecalling/res_pytorch/data_train_r94_chiron_ecoliANDLamda/model/model-nano-brnn2transformer-20190202_step_100000.pt