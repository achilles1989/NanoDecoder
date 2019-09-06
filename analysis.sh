#!/usr/bin/env bash

path_save=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_test_rrwick_read10000/test_read500
#path_save=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_test_r94_na12878_chr20_read10000/test_read500
#path_save=/media/quanc/E/data/chiron_train/res_pytorch/data_test_toy_dataset_read40
#path_save=/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_test_r94_loman_ecoli_read10000/test_read500

path_reference=${path_save}/reference.fasta
num_thread=12

#array_prefix=(albacore_v2.3.3 scrappie_v1.4.0 chiron_v0.4.2 nanodecoder_seq300_stride30_beam1_max100_batch800 nanodecoder_seq300_stride30_beam2_max100_batch500 nanodecoder_seq300_stride60_beam1_max100_batch800 nanodecoder_seq300_stride60_beam2_max100_batch500 nanodecoder_seq300_stride300_beam1_max100_batch800 nanodecoder_seq300_stride300_beam2_max100_batch500)
#array_prefix=(albacore_v2.3.3 scrappie_v1.4.0 chiron_v0.4.2 nanodecoder_seq300_stride60_beam1_max100_batch800 nanodecoder_seq300_stride60_beam2_max100_batch500)
#array_prefix=(albacore_v2.3.3 scrappie_v1.4.0 chiron_v0.4.2 nanodecoder_embed512_seq300_stride30_beam1_max100_batch800 nanodecoder_embed512_seq300_stride30_beam2_max100_batch500 nanodecoder_embed512_seq300_stride60_beam1_max100_batch800 nanodecoder_embed512_seq300_stride60_beam2_max100_batch500 nanodecoder_embed512_seq300_stride300_beam1_max100_batch800 nanodecoder_embed512_seq300_stride300_beam2_max100_batch500)
array_prefix=(nanodecoder_seq300_stride30_beam2_max100_batch500)
#array_prefix=(nanodecoder_seq300_stride30_beam1_max100_batch800 nanodecoder_seq300_stride30_beam2_max100_batch500 nanodecoder_seq300_stride60_beam1_max100_batch800 nanodecoder_seq300_stride300_beam1_max100_batch800)

for model_prefix in ${array_prefix[@]}; do

    if [ -f ${path_save}/02_basecalled_reads/${model_prefix} ]; then
        echo $model_prefix;
        printf "\n"
        minimap2 -ax map-ont -t $num_thread $path_reference ${path_save}/02_basecalled_reads/${model_prefix} > ${path_save}/02_basecalled_reads/${model_prefix}.sam
        samtools view -@ $num_thread -b -o ${path_save}/02_basecalled_reads/${model_prefix}.bam ${path_save}/02_basecalled_reads/${model_prefix}.sam
        jsa.hts.errorAnalysis --bamFile ${path_save}/02_basecalled_reads/${model_prefix}.bam --reference $path_reference

        minimap2 -x map-ont -t $num_thread -c $path_reference ${path_save}/02_basecalled_reads/${model_prefix} > ${path_save}/02_basecalled_reads/${model_prefix}.paf
        python3 scripts/my_read_length_identity.py ${path_save}/02_basecalled_reads/${model_prefix} ${path_save}/02_basecalled_reads/${model_prefix}.paf

        rm ${path_save}/02_basecalled_reads/${model_prefix}.sam
        rm ${path_save}/02_basecalled_reads/${model_prefix}.bam
        rm ${path_save}/02_basecalled_reads/${model_prefix}.paf
    fi

done


#array_prefix=(nanodecoder-seq300-stride300-beam1-max50 nanodecoder-seq300-stride300-beam2-max50 nanodecoder-seq300-stride30-beam1-max50 nanodecoder-seq300-stride30-beam2-max50 nanodecoder-seq300-stride60-beam1-max50 nanodecoder-seq300-stride60-beam2-max50)
#
#for model_prefix in ${array_prefix[@]}; do
#
#    echo $model_prefix;
#    printf "\n"
#    minimap2 -ax map-ont -t $num_thread $path_reference ${path_save}/02_basecalled_reads/${model_prefix} > ${path_save}/02_basecalled_reads/${model_prefix}.sam
#    samtools view -@ $num_thread -b -o ${path_save}/02_basecalled_reads/${model_prefix}.bam ${path_save}/02_basecalled_reads/${model_prefix}.sam
#    jsa.hts.errorAnalysis --bamFile ${path_save}/02_basecalled_reads/${model_prefix}.bam --reference $path_reference
#
#    minimap2 -x map-ont -t $num_thread -c $path_reference ${path_save}/02_basecalled_reads/${model_prefix} > ${path_save}/02_basecalled_reads/${model_prefix}.paf
#    python3 /media/quanc/D/Workspace/github/Basecalling-comparison/my_read_length_identity.py ${path_save}/02_basecalled_reads/${model_prefix} ${path_save}/02_basecalled_reads/${model_prefix}.paf
#
#    rm ${path_save}/02_basecalled_reads/${model_prefix}.sam
#    rm ${path_save}/02_basecalled_reads/${model_prefix}.bam
#    rm ${path_save}/02_basecalled_reads/${model_prefix}.paf
#
#done