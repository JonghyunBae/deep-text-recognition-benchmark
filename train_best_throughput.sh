#!/bin/bash

trans='TPS'
feat='ResNet'
seq='BiLSTM'
pred='Attn'

for trans_al in $trans
do
    for feat_al in $feat
    do
        for seq_al in $seq
        do
            for pred_al in $pred
            do
                echo $trans_al $feat_al $seq_al $pred_al
                CUDA_VISIBLE_DEVICES=0 python3 train_best_throughput.py \
                --train_data ../data_lmdb_release/training --valid_data ../data_lmdb_release/validation \
                --select_data MJ-ST --batch_ratio 0.5-0.5  \
                --Transformation $trans_al --FeatureExtraction $feat_al --SequenceModeling $seq_al --Prediction $pred_al \
                --num_iter 100000 --batch_size 512 --eps 1e-5
            done
        done
    done
done

