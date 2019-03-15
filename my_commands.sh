#!/usr/bin/env bash


python train.py --gpu=0 --n_epoch=30 --batchsize 32 --save_name=ivat_classifier --lower=0 --use_adv=0 --xi_var=15.0 \
  --use_unlabled=1 --alpha=0.001 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 \
  --pretrained_model results_imdb_adaptive/best.model --use_exp_decay=1 --clip=5.0 --batchsize_semi 96 --use_semi_data 1 --use_attn_d 1 | tee classif_logs.txt


# Notes
# perp 110.85 for pretrained LM after 30H
# Kicked off training at 2:03 PM March 13
# Box is 7h ahead, in UTC


# Trained for a while (prolly 4 epochs and got to 93.7). So gunna 70% learning rate and keep going.

export MODEL_PATH="ivat_classifier_4_dev_acc_92.96750133191263_test_acc_93.704.model"


export MODEL_PATH="ivat_classifier_1_dev_acc_92.91422482685135_test_acc_93.76400000000001.model"
python train.py --gpu=0 --n_epoch=30 --save_name=ivat_classifier_finetune --lower=0 --use_adv=0 --xi_var=15.0 \
  --use_unlabled=1 --alpha=0.0005 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 \
  --use_exp_decay=1 --clip=5.0 --batchsize_semi 96 --use_semi_data 1 --use_attn_d 1 \
  --load_trained_lstm models/$MODEL_PATH | tee -a classif_logs_v3.txt


# On new box
python train.py --gpu=0 --n_epoch=30 --save_name=ivat_classifier --lower=0 --use_adv=0 --xi_var=15.0 \
  --use_unlabled=1 --alpha=0.0005 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 \
  --pretrained_model results_imdb_adaptive/best.model --use_exp_decay=1 --clip=5.0 --batchsize_semi 96 --use_semi_data 1 --use_attn_d 1 | tee classif_logs_v2.txt


# ORIG CMD

python train.py --gpu=0 --n_epoch=30 --batchsize 32 --save_name=imdb_model_vat --lower=0 --use_adv=0 --xi_var=15.0  --use_unlabled=1 --alpha=0.001 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 --pretrained_model imdb_pretrained_lm_ijcai.model --use_exp_decay=1 --clip=5.0 --batchsize_semi 96 --use_semi_data 1 --use_attn_d 1

# MY CMD
python train.py --gpu=0 --n_epoch=30 --batchsize 32 --save_name=imdb_model_vat_v2 --lower=0 --use_adv=0 --xi_var=15.0  --use_unlabled=1 --alpha=0.001 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 --pretrained_model best_lang_model_epoch_31.model --use_exp_decay=1 --clip=5.0 --batchsize_semi 96 --use_semi_data 1 --use_attn_d 1
