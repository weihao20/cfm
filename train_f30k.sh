DATASET_NAME='f30k'
DATA_PATH='/tmp/data/'${DATASET_NAME}

CUDA_VISIBLE_DEVICES=0,1,2 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} \
  --logger_name runs/${DATASET_NAME}_butd_region_bert/log --model_name runs/${DATASET_NAME}_butd_region_bert \
  --num_epochs=25 --lr_update=15 --learning_rate=.0005 --precomp_enc_type basic --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1 \
  --rate_sa 0.33 --rate_img_exchange 0.8 --rate_txt_exchange 0.8 \
  --rate_se 0.33 --rate_img_drop 0.1 --rate_txt_drop 0.1 \
  --seed 1234