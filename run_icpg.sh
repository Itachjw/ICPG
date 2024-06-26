DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name icpg \
--img_aug \
--MLM \
--batch_size 64 \
--dataset_name $DATASET_NAME \
--loss_names 'itc+cdm+chm' \
--num_epoch 60