
#find sd_GSCmdV2/ -type f -name '*.csv.npy' -delete

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#python3.6 -O main.py \
#  --batch-size 8 \
#  --duplicate 20 \
#  --COT \
#  --sess COT_session

export CUDA_VISIBLE_DEVICES=0
python3.6 -O main.py \
  --batch-size 32 \
  --workers 8 \
  --lr 0.1 \
  --epochs 20 \
  --seed 11111 \
  --sess baseline

#export CUDA_VISIBLE_DEVICES=0
#python3.6 main.py \
#  --batch-size 8 \
#  --sess debug
