
#find sd_GSCmdV2/ -type f -name '*.bins.npy' -delete
#find sd_GSCmdV2/ -type f -name '*.csv' -delete

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#python3.6 -O main.py \
#  --batch-size 8 \
#  --duplicate 20 \
#  --COT \
#  --sess COT_session

#export CUDA_VISIBLE_DEVICES=0
#python3.6 -O main.py \
#  --batch-size 32 \
#  --workers 16 \
#  --lr 0.1 \
#  --epochs 50 \
#  --seed 11111 \
#  --sess baseline \
#  $@

#export CUDA_VISIBLE_DEVICES=0,1
#python3.6 main.py \
#  --batch-size 32 \
#  --workers 16 \
#  --lr 0.1 \
#  --epochs 100 \
#  --seed 22222 \
#  --sess brevitas \
#  --export_finn \
#  $@

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#python3.6 -O main.py \
#  --batch-size 32 \
#  --workers 16 \
#  --lr 0.1 \
#  --epochs 100 \
#  --seed 11111 \
#  --sess brevitas \
#  --export_finn \
#  $@

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#python3.6 -O main.py \
#  --batch-size 32 \
#  --workers 16 \
#  --lr 0.1 \
#  --epochs 100 \
#  --seed 11111 \
#  --sess brevitas_wsconv \
#  --export_finn \
#  $@

export CUDA_VISIBLE_DEVICES=0
python3.6 -O main.py \
  --batch-size 128 \
  --workers 16 \
  --lr 0.01 \
  --epochs 5 \
  --seed 11111 \
  --sess M5 \
  --export_finn \
  --optimizer SGD \
  $@

export CUDA_VISIBLE_DEVICES=0
python3.6 -O main.py \
  --batch-size 128 \
  --workers 16 \
  --lr 0.01 \
  --epochs 5 \
  --seed 11111 \
  --sess M5_wsconv \
  --export_finn \
  --optimizer SGD \
  $@

#export CUDA_VISIBLE_DEVICES=1
#python3.6 -O main.py \
#  --batch-size 128 \
#  --workers 16 \
#  --lr 0.01 \
#  --epochs 5 \
#  --seed 11111 \
#  --sess M11 \
#  --export_finn \
#  --optimizer Adam \
#  $@

#export CUDA_VISIBLE_DEVICES=2
#python3.6 -O main.py \
#  --batch-size 128 \
#  --workers 16 \
#  --lr 0.1 \
#  --epochs 30 \
#  --seed 11111 \
#  --sess end2end \
#  --export_finn \
#  --optimizer SGD \
#  $@

#python3.6 -O main.py \
#  --batch-size 32 \
#  --workers 16 \
#  --seed 11111 \
#  --sess brevitas \
#  --resume \
#  $@
