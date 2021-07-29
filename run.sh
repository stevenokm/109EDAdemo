export CUDA_VISIBLE_DEVICES=1

#python3.6 -O main.py \
#  --batch-size 8 \
#  --duplicate 20 \
#  --COT \
#  --sess COT_session

python3.6 -O main.py \
  --batch-size 16 \
  --duplicate 20 \
  --sess baseline
