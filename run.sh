
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

#export CUDA_VISIBLE_DEVICES=0
#python3.6 -O main.py \
#  --batch-size 32 \
#  --workers 16 \
#  --lr 0.01 \
#  --epochs 100 \
#  --seed 11111 \
#  --sess brevitas_wsconv \
#  --export_finn \
#  $@

#export CUDA_VISIBLE_DEVICES=2
#python3 -O main.py \
#  --batch-size 128 \
#  --workers 16 \
#  --lr 0.005 \
#  --epochs 20 \
#  --seed 11111 \
#  --sess M5 \
#  --export_finn \
#  --optimizer SGD \
#  $@

#export CUDA_VISIBLE_DEVICES=2
#python3.6 -O main.py \
#  --batch-size 128 \
#  --workers 16 \
#  --lr 0.005 \
#  --epochs 20 \
#  --seed 11111 \
#  --sess M5_wsconv \
#  --export_finn \
#  --optimizer SGD \
#  $@

# for brevitas - finn intergration flow
export CUDA_VISIBLE_DEVICES=2
#date
#python3 -O main.py \
#  --batch-size 128 \
#  --workers 16 \
#  --lr 0.005 \
#  --epochs 100 \
#  --seed 11111 \
#  --sess M5 \
#  --export_finn \
#  --optimizer SGD \
#  --train
#date
#python3 KWS_OURS.py \
#  --ready_for_hls \
#date
#python3 -O main.py \
#  --batch-size 128 \
#  --workers 16 \
#  --lr 0.005 \
#  --epochs 2 \
#  --seed 11111 \
#  --sess M5 \
#  --export_finn \
#  --optimizer SGD \
#  --pynq \
#  --hls_test
#date
#python3 KWS_OURS.py \
#  --skip_ready_for_hls \
#  --cppsim
#date
#python3 -O main.py \
#  --batch-size 128 \
#  --workers 16 \
#  --lr 0.005 \
#  --epochs 2 \
#  --seed 11111 \
#  --sess M5 \
#  --export_finn \
#  --optimizer SGD \
#  --pynq \
#  --cppsim
date
python3 KWS_OURS.py \
  --skip_ready_for_hls \
  --skip_cppsim \
  --rtlsim
date
python3 -O main.py \
  --batch-size 128 \
  --workers 16 \
  --lr 0.005 \
  --epochs 2 \
  --seed 11111 \
  --sess M5 \
  --export_finn \
  --optimizer SGD \
  --pynq \
  --rtlsim
date
#python3 KWS_OURS.py \
#  --skip_ready_for_hls \
#  --skip_cppsim
#  --skip_rtlsim
#date
#python3 -O main.py \
#  --batch-size 128 \
#  --workers 16 \
#  --lr 0.005 \
#  --epochs 2 \
#  --seed 11111 \
#  --sess M5 \
#  --export_finn \
#  --optimizer SGD \
#  --pynq \
#  --fpga
#date
# end for brevitas - finn intergration flow

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

#export CUDA_VISIBLE_DEVICES=1
#python3.6 -O main.py \
#  --batch-size 256 \
#  --workers 16 \
#  --lr 0.02 \
#  --epochs 20 \
#  --duplicate 10 \
#  --seed 11111 \
#  --sess cnv_1w1a \
#  --export_finn \
#  --optimizer Adam \
#  $@

#export CUDA_VISIBLE_DEVICES=1
#python3.6 -O main.py \
#  --batch-size 256 \
#  --workers 16 \
#  --lr 0.1 \
#  --epochs 20 \
#  --duplicate 10 \
#  --seed 11111 \
#  --sess cnv_1w1a_wsconv \
#  --export_finn \
#  --optimizer SGD \
#  $@

#export CUDA_VISIBLE_DEVICES=1
#python3.6 -O main.py \
#  --batch-size 32 \
#  --workers 16 \
#  --seed 11111 \
#  --sess brevitas \
#  --resume \
#  $@

