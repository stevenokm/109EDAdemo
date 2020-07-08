#sudo docker run --gpus 1 --rm -it -v ${PWD}:/home/workspace -w /home/workspace pytorch_env/pytorch:1.4_py36_cu100 python3 main.py -b 8 --duplicate 20 --epochs 1 
#sleep 1
sudo docker run --gpus 1 --rm -it -v ${PWD}:/home/workspace -w /home/workspace pytorch_env/pytorch:1.4_py36_cu101 python3 main.py -b 8 --duplicate 20 --epochs 1
#sleep 1
#sudo docker run --gpus 1 --rm -it -v ${PWD}:/home/workspace -w /home/workspace pytorch_env/pytorch:1.4_py36_cu102 python3 main.py -b 8 --duplicate 20 --epochs 1
