sudo docker run --gpus 1 --rm -it -v ${PWD}:/home/workspace -v ${PWD}/../../SEM:/home/workspace/SEM -w /home/workspace pytorch_env/pytorch:1.4_py36_cu100 \
    python3 main.py --batch-size 4 --duplicate 20
