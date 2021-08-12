# 109 2nd-year EDA project DEMO


## Overview

This repository contains the PyTorch implementation of 109 2nd-year EDA project DEMO.

## Dependencies

Please refer to [requirements.txt](requirements.txt)

## Usage
For test results for alexnet_brevitas:

```bash
python3.6 -O main.py \
   	--batch-size 32 \
   	--workers 16 \
   	--seed 11111 \
   	--sess brevitas \
   	--resume
```
	
For training alexnet_brevitas and export finn-onnx model:

```bash
python3.6 -O main.py \
	--batch-size 32 \
   	--workers 16 \
   	--seed 11111 \
	--lr 0.1 \
	--epochs 100 \
   	--sess brevitas \
   	--train \
	--export_finn
```

## Benchmark on Google Speech Commands (12-cmd)

The following table shows the train accuracy and test accuracy in a 100-epoch training session.

| Model            | Training | Testing  | Checkpoint |
|:-----------------|:--------:|:--------:|:-----------|
| alexnet_brevitas | 76.263%  | 73.794%  | [ckpt.t7.brevitas_11111.pth](https://drive.google.com/file/d/1WoDzrueavxXudQ4rvTz0vob-v5QbPYQv/view?usp=sharing) |

To use pre-trained checkpoint, please download checkpoint `.pth` files into [checkpoint](checkpoint) folder.

## Acknowledgement
The project is adapted from the [COT](https://github.com/henry8527/COT) repository by [henry8527](https://github.com/henry8527).

The models are adapted from the [BinaryNet.pytorch](https://github.com/itayhubara/BinaryNet.pytorch) repository by [itayhubara](https://github.com/itayhubara), [M5](https://arxiv.org/abs/1610.00087) refer from [Speech Command Recognition with torchaudio](https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio_tutorial.html)

The dataset is adapted from the [Speech Command Recognition](https://github.com/douglas125/SpeechCmdRecognition) repository by [douglas125](https://github.com/douglas125)
