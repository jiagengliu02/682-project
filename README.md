# Pytorch Conformer
Pytorch implementation of [conformer](https://arxiv.org/abs/2005.08100) model with training script for end-to-end speech recognition on the LibriSpeech dataset.

## Environment Setup
```bash
conda create -n conformer python==3.10 -y
conda activate conformer
nvidia-smi
# make sure that your cuda version is higher than 12.1
```

```bash
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchaudio==2.2.0
pip install torchmetrics
```

```bash
conda install ffmpeg
python
import torchaudio
torchaudio.list_audio_backends()
# check printed list, if it's empty:
pip install soundfile==0.12.1
```

## Usage


### Train model from scratch:
```
python train.py --data_dir=./data --train_set=train-clean-100 --test_set=test_clean --checkpoint_path=model_best.pt
```
### Resume training from checkpoint
```
python train.py --load_checkpoint --checkpoint_path=model_best.pt
```
### Train with mixed precision: 
```
python train.py --use_amp
```


