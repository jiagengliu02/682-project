import os
import gc
import argparse
import torchaudio
import torch
import torch.nn.functional as F

from torch import nn
from torchmetrics.text.wer import WordErrorRate
from torch.cuda.amp import autocast, GradScaler
from model import ConformerEncoder, LSTMDecoder
from utils import *

import data
import importlib
from data.transforms import (
        Compose, AddLengths, AudioSqueeze, TextPreprocess,
        MaskSpectrogram, ToNumpy, BPEtexts, MelSpectrogram,
        ToGpu, Pad, NormalizedMelSpectrogram
)
from audiomentations import (
    TimeStretch, PitchShift, AddGaussianNoise
)

import youtokentome as yttm
import yaml
from easydict import EasyDict as edict




from torch.utils.data import Dataset, DataLoader, BatchSampler
class WAVLibriSpeech(Dataset):
    def __init__(self, directory, subset):
        # subset might be 'train-clean-100', 'test-clean', etc.
        self.directory = os.path.join(directory, subset)
        self.files = []
        for root, _, filenames in os.walk(self.directory):
            for filename in filenames:
                if filename.endswith(".wav"):
                    self.files.append(os.path.join(root, filename))

    def __getitem__(self, index):
        filepath = self.files[index]
        waveform, sample_rate = torchaudio.load(filepath)
        # Any additional processing can be done here
        return waveform, sample_rate

    def __len__(self):
        return len(self.files)

def no_pad_collate(batch):
    keys = batch[0].keys()
    collated_batch = {key: [] for key in keys}
    for key in keys:
        items = [item[key] for item in batch]
        collated_batch[key] = items
    return collated_batch

def prepare_bpe(config):
    dataset_module = importlib.import_module(f'.{config.dataset.name}', data.__name__)
    # train BPE
    if config.bpe.get('train', False):
        dataset, ids = dataset_module.get_dataset(config, part='bpe', transforms=TextPreprocess())
        train_data_path = 'bpe_texts.txt'
        with open(train_data_path, "w") as f:
            # run ovefr only train part
            for i in ids:
                text = dataset.get_text(i)
                f.write(f"{text}\n")
        yttm.BPE.train(data=train_data_path, vocab_size=config.model.vocab_size, model=config.bpe.model_path)
        os.system(f'rm {train_data_path}')

    bpe = yttm.BPE(model=config.bpe.model_path)
    return bpe

def main(args):

  # train_data = WAVLibriSpeech(directory=args.data_dir, subset=args.train_set)
  # test_data = WAVLibriSpeech(directory=args.data_dir, subset=args.test_set)
  config = args.config
  bpe = prepare_bpe(config)
  transforms_train = Compose([
            TextPreprocess(),
            ToNumpy(),
            BPEtexts(bpe=bpe, dropout_prob=config.bpe.get('dropout_prob', 0.05)),
            AudioSqueeze(),
            AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.015,
                p=0.5
            ),
            TimeStretch(
                min_rate=0.8,
                max_rate=1.25,
                p=0.5
            ),
            PitchShift(
                min_semitones=-4,
                max_semitones=4,
                p=0.5
            )
            # AddLengths()
    ])
  
  transforms_val = Compose([
            TextPreprocess(),
            ToNumpy(),
            BPEtexts(bpe=bpe),
            AudioSqueeze()
    ])

  dataset_module = importlib.import_module(f'.{config.dataset.name}', data.__name__)
  train_dataset = dataset_module.get_dataset(config, transforms=transforms_train, part='train')
  val_dataset = dataset_module.get_dataset(config, transforms=transforms_val, part='val')

  train_loader = DataLoader(train_dataset, num_workers=config.train.get('num_workers', 4),
                batch_size=config.train.get('batch_size', 1), collate_fn=no_pad_collate)

  val_loader = DataLoader(val_dataset, num_workers=config.train.get('num_workers', 4),
                batch_size=1, collate_fn=no_pad_collate)
  
  # if args.smart_batch:
  #   print('Sorting training data for smart batching...')
  #   sorted_train_inds = [ind for ind, _ in sorted(enumerate(train_data), key=lambda x: x[1][0].shape[1])]
  #   sorted_test_inds = [ind for ind, _ in sorted(enumerate(test_data), key=lambda x: x[1][0].shape[1])]
  #   train_loader = DataLoader(dataset=train_data,
  #                                   pin_memory=True,
  #                                   num_workers=args.num_workers,
  #                                   batch_sampler=BatchSampler(sorted_train_inds, batch_size=args.batch_size, drop_last=True),
  #                                   collate_fn=lambda x: preprocess_example(x, 'train'))

  #   test_loader = DataLoader(dataset=test_data,
  #                               pin_memory=True,
  #                               num_workers=args.num_workers,
  #                               batch_sampler=BatchSampler(sorted_test_inds, batch_size=args.batch_size, drop_last=True),
  #                               collate_fn=lambda x: preprocess_example(x, 'valid'))
  # else:
  #   train_loader = DataLoader(dataset=train_data,
  #                                   pin_memory=True,
  #                                   num_workers=args.num_workers,
  #                                   batch_size=args.batch_size,
  #                                   shuffle=True,
  #                                   collate_fn=lambda x: preprocess_example(x, 'train'))

  #   test_loader = DataLoader(dataset=test_data,
  #                               pin_memory=True,
  #                               num_workers=args.num_workers,
  #                               batch_size=args.batch_size,
  #                               shuffle=False,
  #                               collate_fn=lambda x: preprocess_example(x, 'valid'))

  # Declare Models  
  
  encoder = ConformerEncoder(
                      d_input=args.d_input,
                      d_model=args.d_encoder,
                      num_layers=args.encoder_layers,
                      conv_kernel_size=args.conv_kernel_size, 
                      dropout=args.dropout,
                      feed_forward_residual_factor=args.feed_forward_residual_factor,
                      feed_forward_expansion_factor=args.feed_forward_expansion_factor,
                      num_heads=args.attention_heads)
  
  decoder = LSTMDecoder(
                  d_encoder=args.d_encoder, 
                  d_decoder=args.d_decoder, 
                  num_layers=args.decoder_layers)
  char_decoder = GreedyCharacterDecoder().eval()
  criterion = nn.CTCLoss(blank=28, zero_infinity=True)
  optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-4, betas=(.9, .98), eps=1e-05 if args.use_amp else 1e-09, weight_decay=args.weight_decay)
  scheduler = TransformerLrScheduler(optimizer, args.d_encoder, args.warmup_steps)

  # Print model size
  model_size(encoder, 'Encoder')
  model_size(decoder, 'Decoder')

  gc.collect()

  print("???",train_loader)

  # GPU Setup
  if torch.cuda.is_available():
    print('Using GPU')
    gpu = True
    torch.cuda.set_device(args.gpu)
    criterion = criterion.cuda()
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    char_decoder = char_decoder.cuda()
    torch.cuda.empty_cache()
  else:
    gpu = False

  # Mixed Precision Setup
  if args.use_amp:
    print('Using Mixed Precision')
  grad_scaler = GradScaler(enabled=args.use_amp)

  # Initialize Checkpoint 
  if args.load_checkpoint:
    start_epoch, best_loss = load_checkpoint(encoder, decoder, optimizer, scheduler, args.checkpoint_path)
    print(f'Resuming training from checkpoint starting at epoch {start_epoch}.')
  else:
    start_epoch = 0
    best_loss = float('inf')

  # Train Loop
  optimizer.zero_grad()
  for epoch in range(start_epoch, args.epochs):
    torch.cuda.empty_cache()

    #variational noise for regularization
    add_model_noise(encoder, std=args.variational_noise_std, gpu=gpu)
    add_model_noise(decoder, std=args.variational_noise_std, gpu=gpu)

    # Train/Validation loops
    wer, loss = train(encoder, decoder, char_decoder, optimizer, scheduler, criterion, grad_scaler, train_loader, args, gpu=gpu) 
    valid_wer, valid_loss = validate(encoder, decoder, char_decoder, criterion, val_loader, args, gpu=gpu)
    print(f'Epoch {epoch} - Valid WER: {valid_wer}%, Valid Loss: {valid_loss}, Train WER: {wer}%, Train Loss: {loss}')  

    # Save checkpoint 
    # if valid_loss <= best_loss:
    #   print('Validation loss improved, saving checkpoint.')
    #   best_loss = valid_loss
    #   save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch+1, args.checkpoint_path)
import librosa
def audio_to_spectrograms(audios, sample_rate, max_length=None):
    spectrograms = []
    for a in audios:        
        # 计算短时傅里叶变换（STFT）
        D = librosa.stft(a, n_fft=2048, hop_length=512)
        
        # 将幅度谱转换为分贝（dB）单位
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # 如果没有指定最大长度，使用第一个频谱图的长度
        if max_length is None:
            max_length = S_db.shape[1]
        
        # 填充或截断频谱图
        if S_db.shape[1] < max_length:
            pad_width = max_length - S_db.shape[1]
            S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            S_db = S_db[:, :max_length]

        # 将频谱图转换为tensor
        spectrogram = torch.tensor(S_db)
        
        # 添加到列表中
        spectrograms.append(spectrogram)
    
    # 将列表转换为tensor
    spectrograms = torch.stack(spectrograms)
    
    return spectrograms

def train(encoder, decoder, char_decoder, optimizer, scheduler, criterion, grad_scaler, train_loader, args, gpu=True):
  ''' Run a single training epoch '''

  wer = WordErrorRate()
  error_rate = AvgMeter()
  avg_loss = AvgMeter()
  text_transform = TextTransform()

  encoder.train()
  decoder.train()
  
  # print("!!!",encoder)
  
  for i, batch in enumerate(train_loader):
    scheduler.step()
    gc.collect()
    print(batch.keys())
    audios, text, sample_rate = batch['audio'], batch['text'], batch['sample_rate']

    spectrograms = audio_to_spectrograms(audios, sample_rate)
    input_lengths = len(audios)
    label_lengths = len(text)
    labels = text
    references = text
    mask = None
    
    # spectrograms, labels, input_lengths, label_lengths, references, mask = batch 

    # Move to GPU
    if gpu:
      spectrograms = spectrograms.cuda()
      labels = torch.tensor(labels).cuda()
      input_lengths = torch.tensor(input_lengths).cuda()
      label_lengths = torch.tensor(label_lengths).cuda()
      mask = mask.cuda()
    
    # Update models
    with autocast(enabled=args.use_amp):
      outputs = encoder(spectrograms, mask)
      print(outputs)
      outputs = decoder(outputs)
      print(outputs)
      loss = criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)
    grad_scaler.scale(loss).backward()
    if (i+1) % args.accumulate_iters == 0:
      grad_scaler.step(optimizer)
      grad_scaler.update()
      optimizer.zero_grad()
    avg_loss.update(loss.detach().item())

    # Predict words, compute WER
    inds = char_decoder(outputs.detach())
    predictions = []
    for sample in inds:
      predictions.append(text_transform.int_to_text(sample))
    error_rate.update(wer(predictions, references) * 100)

    # Print metrics and predictions 
    if (i+1) % args.report_freq == 0:
      print(f'Step {i+1} - Avg WER: {error_rate.avg}%, Avg Loss: {avg_loss.avg}')   
      print('Sample Predictions: ', predictions)
    del spectrograms, labels, input_lengths, label_lengths, references, outputs, inds, predictions
  return error_rate.avg, avg_loss.avg

def validate(encoder, decoder, char_decoder, criterion, test_loader, args, gpu=True):
  ''' Evaluate model on test dataset. '''

  avg_loss = AvgMeter()
  error_rate = AvgMeter()
  wer = WordErrorRate()
  text_transform = TextTransform()

  encoder.eval()
  decoder.eval()
  for i, batch in enumerate(test_loader):
    gc.collect()
    spectrograms, labels, input_lengths, label_lengths, references, mask = batch 
  
    # Move to GPU
    if gpu:
      spectrograms = spectrograms.cuda()
      labels = labels.cuda()
      input_lengths = torch.tensor(input_lengths).cuda()
      label_lengths = torch.tensor(label_lengths).cuda()
      mask = mask.cuda()

    with torch.no_grad():
      with autocast(enabled=args.use_amp):
        outputs = encoder(spectrograms, mask)
        outputs = decoder(outputs)
        loss = criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)
      avg_loss.update(loss.item())

      inds = char_decoder(outputs.detach())
      predictions = []
      for sample in inds:
        predictions.append(text_transform.int_to_text(sample))
      error_rate.update(wer(predictions, references) * 100)
  return error_rate.avg, avg_loss.avg


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Training model.')
  parser.add_argument('--config', default='configs/train_librispeech.yaml',
                      help='path to config file')
  parser.add_argument('--data_dir', type=str, default='./data', help='location to download data')
  parser.add_argument('--checkpoint_path', type=str, default='model_best.pt', help='path to store/load checkpoints')
  parser.add_argument('--load_checkpoint', action='store_true', default=False, help='resume training from checkpoint')
  parser.add_argument('--train_set', type=str, default='train-clean-100', help='train dataset')
  parser.add_argument('--test_set', type=str, default='test-clean', help='test dataset')
  parser.add_argument('--batch_size', type=int, default=32, help='batch size')
  parser.add_argument('--warmup_steps', type=float, default=10000, help='Multiply by sqrt(d_model) to get max_lr')
  parser.add_argument('--peak_lr_ratio', type=int, default=0.05, help='Number of warmup steps for LR scheduler')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id (optional)')
  parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
  parser.add_argument('--report_freq', type=int, default=100, help='training objective report frequency')
  parser.add_argument('--layers', type=int, default=8, help='total number of layers')
  parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
  parser.add_argument('--use_amp', action='store_true', default=False, help='use mixed precision to train')
  parser.add_argument('--attention_heads', type=int, default=4, help='number of heads to use for multi-head attention')
  parser.add_argument('--d_input', type=int, default=80, help='dimension of the input (num filter banks)')
  parser.add_argument('--d_encoder', type=int, default=144, help='dimension of the encoder')
  parser.add_argument('--d_decoder', type=int, default=320, help='dimension of the decoder')
  parser.add_argument('--encoder_layers', type=int, default=16, help='number of conformer blocks in the encoder')
  parser.add_argument('--decoder_layers', type=int, default=1, help='number of decoder layers')
  parser.add_argument('--conv_kernel_size', type=int, default=31, help='size of kernel for conformer convolution blocks')
  parser.add_argument('--feed_forward_expansion_factor', type=int, default=4, help='expansion factor for conformer feed forward blocks')
  parser.add_argument('--feed_forward_residual_factor', type=int, default=.5, help='residual factor for conformer feed forward blocks')
  parser.add_argument('--dropout', type=float, default=.1, help='dropout factor for conformer model')
  parser.add_argument('--weight_decay', type=float, default=1e-6, help='model weight decay (corresponds to L2 regularization)')
  parser.add_argument('--variational_noise_std', type=float, default=.0001, help='std of noise added to model weights for regularization')
  parser.add_argument('--num_workers', type=int, default=2, help='num_workers for the dataloader')
  parser.add_argument('--smart_batch', type=bool, default=True, help='Use smart batching for faster training')
  parser.add_argument('--accumulate_iters', type=int, default=1, help='Number of iterations to accumulate gradients')
  args = parser.parse_args()
  with open(args.config, 'r') as f:
      config = edict(yaml.safe_load(f))
  args.config = config
  main(args)