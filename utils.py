import torchaudio
import torch
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt


class TextTransform:
    """Map characters to integers and vice versa"""
    def __init__(self):
        self.char_map = {chr(char): i for i, char in enumerate(range(65, 91))}
        self.char_map["'"] = 26
        self.char_map[" "] = 27
        self.index_map = {i: char for char, i in self.char_map.items()}

    def text_to_int(self, text):
        """Map text string to an integer sequence"""
        return [self.char_map[c] for c in text]

    def int_to_text(self, labels):
        """Map integer sequence to text string"""
        return "".join([self.index_map[i] for i in labels if i != 28])


def get_audio_transforms():
    time_masks = [
        torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)
    ]
    train_audio_transform = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
        *time_masks,
    )
    valid_audio_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_mels=80, hop_length=160
    )

    return train_audio_transform, valid_audio_transform


class BatchSampler(object):
    """Sample contiguous, sorted indices. Leads to less padding and faster training."""
    def __init__(self, sorted_inds, batch_size):
        self.sorted_inds = sorted_inds
        self.batch_size = batch_size

    def __iter__(self):
        inds = self.sorted_inds.copy()
        while len(inds):
            to_take = min(self.batch_size, len(inds))
            start_ind = random.randint(0, len(inds) - to_take)
            batch_inds = inds[start_ind : start_ind + to_take]
            del inds[start_ind : start_ind + to_take]
            yield batch_inds

def preprocess_example(data, data_type="train"):
    """Process raw LibriSpeech examples"""
    text_transform = TextTransform()
    train_audio_transform, valid_audio_transform = get_audio_transforms()
    spectrograms, labels, references, input_lengths, label_lengths = [], [], [], [], []

    for waveform, _, utterance, _, _, _ in data:
        spec = (
            train_audio_transform(waveform).squeeze(0).transpose(0, 1) if data_type == "train" else
            valid_audio_transform(waveform).squeeze(0).transpose(0, 1)
        )
        spectrograms.append(spec)
        labels.append(torch.Tensor(text_transform.text_to_int(utterance)))
        references.append(utterance)
        # input_lengths.append(((spec.shape[0] - 1) // 2 - 1) // 2)
        input_lengths.append(spec.shape[0])
        label_lengths.append(len(labels[-1]))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    mask = torch.ones(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1])
    for i, l in enumerate(input_lengths):
        mask[i, :, :l] = 0

    return spectrograms, labels, input_lengths, label_lengths, references, mask.bool()


class TransformerLrScheduler():
    """
    Transformer LR scheduler from "Attention is all you need." https://arxiv.org/abs/1706.03762
    multiplier and warmup_steps taken from conformer paper: https://arxiv.org/abs/2005.08100
    """
    def __init__(self, optimizer, d_model, warmup_steps, multiplier=5):
        self._optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.n_steps = 0
        self.multiplier = multiplier

    def step(self):
        self.n_steps += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self):
        return (
            self.multiplier
            * (self.d_model**-0.5)
            * min(self.n_steps ** (-0.5), self.n_steps * (self.warmup_steps ** (-1.5)))
        )

def model_size(model, name):
    """Print model size in num_params and MB"""
    param_size, num_params, buffer_size = 0, 0, 0
    for param in model.parameters():
        num_params += param.nelement()
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        num_params += buffer.nelement()
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f"{name} - num_params: {round(num_params / 1e6, 2)}M, size: {round(size_all_mb, 2)}MB")


class GreedyCharacterDecoder(nn.Module):
    """Greedy CTC decoder - Argmax logits and remove duplicates."""

    def __init__(self):
        super(GreedyCharacterDecoder, self).__init__()

    def forward(self, x):
        indices = torch.argmax(x, dim=-1)
        return torch.unique_consecutive(indices, dim=-1).tolist()


class AvgMeter:
    """Keep running average for a metric"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = None
        self.sum = None
        self.cnt = 0

    def update(self, val, n=1):
        self.sum = val * n if self.sum is None else self.sum + val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def view_spectrogram(sample):
    """View spectrogram"""
    specgram = sample.transpose(1, 0)
    plt.figure()
    p = plt.imshow(specgram.log2()[:, :].detach().numpy(), cmap="gray")
    plt.show()


def add_model_noise(model, std=0.0001, gpu=True):
    """
    Add variational noise to model weights: https://ieeexplore.ieee.org/abstract/document/548170
    STD may need some fine tuning...
    """
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.size()).cuda() * std if gpu else torch.randn(param.size()) * std
            param.add_(noise)


def load_checkpoint(encoder, decoder, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint does not exist")
    checkpoint = torch.load(checkpoint_path)
    scheduler.n_steps = checkpoint["scheduler_n_steps"]
    scheduler.multiplier = checkpoint["scheduler_multiplier"]
    scheduler.warmup_steps = checkpoint["scheduler_warmup_steps"]
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["valid_loss"]


def save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch, checkpoint_path):
    """Save model checkpoint"""
    torch.save(
        {
            "epoch": epoch,
            "valid_loss": valid_loss,
            "scheduler_n_steps": scheduler.n_steps,
            "scheduler_multiplier": scheduler.multiplier,
            "scheduler_warmup_steps": scheduler.warmup_steps,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )