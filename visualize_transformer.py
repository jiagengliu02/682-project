import os
import gc
import argparse
import torchaudio
import torch
import torch.nn.functional as F

from torch import nn
from torchmetrics.text.wer import WordErrorRate
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from model import ConformerEncoder, LSTMDecoder, LinearDecoder
from utils import *

import matplotlib.pyplot as plt
import gc

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_parser():

    parser = argparse.ArgumentParser("conformer")
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="location to download data"
    )
    parser.add_argument(
        "--load_checkpoint",
        action="store_true",
        default=False,
        help="resume training from checkpoint",
    )
    parser.add_argument(
        "--train_set", type=str, default="train-clean-100", help="train dataset"
    )
    parser.add_argument("--test_set", type=str, default="test-clean", help="test dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--warmup_steps",
        type=float,
        default=10000,
        help="Multiply by sqrt(d_model) to get max_lr",
    )
    parser.add_argument(
        "--peak_lr_ratio",
        type=int,
        default=0.05,
        help="Number of warmup steps for LR scheduler",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id (optional)")
    parser.add_argument("--layers", type=int, default=8, help="total number of layers")
    parser.add_argument(
        "--model_path", type=str, default="saved_models", help="path to save the model"
    )
    parser.add_argument(
        "--use_amp", action="store_true", default=False, help="use mixed precision to train"
    )
    parser.add_argument(
        "--attention_heads",
        type=int,
        default=4,
        help="number of heads to use for multi-head attention",
    )
    parser.add_argument(
        "--d_input", type=int, default=80, help="dimension of the input (num filter banks)"
    )
    parser.add_argument(
        "--d_encoder", type=int, default=144, help="dimension of the encoder"
    )
    parser.add_argument(
        "--d_decoder", type=int, default=320, help="dimension of the decoder"
    )
    parser.add_argument(
        "--linear_decoder", action="store_true", default=False,
    )
    parser.add_argument(
        "--encoder_layers", type=int, default=16,
        help="number of conformer blocks in the encoder",
    )
    parser.add_argument(
        "--decoder_layers", type=int, default=1,
        help="number of decoder layers"
    )
    parser.add_argument(
        "--conv_kernel_size", type=int, default=31,
        help="size of kernel for conformer convolution blocks",
    )
    parser.add_argument(
        "--feed_forward_expansion_factor", type=int, default=4,
        help="expansion factor for conformer feed forward blocks",
    )
    parser.add_argument(
        "--feed_forward_residual_factor",
        type=int,
        default=0.5,
        help="residual factor for conformer feed forward blocks",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="dropout factor for conformer model"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
        help="model weight decay (corresponds to L2 regularization)",
    )
    parser.add_argument(
        "--variational_noise_std",
        type=float,
        default=0.0001,
        help="std of noise added to model weights for regularization",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="num_workers for the dataloader"
    )
    parser.add_argument(
        "--num_data", type=int, default=0, help="number of data for the dataloader"
    )
    parser.add_argument(
        "--smart_batch",
        action="store_true",
        default=False,
        help="Use smart batching for faster training",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        default=False,
        help="Use smart batching for faster training",
    )
    parser.add_argument(
        "--accumulate_iters",
        type=int,
        default=1,
        help="Number of iterations to accumulate gradients",
    )
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument(
        "--tokenize", "-t",
        action="store_true",
        default=False,
        help="Choose tokenlevel or letter level",
    )

    return parser



def plot_results(
    epochs, train_losses, valid_losses, train_wers, valid_wers, output_dir
):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_wers, label="Train WER")
    plt.plot(epochs, valid_wers, label="Valid WER")
    plt.xlabel("Epoch")
    plt.ylabel("WER (%)")
    plt.legend()
    plt.title("WER over Epochs")

    plt.tight_layout()
    plt.savefig(output_dir)
    plt.close()


def load_data(args):
    # Load Data
    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    train_data = torchaudio.datasets.LIBRISPEECH(
        root=args.data_dir, url=args.train_set, download=True
    )
    test_data = torchaudio.datasets.LIBRISPEECH(
        root=args.data_dir, url=args.test_set, download=True
    )

    if args.num_data > 0:
        train_data = Subset(train_data, range(args.num_data))
        test_data = Subset(test_data, range(args.num_data))

    train_preprocessor = Preprocessor(args.train_set, args.data_dir, args.tokenize)
    test_preprocessor = Preprocessor(args.test_set, args.data_dir, args.tokenize)

    train_loader = DataLoader(
        dataset=train_data,
        pin_memory=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_preprocessor.preprocess,
    )

    test_loader = DataLoader(
        dataset=test_data,
        pin_memory=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_preprocessor.preprocess,
    )
    return train_loader, test_loader

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    output_dir = os.path.join(args.output_path, f'Token{args.tokenize}-Linear{args.linear_decoder}-Data{args.num_data}-Hybrid{args.hybrid}')
    os.makedirs(output_dir, exist_ok=True)
    train_loader, test_loader = load_data(args)

    encoder = ConformerEncoder(
        d_input=args.d_input,
        d_model=args.d_encoder,
        num_layers=args.encoder_layers,
        conv_kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
        feed_forward_residual_factor=args.feed_forward_residual_factor,
        feed_forward_expansion_factor=args.feed_forward_expansion_factor,
        num_heads=args.attention_heads,
    )

    if args.tokenize:
        num_classes = 50257
        blank_id = 50256
    else:
        num_classes = 29
        blank_id = 28

    if args.linear_decoder:
        decoder = LinearDecoder(
            d_encoder=args.d_encoder,
            d_decoder=args.d_decoder,
            num_classes=num_classes,
        )
    else:
        decoder = LSTMDecoder(
            d_encoder=args.d_encoder,
            d_decoder=args.d_decoder,
            num_layers=args.decoder_layers,
            num_classes=num_classes,
        )
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
    for param in gpt_model.parameters():
        param.requires_grad = False

    char_decoder = GreedyCharacterDecoder().eval()
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=5e-4,
        betas=(0.9, 0.98),
        eps=1e-05 if args.use_amp else 1e-09,
        weight_decay=args.weight_decay,
    )
    scheduler = TransformerLrScheduler(optimizer, args.d_encoder, args.warmup_steps)

    # Print model size
    model_size(encoder, "Encoder")
    model_size(decoder, "Decoder")
    gc.collect()

    # GPU Setup
    if torch.cuda.is_available():
        print("Using GPU")
        gpu = True
        torch.cuda.set_device(args.gpu)
        criterion = criterion.cuda()
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        char_decoder = char_decoder.cuda()
        gpt_model = gpt_model.cuda()
        torch.cuda.empty_cache()
    else:
        gpu = False

    # Mixed Precision Setup
    if args.use_amp:
        print("Using Mixed Precision")

    _, _ = load_checkpoint(
        encoder, decoder, optimizer, scheduler,
        os.path.join(output_dir, "model_best.pt"),
    )

    encoder.eval()
    decoder.eval()
    batch = next(iter(test_loader))
    spectrograms, labels, input_lengths, label_lengths, references, mask = batch

    # Move to GPU
    if gpu:
        spectrograms = spectrograms.cuda()
        labels = labels.cuda()
        input_lengths = input_lengths.cuda()
        label_lengths = label_lengths.cuda()
        mask = mask.cuda()

    audio_attentions = encoder(spectrograms, mask, output_attentions=True)[1]
    print(type(audio_attentions), len(audio_attentions), audio_attentions[0].shape)
    token_attentions = gpt_model.transformer(input_ids=labels, output_attentions=True)[2]
    print(type(token_attentions), len(token_attentions), token_attentions[0].shape)

    for i in range


if __name__ == "__main__":
    main()
