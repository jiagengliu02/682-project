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

from transformers import GPT2Tokenizer 

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
    parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
    parser.add_argument(
        "--report_freq", type=int, default=100, help="training objective report frequency"
    )
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
        "--accumulate_iters",
        type=int,
        default=1,
        help="Number of iterations to accumulate gradients",
    )
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument(
        "--tokenize",
        "-t",
        action="store_true",
        default=False,
        help="Choose tokenlevel or letter level",
    )

    return parser



def save_results_to_file(
    epoch, train_wer, train_loss, valid_wer, valid_loss, file_path
):
    with open(file_path, "a") as f:
        f.write(
            f"Epoch {epoch} - Valid WER: {valid_wer}%, Valid Loss: {valid_loss}, Train WER: {train_wer}%, Train Loss: {train_loss}\n"
        )


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

    if args.smart_batch:
        print("Sorting training data for smart batching...")
        sorted_train_inds = [
            ind
            for ind, _ in sorted(enumerate(train_data), key=lambda x: x[1][0].shape[1])
        ]
        sorted_test_inds = [
            ind
            for ind, _ in sorted(enumerate(test_data), key=lambda x: x[1][0].shape[1])
        ]
        train_loader = DataLoader(
            dataset=train_data,
            pin_memory=True,
            num_workers=args.num_workers,
            batch_sampler=BatchSampler(sorted_train_inds, batch_size=args.batch_size),
            collate_fn=train_preprocessor.preprocess,
        )

        test_loader = DataLoader(
            dataset=test_data,
            pin_memory=True,
            num_workers=args.num_workers,
            batch_sampler=BatchSampler(sorted_test_inds, batch_size=args.batch_size),
            collate_fn=test_preprocessor.preprocess,
        )
    else:
        train_loader = DataLoader(
            dataset=train_data,
            pin_memory=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
            # collate_fn=lambda x: train_preprocessor.preprocess(x, args.tokenize),
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
    
    output_dir = os.path.join(args.output_path, f'Token{args.tokenize}-Linear{args.linear_decoder}-Data{args.num_data}')
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
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # GPU Setup
    if torch.cuda.is_available():
        print("Using GPU")
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
        print("Using Mixed Precision")
    grad_scaler = GradScaler(enabled=args.use_amp)

    # Initialize Checkpoint
    if args.load_checkpoint:
        start_epoch, best_loss = load_checkpoint(
            encoder, decoder, optimizer, scheduler,
            os.path.join(output_dir, "model_best.pt"),
        )
        print(f"Resuming training from checkpoint starting at epoch {start_epoch}.")
    else:
        start_epoch = 0
        best_loss = float("inf")

    # Train Loop
    optimizer.zero_grad()

    # Initialize lists to store results
    epochs = []
    train_losses, train_wers = [], []
    valid_losses, valid_wers = [], []
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()

        # variational noise for regularization
        add_model_noise(encoder, std=args.variational_noise_std, gpu=gpu)
        add_model_noise(decoder, std=args.variational_noise_std, gpu=gpu)

        # Train/Validation loops
        wer, loss = train(
            encoder, decoder, char_decoder,
            optimizer, scheduler, grad_scaler,
            criterion, train_loader,
            args, tokenizer,
            gpu=gpu,
        )
        valid_wer, valid_loss = validate(
            encoder, decoder, char_decoder,
            criterion, test_loader,
            args, tokenizer, output_dir,
            gpu=gpu,
        )
        print(
            f"Epoch {epoch} - Valid WER: {valid_wer}%, Valid Loss: {valid_loss}, Train WER: {wer}%, Train Loss: {loss}"
        )
        # Save results to file
        save_results_to_file(
            epoch, wer, loss, valid_wer, valid_loss,
            os.path.join(output_dir, "result.txt"),
        )

        # Store results for plotting
        epochs.append(epoch)
        train_losses.append(loss)
        valid_losses.append(valid_loss)
        train_wers.append(wer)
        valid_wers.append(valid_wer)

        if epoch % 20 == 0:
            save_checkpoint(
                encoder, decoder, optimizer, scheduler, valid_loss, epoch + 1,
                os.path.join(output_dir, f"{epoch}.pt"),
            )
        # Save checkpoint
        if valid_loss <= best_loss:
            best_loss = valid_loss
            save_checkpoint(
                encoder, decoder, optimizer, scheduler, valid_loss, epoch + 1,
                os.path.join(output_dir, "model_best.pt"),
            )

    plot_results(
        epochs, train_losses, valid_losses, train_wers, valid_wers,
        os.path.join(output_dir, "result.png"),
    )


def train(
    encoder, decoder, char_decoder,
    optimizer, scheduler, grad_scaler,
    criterion, train_loader,
    args, tokenizer,
    gpu=True,
):
    """Run a single training epoch"""

    wer = WordErrorRate()
    error_rate = AvgMeter()
    avg_loss = AvgMeter()
    text_transform = TextTransform()

    encoder.train()
    decoder.train()
    for i, batch in enumerate(train_loader):
        scheduler.step()
        gc.collect()
        spectrograms, labels, input_lengths, label_lengths, references, mask = batch

        # Move to GPU
        if gpu:
            spectrograms = spectrograms.cuda()
            labels = labels.cuda()
            input_lengths = input_lengths.cuda()
            label_lengths = label_lengths.cuda()
            mask = mask.cuda()

        # Update models
        with autocast(enabled=args.use_amp):
            outputs = encoder(spectrograms, mask)
            outputs = decoder(outputs)
            outputs = F.log_softmax(outputs, dim=-1)

            loss = criterion(
                outputs.transpose(0, 1),
                labels,
                ((input_lengths - 1) // 2 - 1) // 2, # changed
                label_lengths,
            )

        grad_scaler.scale(loss).backward()
        if (i + 1) % args.accumulate_iters == 0:
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
        avg_loss.update(loss.detach().item())

        predictions = []
        if not args.tokenize:
            # Predict words, compute WER
            inds = char_decoder(outputs.detach())
            
            for sample in inds:
                predictions.append(text_transform.int_to_text(sample))
            error_rate.update(wer(predictions, references) * 100)
        else:
            inds = torch.argmax(outputs.detach(), dim=-1)
            for sample in inds:
                predicted_token = tokenizer.decode(sample, skip_special_tokens=True)
                predictions.append(predicted_token)
            error_rate.update(wer(predictions, references) * 100)

        # Print metrics and predictions
        if (i + 1) % args.report_freq == 0:
            # print('\t', outputs.shape, ((input_lengths - 1) // 2 - 1) // 2, input_lengths)
            # print('\t', labels.shape, label_lengths)
            print(f"Step {i + 1} - Avg WER: {error_rate.avg}%, Avg Loss: {avg_loss.avg}")
            print("Sample Predictions: ")
            for j in range(5):
                print(f"\t Prediction: [{predictions[j]}]\t Reference: [{references[j]}]")
        del (
            spectrograms,
            labels,
            input_lengths,
            label_lengths,
            references,
            outputs,
            inds,
            predictions,
        )
    return error_rate.avg, avg_loss.avg


def validate(
    encoder, decoder, char_decoder,
    criterion, test_loader,
    args, tokenizer, output_dir,
    gpu=True
):
    """Evaluate model on test dataset."""

    wer = WordErrorRate()
    avg_loss = AvgMeter()
    error_rate = AvgMeter()
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
            input_lengths = input_lengths.cuda()
            label_lengths = label_lengths.cuda()
            mask = mask.cuda()

        with torch.no_grad():
            with autocast(enabled=args.use_amp):
                outputs = encoder(spectrograms, mask)
                outputs = decoder(outputs)
                outputs = F.log_softmax(outputs, dim=-1)

                loss = criterion(
                    outputs.transpose(0, 1),
                    labels,
                    ((input_lengths - 1) // 2 - 1) // 2, # changed
                    label_lengths,
                )

            avg_loss.update(loss.item())

            predictions = []
            if not args.tokenize:
                # Predict words, compute WER
                inds = char_decoder(outputs.detach())
                
                for sample in inds:
                    predictions.append(text_transform.int_to_text(sample))
                error_rate.update(wer(predictions, references) * 100)
            else:
                inds = torch.argmax(outputs.detach(), dim=-1)
                for sample in inds:
                    predicted_token = tokenizer.decode(sample, skip_special_tokens=True)
                    predictions.append(predicted_token)
                error_rate.update(wer(predictions, references) * 100)

            # inds = char_decoder(outputs.detach())
            # predictions = []
            # for sample in inds:
            #     predictions.append(text_transform.int_to_text(sample))
            # error_rate.update(wer(predictions, references) * 100)

    with open(os.path.join(output_dir, "result.txt"), "a") as f:
        f.write(f"predictions: {predictions}" + "\n" + f"references: {references}")
    return error_rate.avg, avg_loss.avg

if __name__ == "__main__":
    main()
