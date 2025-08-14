import torch
import torch.nn.functional as F 

import math
from tqdm import tqdm

from dataset import create_dataloaders
from model import create_tacotron
from config import get_config

def prepare_spectrogram(mel_spectrograms, r, device, pad_value=-100.0):
    # Calculate the padding value
    pad_size = (r - mel_spectrograms.shape[1]) % r

    if pad_size > 0:
        padded_mel_spectrograms = F.pad(
            mel_spectrograms, mode='constant', value=pad_value,
            pad=(0, 0, 0, pad_size)
        )

        return padded_mel_spectrograms.to(device)
    else: # no need for padding
        return mel_spectrograms.to(device)
    

def train(m=None):
    # Setup the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataloaders
    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(annotations_file="data/metadata.csv", device=device, base_path="./data/")

    config = get_config()

    start_epoch = 0

    # create the model
    tacotron = create_tacotron(cfg=config, device=device)
    if m is not None:
        data = torch.load(f=m, map_location=device)
        model_state_dict = data["model_state_dict"]
        optimizer_state_dict = data["optimizer_state_dict"]
        scheduler_state_dict = data["scheduler_state_dict"]
        
        # Load model parameters
        tacotron.load_state_dict(model_state_dict)

        # Optimizer setup
        optimizer = torch.optim.Adam(tacotron.parameters(), lr=5e-4)
        optimizer.load_state_dict(optimizer_state_dict)

        current_lr = optimizer.param_groups[0]['lr']

        # Scheduler setup
        total_steps = config["epochs"] * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2)
        scheduler.load_state_dict(scheduler_state_dict)
        
        start_epoch = data["epoch"]

    else:
        optimizer = torch.optim.Adam(tacotron.parameters(), lr=5e-4)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, total_steps=len(train_dataloader)*config["epochs"], pct_start=0.35, anneal_strategy='cos', div_factor=25, final_div_factor=100)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2)
    end_epoch = start_epoch + config["epochs"] # Train as much as specified

    loss_fn = torch.nn.L1Loss()
    loss_fn_stop_token = torch.nn.BCEWithLogitsLoss()

    # Start the training
    prev_loss = float("inf")
    for epoch in range(start_epoch, end_epoch):
        print(f"Epoch: {epoch}")
        train_loss = 0
        test_loss = 0
        step = 0
        print("-----------------------------------")
        print(f"Training Loop | Epoch: {epoch}")
        batch_loader = tqdm(train_dataloader)
        for batch in batch_loader:
            tacotron.train()

            spectrograms = batch["spectrogram"].to(device)
            mel_spectrograms = batch["mel_spectrogram"].to(device)
            texts = batch["text"].to(device)
            target_stop_tokens = batch["target_stop_token"].to(device)

            # prepare mel spectrogram for loss calculation
            mel_spectrograms = mel_spectrograms.transpose(1, 2)
            mel_spectrograms = prepare_spectrogram(mel_spectrograms, r=5, device=device)

            spectrograms = spectrograms.transpose(1, 2)
            spectrograms = prepare_spectrogram(spectrograms, r=5, device=device, pad_value=0.001)

            # Forward pass
            model_output = tacotron(x=texts, y=mel_spectrograms)
            
            # Calculate the loss
            min_len = min(mel_spectrograms.shape[1], model_output["mel_outputs"].shape[1])
            mel_loss = loss_fn(model_output["mel_outputs"][:, :min_len, :], mel_spectrograms[:, :min_len, :])
            lin_loss = loss_fn(model_output["linear_outputs"], spectrograms)


            stop_token_loss = loss_fn_stop_token(model_output["stop_token_outputs"], target_stop_tokens)

            total_loss = lin_loss + mel_loss + stop_token_loss
            batch_loader.set_postfix({f"Loss": f"{total_loss}"})

            train_loss += total_loss

            # Optimizer zero_grad 
            optimizer.zero_grad()

            # Loss backward
            total_loss.backward()
            
            # Optimizer step
            torch.nn.utils.clip_grad_norm_(tacotron.parameters(), max_norm=1.0) # Or another suitable value
            optimizer.step()

            # Scheduler step
            scheduler.step()

            # Update the step
            step += 1

        train_loss /= step
        print(f"Train Loss: {train_loss}")
        if epoch % 2 == 0:
            # Do the inference
            step = 0
            tacotron.eval()
            with torch.inference_mode():
                print(f"Testing Loop | Epoch: {epoch}")
                test_batch_loader = tqdm(valid_dataloader)
                for test_batch in test_batch_loader:
                    test_mel_spectrograms = test_batch["mel_spectrogram"].to(device)
                    test_texts = test_batch["text"].to(device)
                    test_spectrograms = test_batch["spectrogram"].to(device)
                    test_target_stop_tokens = test_batch["target_stop_token"].to(device)

                    test_mel_spectrograms = test_mel_spectrograms.transpose(1, 2)
                    test_mel_spectrograms = prepare_spectrogram(test_mel_spectrograms, r=5, device=device)

                    test_spectrograms = test_spectrograms.transpose(1, 2)
                    test_spectrograms = prepare_spectrogram(test_spectrograms, r=5, device=device, pad_value=0.001)
                    # Do the forward pass
                    test_model_output = tacotron(x=test_texts, y=test_mel_spectrograms)

                    # Calculate the loss
                    t_mel_loss = loss_fn(test_model_output["mel_outputs"], test_mel_spectrograms)
                    t_lin_loss = loss_fn(test_model_output["linear_outputs"], test_spectrograms)
                    t_stop_token_loss = loss_fn_stop_token(test_model_output["stop_token_outputs"], test_target_stop_tokens)

                    t_total_loss = t_mel_loss + t_lin_loss + t_stop_token_loss

                    test_batch_loader.set_postfix({f"Loss": f"{t_total_loss}"})

                    test_loss += t_total_loss

                    step += 1
            
                test_loss /= step
                print(f"Test Loss: {test_loss}")

            # Save the model
            if train_loss < prev_loss:
                prev_loss = train_loss
                model_name = f"me{epoch}l{math.floor(train_loss)}.pth"
                torch.save(
                    obj={
                        "train_loss" : train_loss,
                        "test_loss" : test_loss,
                        "optimizer_state_dict" : optimizer.state_dict(),
                        "scheduler_state_dict" : scheduler.state_dict(),
                        "lr" : scheduler.get_last_lr(),
                        "model_state_dict" : tacotron.state_dict(),
                        "epoch" : epoch
                    },
                    f=f"models/{model_name}"
                )
            
    return tacotron

if __name__ == "__main__":
    train()
