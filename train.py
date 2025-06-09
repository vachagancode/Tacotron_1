import torch

import math
from tqdm import tqdm

from dataset import create_dataloaders
from model import create_tacotron
from config import get_config

def train():
    # TODO : Create a pre-trained model adding feature

    # Setup the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataloaders
    train_dataloader, test_dataloader = create_dataloaders(annotations_file="data/metadata.csv", device=device, base_path="./data/")

    config = get_config()

    # create the model
    tacotron = create_tacotron(cfg=config, device=device)

    optimizer = torch.optim.Adam(tacotron.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=len(train_dataloader)*config["epochs"], pct_start=0.35)

    loss_fn = torch.nn.L1Loss()
    # Start the training

    prev_loss = float("inf")
    for epoch in range(config["epochs"]):
        print(f"Epoch: {epoch}")
        train_loss = 0
        test_loss = 0
        step = 0
        print("-----------------------------------")
        print(f"Training Loop | Epoch: {epoch}")
        batch_loader = tqdm(test_dataloader)
        for batch in batch_loader:
            tacotron.train()

            spectrograms = batch["spectrogram"].to(device)
            mel_spectrograms = batch["mel_spectrogram"].to(device)
            texts = batch["text"].to(device)

            # Forward pass
            mel_spectrograms = mel_spectrograms.transpose(1, 2)
            spectrograms = spectrograms.transpose(1, 2)

            model_output = tacotron(x=texts, y=mel_spectrograms)
            
            # Calculate the los
            mel_loss = loss_fn(model_output["mel_outputs"], mel_spectrograms)
            lin_loss = loss_fn(model_output["linear_outputs"], spectrograms)

            total_loss = lin_loss + mel_loss

            batch_loader.set_postfix({f"Loss": f"{total_loss}"})

            train_loss += total_loss

            # Optimizer zero_grad 
            optimizer.zero_grad()

            # Loss backward
            total_loss.backward()
            
            # Optimizer step
            optimizer.step()

            # Scheduler step
            scheduler.step()

            # Update the step
            step += 1

        train_loss /= step
        
        if epoch % 2 == 0:
            # Do the inference
            step = 0
            tacotron.eval()
            with torch.inference_mode():
                print(f"Testing Loop | Epoch: {epoch}")
                for test_batch in tqdm(test_dataloader):
                    test_mel_spectrograms = test_batch["mel_spectrogram"].to(device)
                    test_texts = test_batch["text"].to(device)
                    test_spectrograms = test_batch["spectrogram"].to(device)

                    test_mel_spectrograms = test_mel_spectrograms.transpose(1, 2)
                    test_spectrograms = test_spectrograms.transpose(1, 2)
                    # Do the forward pass
                    test_model_output = tacotron(x=test_texts, y=test_mel_spectrograms)

                    # Calculate the loss
                    t_mel_loss = loss_fn(test_model_output["mel_outputs"], test_mel_spectrograms)
                    t_lin_loss = loss_fn(test_model_output["linear_outputs"], test_spectrograms)

                    t_total_loss = t_mel_loss + t_lin_loss

                    test_loss += t_total_loss

                    step += 1
            
                test_loss /= step

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