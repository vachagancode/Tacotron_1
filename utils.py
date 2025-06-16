import torch
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from dataset import tokenize_data, _all_chars
from model import create_tacotron
from config import get_config

def reform_model(path : str, model_name : str):
    # Load the model data
    data = torch.load(f=path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Save the model with only state_dict
    torch.save(obj={data["model_state_dict"]}, f=f"models/{model_name}")

def predict_spectrogram(model, input_text, device):
    model.to(device)
    
    # Tokenize the input_text
    try:
        tokenized = tokenize_data(input_text)
        tokenized = torch.tensor(tokenized, device=device)
        tokenized = tokenized.unsqueeze(0)
    except:
        print(f"Only follwing symbols are allowed to be included in the text:")
        print(_all_chars)

    model.eval()
    with torch.inference_mode():
        model_output = model(tokenized)

        return {
            "mel_spectrogram" : model_output["mel_outputs"],
            "linear_spectrogram" : model_output["linear_outputs"]
        }
    
def plot_spectrogram(spectrogram, spec_type='linear'):
    spectrogram = spectrogram.detach().cpu().numpy()

    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectrogram, x_axis='time', y_axis='linear' if spec_type == 'linear' else 'log', ax=ax, n_fft=1024)
    ax.set(title=f"{'Mel-' if spec_type == 'log' else ''}Spectrogram generated")
    fig.colorbar(img, ax=ax, format='%+2.f dB')

    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config()
    # Load model data
    data = torch.load("./me34l6.pth", map_location=device)

    tacotron = create_tacotron(config, device)
    tacotron.load_state_dict(data["model_state_dict"])

    input_text = "Hello !"

    predicted_outputs = predict_spectrogram(tacotron, input_text, device=device)
    # Plot a spectrogram
    plot_spectrogram(predicted_outputs["mel_spectrogram"].squeeze(0).transpose(0, 1), spec_type='log')