import os
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd

_pad = '_'
_eos = '~' # End of sequence token
_punctuation = '!\'(),-.:;?[]" ' # Common punctuation
_alphabets = 'abcdefghijklmnopqrstuvwxyz' # Lowercase alphabets
_all_chars = ['p', 'r', 'i', 'n', 't', 'g', ',', ' ', 'h', 'e', 'o', 'l', 'y', 's', 'w', 'c', 'a', 'd', 'f', 'm', 'x', 'b', 'v', '.', 'u', 'k', 'j', '"', '-', ';', '(', 'z', ')', ':', "'", 'q', '!', '?', '|', 'â', 'é', 'à', 'ê', 'ü', 'è', '“', '”', '`', '[', ']', '’']
_all_chars = _all_chars + list(_pad) + list(_eos)
char2id = {char: i for i, char in enumerate(_all_chars)}
id2char = {i: char for i, char in enumerate(_all_chars)}
def tokenize_data(text):
    tokenized = []
    for t in text.lower():
        if t == "'":
            tokenized.append(char2id["'"])
        elif t == '"':
            tokenized.append(char2id['"'])
        else:
            tokenized.append(char2id[t])

    return tokenized

class TacotronDataset(Dataset):
    def __init__(self, annotations_file, device, base_path):
        super().__init__()
        self.annotations_file = annotations_file
        self.device = device

        self.target_size = 500000

        self.sample_rate = 22050

        self.df = pd.read_csv(annotations_file, sep="|")
        self.audio_path = os.path.join(base_path, "wavs/")

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            hop_length=256,
        )
    
    def _pad_or_trim_waveform(self, waveform):
        if waveform.shape[-1] < self.target_size:
            zero_pad = torch.zeros(waveform.shape[0], self.target_size - waveform.shape[-1])
            waveform = torch.cat([waveform, zero_pad], dim=1)
            return waveform
        elif waveform.shape[-1] > self.target_size:
            return waveform[:, :self.target_size]

    def _get_all_characters(self):
        chars = []
        for i in range(len(self.df)):
            print(i)
            print(self.df["ftext"].iloc[i])
            if not isinstance(self.df["ftext"].iloc[i], float):
                for t in self.df["ftext"].iloc[i]:
                    if t.lower() not in chars:
                        chars.append(t.lower())
            else:
                for t in self.df["text"].iloc[i]:
                    if t.lower() not in chars:
                        chars.append(t.lower())
        
        return chars, len(chars)

    def _calculate_average_target_size(self):
        i = 0
        s = 0
        
        try:
            while True:
                audio_name = f"{self.df['name'].iloc[i]}.wav"
                audio_path = os.path.join(self.audio_path, audio_name)
                waveform, _ = torchaudio.load(audio_path)
                s += waveform.shape[-1]
                i += 1
        except:
            s /= i
            s = int(s)
            print(f"Average Target Size: {s}")


    def __getitem__(self, idx):
        audio_name = f"{self.df['name'].iloc[idx]}.wav"
        formatted_text = self.df['ftext'].iloc[idx] if not isinstance(self.df['ftext'].iloc[idx], float) else self.df['text'].iloc[idx]

        tokenized_text = torch.tensor(tokenize_data(formatted_text))

        audio_path = os.path.join(self.audio_path, audio_name)

        # Load the audio
        waveform, sr = torchaudio.load(audio_path)

        # Apply Formatting
        # if waveform.shape[-1] != self.target_size:
        #     waveform = self._pad_or_trim_waveform(waveform)

        mel_spectrogram = self.mel_transform(waveform).squeeze(0)
        mel_spectrogram = self.amp_to_db(mel_spectrogram)

        spectrogram = self.spec_transform(waveform).squeeze(0)

        # Stop token target tensor
        mspec_length  = torch.tensor(mel_spectrogram.shape[-1]) # get the length of time steps
        r = 5

        target_stop_token = torch.zeros(int(torch.ceil(mspec_length / r).item()))
        true_length_index = (mspec_length - 1) // r
        target_stop_token[true_length_index] = 1.0
        
        return spectrogram, mel_spectrogram, tokenized_text, target_stop_token, mspec_length

    def __len__(self):
        return len(self.df)
    
def collate_data(batch):
    spectrograms, mel_spectrograms, text_sequences, target_stop_token, mspec_length = zip(*batch)

    # Text padding
    text_lengths = torch.tensor([len(text) for text in text_sequences], dtype=torch.long)
    max_text_length = max(text_lengths)
    padded_text_sequences = torch.full(
        (len(text_sequences), max_text_length),
        fill_value=char2id[_pad],
        dtype=torch.long
    )

    for i, seq in enumerate(text_sequences):
        padded_text_sequences[i, :len(seq)] = seq

    # Mel-Spectrogram padding
    mel_lengths = torch.tensor([spec.shape[-1] for spec in mel_spectrograms], dtype=torch.long)
    max_mel_length = max(mel_lengths)
    padded_mel_spectrograms = torch.full(
        (len(mel_spectrograms), mel_spectrograms[0].shape[0], max_mel_length),
        fill_value=-100.0,
        dtype=torch.float32
    )

    for i, mel_spec in enumerate(mel_spectrograms):
        padded_mel_spectrograms[i, :, :mel_spec.shape[-1]] = mel_spec

    spec_lengths = torch.tensor([spec.shape[-1] for spec in spectrograms])
    max_spec_length = max(spec_lengths)
    padded_spectrograms = torch.full(
        (len(spectrograms), spectrograms[0].shape[0], max_spec_length),
        fill_value=0.001,
        dtype=torch.float32
    )

    for i, spec in enumerate(spectrograms):
        padded_spectrograms[i, :, :spec.shape[-1]] = spec

    # Pad the target stop token
    padded_target_stop_tokens = pad_sequence(target_stop_token, batch_first=True, padding_value=0.0)
    padded_target_stop_tokens = padded_target_stop_tokens.unsqueeze(-1)

    return {"spectrogram" : padded_spectrograms,"mel_spectrogram": padded_mel_spectrograms, "text": padded_text_sequences, "target_stop_token": padded_target_stop_tokens, "mspec_length": mspec_length}

def create_dataloaders(annotations_file, device, base_path):
    dataset = TacotronDataset(annotations_file, device, base_path)

    train_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=8,
        pin_memory=True,
        collate_fn=collate_data
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        shuffle=False,
        batch_size=8,
        pin_memory=True,
        collate_fn=collate_data
    )

    return train_dataloader, test_dataloader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TacotronDataset(annotations_file="data/metadata.csv", device=device, base_path="./data/")
    train_dataloader, test_dataloader = create_dataloaders(annotations_file="data/metadata.csv", device=device, base_path="./data/")
    # print(dataset[51])
    print(next(iter(train_dataloader)))
    # print(dataset[397][2].shape)