import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
import matplotlib.pyplot as plt
import librosa
from torchaudio.transforms import GriffinLim
import wave
import numpy as np
import IPython.display as ipd

class CharacterEmbeddings(nn.Module):
    def __init__(self, vocab_size : int, embedding_dim: int, device : torch.device):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            device=device
        )

    def forward(self, x):
        return self.embeddings(x)

class PreNet(nn.Module):
    def __init__(self, input_dim : int, hidden_dim : int, output_dim : int, dropout : float):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout

        self.fully_connected_1 = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim)
        self.fully_connected_2 = nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x):
        x = self.fully_connected_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fully_connected_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x
    
class ConvBankWithMaxPool(nn.Module):
    def __init__(self, input_dim : int, K : int):
        super().__init__()
        self.input_dim = input_dim
        self.K = K

        self.conv_bank = nn.ModuleList()
        for k in range(1, self.K + 1):
            conv = nn.Conv1d(
                in_channels=self.input_dim  ,
                out_channels=128,
                kernel_size=k,
                bias=False,
                padding='same'
            )
            self.conv_bank.append(conv)

        self.bn_list = nn.ModuleList([
            nn.BatchNorm1d(num_features=128) for _ in range(1, K + 1)
        ])

        # self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1) TODO: Check if compensating the padding method will work 

    def forward(self, x):
        conv_outputs = []
        for i, (conv, bn) in enumerate(zip(self.conv_bank, self.bn_list)):
            out = conv(x)
            out = bn(out)
            out = F.relu(out)
            conv_outputs.append(out)
        
        conv_outputs = torch.cat(conv_outputs, dim=1)

        return conv_outputs

class ProjectionLayers(nn.Module):
    def __init__(self, input_dim : int, output_dim : int, proj_hidden_1 : int, K : int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj_hidden_1 = proj_hidden_1
        self.K = K

        self.proj_1 = nn.Conv1d(
            in_channels=self.K * 128,
            out_channels=self.proj_hidden_1,
            kernel_size=3,
            bias=False,
            padding='same'
        )

        self.batch_norm_1 = nn.BatchNorm1d(num_features=self.proj_hidden_1)

        self.proj_2 = nn.Conv1d(
            in_channels=self.proj_hidden_1,
            out_channels=self.output_dim,
            kernel_size=3,
            bias=False,
            padding='same'
        )

        self.batch_norm_2 = nn.BatchNorm1d(num_features=self.output_dim)

        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.proj_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)

        x = self.proj_2(x)
        x = self.batch_norm_2(x)

        return x
    
class HighwayNet(nn.Module):
    def __init__(self, dim : int, num_layers : int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim

        self.H = nn.ModuleList([nn.Linear(in_features=self.dim, out_features=self.dim) for _ in range(self.num_layers)])
        self.T = nn.ModuleList([nn.Linear(in_features=self.dim, out_features=self.dim) for _ in range(self.num_layers)])
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        for idx, (H_0, T_0) in enumerate(zip(self.H, self.T)):
            H_out = self.relu(H_0(x))
            T_out = self.sigmoid(T_0(x))
            identity = (1 - T_out) * x

            x = H_out * T_out + identity

        return x
    
class CBHG(nn.Module):
    def __init__(self, input_dim : int, output_dim : int, proj_hidden_1 : int, highway_dim : int, num_highway_layers : int, gru_input_size : int, gru_hidden : int, K : int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj_hidden_1 = proj_hidden_1
        self.highway_dim = highway_dim
        self.num_highway_layers = num_highway_layers
        self.gru_input_size = gru_input_size
        self.gru_hidden = gru_hidden
        self.K = K

        self.conv_bank_with_max_pool = ConvBankWithMaxPool(input_dim=self.input_dim, K=self.K)

        self.projection_layer = ProjectionLayers(input_dim=self.input_dim, output_dim=self.output_dim, proj_hidden_1=self.proj_hidden_1, K=self.K)

        self.residual_connections = lambda x, y : x + y

        self.highway_network = HighwayNet(dim=self.highway_dim, num_layers=self.num_highway_layers)

        self.gru = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=self.gru_hidden,
            batch_first=True,
            bidirectional=True
        )
    def forward(self, x):
        
        # pass through convolutional bank 
        # print(f"Input shape: {x.shape}")
        x_conv = x.transpose(1, 2)
        # print(f"Transposed shape: {x_conv.shape}")
        x = self.conv_bank_with_max_pool(x_conv)
        # print(x.shape)
        x = self.projection_layer(x)

        x = self.residual_connections(x, x_conv)
        
        x = x.transpose(1, 2)

        x = self.highway_network(x)

        x, _ = self.gru(x)

        return x

class LocationSensitiveAttention(nn.Module):
    def __init__(self, encoder_dim : int, decoder_dim : int, attention_dim : int, location_kernel_size : int = 31, location_filters : int = 32):
        super().__init__()
        assert location_kernel_size % 2 == 1, "location_kernel_size should be an odd number"

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.location_kernel_size = location_kernel_size
        self.location_filters = location_filters

        self.encoder_proj = nn.Linear(
            in_features=self.encoder_dim,
            out_features=self.attention_dim,
            bias=False
        )

        self.decoder_proj = nn.Linear(
            in_features=self.decoder_dim,
            out_features=self.attention_dim,
            bias=False
        )

        self.location_proj = nn.Conv1d(
            in_channels=1, # previous attention weights are 1D
            out_channels=self.location_filters,
            kernel_size=self.location_kernel_size,
            padding=(self.location_kernel_size - 1) // 2,
            bias=False
        )

        self.location_to_attention = nn.Linear(
            in_features=self.location_filters,
            out_features=self.attention_dim,
            bias=False
        )

        self.energy_proj = nn.Linear(
            in_features=self.attention_dim,
            out_features=1,
            bias=True
        )

    def forward(self, encoder_outputs, decoder_state, prev_attention_weights):
        """
        Args:
            decoder_state: [batch_size, decoder_dim]
            encoder_outputs: [batch_size, max_text_len, encoder_dim]
            prev_attention_weights: [batch_size, max_text_len]
        Returns:
            context_vector: [batch_size, encoder_dim]
            attention_weights: [batch_size, max_text_len]
        """

        batch_size, max_text_len, encoder_dim = encoder_outputs.shape

        encoder_proj = self.encoder_proj(encoder_outputs) # [B, T, A_D]
        decoder_proj = self.decoder_proj(decoder_state) # [B, A_D]
        decoder_proj = decoder_proj.unsqueeze(1) # [B, 1, A_D]

        prev_attention = prev_attention_weights.unsqueeze(dim=1) # [B, 1, T]
        location_features = self.location_proj(prev_attention)  # [B, F, T]
        location_features = location_features.transpose(1, 2) # [B, T, F]
        location_features = self.location_to_attention(location_features) # [B, T, A_D]

        energies = self.energy_proj(
            F.tanh(encoder_proj + decoder_proj + location_features)
        ).squeeze(-1) # [B, T]

        attention_weights = F.softmax(energies, dim=1)

        context_vector = torch.bmm(input=attention_weights.unsqueeze(dim=1), mat2=encoder_outputs).squeeze(dim=1) # [B, E_D]

        return context_vector, attention_weights


class AttentionRNN(nn.Module):
    def __init__(self, input_size : int, hidden_size : int = 256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRUCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )

    def forward(self, x, h_0=None):
        return self.gru(x, h_0)

class DecoderRNN(nn.Module):
    def __init__(self, prenet_dim : int, context_dim : int, hidden_size : int):
        super().__init__()
        self.prenet_dim = prenet_dim
        self.context_dim = context_dim
        self.hidden_size = hidden_size

        self.gru_1 = nn.GRUCell(
            input_size=self.prenet_dim + self.context_dim,
            hidden_size=self.hidden_size,
        )

        self.gru_2 = nn.GRUCell(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
        )  

        self.layer_norm = nn.LayerNorm(self.hidden_size)


    def forward(self, prenet_output : torch.Tensor, context_output : torch.Tensor, hidden_states : tuple = None):
        
        # Concatenate prenet_output and context_output
        x = torch.cat([prenet_output, context_output], dim=-1)
        
        # Initialize hidden states if are nont
        if hidden_states is None:
            gru1_hidden = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            gru2_hidden = torch.zeros_like(gru1_hidden)
        else:
            gru1_hidden, gru2_hidden = hidden_states

        gru1_hidden = self.gru_1(x, gru1_hidden)

        gru2_input = gru1_hidden
        gru2_hidden = self.gru_2(gru2_input, gru2_hidden)

        
        output = self.layer_norm(gru1_hidden + gru2_hidden)

        return output, (gru1_hidden, gru2_hidden)

class Encoder(nn.Module):
    def __init__(self, 
                 vocab_size : int, 
                 embedding_dim : int, 
                 hidden_dim : int, 
                 output_dim : int, 
                 num_highway_layers : int, 
                 K : int,
                 cbhg_input_dim : int,
                 cbhg_output_dim : int,
                 cbhg_highway_dim : int,
                 cbhg_gru_input_size : int,
                 cbhg_gru_hidden : int,
                 cbhg_proj_1 : int,
                 dropout : float, 
                 device : torch.device
                 ):
        super().__init__()
        """
            This is the Encoder block of the Tacotron 1.
            The Encoder block consists of following blocks:
                - Character Embedding block:
                - PreNet Block:
                - CBHG Block:
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_highway_layers = num_highway_layers
        self.K = K
        self.dropout = dropout

        self.device = device

        # CBHG parameters setup
        self.cbhg_input_dim = cbhg_input_dim
        self.cbhg_output_dim = cbhg_output_dim
        self.cbhg_highway_dim = cbhg_highway_dim
        self.cbhg_gru_input_size = cbhg_gru_input_size
        self.cbhg_gru_hidden = cbhg_gru_hidden
        self.cbhg_proj_1 = cbhg_proj_1

        self.character_embedding = CharacterEmbeddings(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            device=self.device
        )

        self.encoder_pre_net = PreNet(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout
        )

        self.cbhg = CBHG(
            input_dim=self.cbhg_input_dim, 
            output_dim=self.cbhg_output_dim,
            highway_dim=self.cbhg_highway_dim,
            num_highway_layers=self.num_highway_layers,
            gru_input_size=self.cbhg_gru_input_size,
            gru_hidden=self.cbhg_gru_hidden,
            proj_hidden_1=self.cbhg_proj_1,
            K=self.K
        )

    def forward(self, x):
        # The input x should have the batch size as well

        x = self.character_embedding(x)

        x = self.encoder_pre_net(x)
        
        x = self.cbhg(x)

        return x

class Decoder(nn.Module):
    def __init__(self, input_dim : int, hidden_dim : int, output_dim : int, encoder_dim : int, decoder_dim : int, attention_dim : int, mel_channels : int, dropout : float, r : int = 5):
        super().__init__()
        """
            This is the Encoder block of the Tacotron 1.
            The Encoder block consists of following blocks:
                - PreNet Block:
                - Attention RNN:
                - LocationSensitiveAttention Block:
                TODO: Add left blocks
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        
        self.mel_channels = mel_channels

        self.r = r

        self.decoder_pre_net = PreNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dropout=self.dropout
        )

        self.attention_rnn = AttentionRNN(
            input_size=self.output_dim*2,
            hidden_size=self.hidden_dim,
        )

        self.decoder_rnn = DecoderRNN(
            prenet_dim=self.output_dim,
            context_dim=self.encoder_dim,
            hidden_size=self.hidden_dim
        )

        self.location_sensitive_attention = LocationSensitiveAttention(
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim,
            attention_dim=self.attention_dim
        )

        self.mel_out_proj = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.mel_channels * self.r
        )

        self.stop_token_proj = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.mel_channels
        )

        self.go_frame = nn.Parameter(torch.zeros(self.mel_channels))

    def init_attention_weights(self, batch_size, encoder_length, device):
        attention_weights = torch.ones(batch_size, encoder_length, device=device)
        attention_weights[:, 0] = 1.0 # Focus entierly on the first position
        
        return attention_weights

    def forward(self, encoder_outputs, target_mels=None, max_decoder_steps=200,  stop_threashold = 0.5):
        batch_size = encoder_outputs.size(0)
        encoder_length = encoder_outputs.size(1)
        device = encoder_outputs.device

        # Initialize states
        attention_rnn_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        decoder_rnn_hidden = None
        prev_attention_weights = self.init_attention_weights(batch_size, encoder_length, device)

        # Store outputs
        mel_outputs = []
        attention_weights_history = []
        stop_token_outputs = []
        
        # Initial input to decoder's PreNet
        current_mel_input = self.go_frame.unsqueeze(0)
        current_mel_input = current_mel_input.expand(batch_size, -1)
        
        for t in range(max_decoder_steps // self.r):

            prenet_output = self.decoder_pre_net(current_mel_input)

            # Attention RNN
            if t == 0:
                context_vector = torch.zeros_like(prenet_output)

            attention_rnn_hidden = self.attention_rnn(
                torch.cat([prenet_output, context_vector], dim=-1),
                attention_rnn_hidden
            )

            # Location-Sensitive Attention
            context_vector, attention_weights = self.location_sensitive_attention(
                encoder_outputs,
                attention_rnn_hidden, # decoder state is the attention rnn hidden
                prev_attention_weights
            )
            attention_weights_history.append(attention_weights)
            prev_attention_weights = attention_weights # update for the next step

            # Decoder RNN
            output, decoder_rnn_hidden = self.decoder_rnn(
                prenet_output,
                context_vector,
                decoder_rnn_hidden
            )

            # Predict mel frame
            current_mel_output = self.mel_out_proj(output)
            mel_output = current_mel_output.view(batch_size, self.r, self.mel_channels)
            mel_outputs.append(mel_output)


            stop_token_logit = self.stop_token_proj(output)
            stop_token_outputs.append(stop_token_logit)
            
            if target_mels is None: # this means inference mode
                stop_prob = torch.sigmoid(stop_token_logit)
                if torch.all(stop_prob > stop_threashold):
                    break


            if target_mels is not None and t + 1 < target_mels.size(1):
                current_mel_input = target_mels[:, t, :]
            else:
                current_mel_input = mel_output[:, -1, :]
                if target_mels is not None: # training mode but reached the end
                    break

        # mel_outputs = torch.stack(mel_outputs, dim=1)  # [B, num_frames, mel_channels]
        # mel_outputs = mel_outputs.view(mel_outputs.shape[0], mel_outputs.shape[1] * mel_outputs.shape[2], -1)
        # print(mel_outputs.shape)
        mel_outputs = torch.cat(mel_outputs, dim=1)
        attention_weights_history = torch.stack(attention_weights_history, dim=1)
        stop_token_outputs = torch.stack(stop_token_outputs, dim=1)

        return mel_outputs, attention_weights_history, stop_token_outputs
        
class Tacotron(nn.Module):
    def __init__(self, cfg, device : torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.encoder = Encoder(
            vocab_size=self.cfg["encoder"]["vocab_size"],
            embedding_dim=self.cfg["encoder"]["embedding_dim"],
            hidden_dim=self.cfg["encoder"]["hidden_dim"],
            output_dim=self.cfg["encoder"]["output_dim"],
            num_highway_layers=self.cfg["encoder"]["num_highway_layers"],
            device=self.device,
            K=self.cfg["encoder"]["K"],
            dropout=self.cfg["encoder"]["dropout"],
            cbhg_input_dim=self.cfg["encoder"]["cbhg_input_dim"],
            cbhg_output_dim=self.cfg["encoder"]["cbhg_output_dim"],
            cbhg_proj_1=self.cfg["encoder"]["cbhg_proj_1"],
            cbhg_highway_dim=self.cfg["encoder"]["cbhg_highway_dim"],
            cbhg_gru_input_size=self.cfg["encoder"]["cbhg_gru_input_size"],
            cbhg_gru_hidden=self.cfg["encoder"]["cbhg_gru_hidden"]
        )

        self.decoder = Decoder(
            input_dim=self.cfg["decoder"]["input_dim"],
            hidden_dim=self.cfg["decoder"]["hidden_dim"],
            output_dim=self.cfg["decoder"]["output_dim"],
            encoder_dim=self.cfg["decoder"]["encoder_dim"],
            decoder_dim=self.cfg["decoder"]["decoder_dim"],
            attention_dim=self.cfg["decoder"]["attention_dim"],
            mel_channels=self.cfg["decoder"]["mel_channels"],
            dropout=self.cfg["decoder"]["dropout"]
        )

        self.post_processing_cbhg = CBHG(
            input_dim=self.cfg["pp_cbhg"]["input_dim"],
            output_dim=self.cfg["pp_cbhg"]["output_dim"],
            highway_dim=self.cfg["pp_cbhg"]["highway_dim"],
            num_highway_layers=self.cfg["pp_cbhg"]["num_highway_layers"],
            gru_input_size=self.cfg["pp_cbhg"]["gru_input_size"],
            gru_hidden=self.cfg["pp_cbhg"]["gru_hidden"],
            K=self.cfg["pp_cbhg"]["K"],
            proj_hidden_1=self.cfg["pp_cbhg"]["proj_hidden_1"]
        )

        self.linear_proj = nn.Linear(
            in_features=self.cfg["pp_cbhg"]["gru_hidden"] * 2,
            out_features=self.cfg.get("n_fft", 1024) // 2 + 1
        )

    def forward(self, x, y=None):
        encoder_outputs = self.encoder(x)

        decoder_mel_outputs, attention_weights, stop_token_outputs = self.decoder(encoder_outputs=encoder_outputs, max_decoder_steps=y.shape[1] if y is not None else 200, target_mels=y)
        # decoder_mel_outputs = decoder_mel_outputs.transpose(1, 2)
        pp_cbhg_output = self.post_processing_cbhg(decoder_mel_outputs)

        linear_outputs = self.linear_proj(pp_cbhg_output)

        return {
            "mel_outputs" : decoder_mel_outputs,
            "linear_outputs" : linear_outputs,
            "stop_token_outputs" : stop_token_outputs,
            "attention_weights" : attention_weights
        }

def create_tacotron(cfg, device):
    return Tacotron(cfg=cfg, device=device).to(device)

if __name__ == "__main__":

    # # embedding = CharacterEmbeddings(10, 256, torch.device("cpu"))
    # # pre_net = PreNet(256, 128, 128, 0.5)
    x = torch.zeros(size=(10,), dtype=torch.long)
    # print(x.shape)
    # # print(x.shape)
    # # x = embedding(x)
    # # print(x.shape)
    # # x = pre_net(x)
    # # print(x.shape)
    # # x = x.unsqueeze(0)
    # # print(x.shape)
    
    # cbhg = CBHG(
    #     input_dim=128,
    #     output_dim=128,
    #     highway_dim=128,
    #     num_highway_layers=4,
    #     gru_input_size=128,
    #     gru_hidden=128,
    #     K=16,
    #     proj_hidden_1=128
    # )

    # # print(cbhg(x))

    # encoder = Encoder(
    #     vocab_size=10,
    #     embedding_dim=256,
    #     hidden_dim=128,
    #     output_dim=128,
    #     num_highway_layers=4,
    #     device=torch.device("cpu"),
    #     K=16,
    #     dropout=0.5,
    #     cbhg_input_dim=128,
    #     cbhg_output_dim=128,
    #     cbhg_proj_1=128,
    #     cbhg_highway_dim=128,
    #     cbhg_gru_input_size=128,
    #     cbhg_gru_hidden=128
    # )
    # encoder_outputs = encoder(x.unsqueeze(0))

    # decoder = Decoder(
    #     input_dim=80,
    #     hidden_dim=256,
    #     output_dim=256,
    #     encoder_dim=256,
    #     decoder_dim=256,
    #     attention_dim=256,
    #     mel_channels=80,
    #     dropout=0.5
    # )

    # spectrogram = decoder(encoder_outputs)[0]

    griffin_lim_transform = GriffinLim(n_fft=1024, power=1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_config()

    tacotron = create_tacotron(cfg, device)
    # print(x)
    x = x.unsqueeze(0)
    spectrogram = tacotron(x=x, y=None)["linear_outputs"].squeeze(0)

    # print(spectrogram)

    # spectrogram = torch.abs(spectrogram) + 1e-8
    
    # plt.figure(figsize=(10, 4))
    # img = librosa.display.specshow(spectrogram.detach().cpu().numpy().T, y_axis='log', x_axis='time', cmap='inferno')
    # plt.show()
    
    waveform = griffin_lim_transform(spectrogram.detach().T)

    # print(waveform)

    obj = wave.open("output.wav", "w")
    obj.setnchannels(1)  # Mono audio
    obj.setsampwidth(2)   # 16-bit audio
    obj.setframerate(22050)  # 44100 Hz sample rate
    obj.writeframesraw(waveform.numpy().tobytes())
    obj.close()

    # print(waveform)

    ipd.Audio(waveform, rate=22050)