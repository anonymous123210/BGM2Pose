import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .wisppn_resnet import get_wisppn
import torch

class Wipose_LSTM(nn.Module):
    def __init__(self, in_cha=7, out_cha=4, debug=False):
        super().__init__()
        self.in_cha = in_cha
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_cha, 32, kernel_size=2, stride=1, padding = 'same'),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(0.0),
            nn.SiLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=2, stride=1, padding = 'same'),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(0.0),
            nn.SiLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding = 'same'),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2, stride=2),
            nn.Dropout(0.0),
            nn.SiLU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, padding = 'same'), #[bs * time, 1, 240]
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.0),
            nn.SiLU(),
        )
        self.cnn = nn.Sequential(
            self.cnn1,
            self.cnn2,
            self.cnn3,
            self.cnn4,
        )
        self.hidden_size = 256
        self.lstm1 = nn.LSTM(input_size = 256, hidden_size = self.hidden_size, dropout = 0.0, num_layers=3)
        # self.lstm2 = nn.LSTM(input_size = self.hidden_size, hidden_size = self.hidden_size, dropout = 0.0)
        # self.lstm3 = nn.LSTM(input_size = self.hidden_size, hidden_size = self.hidden_size, dropout = 0.0)
        self.seq_len = 12
        self.fc1 = nn.Sequential(
            nn.Linear(self.hidden_size, out_cha),
        )
        self.debug = debug

    def forward(self, x, music_mask): #mel -> (bs, in_cha = 4, time, freq)
        bs, in_cha, time, freq = x.shape
        # print("1:",x.shape)
        if (in_cha == 9) & (self.in_cha==6):
            x = x[:,:6,:,:]
            in_cha = 6
        x = x.permute(0,2,1,3)
        # x = [self.cnn(x[:,i,:,:]).unsqueeze(1) for i in range(self.seq_len)] #input[bs, 4, freq_bins] -> output[bs, 1, 932]
        x = x.reshape(bs * time, in_cha, freq)
        x = self.cnn(x)
        # print("2:", x.shape)
        # x = torch.cat(x, dim = 1) #[bs, 5, 932]
        x = x.reshape(bs, time, 256)
        x = x.permute(1,0,2)
        x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)
        # x, _ = self.lstm3(x)
        # print(x.shape)
        x = x.permute(1,0,2)
        x = self.fc1(x)
        return x