import torch
from torch import nn
from models.S_Former import spatial_transformer
from models.T_Former import temporal_transformer


class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_former = spatial_transformer()
        self.t_former = temporal_transformer()
        self.fc = nn.Linear(512, 7)
               
    def forward(self, x):

        x = self.s_former(x)
        x = self.t_former(x)
        x = self.fc(x)
        return x


class RVT(nn.Module):
    def __init__(self, generate_model):
        super().__init__()
        self.generate_model = generate_model
        self.generate_model.fc = nn.Identity()
        self.rnn = nn.RNN(512, 2)
        
    def forward(self, x, hidden):

        x = self.generate_model(x)
        x= self.rnn(x, hidden)
        return x

if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    model(img)

class RVT_3(nn.Module):
    def __init__(self, generate_model):
        super().__init__()
        self.generate_model = generate_model
        self.generate_model.fc = nn.Identity()
        self.rnn = nn.RNN(512, 3)
        
    def forward(self, x, hidden):

        x = self.generate_model(x)
        x= self.rnn(x, hidden)
        return x

class RVT_3L_3(nn.Module):
    def __init__(self, generate_model):
        super().__init__()
        self.generate_model = generate_model
        self.generate_model.fc = nn.Identity()
        self.rnn1 = nn.RNN(512, 256)
        self.rnn2 = nn.RNN(256, 128)
        self.rnn3 = nn.RNN(128, 3)
        
    def forward(self, x, hidden1, hidden2, hidden3):

        x = self.generate_model(x)
        x, hidden_1= self.rnn1(x, hidden1)
        x, hidden_2= self.rnn2(x, hidden2)
        x, hidden_3= self.rnn3(x, hidden3)
        return x, hidden_1, hidden_2, hidden_3 

if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    model(img)