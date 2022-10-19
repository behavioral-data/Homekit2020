from src.utils import get_logger
from src.models.models.modules import CNNEncoder

logger = get_logger(__name__)

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm


def get_autoencoder_by_name(model_name):
    if model_name == "ConvAutoencoder":
        return ConvAutoencoder
    elif model_name =="RecurrentAutoencoder":
        return RecurrentAutoencoder
    else:
        raise ValueError(f"{model_name} not recognized")

class LSTMEncoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(LSTMEncoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    # x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.permute(1,0,2)

class LSTMDecoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(LSTMDecoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  
  def forward(self, x):
    x = x.repeat(1,self.seq_len, 1)
    # x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    # x = x.reshape((self.seq_len, self.hidden_dim))
    return self.output_layer(x)



class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, return_loss=True):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(seq_len, n_features, embedding_dim)
        self.decoder = LSTMDecoder(seq_len, embedding_dim, n_features)
        self.criterion = nn.MSELoss()
        
        self.name = "RecurrentAutoencoder"
        self.base_model_prefix = self.name

    def forward(self, inputs_embeds,labels):
        pred = self.encoder(inputs_embeds)
        pred = self.decoder(pred)
        return self.criterion(pred,labels) , pred 

class ConvAutoencoder(nn.Module):
    def __init__(self,seq_len=None, n_features=None):
        super(ConvAutoencoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(8, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
        #     nn.Conv1d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
        #     nn.ReLU(True),
        #     nn.MaxPool1d(2, stride=1)  # b, 8, 2, 2
        # )
        self.encoder = CNNEncoder(n_features,seq_len, kernel_sizes=[3,2],
                                  out_channels=[16,8],stride_sizes=[3,2],
                                  max_pool_stride_size= None, max_pool_kernel_size=None)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 3, stride=3),  # b, 8, 15, 15
            nn.ReLU(True),
            # nn.ConvTranspose1d(8, 1, 2, stride=2),  # b, 1, 28, 28
            # nn.ReLU(True),
            # nn.Tanh()
        )

        self.name = "ConvAutoencoder"
        self.base_model_prefix = self.name
        self.criterion = nn.MSELoss()

    def forward(self, inputs_embeds,labels):
        #Padding:
        #Nasty hack, do better later
        PAD_MULT = 12
        seq_len = inputs_embeds.shape[1]
        pad_to = ((seq_len // PAD_MULT) + 1) * PAD_MULT
        pad_len = pad_to-seq_len

        x = F.pad(inputs_embeds,(0,0,0,pad_len))
        x = x.transpose(1,2)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.transpose(1,2)
        x = x[:,:seq_len,:]
        return self.criterion(x,labels) , x 


def run_autoencoder(base_model,
                    task,
                    n_epochs=10,
                    no_wandb=False,
                    notes=None,
                    batch_size=1,
                    learning_rate = 1e-3):
    
    

    train_dataset = task.get_train_dataset()
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,
                        shuffle=True, num_workers=4)
    # eval_dataset = task.get_eval_dataset()
    
    infer_example = train_dataset[0]
    n_timesteps, n_features = infer_example.shape

    model = base_model(seq_len=n_timesteps, n_features=n_features).cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)
    
    config_info = {"n_epochs": n_epochs,
                   "model_type":model.name,
                   "task":task.get_name()}
    if not no_wandb:
        import wandb
        wandb.init(project="flu",
                   entity="mikeamerrill", #TODO make this an argument
                   config=config_info,
                   notes=notes)
        wandb.watch(model)
    
    
    for epoch in tqdm(range(n_epochs)):
        total_loss = 0
        for batch in tqdm(train_dataloader):

            X = Variable(batch).cuda()
            # ===================forward=====================
            output = model(X)
            loss = criterion(output, X)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        total_loss += loss.data

        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, n_epochs, total_loss))