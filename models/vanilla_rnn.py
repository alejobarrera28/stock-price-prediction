import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import config

class VanillaRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int = 1, dropout_rate: float = 0.2):
        super(VanillaRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        self.device = torch.device(config.device)
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        
        rnn_out, _ = self.rnn(x, h_0)
        
        last_output = rnn_out[:, -1, :]
        
        out = self.dropout(last_output)
        out = self.fc(out)
        
        return out
    
    def predict(self, data_loader: DataLoader):
        self.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self(batch_x)
                predictions.append(outputs.cpu().numpy())
                actuals.append(batch_y.cpu().numpy())
        
        return np.concatenate(predictions), np.concatenate(actuals)
    
    def evaluate(self, data_loader: DataLoader, criterion):
        self.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {'loss': avg_loss}
    
    def train_epoch(self, train_loader: DataLoader, optimizer, criterion):
        self.train()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {'train_loss': avg_loss}
    
    def save_model(self, filepath: str):
        model_state = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'model_type': self.__class__.__name__
        }
        torch.save(model_state, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, model_class):
        model_state = torch.load(filepath, map_location='cpu')
        
        model = model_class(
            input_size=model_state['input_size'],
            hidden_size=model_state['hidden_size'],
            num_layers=model_state['num_layers'],
            output_size=model_state['output_size'],
            dropout_rate=model_state['dropout_rate']
        )
        
        model.load_state_dict(model_state['model_state_dict'])
        print(f"Model loaded from {filepath}")
        return model
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': self.__class__.__name__,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }