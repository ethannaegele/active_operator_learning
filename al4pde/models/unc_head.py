import torch
import torch.nn as nn

class FCs(nn.Module):
    """This defines the fully connected layers that are part of the uncertainty
    heads. Mainly useful when num_layers > 1, i.e. when more than one 
    fully connected layer follows the GAP."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int = 1,
                 hidden_channels: int = 256,
                 use_end_activation: bool = True
                 ):
        super().__init__()
        layers = []
        current_in = in_channels
        for _ in range(max(0, num_layers - 1)): # optional additional FC layers
            layers.append(nn.Linear(current_in, hidden_channels))
            layers.append(nn.ReLU())
            current_in = hidden_channels
        
        layers.append(nn.Linear(current_in, out_channels))
        if use_end_activation:
            layers.append(nn.ReLU())

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_layers(x)

class GAP1dAndFC(nn.Module):
    def __init__(self, 
                 in_channels: int,  
                 out_channels: int,
                 num_layers: int = 1,
                 hidden_channels: int = 256,
                 make_channel_first_input: bool = False,
                 use_end_activation: bool = True
                 ):
        
        """Learning Loss style uncertainty head containing Global Average 
        Pooling layer followed by one (or multiple) fully connected layers.
        Several of these layers will be used within a module to produce
        an uncertainty estimate. """
        super().__init__()

        self.make_channel_first_input = make_channel_first_input
        self.gap = nn.AdaptiveAvgPool1d(1) # global average pooling

        self.fcs = FCs(in_channels, 
                       out_channels, 
                       num_layers, 
                       hidden_channels, 
                       use_end_activation=use_end_activation)

    def forward(self, x):
        if self.make_channel_first_input:
            x = x.permute(0, 2, 1) # [B, X, C] -> [B, C, X]
        x = self.gap(x).squeeze(-1)
        x = self.fcs(x)
        return x # shape [B, out_channels]



