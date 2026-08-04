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

class GAP2dAndFC(nn.Module):
    def __init__(self, 
                 in_channels: int,  
                 out_channels: int,
                 num_layers: int = 1,
                 hidden_channels: int = 256,
                 make_channel_first_input: bool = False,
                 use_end_activation: bool = True,
                 time_as_second_dim: bool = False, # True for full traj models
                 ):
        
        """Learning Loss style uncertainty head containing Global Average 
        Pooling layer followed by one (or multiple) fully connected layers.
        Several of these layers will be used within a module to produce
        an uncertainty estimate. """
        super().__init__()

        self.make_channel_first_input = make_channel_first_input
        self.time_as_second_dim = time_as_second_dim
        if time_as_second_dim:
            # average across space X, for each time slice independently -> [T] avg values, dynamically for T
            self.gap = nn.AdaptiveAvgPool2d((1, None)) # global average pooling
        else:
            # average across space X, Y -> scalar
            self.gap = nn.AdaptiveAvgPool2d(1) # [B, C, X, Y] -> [B, C, 1, 1]

        self.fcs = FCs(in_channels, 
                       out_channels, 
                       num_layers, 
                       hidden_channels, 
                       use_end_activation=use_end_activation)

    def forward(self, x):
        if self.make_channel_first_input:
            x = x.permute(0, 3, 1, 2) # [B, X, T, C] -> [B, C, X, T]
        # otherwise, expect shape like [B, C, X, T], C = model width/features/channels
        x = self.gap(x).squeeze(-2) # [B, C, T]
        if not self.time_as_second_dim:
            x = x.squeeze(-1) # [B, C, 1] -> [B, C]
        if self.time_as_second_dim:
            x = x.permute(0, 2, 1) # [B, T, C]
        x = self.fcs(x)
        return x # shape [B, T, out_channels] or [B, out_channels]

