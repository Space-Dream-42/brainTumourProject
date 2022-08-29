import torch
import torch.nn as nn

class SmallSegNet(nn.Module):
    
    def __init__(self, num_modalities=4, num_classes=4):
        super(SmallSegNet, self).__init__()
        self.sm = nn.Softmax(dim=1)
        self.layers = nn.Sequential(
        
        # decrease x,y and increase channels
            # Conv
            nn.Conv3d(num_modalities,32,3,stride=2,padding=0),
            nn.ReLU(),
            
            nn.Conv3d(32,32,3,stride=2,padding=0),
            nn.ReLU(),
        
            # increase x,y and decrease channels
            nn.ConvTranspose3d(32, 32, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose3d(32, num_classes, 3, stride=2, padding=0, output_padding=(0,1,1))
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x