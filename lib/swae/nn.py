import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_filter, d=48, normalize_output=False):
        super().__init__()
        self.num_filter = num_filter
        self.conv1 = nn.Conv2d(1, self.num_filter, 
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.num_filter, self.num_filter, 
                               kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, padding=0)
        self.conv3 = nn.Conv2d(self.num_filter, self.num_filter*2,
                               kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.num_filter*2, self.num_filter*2,
                               kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, padding=0)
        self.conv5 = nn.Conv2d(self.num_filter*2, self.num_filter*4, 
                               kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.num_filter*4, self.num_filter*4,
                               kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, padding=1)        
        
        self.fc1 = nn.Linear(int(self.num_filter*4*4*4), 128)
        self.fc2 = nn.Linear(128, d)

        self.normalize_output = normalize_output
                
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x[None, ...]

        out = F.leaky_relu(self.conv1(x), 0.2)
        out = self.pool1(F.leaky_relu(self.conv2(out), 0.2))
        out = F.leaky_relu(self.conv3(out), 0.2)
        out = self.pool2(F.leaky_relu(self.conv4(out), 0.2))
        out = F.leaky_relu(self.conv5(out), 0.2)
        out = self.pool3(F.leaky_relu(self.conv6(out), 0.2))
        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        if self.normalize_output:
            return F.normalize(out, p=2, dim=-1)
        else:
            return out
    

class Decoder(nn.Module):
    def __init__(self, num_filter, bottleneck_size=48):
        super().__init__()
        self.num_filter = num_filter

        self.fc4 = nn.Linear(bottleneck_size, 128)
        self.fc5 = nn.Linear(128, self.num_filter*4*4*4)
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(self.num_filter*4, self.num_filter*4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.num_filter*4, self.num_filter*4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.num_filter*4, self.num_filter*4, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(self.num_filter*4, self.num_filter*2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.num_filter*2, self.num_filter*2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.num_filter*2, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        d = self.num_filter
        out = torch.relu(self.fc5(self.fc4(x)))
        out = out.view(-1, 4*d, 4, 4)
        
        out = F.leaky_relu(self.conv1(self.upsample(out)), 0.2)
        out = F.leaky_relu(self.conv2(out), 0.2)
        out = F.leaky_relu(self.conv3(self.upsample(out)), 0.2)
        out = F.leaky_relu(self.conv4(out), 0.2)
        out = F.leaky_relu(self.conv5(self.upsample(out)), 0.2)

        out = torch.sigmoid(self.conv6(out))
        return out
    
    
class AE(nn.Module):
    def __init__(self, input_shape, d=48, normalize_output=False):
        super().__init__()
        self.encoder = Encoder(input_shape, d, normalize_output)
        self.decoder = Decoder(input_shape, d)
    
    def forward(self, x):
        z = self.encoder(x)      
        y = self.decoder(z)
        return y