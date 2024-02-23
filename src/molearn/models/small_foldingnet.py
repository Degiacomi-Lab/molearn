from torch import nn
from .foldingnet import *


class Small_Decoder(nn.Module):
    '''
    Decoder Module of FoldingNet
    '''

    def __init__(self, out_points, in_channel=2, **kwargs):
        
        super().__init__()

        # Sample the grids in 2D space
        # xx = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        # yy = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        # self.grid = np.meshgrid(xx, yy)   # (2, 45, 45)
        start_out = (out_points//8) +1

        self.out_points = out_points

        self.layer1 = Decoder_Layer(1,           start_out,    in_channel,3*32)
        self.layer2 = Decoder_Layer(start_out,   start_out*8,  3*32,     3)

    def forward(self, x):
        '''
        x: (B, C) or (B, C, 1)
        '''
        x = x.view(-1, 2, 1)
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class Small_AutoEncoder(AutoEncoder):
    '''
    autoencoder architecture derived from FoldingNet.
    '''
    
    def __init__(self, *args, **kwargs):
        
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Small_Decoder(*args, **kwargs)
    
    
class Big_Skinny_Decoder(nn.Module):
    '''
    Decoder Module of FoldingNet
    '''

    def __init__(self, out_points, latent_dimension=2, **kwargs):
        super(Decoder, self).__init__()
        self.latent_dimension = latent_dimension

        start_out = (out_points//16)+1

        self.out_points = out_points

        self.layer1 = Decoder_Layer(1,           start_out,    latent_dimension,3*16)
        self.layer2 = Decoder_Layer(start_out,   start_out*2,  3*16,     3*4)
        self.layer3 = Decoder_Layer(start_out*2, start_out*4, 3*4,      3*2)
        self.layer4 = Decoder_Layer(start_out*4,start_out*16, 3*2,       3)

    def forward(self, x):
        '''
        x: (B, C)
        '''
        x = x.view(-1, self.latent_dimension, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


if __name__=='__main__':
    print('Nothing to see here')
