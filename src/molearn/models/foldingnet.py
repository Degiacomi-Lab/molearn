import torch
from torch import nn
import torch.nn.functional as F


def index_points(point_clouds, index):
    '''
    Given a batch of tensor and index, select sub-tensor.
    
    :param points_clouds: input points data, [B, N, C]
    :param index: sample index data, [B, N, k]
    :return: indexed points data, [B, N, k, C]
    '''
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points


def knn(x, k):
    '''
    K nearest neighborhood.
    
    :param x: a tensor with size of (B, C, N)
    :param k: the number of nearest neighborhoods
    :return: indices of the k nearest neighborhoods with size of (B, N, k)
    '''
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, 1, N), (B, N, N), (B, N, 1) -> (B, N, N)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


class GraphLayer(nn.Module):
    '''
    Graph layer.
    in_channel: it depends on the input of this network.
    out_channel: given by ourselves.
    '''
    def __init__(self, in_channel, out_channel, k=16):
        super(GraphLayer, self).__init__()
        self.k = k
        self.conv = nn.Conv1d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        '''
        :param x: tensor with size of (B, C, N)
        '''
        # KNN
        knn_idx = knn(x, k=self.k)  # (B, N, k)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, k, C)

        # Local Max Pooling
        x = torch.max(knn_x, dim=2)[0].permute(0, 2, 1)  # (B, N, C)

        # Feature Map
        x = F.relu(self.bn(self.conv(x)))
        return x


class Encoder(nn.Module):
    '''
    Graph based encoder
    '''
    def __init__(self, latent_dimension=2, **kwargs):
        super(Encoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.conv1 = nn.Conv1d(12, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.graph_layer1 = GraphLayer(in_channel=64, out_channel=128, k=16)
        self.graph_layer2 = GraphLayer(in_channel=128, out_channel=1024, k=16)

        self.conv4 = nn.Conv1d(1024, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, latent_dimension,1)

    def forward(self, x):
        b, c, n = x.size()

        # get the covariances, reshape and concatenate with x
        knn_idx = knn(x, k=16)
        knn_x = index_points(x.permute(0, 2, 1), knn_idx)  # (B, N, 16, 3)
        mean = torch.mean(knn_x, dim=2, keepdim=True)
        knn_x = knn_x - mean
        covariances = torch.matmul(knn_x.transpose(2, 3), knn_x).view(b, n, -1).permute(0, 2, 1)
        x = torch.cat([x, covariances], dim=1)  # (B, 12, N)

        # three layer MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # two consecutive graph layers
        x = self.graph_layer1(x)
        x = self.graph_layer2(x)

        x = self.bn4(self.conv4(x))

        x = torch.max(x, dim=-1)[0].unsqueeze(-1)

        x = self.conv5(x)
        return x


class FoldingLayer(nn.Module):
    '''
    The folding operation of FoldingNet
    '''

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 3,1,1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 3,1,1)
        layers.append(out_layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, *args):
        """
        :param grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        # concatenate
        # try:
        #    x = torch.cat([*args], dim=1)
        # except:
        #    for arg in args:
        #        print(arg.shape)
        #    raise
        x = torch.cat([*args], dim=1)
        # shared mlp
        x = self.layers(x)

        return x


class Decoder_Layer(nn.Module):
    '''
    Decoder Module of FoldingNet
    '''

    def __init__(self, in_points, out_points, in_channel, out_channel,**kwargs):
        super(Decoder_Layer, self).__init__()

        # Sample the grids in 2D space
        # xx = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        # yy = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        # self.grid = np.meshgrid(xx, yy)   # (2, 45, 45)
        self.out_points = out_points
        self.grid = torch.linspace(-0.5, 0.5, out_points).view(1,-1)
        # reshape
        # self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 45, 45) -> (2, 45 * 45)
        assert out_points % in_points == 0
        self.m = out_points//in_points

        self.fold1 = FoldingLayer(in_channel + 1, [512, 512, out_channel])
        self.fold2 = FoldingLayer(in_channel + out_channel+1, [512, 512, out_channel])

    def forward(self, x):
        '''
        :param x: (B, C)
        '''
        batch_size = x.shape[0]

        # repeat grid for batch operation
        grid = self.grid.to(x.device)                      # (2, 45 * 45)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)

        # repeat codewords
        x = x.repeat_interleave(self.m, dim=-1)            # (B, 512, 45 * 45)

        # two folding operations
        recon1 = self.fold1(grid,x)
        recon2 = recon1+self.fold2(grid,x, recon1)

        return recon2


class Decoder(nn.Module):
    '''
    Decoder Module of FoldingNet
    '''

    def __init__(self, out_points, latent_dimension=2, **kwargs):
        super(Decoder, self).__init__()
        self.latent_dimension = latent_dimension

        # Sample the grids in 2D space
        # xx = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        # yy = np.linspace(-0.3, 0.3, 45, dtype=np.float32)
        # self.grid = np.meshgrid(xx, yy)   # (2, 45, 45)

        start_out = (out_points//128) +1

        self.out_points = out_points

        self.layer1 = Decoder_Layer(1,           start_out,    latent_dimension,3*128)
        self.layer2 = Decoder_Layer(start_out,   start_out*8,  3*128,     3*16)
        self.layer3 = Decoder_Layer(start_out*8, start_out*32, 3*16,      3*4)
        self.layer4 = Decoder_Layer(start_out*32,start_out*128,3*4,       3)

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


class AutoEncoder(nn.Module):
    '''
    Autoencoder architecture derived from FoldingNet.
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Decoder(*args, **kwargs)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__=='__main__':
    print('Nothing to see here')
