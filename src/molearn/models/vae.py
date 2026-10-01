import torch
from torch import nn
import torch.nn.functional as F

from molearn.models.foldingnet import AutoEncoder, GraphLayer, knn, index_points


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
        self.conv_mu = nn.Conv1d(512, latent_dimension,1)
        self.conv_logvar = nn.Conv1d(512, latent_dimension,1)

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

        mu = self.conv_mu(x).squeeze(-1)
        logvar = self.conv_logvar(x).squeeze(-1)
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std) 
        z = eps * std + mu

        return z.squeeze(-1), mu, logvar


class VAE(AutoEncoder):
    """
    Variational autoencoder architecture derived from FoldingNet.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(*args, **kwargs)

    def forward(self, x):
        z, _, _ = self.encode(x)
        x_rec = self.decode(z)
        return x_rec


if __name__=='__main__':
    print('Nothing to see here')
