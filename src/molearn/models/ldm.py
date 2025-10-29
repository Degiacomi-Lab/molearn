import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import os
from collections import deque


class PosEmbed(nn.Module):
    def __init__(self, dim):
        super(PosEmbed, self).__init__()
        assert dim % 2 == 0, "Odd number of positional embedding dimensions is not supported"
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        scaling_factor = math.log(10_000) / (half_dim - 1)

        embedding = torch.exp(torch.arange(half_dim) * scaling_factor * -1) # exponential *decay* - had a bug here where i didn't multiply with -1 and the model was worse
        if torch.cuda.is_available():
          embedding = embedding.to(device)

        embeddings = torch.outer(t, embedding) # embedding per t

        sines = torch.sin(embeddings)
        cosines = torch.cos(embeddings)

        embeddings = torch.cat([sines, cosines], dim=1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(emb_dim, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, emb):
        x = self.relu(self.conv1(x))
        emb = self.fc(emb)
        x_h = x + emb[:, :, None] # one embedding scalar per channel
        x_h = self.relu(self.conv2(x_h))
        return x_h


class LDM(nn.Module):
    def __init__(self, emb_dim=128, base_channels=32):
        super(LDM, self).__init__()

        self.emb = PosEmbed(emb_dim)

        self.inc = Block(1, base_channels, emb_dim)
        self.down1 = Block(base_channels, base_channels * 2, emb_dim)

        self.down2 = Block(base_channels * 2, base_channels * 4, emb_dim)

        self.bottleneck = Block(base_channels * 4, base_channels * 4, emb_dim)

        self.upconv1 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up1 = Block(base_channels * 4, base_channels * 2, emb_dim)

        self.upconv2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up2 = Block(base_channels * 2, base_channels, emb_dim)

        self.outc = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x, t):
        emb = self.emb(t)

        x1 = self.inc(x, emb)

        x2_pooled = F.max_pool1d(x1, 2)
        x2 = self.down1(x2_pooled, emb)

        x3_pooled = F.max_pool1d(x2, 2)
        x3 = self.down2(x3_pooled, emb)

        x_mid = self.bottleneck(x3, emb)

        x = self.upconv1(x_mid)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x, emb)

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x, emb)

        output = self.outc(x)
        return output



class MiniLDM(nn.Module):
    """
        MiniLDM is a dummy LDM used in a proof of concept notebook "example_train_ldm.ipynb
        In practice, LDM should be used with a latent space in R^d where d >> 2
    """
    def __init__(self, emb_dim=16, base_channels=32):
        super(MiniLDM, self).__init__()

        self.emb = PosEmbed(emb_dim)

        self.inc = Block(1, base_channels, emb_dim)
        self.down1 = Block(base_channels, base_channels * 2, emb_dim)

        self.down2 = Block(base_channels * 2, base_channels * 4, emb_dim)

        self.bottleneck = Block(base_channels * 4, base_channels * 4, emb_dim)

        self.upconv1 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=1, stride=1) 
        self.up1 = Block(base_channels * 4, base_channels * 2, emb_dim)

        self.upconv2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=1, stride=1) 
        self.up2 = Block(base_channels * 2, base_channels, emb_dim)

        self.outc = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x, t):
        emb = self.emb(t)

        x1 = self.inc(x, emb)

        x2_pooled = F.max_pool1d(x1, 1) 
        x2 = self.down1(x2_pooled, emb)

        x3_pooled = F.max_pool1d(x2, 1) 
        x3 = self.down2(x3_pooled, emb)

        x_mid = self.bottleneck(x3, emb)
        x = self.upconv1(x_mid)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x, emb)

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x, emb)

        output = self.outc(x)
        return output


def forward(x_0, t, alphas_bar, device):
    t_i = t - 1 # zero indexing!
    x_0 = x_0.to(device)
    eps = torch.randn_like(x_0, device=device)
    x_0_scaled = torch.sqrt(alphas_bar[t_i]).view(-1, 1, 1) * x_0
    eps_scaled = torch.sqrt(1 - alphas_bar[t_i]).view(-1, 1, 1) * eps
    noisy_x_0 = x_0_scaled + eps_scaled
    return noisy_x_0, eps


def get_schedules(device, T=1000, start=0.0001, end=0.02):
  betas = torch.linspace(start, end, T, device=device)
  alphas = 1.0 - betas
  alphas_bar = torch.cumprod(alphas, axis=0)
  return {"T" : T, "betas" : betas, "alphas" : alphas, "alphas_bar" : alphas_bar}


def train(model, dataloader, schedules, epochs, loss_function, optimizer, device, checkpoint_path='checkpoints'):
    optimizer = optimizer(model.parameters(), lr=1e-4)

    os.makedirs(checkpoint_path, exist_ok=True)
    recent_losses = deque(maxlen=1000)

    for epoch in range(epochs):
        for step, vector in enumerate(dataloader):
            vector = vector[0].to(device)
            vector = vector[:, None, :]
            # sample x_t
            t = torch.randint(low=1, high=schedules['T'] + 1, size=(vector.size(0),), device=device)
            x_t, noise = forward(vector, t, schedules['alphas_bar'], device=device)

            predicted_noise = model(x_t, t) # Predict the noise
            loss = loss_function(predicted_noise, noise) # Learn the noise
            recent_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                if len(recent_losses) > 0:
                    # Calculate the average loss over the stored steps
                    avg_loss_1000 = sum(recent_losses) / len(recent_losses)
                else:
                    avg_loss_1000 = 0.0

                print(f"Epoch {epoch} | Step {step}/{len(dataloader)} | Current Loss: {loss.item():.4f} | Avg. 1000-Step Loss: {avg_loss_1000:.4f}")

            
        checkpoint_name = f'ldm_epoch_{epoch:03d}.pt'
        save_path = os.path.join(checkpoint_path, checkpoint_name)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, save_path)

        print(f"Saved checkpoint for Epoch {epoch} to {save_path}")
