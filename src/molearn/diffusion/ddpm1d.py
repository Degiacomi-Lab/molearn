import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils1d import *
from modules import UNet1d


class Diffusion1d:
    '''
    contains diffusion tools
    linear noise schedule here
    '''
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device=torch.device('cpu')):
        '''
        replace img_size with something better, num_dimensions
        :param noise_steps: num of sampling steps, 1000 is proposed in some of the original papers
        :param beta_start: beta start
        :param beta_end: beta end
        '''
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)


    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)




    def noise_images(self, x, t):
        '''
        x_t = \sqrt{\hat \alpha_t}x_0 + sqrt{1-\hat \alpha_t}\epsilon
        '''
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ


    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    def sample(self, model, n):
        '''
        take the model we'll use for sampling

        Algorithm 2 - sampling:
        1: $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        2: $\mathbf{for} t = T, ..., 1 \mathbf{do}$
        3: \tab $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, mathbf{I}) \mathbif{if} t > 1, \mathbf{else z} = \mathbf{0}$
        4: \tab $\mathbf{x}_{t-1} = \frac{1}{\sqrt{{\alpha}_t}}\left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\hat\alpha_t}}\epsilon_\theta\left(\mathbf{x}_t,t\right) \right) +
        \sigma_t\mathbf{z}$ 
        5: $\mathbf{ end for}$
        6: $\mathbf{return x}_0$

        in human:
            x_T~N(0,I)
            for t = T,...,1 do
                z~N(0,I) if t > 1, else z = 0
                x_(t-1) = 1/sqrt(alpha_t) \left( x_t - (1-alpha_t)/(sqrt(1 - alphahat_t) * epsilon_0(x_t, t))\right) + sigma_t*z

        '''
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        #x = (x * 255).type(torch.uint8) don't need to be scaled between 0 and 255
        return x



def train(args):
    setup_logging(args.run_name) # make folders
    device = args.device

    #dataloader = get_data(args)

    data = args.data
    train_data, valid_data = data.get_dataloader(batch_size=args.batch_size, validation_split=0.1, pin_memory=True)



    model = UNet1d().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion1d(img_size=args.image_size, device=device)
    #logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(train_data)

    for epoch in range(args.epochs):
        #logging.info(f"Starting epoch {epoch}:")
        print(f'Starting epoch {epoch}:')
        #pbar = tqdm(dataloader)
        total_train_loss = 0.0
        model.train()
        for i, images  in enumerate(train_data):
            images = images[0].to(device)
            #sample random timesteps
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            #noise images, and get that noise
            x_t, noise = diffusion.noise_images(images, t)
            #predict noise
            predicted_noise = model(x_t, t)
            #mse between actual noise and predicted noise
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss+=loss.item()*images.shape[0]
            #pbar.set_postfix(MSE=loss.item())

        model.eval()
        with torch.no_grad():
            for i, images in enumerate(valid_data):
                images = images[0].to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)
                total_valid_loss+=loss.item()*images.shape[0]

            sampled_images = diffusion.sample(model, n=10)

            ndarr = sampled_images.permute(0, 2, 1).to('cpu').numpy()
            np.save(os.path.join('results', args.run_name, f'{epoch}'), ndarr)
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional_protein"

    import sys
    sys.path.insert(0, '/home2/wppj21/Workshop/molearn/src')
    import molearn
    data = molearn.PDBData()
    data.import_pdb(args.dataset_path)
    data.atomselect(atoms = ['CA', 'C', 'CB', 'O', 'N',])

    args.data = data
    args.epochs = 500
    args.batch_size = 12
    args.image_size = data._mol.coordinates.shape[1]
    args.dataset_path = "/projects/cgw/proteins/molearn/MurD_closed_open.pdb"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()



