#Implementing the basic GAN (only using simple Linear layers (Dense Layer))
#Experimenting the Network to generate hand written digit images. 


import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm # it shows the progress of out training loop. 
from torchvision.utils import make_grid  


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images in a uniform grid
    '''
    image_tensor = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_tensor[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def generate_noise(batch_size, noise_dim, device):
  return torch.randn((batch_size, noise_dim), device=device)


# Generator Code
class Generator(nn.Module):
  
  def __init__(self, noise_dim=10, hidden_dim=128, image_dim=28*28):
    super().__init__()
    self.gen = nn.Sequential(
        self.generator_Block(noise_dim, hidden_dim),
        self.generator_Block(hidden_dim, 2*hidden_dim),
        self.generator_Block(2*hidden_dim, 4*hidden_dim),
        self.generator_Block(4*hidden_dim, 8*hidden_dim),
        self.generator_Block(8*hidden_dim, image_dim, last_layer=True)
    )

  def generator_Block(self, input_dim, output_dim, last_layer=False):

    if not last_layer:
      return nn.Sequential(
      nn.Linear(input_dim, output_dim),
      nn.BatchNorm1d(output_dim),
      nn.ReLU()
      )
    else:
      return nn.Sequential(
      nn.Linear(input_dim, output_dim),
      nn.Sigmoid()
      )

  def forward(self, noise):
    return self.gen(noise)

#Discriminator Code
class Discriminator(nn.Module):
  def __init__(self, image_dim=28*28, hidden_dim=128):
    super().__init__()

    self.disc = nn.Sequential(
        self.discriminator_Block(image_dim, 4*hidden_dim),
        self.discriminator_Block(4*hidden_dim, 2*hidden_dim),
        self.discriminator_Block(2*hidden_dim, hidden_dim),
        nn.Linear(hidden_dim, 1)
    )

  def discriminator_Block(self, input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2)
    )

  def forward(self, image):
    return self.disc(image)



# Setting parameters
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
noise_dim = 64
batch_size = 128
learning_rate = 0.00001
device = 'cuda'
# Loading MNIST dataset 
dataloader = DataLoader(
    MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)

gen = Generator(noise_dim).to(device)
gen_optimizer = torch.optim.Adam(gen.parameters(), lr=learning_rate)
disc = Discriminator().to(device) 
disc_optimizer = torch.optim.Adam(disc.parameters(), lr=learning_rate)

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):

  noise = generate_noise(num_images, z_dim, device)
  fake_images = gen(noise)
  real_images = real

  disc_fake_out = disc(fake_images)
  disc_real_out = disc(real_images)

  #Discriminator should be able to discriminate between real and fake images. So, Ideally it should output 1 for real images and 0 for fake images. 
  fake_loss = criterion(disc_fake_out, torch.zeros_like(disc_fake_out))
  real_loss = criterion(disc_real_out, torch.ones_like(disc_real_out))

  #taking the average of both losses of real and fake 
  disc_loss = (fake_loss + real_loss)/2

  return disc_loss

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):

  noise = generate_noise(num_images, z_dim, device)
  fake_images = gen(noise)
  disc_out = disc(fake_images)
  #the generator try to generate fake images hopping that it can fool the discriminator to output 1, meaning that the input image is real!
  loss = criterion(disc_out, torch.ones_like(disc_out))  

  return loss


for epoch in range(n_epochs):

  discriminator_loss=[]
  generator_loss=[]
  for real, _ in tqdm(dataloader):
    batch_size = real.shape[0]
    real = real.reshape(batch_size, -1).to(device)

    #Update discriminator
    disc_optimizer.zero_grad()
    disc_loss = get_disc_loss(gen, disc, criterion, real, batch_size, noise_dim, device)
    disc_loss.backward(retain_graph=True)
    disc_optimizer.step()

    #Update Generator
    gen_optimizer.zero_grad()
    gen_loss = get_gen_loss(gen, disc, criterion, batch_size, noise_dim, device)
    gen_loss.backward()
    gen_optimizer.step()


    discriminator_loss.append(disc_loss.item())
    generator_loss.append(gen_loss.item())

  #visualizing the result for each epoch
  print(f"epoch {epoch}: Generator loss: {np.array(generator_loss).mean()}, discriminator loss: {np.array(discriminator_loss).mean()}")
  fake_noise = generate_noise(batch_size, noise_dim, device=device)
  fake = gen(fake_noise)
  show_tensor_images(fake)
  show_tensor_images(real)


# plot discriminator loss and generator loss over time. 
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_loss,label="G")
plt.plot(discriminator_loss,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
