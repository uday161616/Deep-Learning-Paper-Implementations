import torch
import random
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02) # mentioned in the paper. Section 4. 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen = Generator().to(device)
init_weights(gen)
disc = Discriminator().to(device)
init_weights(disc)
criterion = nn.BCELoss().to(device)

# Hyperparameters
beta1 = 0.5 # mentioned in the paper. Section 4
lr = 2e-4
img_size = 64
batch_size = 128
z_dim = 100
num_epochs = 5

gen_optim = optim.Adam(gen.parameters(), lr = lr, betas = (beta1, 0.999))
disc_optim = optim.Adam(disc.parameters(), lr = lr, betas = (beta1, 0.999))


transforms = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
    ]
)

dataset = datasets.ImageFolder(r"DCGAN\data", transform = transforms)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

fixed_noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)

writer_real = SummaryWriter(r"DCGAN\logs\real")
writer_fake = SummaryWriter(r"DCGAN\logs\fake")
step = 0

gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        disc_optim.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        gen_optim.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1





