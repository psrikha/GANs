import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt

dataset = datasets.CIFAR10(root="./data", download=False, transform=transforms.Compose([
    transforms.Resize(64),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle=True, num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
noise_dimension = 100 
realval = 1 
fakeval = 0 

randomseed = random.randint(10, 100000)
random.seed(randomseed)
torch.manual_seed(randomseed)

def weights_initialize(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, passed_input):
        discriminator_output = self.main(passed_input)
        return discriminator_output.view(-1, 1).squeeze(1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dimension, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, passed_input):
        generator_output = self.main(passed_input)
        return generator_output

discriminator = Discriminator().to(device)
generator = Generator().to(device)
discriminator.apply(weights_initialize)
generator.apply(weights_initialize)

generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

noise = torch.randn(128, noise_dimension, 1, 1, device=device)
gen_loss_list = []
dis_loss_list = []
counter = 0
counter_list = []

num_epochs = 10
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        counter += 1
        counter_list.append(counter)
        
        real_data = data[0].to(device)
        size_of_batch = real_data.size(0)
        labels_tensor = torch.full((size_of_batch,), realval, device = device).float()
        discriminator.zero_grad()
        dis_output = discriminator(real_data ).float()
        dis_real_error = criterion(dis_output, labels_tensor)
        dis_real_error.backward()
        dis_real_output_mean = dis_output.mean().item()

        labels_tensor.fill_(fakeval).float()
        noise = torch.randn(size_of_batch, noise_dimension, 1, 1, device=device)
        fake_data = generator(noise)
        dis_output = discriminator(fake_data.detach()).float()
        dis_fake_error = criterion(dis_output, labels_tensor)
        dis_fake_error.backward()
        dis_fake_output_mean = dis_output.mean().item()
        disriminator_optimizer.step()
        final_dis_error = dis_real_error + dis_fake_error
        dis_loss_list.append(final_dis_error.item())
        
        labels_tensor.fill_(realval).float()
        generator.zero_grad()
        gen_output = discriminator(fake_data).float()
        gen_error = criterion(gen_output, labels_tensor)
        gen_loss_list.append(gen_error.item())
        gen_error.backward()
        gen_output_mean = gen_output.mean().item()
        generator_optimizer.step()

        
        print('[%d/%d][%d/%d] DisLoss: %.4f GenLoss: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % 
              (epoch, num_epochs, i, len(dataloader), final_dis_error.item(), 
               gen_error.item(), dis_real_output_mean, dis_fake_output_mean, gen_output_mean ))
        
    fake_data = generator(noise)
    vutils.save_image(real_data,'DCganOutput/real_samples.png',normalize=True)
    vutils.save_image(fake_data.detach(),'DCganOutput/fake_samples_epoch_%03d.png' % (epoch), normalize=True)

plt.plot(counter_list, gen_loss_list, 'r.', label='Generator')
plt.plot(counter_list, dis_loss_list, 'g.', label='Discriminator')
plt.title("DCGAN Loss of Generator and Discriminator ")
plt.xlabel("Batch Number")
plt.ylabel("Binary Cross Entropy Loss")
plt.legend(loc="best")
plt.show()











