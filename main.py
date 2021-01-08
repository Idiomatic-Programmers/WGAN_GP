import torch
from torchvision import transforms
from datasets.data import FashionMNIST, DataLoader
from models.modules import Generator, Discriminator, initialise_weights
from torch.utils.tensorboard import SummaryWriter
import torchvision
from models.utils import gp

torch.cuda.empty_cache()

hyperparameters = {
    'load_size': 64,
    'batch_size': 32,
    'channels': 1,
    'epoch': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lr': 1e-4,
    'z_dim': 100,
    'critic_iterations': 5,
    'weights_clip': 0.01,
    'lamda': 10,
    'num_classes': 10,
    'embed_size': 100
}


def get_data_loader(config):

    transformations = transforms.Compose([
        transforms.Resize(config['load_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(config['channels'])], [0.5 for _ in range(config['channels'])])
    ])

    dataset = FashionMNIST()("datasets", transformations)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)


def get_models(config):
    disc = Discriminator(config['channels'], config['load_size'], config['num_classes'], 64).to(config['device'])
    gen = Generator(config['z_dim'], config['channels'], config['load_size'], config['num_classes'], config['embed_size']).to(config['device'])
    initialise_weights(gen)
    initialise_weights(disc)

    return gen, disc


def train(config):
    data_loader = get_data_loader(config)
    gen, critic = get_models(config)

    optimiser_G = torch.optim.Adam(gen.parameters(), lr=config['lr'], betas=(0.0, 0.9))
    optimiser_C = torch.optim.Adam(critic.parameters(), lr=config['lr'], betas=(0.0, 0.9))

    # fixed_noise = torch.randn(config['batch_size'], config['z_dim'], 1, 1).to(config['device'])

    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")

    step = 0

    gen.train()
    critic.train()

    for epoch in range(config['epoch']):
        for batch, (real, label) in enumerate(data_loader):
            real = real.to(config['device'])
            label = label.to(config['device'])
            batch_size = real.size(0)

            # Train Discriminator --> max E(D(X)) - E(D(G(X)))
            for _ in range(config['critic_iterations']):
                noise = torch.randn((batch_size, config['z_dim'], 1, 1)).to(config['device'])
                fake = gen(noise, label)

                critic_real = critic(real, label).reshape(-1)
                critic_fake = critic(fake, label).reshape(-1)
                gradient_penality = gp(critic, real, fake, label, config['device'])

                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config['lamda']*gradient_penality
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                optimiser_C.step()

                for p in critic.parameters():
                    p.data.clamp_(-config['weights_clip'], config['weights_clip'])

            # Train Generator ---> max E(D(X))
            output = critic(fake, label).reshape(-1)
            loss_gen = -torch.mean(output)
            gen.zero_grad()
            loss_gen.backward()
            optimiser_G.step()

            # Print losses occasionally and print to tensorboard
            if batch % 100 == 0:
                print(
                    f"Epoch [{epoch}/{config['epoch']}] Batch {batch}/{len(data_loader)} \
                              Loss D: {loss_critic.item():.4f}, loss G: {loss_gen.item():.4f}"
                )

                with torch.no_grad():
                    fake = gen(noise, label)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:config['batch_size']], normalize=True
                    )
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:config['batch_size']], normalize=True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step += 1


train(hyperparameters)
