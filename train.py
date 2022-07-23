import argparse
import time
import logging
from multiprocessing import cpu_count

import torch
from torchvision import datasets, transforms

from nets import Generator, Discriminator


logging.basicConfig(level=logging.INFO)


def g_loss_fn(output, target):
    """
    Generator loss function

    'Early in learning, when G is poor, D can reject samples with
    high confidence because they are clearly different from
    the training data. In this case, log(1 − D(G(z))) saturates.
    Rather than training G to minimize log(1 − D(G(z)))
    we can train G to maximize log D(G(z)).'
    :return:
    """
    return -torch.mean(torch.log(output))


def d_loss_fn(output, target):
    """
    Discriminator loss function
    :param output:
    :param target:
    :return:
    """
    return torch.mean(torch.log(output) + torch.log(1 - target))


def get_random_batch(batch_size, hidden_size, device):
    return torch.randn((batch_size, hidden_size), device=device)


def main(args):
    torch.manual_seed(args.seed)
    logger = logging.getLogger('GANs')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    k = args.k
    n_layers = args.n_layers
    g_in_features = args.in_features
    g_out_features = 28*28
    d_in_features = 28*28
    d_out_features = 1

    train_kwargs = {'batch_size': batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)

    generator = Generator(g_in_features, g_out_features, n_layers)
    discriminator = Discriminator(d_in_features, d_out_features, n_layers)

    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=lr
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=lr
    )

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        t = time.time()
        d_loss_temp = list()
        for i in range(k):
            discriminator.zero_grad()
            imgs, _ = next(iter(train_loader))
            imgs = imgs.to(device)

            random_data = get_random_batch(batch_size, g_in_features, device)

            # get outputs
            p_data = discriminator(imgs)
            generated_imgs = generator(random_data)
            p_noise = discriminator(generated_imgs)

            # calculate loss of discriminator
            d_loss = d_loss_fn(output=p_data, target=p_noise)
            d_loss_temp.append(d_loss.item())
            d_loss.backward()
            d_optimizer.step()

        d_loss = torch.mean(d_loss_temp)

        generator.zero_grad()

        random_data = get_random_batch(batch_size, g_in_features, device)

        # get outputs
        generated_imgs = generator(random_data)
        prob = discriminator(generated_imgs)

        # calculate loss of generator
        g_loss = g_loss_fn(prob, prob)
        g_loss.backward()
        g_optimizer.step()

        logger.info(f'[RESULT]: Train. Epoch: {epoch}/{epochs} G loss: {g_loss:.5f} D loss: {d_loss:.5f} time: {(time.time() - t):.5f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0., help='SGD weight decay')

    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--in_features', type=int, default=64, help='number of input features of generator')
    parser.add_argument('--k', type=int, default=1, help='number of steps to apply to the discriminator')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--val_size', type=float, default=0.3, help='part of the valid dataset')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='number of processes working on cpu')

    parser.add_argument('--verbose_step', type=int, default=1, help='period of verbose step')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose')

    args = parser.parse_args()

    main(args)
