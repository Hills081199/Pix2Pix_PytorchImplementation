import config
import argparse
import torch
import torch.nn as nn
import transforms as T

from torch.utils.data import DataLoader
from model.discriminator import Discriminator
from model.generator import Generator
from cityscapes import CityScapes
from loss import DLoss, GLoss


if __name__ == '__main__':
    # parser = argparse.ArgumentParser
    # parser.add
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transforms = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    #dataset and dataloader
    data_path = config.data_path
    dataset = CityScapes(data_path, transforms)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    #Get model
    G = Generator().to(device)
    D = Discriminator().to(device)

    #optimizers 
    g_optimizer = torch.optim.Adam(G.parameters(), lr=config.learning_rate)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=config.learning_rate)

    #loss functions
    g_loss = GLoss(alpha=100)
    d_loss = DLoss()

    for epoch in range(config.epochs):
        g_loss_epoch = 0
        d_loss_epoch = 0
        for imgA, imgB in dataloader:
            imgA = imgA.to(device)
            imgB = imgB.to(device)

            # calculate G Loss
            fakeB = G(imgA)
            fake_pred = D(fakeB, imgA)
            loss_g = g_loss(fakeB, imgB, fake_pred)

            # calculate D Loss
            fakeB = G(imgA).detach()
            fake_pred = D(fakeB, imgA)
            real_pred = D(imgB, imgA)
            loss_d = d_loss(fake_pred, real_pred)

            # backpro g
            g_optimizer.zero_grad()
            loss_g.backward()
            g_optimizer.step()

            # backpro d
            d_optimizer.zero_grad()
            loss_d.backward()
            d_optimizer.step()

            g_loss_epoch += loss_g.item()
            d_loss_epoch += loss_d.item()

            break

        g_loss_epoch = g_loss_epoch / len(dataloader)
        d_loss_epoch = d_loss_epoch / len(dataloader)

        print("Epoch {} / {} : G_LOSS : {:.3f}, D_LOSS: {:.3f}".format(epoch+1, config.epochs, g_loss_epoch, d_loss_epoch))
        torch.save(G.state_dict(), "./model_state_dicts/G_at_epoch_{}.pth".format(epoch+1))
        torch.save(D.state_dict(), "./model_state_dicts/D_at_epoch_{}.pth".format(epoch+1))