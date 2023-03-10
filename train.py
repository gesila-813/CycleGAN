import torch
from dataset import FaceModelFaceDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator

count = 0

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (face, faceModel) in enumerate(loop):
        face = face.to(config.DEVICE)
        faceModel = faceModel.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_faceModel = gen_H(face)
            D_H_real = disc_H(faceModel)
            D_H_fake = disc_H(fake_faceModel.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_face = gen_Z(faceModel)
            D_Z_real = disc_Z(face)
            D_Z_fake = disc_Z(fake_face.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it togethor
            D_loss = (D_H_loss + D_Z_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_faceModel)
            D_Z_fake = disc_Z(fake_face)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_face = gen_Z(fake_faceModel)
            cycle_faceModel = gen_H(fake_face)
            cycle_face_loss = l1(face, cycle_face)
            cycle_faceModel_loss = l1(faceModel, cycle_faceModel)

            
            # identity_face = gen_Z(face)
            # identity_faceModel = gen_H(faceModel)
            # identity_face_loss = l1(face, identity_face)
            # identity_faceModel_loss = l1(faceModel, identity_faceModel)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_face_loss * config.LAMBDA_CYCLE
                + cycle_faceModel_loss * config.LAMBDA_CYCLE
                # + identity_faceModel_loss * config.LAMBDA_IDENTITY
                # + identity_face_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        torch.cuda.empty_cache()

        if idx % config.BATCH_SIZE == 0:
            save_image(fake_faceModel*0.5+0.5, f"result/{epoch}_fake_face_{idx}.png")
            save_image(fake_face*0.5+0.5, f"result/{epoch}_fake_model_{idx}.png")
        
        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))
        torch.cuda.empty_cache()


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = FaceModelFaceDataset(
        root_faceModel=config.TRAIN_DIR+"/train_face", root_face=config.TRAIN_DIR+"/train_face_model",    transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, epoch)

        if config.SAVE_MODEL:
            if (epoch % 50) == 0:
                save_checkpoint(gen_H, opt_gen, filename=str(epoch) + config.CHECKPOINT_GEN_H)
                save_checkpoint(gen_Z, opt_gen, filename=str(epoch) + config.CHECKPOINT_GEN_Z)
                save_checkpoint(disc_H, opt_disc, filename=str(epoch) + config.CHECKPOINT_CRITIC_H)
                save_checkpoint(disc_Z, opt_disc, filename=str(epoch) + config.CHECKPOINT_CRITIC_Z)
            else:
                save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
                save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
                save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
                save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()