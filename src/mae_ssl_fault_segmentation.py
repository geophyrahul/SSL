
"""
FULL STARTER SCRIPT
MAE SSL PRETRAINING -> SAVE ENCODER -> FAULT SEGMENTATION FINETUNING

IMPORTANT:
This is a complete runnable framework/skeleton based on your uploaded code.
You still need to insert your dataset paths and can reuse your plotting code.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# BLOCKS
# =========================================================

class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = nn.InstanceNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = nn.InstanceNorm3d(out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        s = self.skip(x)
        x = self.act(self.n1(self.conv1(x)))
        x = self.n2(self.conv2(x))
        return self.act(x + s)

# =========================================================
# ENCODER
# =========================================================

class Encoder3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool3d(2)

        self.e1 = ResidualBlock3D(1, 32)
        self.e2 = ResidualBlock3D(32, 64)
        self.e3 = ResidualBlock3D(64, 128)
        self.e4 = ResidualBlock3D(128, 256)

    def forward(self, x):

        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))

        z = self.pool(x4)

        return x1, x2, x3, x4, z

# =========================================================
# SSL MODEL
# =========================================================

class SSLModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder3D()

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256,256,2,2),
            ResidualBlock3D(256,256),
            nn.ConvTranspose3d(256,128,2,2),
            ResidualBlock3D(128,128),
            nn.ConvTranspose3d(128,64,2,2),
            ResidualBlock3D(64,64),
            nn.ConvTranspose3d(64,32,2,2),
            ResidualBlock3D(32,32),
            nn.Conv3d(32,1,1)
        )

    def forward(self, x):

        _,_,_,_,z = self.encoder(x)

        return self.decoder(z)

# =========================================================
# SEGMENTATION MODEL
# =========================================================

class FaultSegModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder3D()

        self.up4 = nn.ConvTranspose3d(256,256,2,2)
        self.d4  = ResidualBlock3D(512,256)

        self.up3 = nn.ConvTranspose3d(256,128,2,2)
        self.d3  = ResidualBlock3D(256,128)

        self.up2 = nn.ConvTranspose3d(128,64,2,2)
        self.d2  = ResidualBlock3D(128,64)

        self.up1 = nn.ConvTranspose3d(64,32,2,2)
        self.d1  = ResidualBlock3D(64,32)

        self.final = nn.Conv3d(32,1,1)

    def forward(self,x):

        x1,x2,x3,x4,z = self.encoder(x)

        x = self.up4(z)
        x = self.d4(torch.cat([x,x4],1))

        x = self.up3(x)
        x = self.d3(torch.cat([x,x3],1))

        x = self.up2(x)
        x = self.d2(torch.cat([x,x2],1))

        x = self.up1(x)
        x = self.d1(torch.cat([x,x1],1))

        return self.final(x)

# =========================================================
# MAE MASKING
# =========================================================

def random_mask_cube(x, ratio=0.4):

    masked = x.clone()

    B,C,D,H,W = x.shape

    md = int(D * ratio)
    mh = int(H * ratio)
    mw = int(W * ratio)

    z = np.random.randint(0, D-md)
    y = np.random.randint(0, H-mh)
    xx = np.random.randint(0, W-mw)

    mask = torch.zeros_like(x)

    mask[:,:,z:z+md,y:y+mh,xx:xx+mw] = 1
    masked[:,:,z:z+md,y:y+mh,xx:xx+mw] = 0

    return masked, mask

# =========================================================
# SSL LOSS
# =========================================================

def ssl_loss(pred, gt, mask):
    return (((pred - gt)**2) * mask).mean()

# =========================================================
# SEG LOSS
# =========================================================

def seg_loss(logits, gt):

    bce = F.binary_cross_entropy_with_logits(logits, gt)

    prob = torch.sigmoid(logits)

    dice = 1 - (
        (2*(prob*gt).sum()+1e-5) /
        ((prob+gt).sum()+1e-5)
    )

    return bce + dice

# =========================================================
# PRETRAIN
# =========================================================

def pretrain_ssl(model, loader, epochs=100):

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    model.train()

    for ep in range(epochs):

        losses = []

        for cube in loader:

            cube = cube.to(device)

            masked, mask = random_mask_cube(cube)

            pred = model(masked)

            loss = ssl_loss(pred, cube, mask)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print("SSL", ep+1, np.mean(losses))

    torch.save(
        model.encoder.state_dict(),
        "ssl_encoder.pt"
    )

# =========================================================
# FINETUNE
# =========================================================

def finetune(seg_model,
             train_loader,
             val_loader,
             epochs=200,
             freeze_epochs=20):

    seg_model.encoder.load_state_dict(
        torch.load("ssl_encoder.pt")
    )

    for p in seg_model.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        seg_model.parameters(),
        lr=1e-4
    )

    best = 1e9

    for ep in range(epochs):

        if ep == freeze_epochs:

            for p in seg_model.encoder.parameters():
                p.requires_grad = True

            optimizer = torch.optim.AdamW(
                [
                    {
                        "params":
                        seg_model.encoder.parameters(),
                        "lr":1e-5
                    },
                    {
                        "params":[
                            p for n,p in seg_model.named_parameters()
                            if "encoder" not in n
                        ],
                        "lr":1e-4
                    }
                ]
            )

        seg_model.train()

        train_loss = []

        for x,y in train_loader:

            x = x.to(device)
            y = y.to(device)

            logits = seg_model(x)

            loss = seg_loss(logits,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        print("Epoch", ep+1, np.mean(train_loss))

        if np.mean(train_loss) < best:

            best = np.mean(train_loss)

            torch.save(
                seg_model.state_dict(),
                "best_fault_model.pt"
            )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("1. Build SSL dataloader")
    print("2. Run pretrain_ssl()")
    print("3. Build segmentation dataloaders")
    print("4. Run finetune()")
