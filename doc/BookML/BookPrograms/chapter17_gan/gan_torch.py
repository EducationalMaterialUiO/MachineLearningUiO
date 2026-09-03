"""Chapter 17: a generative adversarial network on MNIST, in PyTorch.

The structure mirrors the two-dimensional programs of ``gan.py``: two networks,
two optimisers, and one alternating loop.  Only the architectures and the data
loader change.  The discriminator returns a *logit*, never a probability, so
that the losses can be written with ``BCEWithLogitsLoss`` and evaluated
stably; the sigmoid of Eq. (17.dstar) is applied only when we want to look at
$D(\\bm{x})$ itself.

Usage
-----
    python gan_torch.py --epochs 50                  # non-saturating GAN
    python gan_torch.py --epochs 50 --smooth 0.9     # one-sided label smoothing
    python gan_torch.py --epochs 50 --loss wgangp    # Wasserstein critic
    python gan_torch.py --epochs 50 --arch dcgan     # convolutional
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# 1.  architectures
# ---------------------------------------------------------------------------
def make_generator_fc(k_z=100, n_out=784, widths=(256, 512, 1024)):
    """z in R^k -> image in [-1,1]^784, Eq. (17.generator)."""
    layers, n_in = [], k_z
    for w in widths:
        layers += [nn.Linear(n_in, w), nn.BatchNorm1d(w),
                   nn.LeakyReLU(0.2, inplace=True)]
        n_in = w
    layers += [nn.Linear(n_in, n_out), nn.Tanh()]
    return nn.Sequential(*layers)


def make_discriminator_fc(n_in=784, widths=(1024, 512, 256), p_drop=0.3):
    """image -> logit u(x); D(x) = sigmoid(u(x)).  No batch norm in D."""
    layers = []
    for w in widths:
        layers += [nn.Linear(n_in, w), nn.LeakyReLU(0.2, inplace=True),
                   nn.Dropout(p_drop)]
        n_in = w
    layers += [nn.Linear(n_in, 1)]
    return nn.Sequential(*layers)


class GeneratorDC(nn.Module):
    """DCGAN generator: transposed convolutions, 1 x 28 x 28 output."""

    def __init__(self, k_z=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(k_z, ngf * 4, 7, 1, 0, bias=False),   # 7x7
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 14x14
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False),     # 28x28
            nn.Tanh())

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


class DiscriminatorDC(nn.Module):
    """DCGAN discriminator: strided convolutions, no pooling, no batch norm
    on the input layer (Radford et al. 2016)."""

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),                  # 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),            # 7x7
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False))              # 1x1

    def forward(self, x):
        return self.net(x).view(-1, 1)


def init_dcgan(m):
    """N(0, 0.02^2) initialisation, the DCGAN convention."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# 2.  the Wasserstein gradient penalty, Eq. (17.wgangp)
# ---------------------------------------------------------------------------
def gradient_penalty(D, x_real, x_fake):
    eps = torch.rand(x_real.size(0), *([1] * (x_real.dim() - 1)),
                     device=x_real.device)
    xhat = (eps * x_real + (1 - eps) * x_fake).requires_grad_(True)
    u = D(xhat)
    g = torch.autograd.grad(u.sum(), xhat, create_graph=True)[0]
    return ((g.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# 3.  training
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))])
    data = datasets.MNIST(args.data, train=True, download=True, transform=tf)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True,
                        drop_last=True, num_workers=2)

    conv = args.arch == "dcgan"
    if conv:
        G = GeneratorDC(args.k_z).to(device)
        D = DiscriminatorDC().to(device)
        G.apply(init_dcgan)
        D.apply(init_dcgan)
    else:
        G = make_generator_fc(args.k_z).to(device)
        D = make_discriminator_fc().to(device)

    # beta_1 = 0.5: the momentum of the default Adam destabilises the game
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(64, args.k_z, device=device)

    for epoch in range(1, args.epochs + 1):
        s_d = s_g = s_dr = s_df = 0.0
        for x, _ in loader:
            x = x.to(device)
            if not conv:
                x = x.flatten(1)
            n = x.size(0)

            # ---- discriminator (or critic) step -----------------------------
            for _ in range(args.n_critic):
                z = torch.randn(n, args.k_z, device=device)
                x_fake = G(z).detach()          # no gradient into G here
                u_real, u_fake = D(x), D(x_fake)
                if args.loss == "wgangp":
                    loss_d = (u_fake.mean() - u_real.mean()
                              + args.lam * gradient_penalty(D, x, x_fake))
                else:
                    loss_d = (bce(u_real, torch.full_like(u_real, args.smooth))
                              + bce(u_fake, torch.zeros_like(u_fake)))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            # ---- generator step ---------------------------------------------
            z = torch.randn(n, args.k_z, device=device)
            u_fake = D(G(z))
            if args.loss == "wgangp":
                loss_g = -u_fake.mean()
            elif args.loss == "sat":
                # the original objective: minimise log(1 - D(G(z)))
                loss_g = -bce(u_fake, torch.zeros_like(u_fake))
            else:
                # non-saturating: maximise log D(G(z)),  Eq. (17.nonsat)
                loss_g = bce(u_fake, torch.ones_like(u_fake))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            s_d += loss_d.item()
            s_g += loss_g.item()
            s_dr += torch.sigmoid(u_real).mean().item()
            s_df += torch.sigmoid(u_fake).mean().item()

        m = len(loader)
        print(f"epoch {epoch:03d}  L_D {s_d/m:7.4f}  L_G {s_g/m:7.4f}"
              f"  D(x) {s_dr/m:.3f}  D(G(z)) {s_df/m:.3f}", flush=True)

        with torch.no_grad():
            G.eval()
            img = G(fixed_z).view(-1, 1, 28, 28)
            G.train()
        save_image((img + 1) / 2, f"samples_epoch{epoch:03d}.png", nrow=8)

    torch.save({"G": G.state_dict(), "D": D.state_dict()}, "gan_mnist.pt")
    return G, D


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--k_z", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--smooth", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--n_critic", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data", default="./data")
    p.add_argument("--arch", choices=["fc", "dcgan"], default="fc")
    p.add_argument("--loss", choices=["nonsat", "sat", "wgangp"],
                   default="nonsat")
    args = p.parse_args()
    if args.loss == "wgangp" and args.n_critic == 1:
        args.n_critic = 5           # the critic must be near-optimal
    train(args)
