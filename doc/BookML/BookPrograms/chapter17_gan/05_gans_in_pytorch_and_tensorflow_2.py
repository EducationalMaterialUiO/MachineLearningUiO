"""Chapter 17: listing 5, from the section on gans in pytorch and tensorflow.

Extracted from doc/BookML/chapter17.tex.
"""

G, D = make_generator_fc(), make_discriminator_fc()
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()

for x, _ in loader:                            # x in [-1,1], flattened
    n = x.size(0)

    # --- discriminator: gradient ascent on V, Eq. (17.dstep) --------------
    z = torch.randn(n, 100)
    x_fake = G(z).detach()                     # no gradient into G here
    u_real, u_fake = D(x), D(x_fake)
    loss_d = (bce(u_real, torch.full_like(u_real, smooth))   # smooth = 1.0 or 0.9
              + bce(u_fake, torch.zeros_like(u_fake)))       # Eq. (17.dloss)
    opt_d.zero_grad(set_to_none=True)
    loss_d.backward()
    opt_d.step()

    # --- generator: non-saturating loss, Eq. (17.nonsat) ------------------
    z = torch.randn(n, 100)
    loss_g = bce(D(G(z)), torch.ones(n, 1))    # = -log D(G(z))
    opt_g.zero_grad(set_to_none=True)
    loss_g.backward()
    opt_g.step()
