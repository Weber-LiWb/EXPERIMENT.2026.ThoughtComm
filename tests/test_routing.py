import torch
from thoughtcomm.thoughts.routing import agreement_alpha, build_weight_vector, route_latents

def test_route_shapes():
    A, nz = 3, 5
    mask = torch.tensor([[1,0,1,0,0],[0,1,1,0,0],[0,0,1,1,0]], dtype=torch.bool)
    alpha = agreement_alpha(mask)
    w = build_weight_vector(alpha, mode="linear")
    z = torch.arange(nz).float()
    zt = route_latents(z, mask, w)
    assert zt.shape == (A, nz)
