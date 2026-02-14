import torch
from thoughtcomm.thoughts.adapter import PrefixAdapter

def test_adapter_shape():
    nz, d, m = 16, 32, 1
    g = PrefixAdapter(nz=nz, d=d, m=m, hidden=64, depth=2)
    z = torch.randn(nz)
    out = g(z)
    assert out.shape == (1, m, d)
