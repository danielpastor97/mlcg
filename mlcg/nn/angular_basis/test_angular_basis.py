from e3nn.util.test import assert_equivariant
from mlcg.nn.angular_basis import SphericalHarmonics
import torch


def test_sph_equivariance():
    TOLERANCE = 5e-6
    sph = SphericalHarmonics(lmax=4)

    def adapt(x):
        # change the layout of out so that it fits into assert_equivariant
        out = sph(x)
        return torch.hstack(out)

    res = assert_equivariant(
        adapt, irreps_in=sph.irreps_in, irreps_out=sph.irreps_out
    )
    for (parity_k, did_translate), max_abs_err in res.items():
        print(f"parity={parity_k}  translated={did_translate}")
        assert max_abs_err < TOLERANCE
