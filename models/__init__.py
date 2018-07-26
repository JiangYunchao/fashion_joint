"""Networks for fashion net."""
from .fashionnet import FashionNet, FashionNetDeploy
#from .fashionnet_coarse import FashionNetCoarse, FashionNetCoarseDeploy
#from .fashionnet_core import FashionNetCore, FashionNetCoreDeploy
#from .fashionnet_cf import FashionNetCF, FashionNetCFDeploy
#from .fashionnet_2c import FashionNet2C, FashionNet2CDeploy
#from .fashionnet_cp import FashionNetCP, FashionNetCPDeploy
#from .fashionnet_pitf import FashionNetPITF, FashionNetPITFDeploy
from .fashionsolver import FashionNetSolver

__all__ = [
    'FashionNet',
    'FashionNetDeploy',
    'FashionNetCoarse',
    'FashionNetCoarseDeploy',
    'FashionNetCF',
    'FashionNetCFDeploy',
    'FashionNet2C',
    'FashionNet2CDeploy',
    'FashionNetCP',
    'FashionNetCPDeploy',
    'FashionNetPITF',
    'FashionNetPITFDeploy',
    'FashionNetCore',
    'FashionNetCoreDeploy',
    'FashionNetSolver',
]
