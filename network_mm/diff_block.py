




import torch.nn as nn
from network_mm.ffns import FCODE


class DiffBlock(nn.Module):
    def __init__(self, dim, ode_dim, args=None):
        super().__init__()
        if args is None:
            raise ValueError("DiffBlock requires explicit args; parse CLI/config in the entrypoint.")
        self.args = args
        self.blocks = nn.ModuleList()

        diff_type = self.args.diff_type


        for e in diff_type.split('_'):
            e, act = e.split('@')
            if e == None:
                None
            elif e == 'fcode':
                self.blocks.append(FCODE(dim, act, args=self.args))
            else:
                raise NotImplementedError

    def forward(self, x, z0=None):
        # input: [b,n,c]

        # identity = x
        outlist = []
        for block in self.blocks:
            if z0 is not None: # for CDE
                out = block(x, z0=z0)
            else: 
                out = block(x)
            outlist.append(out)
        
        out = sum(outlist)

        return out
