import numpy as np
import torch
import torch.nn as nn

from module.TransfomerEncoder import _TransformerEncode
from module.unimol_tools.models import UniMolModel
# src_tokens,
#         src_distance,
#         src_coord,
#         src_edge_type,
class Multimodel(nn.Module):
    def __init__(self, input_dim, embd_dim, ff_dim, num_layer, num_head) -> None:
        super(Multimodel, self).__init__()
        self.peakEncoder = _TransformerEncode(input_dim, embd_dim, ff_dim, num_head, num_layer)
        self.smilesEncoder = UniMolModel(output_dim = 1, data_type = "molecule")
        self.l2_norm = nn.functional.normalize

    def forward(self,smiles_input,peaks_input):
        peak_embd = self.peakEncoder(peaks_input)
        smiles_embd = self.smilesEncoder(**smiles_input,eturn_repr=True)
        peak_embd = self.l2_norm(peak_embd, dim=-1)
        smiles_embd = self.l2_norm(smiles_embd, dim=-1)
        return smiles_embd,peak_embd

        
