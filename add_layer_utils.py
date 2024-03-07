import torch
from torch import nn

def add_unit_init_linear(in_features, out_features, bias=False, init_type="unit"):
    
    if init_type.lower() not in ["unit", "he"]:
        assert "invaild initalize type of adding layer!!"
        
    if in_features != out_features:
        assert "In current version, in feature and out feature must same!!"
    
    new_layer = nn.Linear(in_features, out_features, bias=bias)
    
    if init_type == "unit":
        new_layer.weight.data = torch.nn.Parameter(torch.eye(in_features))
        
    
    return new_layer
    