import os
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from model import BASNet

def export_onnx(model_pth):
    dummy_input = torch.randn(1, 3, 256, 256, device='cuda')
    input_names = ["input"]
    # output_names = ["d1", "d2", "d3", "d4","d5", "d6", "d7", "d8"]
    output_names = ["d1"]
    # -------------- 3. model define -------------------------------------
    print("...load BASNet...")
    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_pth))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    torch.onnx.export(net, 
                    dummy_input, 
                    "basnet_human.onnx",
                    verbose=False,
                    input_names=input_names, 
                    output_names=output_names)


if __name__ == '__main__':

    model_dir = './saved_models/basnet_bsi/basnet_bsi_itr_64000_train_1.749688_tar_0.110722.pth'
    export_onnx(model_dir)

