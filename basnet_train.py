import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet

import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def bce_ssim_loss(pred,target):

	bce_out = bce_loss(pred,target)
	ssim_out = 1 - ssim_loss(pred,target)
	iou_out = iou_loss(pred,target)

	loss = bce_out + ssim_out + iou_out

	return loss

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):

	loss0 = bce_ssim_loss(d0,labels_v)
	loss1 = bce_ssim_loss(d1,labels_v)
	loss2 = bce_ssim_loss(d2,labels_v)
	loss3 = bce_ssim_loss(d3,labels_v)
	loss4 = bce_ssim_loss(d4,labels_v)
	loss5 = bce_ssim_loss(d5,labels_v)
	loss6 = bce_ssim_loss(d6,labels_v)
	loss7 = bce_ssim_loss(d7,labels_v)
	#ssim0 = 1 - ssim_loss(d0,labels_v)

	# iou0 = iou_loss(d0,labels_v)
	#loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7#+ 5.0*lossa
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(
        loss0.item(),
        loss1.item(),
        loss2.item(),
        loss3.item(),
        loss4.item(),
        loss5.item(),
        loss6.item()
    ))
	# print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

	return loss0, loss


# ------- 2. set the directory of training dataset --------

data_dir = '/data/fangcheng.ji/datasets/human_segmentation'
tra_image_dir = '/images/'
tra_label_dir = '/masks/'

model_dir = "./saved_models/basnet_bsi/"
load_model_name = "basnet_bsi_itr_6000_train_4.073827_tar_0.397622.pth"

writer = SummaryWriter('./log')

epoch_num = 100000
batch_size_train = 14
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*')
tra_lbl_name_list = [x.replace(tra_image_dir, tra_label_dir).replace('.' + x.split('.')[-1], '.png') \
                     for x in tra_img_name_list]

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        RandomCrop(224),
        ToTensorLab(flag=0)]))

salobj_dataloader = DataLoader(
    salobj_dataset,
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

# ------- 3. define model --------
# define the net
net = BASNet(3, 1, device="cuda")
if load_model_name:
    print(f"loading weight from {load_model_name}")
    net.load_state_dict(torch.load(model_dir + load_model_name))

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 6000
ite_num4val = 0
running_loss = 0.0
running_tar_loss = 0.0

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs_v = inputs.to(net.device)
        labels_v = labels.to(net.device)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.item()
        running_tar_loss += loss2.item()

        # tensorboard writer
        writer.add_scalar("Loss/loss_d0", loss2.item(), ite_num)
        writer.add_scalar("Loss/loss_total", loss.item(), ite_num)

        writer.add_scalar("Loss/running_loss_mean", running_loss / ite_num4val, ite_num)
        writer.add_scalar("Loss/tar_loss_mean", running_tar_loss / ite_num4val, ite_num)

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, d7, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1,
            epoch_num,
            (i + 1) * batch_size_train,
            train_num,
            ite_num,
            running_loss / ite_num4val,
            running_tar_loss / ite_num4val
        ))

        if ite_num % 2000 == 0:  # save model every 2000 iterations

            torch.save(net.state_dict(), model_dir + "basnet_bsi_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

print('-------------Congratulations! Training Done!!!-------------')
