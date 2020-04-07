import os
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import argparse
from collections import OrderedDict
from PIL import Image
from loss import *
from preprocess import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=3)
parser.add_argument('--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--save_path', default='./model.pth')
parser.add_argument('--img_folder', default='./output')
parser.add_argument('--palette_data', default=[0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192])
parser.add_argument('--is_aux', default=True)
parser.add_argument('--aux_weight', default=0.4)
parser.add_argument('--n_classes', default=21)

args = parser.parse_args()

print("-----------config")
for arg in vars(args):
	print("{:} : {:}".format(arg, getattr(args, arg)))
print("-----------------")

transform = transforms.Compose([
	Crop_128(),
	transforms.ToTensor()
])

target_transform = transforms.Compose([
	Crop_128(),
	Target_Preprocess()
])

if not os.path.exists('./data'):
	os.makedirs('./data')

if not os.path.exists(args.img_folder):
	os.makedirs(args.img_folder)

if not os.path.exists(args.img_folder+"/train"):
	os.makedirs(args.img_folder+"/train")

if not os.path.exists(args.img_folder+"/val"):
	os.makedirs(args.img_folder+"/val")

trainset = torchvision.datasets.VOCSegmentation(root='./data', image_set='train', transform=transform, target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)

testset = torchvision.datasets.VOCSegmentation(root='./data', image_set='val', transform=transform, target_transform=target_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=True)
net.load_state_dict(torch.load(args.save_path))

net.to(args.device)

criterion = CrossEntropyLoss_aux(aux_weight=args.aux_weight)

train_loss = AverageMeter()
train_accuracy = AverageMeter()
train_tp = torch.zeros(args.n_classes)
train_fp = torch.zeros(args.n_classes)
train_fn = torch.zeros(args.n_classes)

with torch.no_grad():
	for i, data in enumerate(tqdm(trainloader)):
		inputs, targets = data[0].to(args.device), data[1].to(args.device)
		outputs = net(inputs)
		loss = criterion(outputs, targets)


		if args.is_aux:
			outputs = outputs["out"]

		train_loss.update(loss.item(), args.batch_size)
		train_accuracy.update(torch.sum(torch.max(outputs, dim=1)[1]==targets).type(torch.float)/(targets.shape[0]*targets.shape[1]*targets.shape[2]), args.batch_size)

		predicts = torch.max(outputs, dim=1)[1]
		for j in range(args.n_classes):
			train_tp[j]+=(torch.sum((predicts==j) & (targets==j)))
			train_fp[j]+=(torch.sum((predicts==j) & (targets!=j)))
			train_fn[j]+=(torch.sum((predicts!=j) & (targets==j)))

		if not os.path.exists(args.img_folder):
			os.makedirs(args.img_folder)

		for j in range(outputs.shape[0]):
			out_path = args.img_folder+'/train/{:}_output.png'.format(i*args.batch_size+j)
			out_arr = np.array(torch.max(outputs[j], dim=0)[1].cpu(), dtype='uint8')
			out_img = Image.fromarray(out_arr, mode='P')
			out_img.putpalette(args.palette_data)
			out_img.save(out_path)

train_iou = .0
for j in range(args.n_classes):
	train_iou += train_tp[j]/(train_tp[j]+train_fp[j]+train_fn[j])/args.n_classes


test_loss = AverageMeter()
test_accuracy = AverageMeter()
test_tp = torch.zeros(args.n_classes)
test_fp = torch.zeros(args.n_classes)
test_fn = torch.zeros(args.n_classes)
with torch.no_grad():
	for i, data in enumerate(tqdm(testloader)):
		inputs, targets = data[0].to(args.device), data[1].to(args.device)
		outputs = net(inputs)
		loss = criterion(outputs, targets)


		if args.is_aux:
			outputs = outputs["out"]

		test_loss.update(loss.item(), args.batch_size)
		test_accuracy.update(torch.sum(torch.max(outputs, dim=1)[1]==targets).type(torch.float)/(targets.shape[0]*targets.shape[1]*targets.shape[2]), args.batch_size)

		if not os.path.exists(args.img_folder):
			os.makedirs(args.img_folder)

		for j in range(outputs.shape[0]):
			out_path = args.img_folder+'/val/{:}_output.png'.format(i*args.batch_size+j)
			out_arr = np.array(torch.max(outputs[j], dim=0)[1].cpu(), dtype='uint8')
			out_img = Image.fromarray(out_arr, mode='P')
			out_img.putpalette(args.palette_data)
			out_img.save(out_path)

		predicts = torch.max(outputs, dim=1)[1]
		for j in range(args.n_classes):
			test_tp[j]+=(torch.sum((predicts==j) & (targets==j)))
			test_fp[j]+=(torch.sum((predicts==j) & (targets!=j)))
			test_fn[j]+=(torch.sum((predicts!=j) & (targets==j)))

test_iou = .0
for j in range(args.n_classes):
	test_iou += test_tp[j]/(test_tp[j]+test_fp[j]+test_fn[j])/args.n_classes

print('train_loss: {:.3f}, train_accuracy: {:.3f}, test_loss: {:.3f}, test_accuracy: {:.3f}, train_iou: {:.3f}, test_iou: {:.3f}'.format(train_loss.avg, test_loss.avg, train_accuracy.avg, test_accuracy.avg, train_iou, test_iou))
