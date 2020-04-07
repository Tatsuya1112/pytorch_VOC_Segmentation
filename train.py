import os
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import argparse
from preprocess import *
from utils import *
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2)
parser.add_argument('--epochs', default=30)
parser.add_argument('--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--save_path', default='./model.pth')
parser.add_argument('--weight_decay', default=1e-4)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--is_aux', default=True)
parser.add_argument('--aux_weight', default=0.4)
parser.add_argument('--pretrained', default=True)
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

trainset = torchvision.datasets.VOCSegmentation(root='../pytorch_VOCSegmentation/data', image_set='train', transform=transform, target_transform=target_transform)
testset = torchvision.datasets.VOCSegmentation(root='../pytorch_VOCSegmentation/data', image_set='val', transform=transform, target_transform=target_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True)

net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=True)
net.classifier[4] =  nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
net.aux_classifier[4] =  nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
net.classifier[4].apply(init_weights)
net.aux_classifier[4].apply(init_weights)

net.to(args.device)

criterion = CrossEntropyLoss_aux(aux_weight=args.aux_weight)

optimizer = optim.SGD([
	{"params" : net.backbone.parameters(), "lr" : 1e-3},
	{"params" : net.classifier[0].parameters(), "lr" : 1e-3},
	{"params" : net.classifier[1].parameters(), "lr" : 1e-3},
	{"params" : net.classifier[2].parameters(), "lr" : 1e-3},
	{"params" : net.classifier[3].parameters(), "lr" : 1e-3},
	{"params" : net.classifier[4].parameters(), "lr" : 1e-2},
	{"params" : net.aux_classifier[0].parameters(), "lr" : 1e-3},
	{"params" : net.aux_classifier[1].parameters(), "lr" : 1e-3},
	{"params" : net.aux_classifier[2].parameters(), "lr" : 1e-3},
	{"params" : net.aux_classifier[3].parameters(), "lr" : 1e-3},
	{"params" : net.aux_classifier[4].parameters(), "lr" : 1e-2},
], momentum=args.momentum, weight_decay=args.weight_decay)

lambda1 = lambda epoch: epoch // 30
lambda2 = lambda epoch: 0.95 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])

for epoch in range(args.epochs):

	train_loss = AverageMeter()
	train_accuracy = AverageMeter()
	train_tp = torch.zeros(args.n_classes)
	train_fp = torch.zeros(args.n_classes)
	train_fn = torch.zeros(args.n_classes)


	for i, data in enumerate(tqdm(trainloader)):
		# if i>1:
			# continue
		inputs, targets = data[0].to(args.device), data[1].to(args.device)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if args.is_aux:
			outputs = outputs["out"]

		predicts = torch.max(outputs, dim=1)[1]
		for j in range(args.n_classes):
			train_tp[j]+=(torch.sum((predicts==j) & (targets==j)))
			train_fp[j]+=(torch.sum((predicts==j) & (targets!=j)))
			train_fn[j]+=(torch.sum((predicts!=j) & (targets==j)))


	scheduler.step()

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
			# if i>1:
				# continue
			inputs, targets = data[0].to(args.device), data[1].to(args.device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			if args.is_aux:
				outputs = outputs["out"]

			predicts = torch.max(outputs, dim=1)[1]
			for j in range(args.n_classes):
				test_tp[j]+=(torch.sum((predicts==j) & (targets==j)))
				test_fp[j]+=(torch.sum((predicts==j) & (targets!=j)))
				test_fn[j]+=(torch.sum((predicts!=j) & (targets==j)))

	test_iou = .0
	for j in range(args.n_classes):
		test_iou += test_tp[j]/(test_tp[j]+test_fp[j]+test_fn[j])/args.n_classes



	print('epoch: {:}, train_loss: {:.3f}, train_accuracy: {:.3f}, test_loss: {:.3f}, test_accuracy: {:.3f}, train_mIoU: {:.3f}, test_mIoU: {:.3f}'.format(epoch, train_loss.avg, train_accuracy.avg, test_loss.avg, test_accuracy.avg, train_iou, test_iou))

	torch.save(net.state_dict(), args.save_path)
