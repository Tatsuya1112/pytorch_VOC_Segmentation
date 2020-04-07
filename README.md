# pytorch_VOC_Segmentation

Pascal VOC2012のデータをpytorchを用いてSemantic Segmentationします。

![18](https://user-images.githubusercontent.com/45190789/78622440-1c2d3480-78c0-11ea-8ac0-19121c5b1b0e.png)

```
pytorch_VOC_Segmentation/
　├ train.py
　├ test.py
　├ loss.py
　├ utils.py
　├ data/
　└ output/
　    ├ train/
　    └ val/
```

# Requirements

```  
argparse
numpy
torch
torchvision
tqdm
```

# Dataset

torchvisonのVOCSegmentationデータセットをダウンロードします

```python
trainset = torchvision.datasets.VOCSegmentation(root='./data', image_set='train', transform=transform, target_transform=target_transform)
testset = torchvision.datasets.VOCSegmentation(root='./data', image_set='val', transform=transform, target_transform=target_transform)

```


# Model

torchvisionよりCOCO train2017で訓練済みのdeeplabv3_resnet101をダウンロードします。

```python
net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=True)
```

classifier層とaux_classifier層の最後のConv2dの部分を初期化して学習し直します。

```python
def init_weights(m):
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)

net.classifier[4] =  nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
net.aux_classifier[4] =  nn.Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
net.classifier[4].apply(init_weights)
net.aux_classifier[4].apply(init_weights)
```

# Loss

(classification層の出力のクロスエントロピーロス) + (aux_classification層の出力のクロスエントロピーロス×0.4)

# Result

| model | pixelwise accuracy | loss | mIoU |
| ---- | ---- | ---- | ---- |
| deeplabv3_resnet101 | 0.972 | 0.103 | 00 |

# References

つくりながら学ぶ！PyTorchによる発展ディープラーニング　　　　　　　　　　
https://book.mynavi.jp/ec/products/detail/id=104855
