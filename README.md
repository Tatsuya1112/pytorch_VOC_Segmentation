# pytorch_VOC_Segmentation

Pascal VOC2012のデータをpytorchを用いてSemantic Segmentationします。

![18](https://user-images.githubusercontent.com/45190789/78622440-1c2d3480-78c0-11ea-8ac0-19121c5b1b0e.png)

```
pytorch_VOC_Segmentation/
　├ train.py
　├ test.py
　├ loss.py
　├ utils.py
　├ preprocess.py
　├ model.pth
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

torchvisonのVOCSegmentationデータセットをダウンロードします。

```python
trainset = torchvision.datasets.VOCSegmentation(root='./data', image_set='train', transform=transform, target_transform=target_transform)
testset = torchvision.datasets.VOCSegmentation(root='./data', image_set='val', transform=transform, target_transform=target_transform)

```

# Preprocess

``` python
class Crop_128:
    def __call__(self, x):
        W, H = x.size
        D = min(W, H)
        x = x.crop(((W-D)/2, (H-D)/2, (W+D)/2, (H+D)/2))
        x = x.resize((128, 128))
        return x
```

128×128のサイズに画像を一律にクロップします。

# Model

torchvisionよりCOCO train2017で学習済みのdeeplabv3_resnet101をダウンロードします。

```python
net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=True)
```

classifier層とaux_classifier層の最後のConv2dの部分の重みを初期化して学習し直します。

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

| model | loss | pixel-wise accuracy | mIoU |
| ---- | ---- | ---- | ---- |
| deeplabv3_resnet101 | 1.764 | 0.776 | 0.373 |

![19](https://user-images.githubusercontent.com/45190789/78628776-26572f00-78d0-11ea-8cef-b23ad1476101.png)
![20](https://user-images.githubusercontent.com/45190789/78628779-28b98900-78d0-11ea-94cd-e67c5aacc4e2.png)
![21](https://user-images.githubusercontent.com/45190789/78628785-2b1be300-78d0-11ea-8d17-dfd0274890fd.png)

