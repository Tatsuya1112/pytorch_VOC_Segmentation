# pytorch_VOC_Segmentation

Pascal VOC2012のデータをpytorchを用いてSemantic Segmentationします。

![18](https://user-images.githubusercontent.com/45190789/78622440-1c2d3480-78c0-11ea-8ac0-19121c5b1b0e.png)

# Requirements

```  
argparse
numpy
torch
torchvision
tqdm
```

# Datasets

torchvisonのVOCSegmentationデータセットをダウンロードします

```python
trainset = torchvision.datasets.VOCSegmentation(root='./data', image_set='train', transform=transform, target_transform=target_transform)
testset = torchvision.datasets.VOCSegmentation(root='./data', image_set='val', transform=transform, target_transform=target_transform)

```


# Models

COCO train2017で訓練済みのdeeplabv3_resnet101を用います。

```python
net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=True)
```

### TwoLayerNet

2層の全結合層のみから成るモデル


```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                   [-1, 50]          39,250
              ReLU-2                   [-1, 50]               0
            Linear-3                   [-1, 10]             510
================================================================
Total params: 39,760
Trainable params: 39,760
Non-trainable params: 0
----------------------------------------------------------------
```

### SimpleConvNet

1層の畳み込み層と2層の全結合層から成るモデル

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 30, 24, 24]             780
              ReLU-2           [-1, 30, 24, 24]               0
         MaxPool2d-3           [-1, 30, 12, 12]               0
            Linear-4                  [-1, 100]         432,100
              ReLU-5                  [-1, 100]               0
            Linear-6                   [-1, 10]           1,010
================================================================
Total params: 433,890
Trainable params: 433,890
Non-trainable params: 0
----------------------------------------------------------------
```

# Loss

classification
aux_classification *0.4

CrossEntropyLoss
aux_loss = True

# Results

| model | pixelwise accuracy | loss | mIoU |
| ---- | ---- | ---- | ---- |
| TwoLayerNet | 0.972 | 0.103 | 00 |
| SimpleConvNet | 0.988 | 0.059 | 00 |

# References

ゼロから作るDeep Learning――Pythonで学ぶディープラーニングの理論と実装
https://www.oreilly.co.jp/books/9784873117584/
