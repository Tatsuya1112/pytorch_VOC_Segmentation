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

# Dataset

torchvisonのVOCSegmentationデータセットをダウンロードします

```python
trainset = torchvision.datasets.VOCSegmentation(root='./data', image_set='train', transform=transform, target_transform=target_transform)
testset = torchvision.datasets.VOCSegmentation(root='./data', image_set='val', transform=transform, target_transform=target_transform)

```


# Model

COCO train2017で訓練済みのdeeplabv3_resnet101を用います。

```python
net = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=True)
```

## backbone

## classifier

```
DeepLabHead(
  (0): ASPP(
    (convs): ModuleList(
      (0): Sequential(
        (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (1): ASPPConv(
        (0): Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (2): ASPPConv(
        (0): Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(24, 24), dilation=(24, 24), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (3): ASPPConv(
        (0): Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(36, 36), dilation=(36, 36), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (4): ASPPPooling(
        (0): AdaptiveAvgPool2d(output_size=1)
        (1): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU()
      )
    )
    (project): Sequential(
      (0): Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.5, inplace=False)
    )
  )
  (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (3): ReLU()
  (4): Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
)
```

## aux_classifier

# Loss

(classification層からの出力のクロスロピーロス)　+ (aux_classification層の出力のクロスエントロピーロス*0.4)

# Result

| model | pixelwise accuracy | loss | mIoU |
| ---- | ---- | ---- | ---- |
| TwoLayerNet | 0.972 | 0.103 | 00 |
| SimpleConvNet | 0.988 | 0.059 | 00 |

# References

ゼロから作るDeep Learning――Pythonで学ぶディープラーニングの理論と実装
https://www.oreilly.co.jp/books/9784873117584/
