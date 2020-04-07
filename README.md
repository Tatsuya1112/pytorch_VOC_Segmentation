# pytorch_VOC_Segmentation

![あ無qqs](https://user-images.githubusercontent.com/45190789/77528170-cbd1d200-6ed0-11ea-8954-203884445987.png)

# Requirements

```
torch
torchvision
tqdm
```

# Datasets

dataフォルダを作成し、torchvisonのMNISTデータセットをダウンロードします

```python
if not os.path.exists('./data'):
    os.makedirs('./data')

trainset = torchvision.datasets.MNIST(root='./data', download=True, train=True, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root='./data', download=True, train=False, transform=transforms.ToTensor())
```

# Models

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

# Results

| model | accuracy | cross-entropy-loss |
| ---- | ---- | ---- |
| TwoLayerNet | 0.972 | 0.103 |
| SimpleConvNet | 0.988 | 0.059 |

# References

ゼロから作るDeep Learning――Pythonで学ぶディープラーニングの理論と実装
https://www.oreilly.co.jp/books/9784873117584/
