## Requirements

1. The project was implemented and tested in Python 3.5 and Pytorch 0.4. The higher versions should work after minor modification.
2. Other common modules like numpy, pandas and seaborn for visualization.
3. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.

## Datasets

Our proposed Chinese Character dataset is accessible on [link](https://drive.google.com/drive/folders/1ShCuKkRx0Oeso4qRdHkiATFFXmDV4S5B?usp=sharing)


## Implementation details

### data preparation

build train/validation/test sets,

```
1-make_chinese1_list.py
2-make_letter_list.py
3-make_tiny_letter_list.py
```


### training
```
4-train_capital_letter_resnet20.py
5-train_chinese1_resnet20.py
6-train_tiny_letter_resnet20.py
```
### explanation generation
```
7-FARCE_capital_letter.py
8-FARCE_chinese.py
9-FARCE_mnist.py
```
## Time and Space

All experiments were run on NVIDIA TITAN Xp 


model     | #GPUs | train time |
---------|--------|-----|
train_mnist_resnet20     | 1 | ~10min    | 
train_capital_letter_resnet20    | 1 | ~7min    |
train_chinese1_resnet20    | 1 | ~10min    | 
FARCE_mnist     | 1 | ~20min   |
FARCE_capital_letter     | 2 | ~20min    |
FARCE_chinese     | 1 | ~20min   |


## Further Reading
Methodology described in (Zhao, Yunxia, 2020)[https://arxiv.org/abs/2007.05684]
