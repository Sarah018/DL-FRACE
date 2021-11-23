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
make_letter_list.py
make_chinese1_list.py
```


### training
```
train_mnist_resnet20.py
train_capital_letter_resnet20.py
train_chinese1_resnet20.py
```
### explanation generation
```
FARCE_mnist.py
FARCE_capital_letter.py
FARCE_chinese.py
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

