# FSS-1000: A 1000 Class Dataset for Few-shot Segmentation 

<img src='examples/example.png' align="left">

We provide our dataset and PyTorch implementation for relation network benchmark. Details are in our [paper](https://arxiv.org/abs/1907.12347). 

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- PyTorch 0.4+

## FSS-1000 Dataset
- Google drive: [download here](https://drive.google.com/open?id=16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI)
- Online Preview: Coming soon

## Getting Started
### Testing
First, download pretrained model [here](https://drive.google.com/open?id=1Vk0Pq8vOZrfrDtCISMcJmAQnt9jkXfPn).

```
python autolabel.py -sd imgs/example/support -td imgs/example/query
```

- Set option ```-sd``` to the support directory and the script will input them as support set. 
- Set option ```-td``` to the path of your query images.
- Results will be saved under ```./results```
  
### Testing your own data
- Label 5 support images following the format in ```imgs/example/support/```.  
- Set your support and query path accordingly.

### Training

Arrange the dataset as described in ```get_oneshot_batch()``` in ```training.py```, then run

```
python training.py
```

## Citing

If you use this repository, dataset or want to reference our work, please use the following BibTeX entry.

```
@article{FSS1000,
Author = {Xiang Li and Tianhan Wei and Yau Pun Chen and Yu-Wing Tai and Chi-Keung Tang},
Title = {FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation},
Year = {2020},
Journal = {CVPR},
}
```
