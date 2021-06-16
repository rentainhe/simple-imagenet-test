# simple-imagenet-test
A simple test code for testing model on Imagenet

## Usage
### 0. Requirements
- pytorch >= 1.7.0
- torchvision >= 0.8.1

### 1. Data Preparation
we've already provide the Imagenet val label in this project: [val_label.txt](https://github.com/rentainhe/simple-imagenet-test/blob/master/val_label.txt)
```
ILSVRC2012_val_00000001.JPEG 65
ILSVRC2012_val_00000002.JPEG 970
ILSVRC2012_val_00000003.JPEG 230
......
ILSVRC2012_val_00049998.JPEG 232
ILSVRC2012_val_00049999.JPEG 982
ILSVRC2012_val_00050000.JPEG 355
```
You only need to prepare the __Imagenet val dataset__ which you can download from [official website](https://www.image-net.org/)

The directory structure should be:
```
│ILSVRC2012/
├──val/
├── ILSVRC2012_val_00000293.JPEG
├── ILSVRC2012_val_00002138.JPEG
├── ......
```

### 2. Quick Start
a quick start of testing the accuracy of `ResNext101` on Imagenet val dataset
```bash
$ python main.py --data-path /path/to/imagenet/val-dataset/
```
You need to specify `--data-path`

Addition args:
- `--gpu=str` set the specified GPU for testing, e.g. `--gpu 0` to set `device:0` for testing

- `--batch-size=int` set the testing batch size

- `--output=str` set the path to store the config file

### 3. Details
You only need to prepare the model and weight for testing, other details will be released later
