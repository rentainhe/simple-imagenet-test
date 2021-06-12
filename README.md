# simple-imagenet-test
A simple test code for testing model on Imagenet

## Usage
### 1. Prepare Imagenet Dataset
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

You can put it in any folder you want like this:
```
ILSVRC2012_val_00003579.JPEG
ILSVRC2012_val_00000379.JPEG
ILSVRC2012_val_00000123.JPEG
...
ILSVRC2012_val_00036838.JPEG
ILSVRC2012_val_00016765.JPEG
ILSVRC2012_val_00037741.JPEG
```

### 2. Quick Start
a quick start of testing the accuracy of `ResNext101` on Imagenet val dataset
```bash
$ python main.py --data-path /.../imagenet/val/
```
You need to specify `--data-path`
