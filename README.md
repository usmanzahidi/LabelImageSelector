# Label Image Selector

Given an image set, it selects most distinct images that are suitable for labelling and training of Deep Learning models

## Requirements
`python3` `torchvision` `pickle` `numpy` `opencv-python` `scikit-image` `matplotlib`
`detectron2`

## Installation

```
https://github.com/usmanzahidi/LabelImageSelector.git
```

## Usage

```bash
usage: main.py [-i PATH] [-o PATH] [-n NUMBER] [--entropy BOOLEAN]
-i rgb   image folder path i.e. unlabelled image pool
-o output folder path
-n number of images required for labelling
--entropy include maximum entropy measure in image selection
--no-entropy only use TSNE for image selection
```

## Example:

```bash
-i ./images/test/rgb -o ./images/test/selected -n 51 --entropy|--no-entropy
```

## Reference:
```bash
Zahidi, Usman A., Cielniak, Grzegorz, ”Active Learning for Crop-Weed Discrimination by
Image Classification from Convolutional Neural Network’s Feature Pyramid Levels”, In
Computer Vision Systems, ISBN:978-3-030-87156-7,pages= 245–257,Springer
International Publishing, September 2021.
https://link.springer.com/chapter/10.1007%2F978-3-030-87156-7_20
```