<div>
 
**License plates segmentation and rectification project

</div>


The main features of this project:

 - Google Colabs training code for License Plates segmentation DL-models (based on [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch), using [Pytorch Lightning](https://www.pytorchlightning.ai/) as a framework)
 - Python code for rectifying detected plates (using [OpenCV](https://opencv.org/)) into flat images to be read by OCR models later
 - Helper function converting CoCo (and [Via](https://www.robots.ox.ac.uk/~vgg/software/via/)) annotations into masks
 - Helper funciton optimising Pytorch and Lightning models into TensorRT scripts
 

### 📋 Table of content
1. [Training models](#training)
2. [Rectifying masks](#rectifying)
3. [Helper scripts](#helper)
    1. [Annotation converters](#convert_annotation)
    2. [Models converters](#convert_models)
4. [License](#license)


### ⏳ Training models <a name="training"></a>

Two Colab notebooks are provided to train segmentation models for License Plates: [lpr_seg.ipynb](https://github.com/IncrediBlame/lpr_demo/blob/master/lpr_seg.ipynb) and [lpr_seg_crops.ipynb](https://github.com/IncrediBlame/lpr_demo/blob/master/lpr_seg_crops.ipynb)

The former one trains a model for full-scale images with possibly several license plates on every image. It only requires a train set of images and masks to be operational.

The latter one uses the same images but applies a custom augmentation to them: it randomly rotates an image by a small angle, then randomly crops only one license plate from its bounding box with a random padding from each side, then rescales the obtained plate to 192x192 image, and finally generates a high precision mask. This notebook requires a set of images plus a CoCo 1.0 json annotation file with precise coordinates. It doesn't use masks.

#### Disclaimer: Notebooks demonstrate excertps of the dataset we used, but this project does not provide the actual dataset. You will have to find or collate your own dataset to train your models. There are several open LPR datasets available on the Internet.


### 💡 Rectifying masks <a name="rectifying"></a>

Rectifying is achieved in several stages of filtering.

Given the original image.

<p align="center"><img src="https://github.com/IncrediBlame/lpr_demo/blob/master/pics/1.png" width="512" /></p>

We threshold it several times.

<p align="center"><img src="https://github.com/IncrediBlame/lpr_demo/blob/master/pics/2.png" width="256" /></p>

Then separate the main component to remove noise.

<p align="center"><img src="https://github.com/IncrediBlame/lpr_demo/blob/master/pics/3.png" width="256" /></p>

Then separate digits (optional).

<p align="center"><img src="https://github.com/IncrediBlame/lpr_demo/blob/master/pics/4.png" width="256" /></p>

To determine their directions (optional).

<p align="center"><img src="https://github.com/IncrediBlame/lpr_demo/blob/master/pics/5.png" width="256" /></p>

Directions are used to split top/bottom/left/right sides of the mask.

<p align="center"><img src="https://github.com/IncrediBlame/lpr_demo/blob/master/pics/6.png" width="256" /></p>

Which are then used to warp-rectify the image.

<p align="center"><img src="https://github.com/IncrediBlame/lpr_demo/blob/master/pics/7.png" width="256" /></p>


Script [rectify_bundle.py](https://github.com/IncrediBlame/lpr_demo/blob/master/rectify_bundle.py) is used to rectify images from a directory and provided as a usage example. Script [rectify_one.py](https://github.com/IncrediBlame/lpr_demo/blob/master/rectify_one.py) takes one image and one mask with possibly multiple License Plates and slices it into crops. Then it calls [get_corners.py](https://github.com/IncrediBlame/lpr_demo/blob/master/get_corners.py) on each of them. The latter two could be applied to your use-case.


### 📦 Helper scripts <a name="helper"></a>

#### 1. Annotation converters <a name="convert_annotation"></a>

Script [helpers/annotation2mask.py](https://github.com/IncrediBlame/lpr_demo/blob/master/helpers/annotation2mask.py) converts Via annotations into masks.

Script [helpers/coco2crop.py](https://github.com/IncrediBlame/lpr_demo/blob/master/helpers/coco2crop.py) converts CoCo annotations into crops of individual License Plates.

Script [helpers/coco2mask.py](https://github.com/IncrediBlame/lpr_demo/blob/master/helpers/coco2mask.py) converts CoCo annotations into masks.

Script [helpers/mask2crop.py](https://github.com/IncrediBlame/lpr_demo/blob/master/helpers/mask2crop.py) converts masks into crops of individual License Plates.

Script [helpers/pascalmask2mask.py](https://github.com/IncrediBlame/lpr_demo/blob/master/helpers/pascalmask2mask.py) converts masks in Pascal VOC format to normal masks for training.

#### 2. Models converters <a name="convert_models"></a>

Script [helpers/lightning2model.py](https://github.com/IncrediBlame/lpr_demo/blob/master/helpers/lightning2model.py) converts Pytorch Lightning model to normal Pytorch.

Script [helpers/model2trt.py](https://github.com/IncrediBlame/lpr_demo/blob/master/helpers/model2trt.py) converts normal Pytorch model into TensorRT model.


There are more utils in the [helpers/](https://github.com/IncrediBlame/lpr_demo/blob/master/helpers/) directory, which the reader can explore.


### 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/IncrediBlame/lpr_demo/blob/master/LICENSE)
