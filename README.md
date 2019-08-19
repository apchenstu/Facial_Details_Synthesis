This is the code repo of facial details synthesis from single input image. Paper: [here](https://arxiv.org/abs/1903.10873), Supplemental Material: [here](https://github.com/apchenstu/Facial_Details_Synthesis/blob/master/src/imgs/Supplemental_Material.pdf).

This repository consists 5 individual parts: *DFDN*, *emotionNet*, *landmarkDetector*, *proxyEstimator* and *faceRender*.  The DFDN is based on junyanz's [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), for the landmark and expression detector, we use a simplify version of [openFace](https://github.com/TadasBaltrusaitis/OpenFace), and our proxyEstimator is modified based on [patrikhuber](https://github.com/patrikhuber)'s fantastic work [eos](https://github.com/patrikhuber/eos) .  We want to thank each of them for their kindly work.



# Facial Details Synthesis
### [Anpei Chen*](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+A), [Zhang Chen*](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Guli Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+G), [Ziheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Kenny Mitchell](https://arxiv.org/search/cs?searchtype=author&query=Mitchell%2C+K), [Jingyi Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+J)

We present a single-image 3D face synthesis technique that can handle challenging facial expressions while recovering fine geometric details. Our technique employs expression analysis for proxy face geometry generation and combines supervised and unsupervised learning for facial detail synthesis. On proxy generation, we conduct emotion prediction to determine a new expression-informed proxy. On detail synthesis, we present a Deep Facial Detail Net (DFDN) based on Conditional Generative Adversarial Net (CGAN) that employs both geometry and appearance loss functions. For geometry, we capture 366 high-quality 3D scans from 122 different subjects under 3 facial expressions. For appearance, we use additional 163K in-the-wild face images and apply image-based rendering to accommodate lighting variations. Comprehensive experiments demonstrate that our framework can produce high-quality 3D faces with realistic details under challenging facial expressions. 

![](https://github.com/apchenstu/Facial_Details_Synthesis/blob/master/src/imgs/teaser.png)


# Features
 - **Functionality**
	 * proxy estimation with expression/emotion prior
	 * facial details prediction, i.e. winkles
	 * results visualizer or facial render
- **Input**: single image or images folder
- **Output**: proxy mesh & texture, details displacementMap and normalMap
- **OS**: Window 10

## Set up environment


 1. Install window version *Anaconda Python3.7* and *pytorch*
 2. [Optional] Install *tensorflow* and *keras* if you want to use emotion prior


## Released version
 

 1. Download the released package. [released version](https://1drv.ms/u/s!AjyDwSVHuwr8omaBIMsNku1KDPqq?e=C11URL)
 2. Download models and pre-train weights. 
 
     [DFDN checkpoints](https://1drv.ms/u/s!AjyDwSVHuwr8omMGWNP0PA-X0ASx?e=E1vWrY), unzip to `./DFDN/checkpoints`
     
     [landmork models](https://1drv.ms/u/s!AjyDwSVHuwr8omVnsY5ophd4yxIr?e=XbVjUr), unzip to `./landmarkDetector`
     
     [Optional] [emotionNet checkpoints](https://1drv.ms/u/s!AjyDwSVHuwr8omF7lTcbT6GcxcpN?e=P4kH7N), unzip to `./emotionNet/checkpoints`
     
 3. Install BFM2017
 
    - install eos by `pip install --force-reinstall eos-py==0.16.1`
    - Download [BFM2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html) and copy `model2017-1_bfm_nomouth.h5` to `./proxyEstimator/bfm2017/`.

    - Run `python convert-bfm2017-to-eos.py` to generate `bfm2017-1_bfm_nomouth.bin` in `./proxyEstimator/bfm2017/` folder.

 5. Have fun!

## Usage

* For proxy estimation, 

  ```
  python proxyPredictor.py -i path/to/input/image -o path/to/output/folder [--FAC 1][--emotion 1]
  ```
  
  - For batch processing, you can set `-i` to a image folder.

  - For prior features, you can optional choose one of those two priors: 
      with facial coding features, type `--FAC 1`, 
      with emotion features type `--emotion 1`.

  example: `python proxyPredictor.py -i ./samples/proxy -o ./results`

- For facial details estimation,

  ```
  python facialDetails.py -i path/to/input/image -o path/to/output/folder
  ```
  
  example: 
  
  `python facialDetails.py -i ./samples/details/019615.jpg -o ./results`
  
  `python facialDetails.py -i ./samples/details -o ./results`


    
## Compiling
we suggest you directly download the released package for convenient, if you are interested in compile the source code, please follow the following guidelines:

**on the way .....**

**the visualizer only support mesh + normalMap, the render will also support displacementMap in near future** 


## Citation

If you find this code useful to your research, please consider citing:
```
@InProceedings{Chen_2019_ICCV,  
author = {Chen, Anpei and Chen, Zhang and Zhang, Guli and Mitchell, Kenny and Yu, Jingyi},  
title = {Photo-Realistic Facial Details Synthesis From Single Image},  
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},  
month = {Oct},  
year = {2019}  
}
```
