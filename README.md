



This is the code repo for *Facial Details Synthesis From Single Input Image*. [[Paper](https://arxiv.org/abs/1903.10873)] [[Supplemental Material](https://github.com/apchenstu/Facial_Details_Synthesis/blob/master/src/imgs/Supplemental_Material.pdf)]

This repository consists of 5 individual parts: *DFDN*, *emotionNet*, *landmarkDetector*, *proxyEstimator* and *faceRender*. 

 - *DFDN* is used to estimate displacement map, and its network architecture is based on junyanz's [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
 - For *landmarkDetector* and FACS-based expression detector (you can choose between this and *emotionNet*), we use a simplified version of [openFace](https://github.com/TadasBaltrusaitis/OpenFace)
 - [*proxyEstimator*](https://github.com/LansburyCH/eos-expression-aware-proxy/tree/d8d4c7dfec4784c4f02dc8299bb73b80f81a6110) is used to generate proxy mesh using expression/emotion prior. It is modified based on [patrikhuber](https://github.com/patrikhuber)'s fantastic work [eos](https://github.com/patrikhuber/eos)
 - [*faceRender*](https://github.com/gg-z/face_rendering/tree/41b5ea992246dc02768cde715dd39873f0411e13) is used for interactive rendering

We would like to thank each of the related projects for their great work.



# Facial Details Synthesis
### [Anpei Chen*](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+A), [Zhang Chen*](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Guli Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+G), [Ziheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Kenny Mitchell](https://arxiv.org/search/cs?searchtype=author&query=Mitchell%2C+K), [Jingyi Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+J)

We present a single-image 3D face synthesis technique that can handle challenging facial expressions while recovering fine geometric details. Our technique employs expression analysis for proxy face geometry generation and combines supervised and unsupervised learning for facial detail synthesis. On proxy generation, we conduct emotion prediction to determine a new expression-informed proxy. On detail synthesis, we present a Deep Facial Detail Net (DFDN) based on Conditional Generative Adversarial Net (CGAN) that employs both geometry and appearance loss functions. For geometry, we capture 366 high-quality 3D scans from 122 different subjects under 3 facial expressions. For appearance, we use additional 163K in-the-wild face images and apply image-based rendering to accommodate lighting variations. Comprehensive experiments demonstrate that our framework can produce high-quality 3D faces with realistic details under challenging facial expressions. 

![](https://github.com/apchenstu/Facial_Details_Synthesis/blob/master/src/imgs/teaser.png)


## Features
 - **Functionality**
	 * Proxy estimation with expression/emotion prior
	 * Facial details prediction, i.e. winkles
	 * Renderer for results (proxy mesh + normalMap/displacementMap)
- **Input**: Single image or image folder
- **Output**: Proxy mesh & texture, detailed displacementMap and normalMap
- **OS**: Windows 10

## Set up environment

 1. Install windows version of *Anaconda Python3.7* and *pytorch*
 2. [Optional] Install *tensorflow* and *keras* if you want to use emotion prior (*emotionNet*)


## Released version
 

 1. Download the released package. 
 
     Release v0.1.0 [[Google Drive](https://drive.google.com/file/d/1n1gB4bb9TOiFgp8IqfscqFOS3LHzKUIN/view?usp=sharing), [OneDrive](https://1drv.ms/u/s!Ard0t_p4QWIMc5auQCu-G4uKQDo?e=8C9378)]
 2. Download models and pre-trained weights. 
 
     DFDN checkpoints [[Google Drive](https://drive.google.com/file/d/1taK985IJr3m15HG1S7k70bvI-SuusYom/view?usp=sharing), [OneDrive](https://1drv.ms/u/s!Ard0t_p4QWIMeVDOvgXWwZJG57I?e=K1tMIV)] unzip to `./DFDN/checkpoints`
     
     landmark models [[Google Drive](https://drive.google.com/file/d/1rNNkXf372XvtBNiMu4kJe27p9v7nRKgX/view?usp=sharing), [OneDrive](https://1drv.ms/u/s!Ard0t_p4QWIMeJG_0W5UOwTPIM4?e=MSA8BM)] unzip to `./landmarkDetector`
     
     [Optional] emotionNet checkpoints [[Google Drive](https://drive.google.com/file/d/1pTz0wqJLwa_QQxxownu5KsifocDLBF-6/view?usp=sharing), [OneDrive](https://1drv.ms/u/s!Ard0t_p4QWIMdwW7nEx0pV0w56I?e=rQta0m)] unzip to `./emotionNet/checkpoints`
     
 3. Install BFM2017
 
    - Install eos-py by `pip install --force-reinstall eos-py==0.16.1`
    - Download [BFM2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html) and copy `model2017-1_bfm_nomouth.h5` to `./proxyEstimator/bfm2017/`

    - Run `python convert-bfm2017-to-eos.py` to generate `bfm2017-1_bfm_nomouth.bin` in `./proxyEstimator/bfm2017/` folder

 5. Have fun!

## Usage

* For proxy estimation, 

  ```
  python proxyPredictor.py -i path/to/input/image -o path/to/output/folder [--FAC 1][--emotion 1]
  ```
  
  - For batch processing, you can set `-i` to a image folder.

  - For prior features, you can optional choose one of those two priors: 
      FACS-based expression prior, `--FAC 1`, 
      emotion prior, `--emotion 1`.

  example: `python proxyPredictor.py -i ./samples/proxy -o ./results`

- For facial details estimation,

  ```
  python facialDetails.py -i path/to/input/image -o path/to/output/folder
  ```
  
  example: 
  
  `python facialDetails.py -i ./samples/details/019615.jpg -o ./results`
  
  `python facialDetails.py -i ./samples/details -o ./results`


    
## Compiling
We suggest you directly download the released package for convenience. If you are interested in compiling the source code, please go through the following guidelines.

 1. First, clone the source code,

     `git clone https://github.com/apchenstu/Facial_Details_Synthesis.git --recursive`

 2. cd to the root of each individual model then start compiling,

    **landmarkDetector**
     - Executing the `download_libraries.ps1` and `download_models.ps1` with PowerShell script.

	- Open `OpenFace.sln` using Visual Studio and compile the code.

      After compiling, the excuse file would located in `/x64/Release/FaceLandmarkImg.exe`

	**textureRender**
    - install with
	  ```
       mkdir build && cd build
       cmake -A X64 -D CMAKE_PREFIX_PATH=../thirds ../src
      ```
      
     - Open `textureRender.sln` using Visual Studio and compile the code.
     
       After compiling, the excuse file would located in  `Release/textureRender.exe`

    **proxyEstimator**

    - install [vcpkg](https://github.com/Microsoft/vcpkg/)
    
    - install package under vcpkg folder: `./vcpkg install opencv boost --triplet x64-windows`
    
    
    - Install with, 
      ```
      mkdir build && cd build
      cmake .. -A X64 -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]\scripts\buildsystems\vcpkg.cmake
      ```
      
    - Open `eos.sln` using Visual Studio and compile the code.
    
      After compiling, the excuse file would located in  `Release/textureRender.exe`
    
      For more details, please refer to this [repo](https://github.com/LansburyCH/eos-expression-aware-proxy/tree/d8d4c7dfec4784c4f02dc8299bb73b80f81a6110).


    **faceRender**

    - Install with
	  ```
       mkdir build && cd build
       cmake -A X64 -D CMAKE_PREFIX_PATH=../thirds ../src
      ```
     - Open `eos.sln` using Visual Studio and compile the code.
    
       After compiling, the excuse file would located in `build\Release`
   
       Note: The visualizer currently only supports mesh + normalMap, but will also support displacementMap in the near future.

      After compiling, please download [DFDN checkpoints](https://1drv.ms/u/s!AjyDwSVHuwr8omMGWNP0PA-X0ASx?e=E1vWrY), unzip to `./DFDN/checkpoints`. Then you are free to use.

**Others**

On the way .....


## Citation

If you find this code useful to your research, please consider citing:
```
@article{chen2019photo,
  title     = {Photo-Realistic Facial Details Synthesis from Single Image},
  author    = {Anpei Chen, Zhang Chen, Guli Zhang, Ziheng Zhang, Kenny Mitchell and Jingyi Yu},
  journal   = {arXiv preprint arXiv:1903.10873},
  year      = {2019}
}
```
