~~~
## NOTICE: THE SOFTWARE REPOSITORY HAS MOST OF ITS CODES EITEHR IN COMPILED FORM OR PROTECTED WITH PASSOWRD TO MAINTAIN CONFIDENTIALITY DURING REVIEW PROCESS. ALL SOURCE CODES AND RELATED DATASETS WILL BE FULLY RELEASED UNDER SUITABLE OPENSOURCE LISCENCE ONCE THE MAIN MANUSCRIPT IS ACCEPTED FOR PUBLICATION
~~~

This github repository is the software section of manuscript (under review) entitled "CRISP: A deep learning architecture for GC×GC-TOFMS contour ROI identification, simulation, and analysis of imaging metabolomics"  

# General Overview of CRISP Software

<img src="/images/graphical_summary.jpg" alt="CRISP Software Overview"/>


## CRISP: A Deep Learning approach for GC×GC-TOFMS Contour ROI Identification, Simulation and untargeted metabolomics Profiling 

**1. GENERAL INFORMATION**
  
CRISP is the novel platform that integrates multiple existing and new state-of-art deep learning (DL) platforms for untargeted profiling of Two-Dimensional Gas Chromatography and Time-of-Flight Mass Spectrometry (GC×GC-TOFMS)-generated data in terms of contour  data. The data can be subjected to analysis on three different modes (Normal, Gradient and Holistically-Nested Edge Detection ‘HNED”) based on user’s preference.  The open source software toolkit is responsible for feature detection, simulation, data enhancement and classification of GC×GC-TOFMS-generated contour for the untargeted metabolite profiling. The integrated platform has novel ability to automatically construct aggregate feature representative contour(AFRC) from group of supplied contour data belonging to different classes/groups to reduce manual selection biasness and subsequently auto-identify contrasting region of interest (ROIs) within these contour based on AFRIs. These ROIs can be algorithmically detected and stacked together, using our own algorithm termed as “DeepStacking*” to create a new representative feature map that maximizes the contrasting features and maximizes the profiling accuracy for each group during classifier training. The Integrated Generative Adversarial Network (GAN) is the main engine that simulates the synthetic Contour data based on the real samples. The Integrated GAN in use is a modified version of SGAN/WGAN that minimizes gradient vanishing and Lipschitz Constraint contained in general SGAN and WGANs. The synthesized output are computed for frechet inception distance (FID) to evaluate their quality of likeliness to the source data. The integrated modules has the framework for efficient high resolution contour image synthesis (64×64 - 512×512) resolution along with customizable contour intensity manipulation. The synthesized images can subjected to contour image super resolution (SR) model based on Cascading Residual Network. The Contour Profiler’s SR network is fully customizable. It has be customized by default and trained using our own Contour-like HR dataset to maximize the model’s capabilities to enhance contour image data. The Classifier is collection of customized version of state-of-art Convolutional neural network (CNN) including VGG16/VGG19/Inception/RasNet/DenseNet for detecting features and obtaining maximum efficient models.  Instead of usual 224×224 Input dimensions, the model is able to input (224×244 or 512×512) dimensions along with customizable model optimizer feature to maximize the contour feature detection capabilities. All models once trained can be restored, continued training and used independently to inference their corresponding inputs. All models created by Contour Profiler have their summary, ROIs Deep stacking co-ordinates, logs and training history including models losses, FIDs*  and QScores*  stored whenever applicable. Taken together, the ContourProfiler provides an all-in-one open source integrated platform for advanced profiling of non-targeted GC×GC-TOFMS Contour data. [Note: * see article for details] 
  
**2. SOFTWARE ARCHITECTURE AND USES**
  
The ContourProfiler is designed for GC×GC-TOFMS and consists of four major components: 
A.                ROI & Deep Stacking 
B.                Contour Simulation & Synthesis 
C.                Coutour Enhancement 
D.                Contour Classifier training & Inference  
Each module can be used independently or in combination for opted outcome. However, the method can be applied for any other contour images of similar 2D datatype. Please Note once all step A-D should be done for same Mode (Normal/Gradient/HNED) 
  
## A. ROI & Deep Stacking 
·Manually choose ROI (1A.  Manual ROI selection): This section allows the user to manually select single ROI graphically or based on input co-ordinates. It allows cropping of the region to get single desired ROI with multiple additional options such as resizing, applying image filters and output file type selection options. The extraction of initial contour data can be done in three different modes: Normal (same as Input); Gradient (Image gradient mode); and Holistically-Nested Edge detection (HNED mode which used pre-trained Neural network for holistic edge-based feature extraction). Please note, once the Contour data is extracted in certain mode, same mode setting should be followed for all contour groups and for future inferencing. Selection of ROI for only single image per group folder is required, all images in the folder will be extracted for same ROIs automatically. 
·General Protocol: (1A.  Manual ROI selection tab) Select single images from each group à Select single ROI/entire image from Contour à  Extract ROIs (new folder will be created with extracted ROIs for all Contours). (Can apply additional mode filters like resizing, HNED/NMRAL/Gradient as described earlier) 
    
·IAFR construction (1B. Build IAFR): This is a novel procedure which allows generalization of the groups of contour images belong to pre-defined profile in terms of weighted aggregate feature extraction. This allows dominant features to represent exclusively in the final single representative image and scarce or outliner features will be minimized algorithmically. Thus, resulting in a final AFRI, which will clearly represent common/average dominant major features present in the group, and reduces manual selection bias. The parameters such as FPS, weight accumulation factor, and cyclic image duration will control the amount of features to be represented in the final AFRI image. Several image augmentation features (e.g.: Blur, Noise, erosion, dilation) can be applied during the construction of AFRI image to enhance its coverage of the profile feature. The AFRI image can be automatically fetched into the subsequent module for AutoROI detection & DeepStacking. 
·General protocol:  (1B. Build IAFR tab) Select any image from the Cropped ROI folder from 1A for two different classes of Contours à Set the parameters (or use default) and apply filters if desired à Process AFRI 

·Multiple ROIs detection & Feature-based dataset construction (1C. AutoROIs and Deep Stacking): 

This modules does multiple important task related to final database construction for Contour simulation and classification. The module has several customized models/algorithmic means to identify multiple ROIs between two AFRI images. These include retrained: VGG16 (more preferred CNN model), SIAMESE, SSIM, PNSR, Hamming and FID (details in publication paper). The top-N number of ROIs are identified based on user settings: Similarity threshold, scanning window size, window overlapping amounts. During processing, the scanning window from left to right will evaluate the features of Contour data in AFRI images and output the scores of each window area and a list of co-ordinates with scores will be generated for top-N least similar regions. There is option to view graph of the scores along the R1 dimension of the contour data. Then the contour feature images for all individual contours in each group can be generated as “DeepStacked” images using the DeepStack dataset builder. This feature stacked dataset can be used for simulation instead of simulating entire contour plots for amplifying the contrasting features between different profiles. Once, the Deepstacking dataset is generated, it can be used for simulation the feature vectors that will be further used for classification. 
·General Protocol:  (1C. AutoROIs and Deep Stacking Tab) Select single images from each group à  Select AFRI contour image for each class  à   Set the correct Scoring method (default is VGG16) à Set the window shifting parameters à Run the Compute ROIs à Build DeepStacking Dataset 
[Note: *You can skip this section, if you want to use entire contour plot for the simulation and classification. However, if you use this option, the same ROIs deep stacking configuration should be used for all steps in Contour Profiler]. 
    
## B. Contour modeling & Generator (Contour Synthesis & Inference): 
The ContourProfiler has advance Contour simulation neural nets based on Generative adversteral network (GANs) with multiple customizable features.  The generator is able to train model from 64×64 - 512×512 pixel contours with limited sample size and good outcome. The larger sample size can provide better results based on diversity of the original images. There is option to set hyper parameters and details options for setting FIDs, Z-vectors, RELU, Learning rates and model Optimizers (B. Contour synthesis à 1. Hyper parameters). The image augmentation (B. Contour synthesis à 2. Augmentation) tab allows addition of multiple different filters (e.g: dilation, distortion, erosion, contrast, brightness, noise, edge sharpening etc.) to initially expand the diversity of limited contours being fed to generator without actually creating any significant change in the overall profile of the input contours. This enhances the dataset for generator to train better especially in low-sample condition. The real-time graphs of model loss and preview of the model along with FID curves will be shown during training for getting status of the generator. The QScore (which forms QCurve) is a customized matrix that will tentatively indicate quality of the output (0.0000-1.0000) mainly based on the sharpness/blurriness feature of the synthetic image during training as compared to the original contour image distribution. If the sharpness of images is relatively similar to that of source data, the QScore will approximately be 1.00. This is not an absolutely reliable score but gives user the trend of ongoing simulation training as real-time model feature update. The training models and its history can be stored/restored at any point of the training along with the configuration. Once the model is trained enough (based of FID curves, Model loss and preview images) the model can be used for synthesizing entirely new contour plots without requiring original datasets (B. Contour synthesis à 3. Synthesize) in a customizable grid of 1×1 – 10×10 shape. There is also a novel advance option to control the intensity of entities in the synthesized Contour which in real situation often relates to the concentration of metabolites. This feature works best when the source images have minimum of background noise or column bleeding.  The image augmentation feature can be applied to the synthesized output contours as well which will assist increasing diversity of the synthesized samples. The advance features of manipulating Z-vector cab lead to higher diversity of the generated samples. Please note, do not mix different extraction modes (Normal/HEDN/gradient) while Contour simulation and Training. It must be same with exact same ROIs/DeepStacking (if applied earlier). The source data is not required once the model is trained enough. All models are compatible for continued training. 
·General protocol:  (B. Contour Simulation | Synthesis) For Training, Select the folder for Source images (Source path) à Set the path for trained models (model path) and provide unique Model id (Model id) à set path for storing preview images (Preview images path) à Setup Hyper parameters & Image augmentation à Start training (the models can be immediately stored and preview images updates during training using hotkeys in simulation preview window). 
For Synthesis, Select the trained model (model path) and provide the exact Model ID used for training & storage à Set the output location (Result images) à Set the required parameters (Z-dim, Contour intensity factors “see article for details”) à Apply image augmentation (optional) for each simulated images à Start synthesis (Synths. Contour button). 
  
## C. Contour Super resolution 
 A super resolution (SR) algorithm uses deep neural nets to clarify, sharpen, and upscale the digital image without losing its content and defining characteristics. This module is responsible for enhancement of image resolution based on our own customized super resolution model trained on our novel contour-like datasets, which enabled production of high-quality simulated contour dataset for training classifier. The module is based on modified version of Cascading Residual Network, which gives a very fast and accurate upscaling of images. The SR model has been trained on custom contour-like datasets which enables more precise upscaling and enhancement of contour images. However, the software provides full-customizable option to build and train on your own custom datasets for building database-specific SR models. The default upscaling is 2x.   
·General protocol: (C. Super resolution Tab) For training, selected dataset (SR training dataset) à Setup hyper parameters (epochs, Learning rates, batch size) à Train models. For Upscaling, select the folder containing low resolution images (LR input) à Select the folder containing trained models (SR model path) à Input name of the train model (SR model id) à Press upscaling button à Up scaled images will be stored on SR output path. 
  
## D.  Contour Classifier Training and Inference 
This is the final module of Contour Profiler which uses customized version of state-of-the-art deep convolutional neural network (D-CNN) such as VGG16, VGG19 Inception V3, DenseNet, and RasNet for classification of Contour using Transfer learning. The technique is applicable for relatively small amount of samples. This module has two submodules: (i) Trainer, which can train on input multiple classes of GC×GC-MS data. The data sets are separated into training and validation sets for evaluating both training and validation accuracy, receiver operator curves and model loss function. The training module was a built-in image augmentation option, which can randomly conduct multiple image augmentation operation (e.g. image shearing, skewing, and distortion) for generating more diversity for the training network. (ii) Inference, this is the ultimate terminal point of the ContourGAN platform. This uses the trained classifier model to conduct inference on the unknown samples and preset threshold. The GUI has real-time visualization of classification and validation accuracy, model loss function which enables users to instantly view the training status. The models can be continued training with updated datasets and training configuration can be stored for future use and unknown sample inferencing. A report file is generated consisting the source images for inferencing and tagged images (optional) with their corresponding classification confidence. 
· General Protocol: (D. Classification Training| Inference tab) For training, Set the source image dataset (Source images)à Set the Model storage path (Model path) and Unique classifier model ID (Model id) à Select the Dense Neural Net for Transfer learning model (default VGG)à Set the hyper parameters (Learning rate/decay, Optimizers, batch size, Training Iterations… etc.) à Set image augmentation (options in image augmentation tab) à Start training. For Inference, Select the trained model path (model path) à Input name of trained classifier model ID (Model id) à Set Input image contour source (Test images) à Set output inference result folder (Tagged images) & report file path and classification threshold and report file output format (Under Inferencing option tab)  à Start Inferencing [Note: The preprocessing of data for inferencing should be done in same manner as the model was trained for. i.e.: Full frame or ROI input in same image mode. For the Deep stacked contour images, the inference input should be of exact same co-ordinates DeepStacked images as used for training model. Use the (1A & C. AutoROIs and Deep Stacking Tab) and load the deep stacking data co-ordinates and preprocess the Input images before inputting for the inferencing. 


## CRISP Classifier Training | Inference module has builtin database construction module which allows generation of custom datasets

```
The images can be multipel different classes : orignial, simulated or mixed contours of same type (Avoid mixing Deepstacked and full-conotur images)

Dataset class name annotation rules

--> Classes are represented by their folder name (e.g: 01Normal, 02ESRD-DM) with prefix 
-->The folder name should start with number "01Class,02Class,03Class..." which will be the same order of code (0x) marked during inferencing/classification for unknown samples

Custom classifier database construction hints:
Sample classifier dataset construction is ( ./CRISP_rootfolder/classifier_data/training_dataest_construct_example/mixCB_simulated_deepstacked_512res ) 

CRISP--> [ d)Classifier|Inference ]  -->  Build dataset --> whole dataset cosntruction option
basepath for entire classes          -->  ./CRISP_rootfolder/classifier_data/training_dataest_construct_example/mixCB_simulated_deepstacked_512res
dataset build path                   -->  ./CRISP_rootfolder/classifier_data/outputs/sample_dataset/
datset basename                      -->  MixCB_simulated_deepstacked_512res_dataset
Training dataset ratio               -->  85:15 default (i.e.: 85% training and 15% validation)
```

## Requirments installation procedure

Install the requirement for the minimum GPU version of the python

```
 pip install -r requirements_gpu.txt

```

Install the requirement for the minimum CPU version of the python


```
 pip install -r requirements_cpu.txt

```

## The stand alone windows package of CPU version of python (which is slow but relatively simple than GPU version) google drive link

```
https://github.com/vivekmathema/GCxGC-CRISP/edit/main/README.md  
```



## CRISP Command line parameters infromation
------------------------------------------------------------------------------------------------------------------
python3 CRISP.py    [-h | --help]              
                                      [--gui_type GUI_TYPE]
                                      [--config_run CONFIG_RUN]              
                                      [--config_fpath CONFIG_FPATH] 
                                      [--run_session RUN_SESSION]


optional arguments:

  -h, --help          Shows this command line help message and exits CRISP

  --gui_type GUI_TYPE   
                          Use different GUI Schemes. Five types available [ 0: Breeze, 1: Oxygen, 2: QtCurve, 3: Windows, 4:Fusion ]

  --config_run CONFIG_RUN
                        [Set 0: false, 1:true]. Run ContourProfiler in GUI mode ONLY. No configuration modules will be run and CRISP will open GUI with default settings

  --config_fpath CONFIG_FPATH
                        full pathname of the configuration file to run. The Confiruation file will be run without any user input or confrimation

  --run_session RUN_SESSION
                        [None,gan_train, gan_syn, train_sr, sr_inf, cls_train, cls_inf] | None : Only loads gui with selected configuration. Following modes are avaialble

                        gan_train  : Load and run gui for GAN model training
                        gan_syn    : Load and run gui for GAN synthesis
                        train_sr   : Load and run gui for contour super resoultion training
                        sr_inf     : Load and run gui for contour image enhancement
                        cls_train  : Load and run gui for classifier training
                        cls_inf    : Load and run gui for classifier inferencing
                        
                        NOTE:  Commandline configuration run is not currently available for ROIs and DeepStacking. 
                                   Due to large numbers of parameters the defination of  each parameter is commented in configuration file itself. 
                                   The Definations of most parameters are presented as tool tip text in status bar of GUI nterface.

------------------------------------------------------------------------------------------------------------------

