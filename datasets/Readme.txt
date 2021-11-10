These are the sample datasets used for training the Models of GANs and Classifiers.

The GAN models are traiend only uisng Original Datasets (NOT Simulates)
The Classifier Models are traiend using both Original and 10x simulated models

The 10x simulated datasets refres to the dataset which has been increased approximately 10-folds by adumenting GAN-bases synthesized contour images.

The simulated dataset should only be used for training classifier modles. However, the CLassifier can also be trained using only original dataset


Google drive download link:  https://drive.google.com/drive/u/1/folders/1IpDga3MZoIh6Ibo2M4G0F_Wqq4nP54nI


For AFRC and ROIs Deepstacking only Original dataset should be used.
Abbreviations
10x   --> referes to approximately 10 times larger dataset vy adding GAN-generated contour images fo same class
DS    --> Deepstacked dataset ( using default provided ROIs )
MicCB --> Mixed column bleed (samples of both low and high column bleed)
Fullframe --> Whole contour (not deep stacked)