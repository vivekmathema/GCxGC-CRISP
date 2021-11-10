These samples are deepstacked version of the whole countour using the ROI datafile: CRISP_root\roids_data\Normal-vs-ESRD_ROIs.data
[ Hint:  Goto "1C: AutoROI & DeepStacking" module of CRISP  --> Set Test Image folder path in 'Class-A image'  (ignore Class-B image, as you may only have all images in one folder  --> In "DS ROIs filepath" Select ROI datafile: CRISP_root\roids_data\Normal-vs-ESRD_ROIs.data  -->  Press "Load ROIs for DeepStacking" ( this loads the ROI Co-ordinates which will appear in the list box ) --> press "Build DeepStack data" ( will generate the Deepstacked version of the all images in test folder under a new folder named "stacked_images" inside the Test Image folder. Finally, use location of this stacked folder in "D) Classifier  --> "test images" along with the Trained model for deep stacked images for inferencing. During each step you can see the console screen for what is being uploaded or processed. ]


==>The unknown GCxGC-TOFMS contour samples extracted using same protocol should be pre-processed accordingly before using the pre-traiend model for further training or inferencing.

==>Model to apply for these samples:  .\classifier_data\trained_weights\MixCB_DS_simulated\MixCB-DS-Simulated
