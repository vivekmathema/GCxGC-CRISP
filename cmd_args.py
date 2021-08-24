#Command line arguments for COntourGANs Synthesis segment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gui_type'      , type=int, default= 4                         , help='Use different GUI Schemes [ 0: Breeze, 1: Oxygen, 2: QtCurve, 3: Windows, 4:Fusion ]') #
parser.add_argument('--config_run'    , type=int, default= 0                         , help='[0: false, 1:true]]. Run ContourProfiler in GUI mode ONLY. No configuration modules will be run') #self.model_id 
parser.add_argument('--config_fpath'  , type=str, default= './config/default.conf'   , help='full pathname of the configuration file to run') #self.model_id 
parser.add_argument('--run_session'   , type=str, default= None                      
                                      , help=  '[None,gan_train, gan_syn, train_sr, sr_inf, cls_train, cls_inf]\
                                               | None       : Only load gui with selected configuration, \
                                               | gan_train  : Load and run gui for GAN model training, \
                                               | gan_syn    : Load and run gui for GAN synthesis,\
                                               | train_sr   : Load and run gui for contour super resoultion training,\
                                               | sr_inf     : Load and run gui for contour image enhancement,\
                                               | cls_train  : Load and run gui for classifier training,\
                                               | cls_inf    : Load and run gui for classifier inferencing,\
                                               [NOTE: Commandline configuration run is not currently avaiable for ROIs and DeepStacking]' )

     
args = parser.parse_args()

