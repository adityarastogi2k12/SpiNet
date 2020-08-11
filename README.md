# SpiNet
This repository contains the official implementation for SpiNet: A Deep Neural Network for Schatten p-norm Regularized Medical Image Reconstruction

## Training Data

The training data for SpiNet consists of brain scans of four patients and 360 images in total. The testing data consists of 164 images from one patients.

Hemant et al. <a href="#modl">[1]</a>. have released the parallel imaging dataset used in their MoDL paper <a href="#modl">[1]</a>. You can download the full dataset from the below link:

Download Link for file "dataset.hdf5" : https://drive.google.com/file/d/1qp-l9kJbRfQU1W5wCjOQZi7I3T6jwA37/view?usp=sharing


## Files
> The repository consists of the following files :-
1. **R16.mat** :- This file consists of the radial golden angle undersampling mask for 16 X undersampling rate.
2. **MODL** :- This folder consists of the saved model, training, testing, model generation and support funtion files for MODL. The files present in the folder     are as follows :-
    - trn_old_R16.py :- This is the main training file for MODL
    - tst_old_R16.py :- This is the main testing file for MODL
    - model_old.py :- Model generating file for MODL
    - supportFunctions_old_R16.py :- Support function file for MODL used for reading and preprocessing training and testing data, generating PSNR values, loading       saved model etc.

3. **SpiNet** :- This folder consists of the saved model, training, testing, model generation and support funtion files for SpiNet. The files present in the        folder are as follows :-

      - trn_R16_8.py :- This is the main training file for SpiNet
      - tst_R16_8.py :- This is the main testing file for SpiNet
      - model_p_R6_8.py :- Model generating file for SpiNet
      - supportFunctions_R16.py :- Support function file for SpiNet used for reading and preprocessing training and testing data, generating PSNR values, loading         saved model etc.

**Note that dataset.hdf5 and R16.mat should be in MODL/SpiNet folder to while executing the training or testing code.**



## References

<b id="my_anchor">[1].</b> MoDL: Model Based Deep Learning Architecture for Inverse Problems  by H.K. Aggarwal, M.P Mani, and Mathews Jacob in IEEE Transactions on Medical Imaging,  2018 
