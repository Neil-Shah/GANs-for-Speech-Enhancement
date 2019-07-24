# GANs-for-Speech-Enhancement
Generative Adversarial Network implemented for the Time-Frequency based Speech Enhancement

This repository is an implementation of an ICASSP 2018 paper titled, 'TIME-FREQUENCY MASKING-BASED SPEECH ENHANCEMENT USING GENERATIVE
ADVERSARIAL NETWORK'.

Procedure:
1) Extract features for training, validation, and testing using the '.m' scripts stored in the folder 'feature_extraction_codes'
2) Once the training, validation and testing features are extracted, the attached 'training_gan.py' and 'testing_gan.py' files could be used to get extracted enhanced speech masks
3) Once the 'testing_gan.py' script is run, the script 'Reconstruction_from_IRMs.m' could be used for getting the estimated spectrum and then estimated enhanced waveform from the mask predicted by the 'testing_gan.py' script.
