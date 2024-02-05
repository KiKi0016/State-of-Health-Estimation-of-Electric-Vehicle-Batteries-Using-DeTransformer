# State-of-Health-Estimation-of-Electric-Vehicle-Batteries-Using-DeTransformer
Deep learning of lithium-ion battery SOH using the DeTransformer model learns the aging characteristics of the battery and then makes predictions about the battery SOH in order to monitor the health of batteries in electric vehicles.

# Prerequisites
### Dependencies
```
python==3.10.12
CUDA==11.8
pytorch==2.1.0
numpy==1.26.0
pandas==2.1.1
matplotlib==3.8.0
h5py==3.10.0
scipy==1.11.3
tqdm==4.66.1
tensorboardX==2.6.2.2
```


# Dataset  

We employed a publicly available dataset from the paper ['Data driven prediciton of battery cycle life before capacity degradation' by K.A. Severson, P.M. Attia, et al](https://www.nature.com/articles/s41560-019-0356-8). 

The dataset are available at ['TOYOTA RESEARCH INSTITUTE'](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204). 

-'2017-05-12_batchdata_updated_struct_errorcorrect.mat'
-'2017-06-30_batchdata_updated_struct_errorcorrect.mat'
-'2018-04-12_batchdata_updated_struct_errorcorrect.mat'

The dataset consists of 124 commercially available lithium-ion phosphate (LFP) cells/graphite batteries (A123 Systems, model APR18650M1A, 1.1 Ah nominal capacity). 

This dataset was specifically curated to investigate the aging process of lithium-ion batteries under different fast charging conditions.

The dataset folder provides related codes for calculating SOH based on battery SOC.


# Experiment

### Training
```
python DeTransformer.py
```

### Test
```
python test.py
```


# Visualisation
We provide the drawing code about the prediction results in the visualisation folder to see more intuitively the performance of the DeTransformer model for SOH and RUL prediction of lithium batteries.
