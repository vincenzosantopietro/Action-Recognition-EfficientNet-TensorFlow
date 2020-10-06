# Action recognition from video with EfficientNet models
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/vincenzosantopietro/action-recognition-efficientnet/graphs/commit-activity)  [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://GitHub.com/Naereen/StrapDown.js/issues/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)


This repository contains an action recognition project with the new networks (EfficientNetB<0,1,2,3,4,5,6,7>) recently integrated in tf.keras.application module.
The dataset used for this experiment is UCF101, already pre-processed. 
The image_dataset_from_directory routine is used and this is only available in tf-nightly module at the moment.
While training, the network stores the best weights in a dedicated folder, as well as logs that can be visualized through TensorBoard.
## Environment
The yaml file exported from my conda environment is available in the repository for the sake of reproducibility.  
```shell script
    cd resources
    conda env create -f environment.yml
    conda activate aiml
```
## Train your networks
You can easily start the training process of your favourite EfficientNet model with the following command.

```shell script
    python main.py --batch_size 32 --epochs 10 --efficientnet_id 1
```

This will create an EfficientNetB1 model and train it on the UCF101 dataset. This task is not so hard to learn. You can expect high scores (accuracy greater than 80% in less than 5 epochs). The training time depends on your hardware infrastructure: on my "poor" Nvidia GTX 1080 8GB it takes a couple of hours or so to complete a few epochs but i'm limited to batch size 8. Why ? Because if I increase it I get an OOM (OutOfMemory) Exception. By the way, if you have more memory on your GPU you won't have any trouble.

## Infererence on a sample video
This is under development

## TODO 💥
 - Inference script on custom video file
 - Support for Webcam 

[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/Naereen/badges/)
