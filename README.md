# SimCLR-in-TensorFlow-2
(Minimally) implements SimCLR ([A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) by Chen et al.) in TensorFlow 2. Uses many delicious pieces of `tf.keras` and TensorFlow's core APIs. A report is available [here](https://app.wandb.ai/sayakpaul/simclr/reports/Towards-self-supervised-image-understanding-with-SimCLR--VmlldzoxMDI5NDM).

## Acknowledgements
I did not code everything from scratch. This particular research paper felt super amazing to read and often felt natural to understand, that's why I wanted to try it out myself and come up with a minimal implementation. I reused the works of the following for different purposes -
- Data augmentation policies comes from here: https://github.com/google-research/simclr/blob/master/data_util.py.
- Loss function comes from here: https://github.com/sthalles/SimCLR-tensorflow/blob/master/utils/losses.py.
- TSNE visualization referred from here: https://github.com/thunderInfy/simclr/blob/master/resnet-simclr.py.

Following are the articles I studied for understanding SimCLR other than the paper:
- [Understanding SimCLR — A Simple Framework for Contrastive Learning of Visual Representations with Code](https://medium.com/analytics-vidhya/understanding-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-d544a9003f3c)
- [Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://sthalles.github.io/simple-self-supervised-learning/)
- [Illustrated SimCLR](https://amitness.com/2020/03/illustrated-simclr/) (This one does an amazing job at explaining the loss function" NT-XEnt Loss)

Thanks a ton to the **ML-GDE program** for providing the GCP Credits using which I could run the experiments, store the intermediate results on GCS buckets as necessary. _All the notebooks can be run on Colab though_.

## Dataset
- Subset of ImageNet: https://github.com/thunderInfy/imagenet-5-categories

## Architecture
```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_2 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
resnet50 (Model)             (None, 7, 7, 2048)        23587712
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048)              0
_________________________________________________________________
dense (Dense)                (None, 256)               524544
_________________________________________________________________
activation (Activation)      (None, 256)               0
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896
_________________________________________________________________
activation_1 (Activation)    (None, 128)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                6450
=================================================================
Total params: 24,151,602
Trainable params: 24,098,482
Non-trainable params: 53,120
```

## Contrastive learning progress
![](https://i.ibb.co/9yM4RzQ/download.png)

## Training with 10% training data using the learned representations (linear evaluation)
![](https://i.ibb.co/GV44Xsk/download-1.png)

```
loss: 1.1009 - accuracy: 0.5840 - val_loss: 1.1486 - val_accuracy: 0.5280
```

This is when I only took the base encoder network i.e. _without any non-linear projections_. I presented results with different projection heads as well (available [here](https://github.com/sayakpaul/SimCLR-in-TensorFlow-2/blob/master/Linear_Evaluation_Imagenet_Subset.ipynb)) but this one came to be the best.

## Learned representations with TSNE
![](https://i.ibb.co/ckz1mbv/download-2.png)

This is when I only took the base encoder network i.e. _without any non-linear projections_. I presented results with different projection heads as well (available [here](https://github.com/sayakpaul/SimCLR-in-TensorFlow-2/blob/master/Linear_Evaluation_Imagenet_Subset.ipynb)) but this one came to be the best.

## Supervised training with the full training dataset

Here's the architecture that was used:

```

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_4 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
resnet50 (Model)             (None, 7, 7, 2048)        23587712
_________________________________________________________________
global_average_pooling2d_1 ( (None, 2048)              0
_________________________________________________________________
dense_1 (Dense)              (None, 256)               524544
_________________________________________________________________
activation (Activation)      (None, 256)               0
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 1285
=================================================================
Total params: 24,113,541
Trainable params: 24,060,421
Non-trainable params: 53,120
```

![](https://i.ibb.co/WVwpZJk/download-3.png)

```
loss: 0.6623 - accuracy: 0.7528 - val_loss: 1.0171 - val_accuracy: 0.6440
```

We see a 12% increase here. The accuracy with the SimCLR framework could further be increased with better pre-training in terms of the following aspect:
- More unsupervised data. If we could gather a larger corpurs of images for the pre-training task (think of ImageNet) that would have definitely helped.
- I only trained using the SimCLR framework for 200 epochs. Longer training could have definitely helped.
- Architectural considerations and hyperparameter tuning:
	- Temperature (tau) (I used 0.1)
	- Mix and match between the different augmentation policies shown in the paper and the strength of the color distortion.
	- Different projection heads.

_SimCLR benefits from larger data_. Ting Chen (the first author of the paper) suggested to go for an augmentation policy (when using custom datasets) that's not too easy nor too hard for the contrastive task i.e. the contrastive accuracy should be high (e.g. > 80%).

## Pre-trained weights
Available here - `Pretrained_Weights`.
