CSCE 5215
Project 2
Troy Krupinski

Purpose:
This project contains Image classification on the realistic CelebA dataset CelebA
Dataset (cuhk.edu.hk )large-scale dataset. This dataset contains images, annotations, and
landmark measurements for 10,177 celebrities. The version of the CelebA dataset that you need
to use is the aligned and cropped version. This version has been pre-processed in the manner
indicated by its title. Three partitions, namely training, validation and test are available with this
version of the dataset. Accuracy on the validation partition will serve as the basis for
determining the value of important hyperparameters such as the sampling fraction (it may not
be possible to train on the entire training partition as it is too large) and the number of epochs.
Once these hyperparameter values are determined, record the accuracy on the test partition.
During the course of this project, you will gain expertise in answering queries on Image datasets
via Image classification models. These models will be built with the Keras deep learning
framework which is one of the widely used frameworks for deep learning.
Before we explore actual project requirements, we will summarize two major tools that we will
be using for this project. The tools are:
1. Use of pretrained models. We will make use of the vgg16 pretrained model. To become
familiar with pretrained models, review and run the vgg16 pretrained model using the
Tutorial 3 notebook available from Canvas. For the large scale CelebA dataset with over
200,000 images a pretrained model is vital to keep training time within reasonable limits
without compromising on accuracy.
2. Use of multi target classification. The CelebA dataset contains multiple annotations, each of
which is a potential classification target. Rather than training multiple models for each target
we will use a single model that can be deployed on multiple targets at the same time (see
Tutorial 3). This tutorial not only exposes you to multi target classification but also to data
frame sampling which will also help to speed up the training process significantly.
