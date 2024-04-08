## Part B

In this part we have been asked to use a pre-trained model instead of training the model from the scratch. So, I have loaded a model(resnet50) from torchvision which is pre-trained on the ImageNet dataset which is somewhat similar to iNaturalist data set.

### Working:
I have loaded a pre trained model i.e. ResNet50 and then fine tuned it using the same data i.e naturalist data. I have used the eights which I got from from training the model on ImageNet data. I also have done the resizing of the image to get the dimension of image in ImageNet data equal to image in my data. I also have made the number of neurons in output layer equal to 10 which was 1000 before to match the number of classes in naturalist dataset which has only 10 classes. Then in fine tuning part I have used different strategies and fine-tuned the model using the iNaturalist dataset, that strategies are given as:
1. Freezing all layers except the last layer
2. Freezing upto k layers and fine tuning the rest
3. Freezing the rest

After that I have done the process by freezing 70, 80 and 90 percent of the layers and also printed the validation acccuracies for different run count. After running what I observed is that I am getting validation accuracies around 70% in 5-6 epochs only, from here we can say that the pre trained model is more effecient than the model we trained from scratch.

