# SIIM-ISIC Melanoma Classification


## 1. General

Competition Name: [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification)

Team Name: Yuval & nosound

Private Leaderboard Score: 0.9441 (Score of With - Context Submission)

Private Leaderboard Place: 27

The With-Context Prize

Members:



*   Yuval Reina, Tel - Aviv, Israel, [Yuval.Reina@gmail.com](mailto:Yuval.Reina@gmail.com)
*   Zahar Chikishev, Berlin, Germany, zahar.chikishev@gmail.com


## 2. Background on us:


### Yuval Reina:


### Academic and professional background

I hold a BSc in Electrical Engineering and an MBA. I have worked for the last 30 years as an R&D engineer and executive manager in different areas, all related to H/W design and RF (no AI or ML). 

ML knowledge comes from online courses (Coursera, Udemy), online articles and Kaggle competitions - Kaggle GM.

**Why this competition?**

I wanted to have some experience with medical imaging


### Time spent on the competition

I spent about 7H a week on this competition, which sums up to around 70H overall. 


### Zahar Chikishev:


### Academic and professional background

M.Sc. in applied math, I have about 10 years of industry experience in software and algorithms development. For the last year and a half I have been a full-time kaggler.

**Why this competition?**

This competition had generally fitted me well, leveraging my previous experience with image classification competition, e.g. the Recursion Cellular Classification competition, where we also teamed up with Yuval and achieved 7th place. 


### Time spent on the competition

I spent about 4-5 working weeks on the competition. 


## 3. Summary

Our solution is based on two step model + Ensemble:



1. Base model for feature extraction per image
2. Transformer model - combining all the output features from a patient and predict per image. 
3. The 2nd stage also included some post - processing and ensembling.


### Base Model:

As base model we used a models from the [EfficientNet](https://arxiv.org/abs/1905.11946) family[6]:



*   EfficientNet b3 
*   EfficientNet b4 
*   EfficientNet b5 
*   EfficientNet b6 
*   EfficientNet b7 

All models were pre-trained on imagenet using noisy student algorithm. The models and weights are from  [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)[2].

The input to these model is the image and meta-data such as age, sex, and anatomic Site. The meta-data is  processed by a small fully connected network and it’s output is concatenated to the input of the classification layer of the original EfficientNet network. This vector is going through a linear layer with output size of 256 to create the “features”, and then after an activation layer to the final linear classification layer. 

This network has 8 outputs and tries to classify the diagnosis label (there are actually more than 8 possible diagnoses, but some don’t have enough examples). 


### Transformer Models:

The input to the Transformer models are a stack of features from all images belonging to the same patient + the metadata for these images.

The transformer is a stack of 4 transformer encoder layers with self attention as described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) [1]. Each transformer encoder layer uses 4 self attention heads. 

The output of the transformer is a N*C, where N is the number of input feature vectors (the number of images) and C is the number of classes (8 in this case). Hence, the transformer predicts the class of each feature vector simultaneously, using the information from all other feature vectors.

The metadata is added using a “transformer style”, i.e. each parameter is transformed to a vector (size 256) using an embedding matrix and then added to the feature vector. for continuous values (like age) the embedding matrix was replaced by a 2 layer fully connected network.   


### Ensembling the output of all networks:

The data was split to 3 folds, 3 times (using 3 different seeds for splitting), and the inference was done using 16 (or 12) TTAs. giving 144 predictions from each model. These were averaged and then the outputs of all the models were averaged. All averaging was done on the outputs before softmax and there form it is actually geometric averaging.


### Training

The heavy lifting was the training and inference of the base models. This was done on a server with 2 GPUs – Tesla V100, Titan RTX that worked in parallel on different tasks. Training one fold of one model took 3H (B3)  to 11H (B7, B6 large images) on the Tesla and 20% more on the Titan, this sums up to about one day for B3 and 3.5 days for B7. Inferencing all the training data  for 12 TTA’s + test data for 16 TTA’s to get the features for the next level took  another 4h - 14h. The transformer training took less than 1H for the full model (3 folds*3seed). 

The total time it took to train all models and folds is about 2.5W for one Tesla (~1.5W using the 2 GPUs).


## 4. Models and features


### Base models

As base models we tried various types of models (pre-trained on Imagenet):



*   Densenets – 121, 161, 169, 201
*   EfficientNet B0, B3 , B4, B5, B6, B7 with and with noisy student pre-training and with normal pretraining 
*   ResNet 101
*   Xception 

At the end we used EfficientNet as it was best when judging accuracy/time

The noisy student version performed better than the normal one.

We also tried different image sizes and ended up using a `400*600` images in most cases, except one were we used `600*900` with the B6 network.



#### Metadata

As was described above the metadata was processed by a small fully connected nn and its output was concatenated to the output of the EfficientNet network (after removing the original top layer).

We also tried a network without metadata, and used the metadata as targets, i.e. this network predicted the diagnosis, but also the sex, age and anatomic site. The final predictions (including transformer) when using this approach weren't as good as the metadata as input approach. 


#### Model’s output

Although the task at hand is to predict melanoma yes/no, it is better to let the network choose the diagnosis among a few possible options. This lets the network “understand” more about the image. The final prediction is the value of the Melanoma output after doing softmax on the output vector. 


#### Features

The final layer in this model is a linear layer with 256 inputs and 8 outputs, we use the input to this layer as features. 


#### Augmentation

The following augmentations where used while training and inference:

Random resize + crop

Random rotation

Random flip

Random color jitter (brightness, contrast, saturation, hue)

[Cutout](https://arxiv.org/abs/1708.04552)[3] - erasing a small rectangle in the image

Hair - Randomly adding “hair like” lines to the image

Metadata augmentation - adding random noise to the metadata as was done in the [1st place solution in ISIC 2019 challenge](https://www.sciencedirect.com/science/article/pii/S2215016120300832?via%3Dihub) [7].


#### TTA

For inference each image was augmented differently 16 times and the final prediction was the average. These augmentations were also used for extracting 16 different features vectors per test image.

The same was done to extract 12 features vectors for the train images (12 and not 16 because of time limits).


### Transformer Network

The input to the transformer network is the features from all the images from one patient.

The inspiration for this kind of model came from a previous competition in which we participated in RSNA[ Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection). In that competition, all the top solutions (including ours) used a two stage network approach (although none of them were transformers).

Using a transformer seems appropriate in this case because transformers are built to seek relationships between embedding (feature) vectors in it’s input.

As this is not a full seq2seq task, we only used encoder layers. The transformer is a stack of 4 encoder layers with 4 attention heads in each layer. (we also tested higher numbers of layers and attention heads - no performance improvement).


#### metadata

The metadata was incorporated in the network by adding “metadata vectors” to the input vectors - each value was transformed to a vector size 256 and added. The discrete values’  transformation was done using a trainable embedding matrix and the continuous values using a small nn.


#### output

The output of this network is a matrix of size N*C where N - the number of images, C - the number of classes. Which means it decides on all the images of the patient at once.     


#### Limit the input size

A transformer can be trained on different number of feature vectors, by using padding. But when the range of numbers is very large, from a couple of hundred images for some patients to a handful for others, this may cause some implementation issues (like in calculating the loss). To simplify these issues, we limited N to 24 feature vectors, and for each patient we randomly divided the images to groups of size up to 24. 

This might degrade the prediction as the most “similar” images might accidentally fall into different groups, but as we use TTA, this issue is almost solved. 


#### Augmentation

From the base model we extract a number of feature vectors (12 for train and 16 for test) using different augmentation for the images and metadata. In the training and inference steps of the transformer model we randomly choose one of these vectors.

Another augmentation is the random grouping as stated above.


## 5. Training and Inferencing

The original 2020 competition data is highly unbalanced, there are only 2-3% of positive targets in the train and test data. Although we were able to train the base model using uneven sampling, the best way to get good training was to add the data from ISIC 2019 competition which has a much higher percentage of melanoma images. 

We split the training data to 3 folds keeping all the images from the same patient in the same fold, and making sure each fold has a similar number of patients with melanoma. The ISIC2019’s data was also split evenly between the folds. The same folds were kept for the base and the transformer models. 

To get more diversity we had 3 different splits using 3 seeds 


### Preprocessing

All images were resized to an aspect ratio of 1:1.5, which was the most popular aspect ratio of the images in the original dataset. We prepared 3 image datasets of sizes `300*450`, `400*600`, `600*900`. Most of the models were trained using the `400*600` dataset, as 300*450 gave inferior results and the `600*900` didn’t improve the results enough.

For the metadata we had to set the same terminology for the 2020 and 2019 datasets.


### Loss Function

The loss function we used was cross entropy. Although the task is to predict only melanoma we found it is better to predict the diagnosis which was split to 8 different classes one of which was melanoma. The final prediction was the value for the melanoma class, after a softmax function on all classes. We also tried a binary cross entropy on the melanoma class alone and a combination between the two, but using cross entropy gave the best results.

The same loss was used for the base model and the transformer, but in the transformer we needed to regularize for the different number of predictions in each batch resulting from the different number of images for each patient. 

We also tried using focal loss which didn’t improve the results, but we left one of the transformer models which was trained with focal loss in the ensemble (A model with cross entropy loss gave the similar CV and LB).


#### Training the transformer model 

The transformer model was trained in two steps. For the first step we used the data from both competitions (2019, 2020). For the 2019 competition we don’t have information about the patient, and each image got a different dummy patient_id, meaning the transformer didn’t learn much from these images. In the 2nd stage we fine-tuned the transformer using only the 2020 competition’s data.

In both steps we used a sampler that over sampled the larger groups.


#### Inference 

As stated above, the inference was done using TTA. For the base model we used 12-16 different augmentations and for the transformer model 32.


### Ensembling

For our final submissions we used 2 ensembles:


##### Without Context Submission:



1. EfficientNet B3 noisy student image size 400*600
2. EfficientNet B4 noisy student image size 400*600
3. EfficientNet B5 noisy student image size 400*600
4. EfficientNet B6 noisy student image size **600*900**
5. EfficientNet B7 noisy student image size 400*600


##### With Context

All the “without context” model +



*   Transformer on features from A.
*   Transformer on features from B.
*   Transformer on features from C using focal loss
*   Transformer on features from D.
*   Transformer on features from E.


## 6. Interesting findings


### CV and LB

Although the number of images in the competition was very large, the number of different patients wasn’t large enough to give a stable and reliable CV and LB. And the correlation between the two was low. At the end we trusted neither and we submitted the models we felt were the most robust.


### What didn’t work



1. Using different sizes of images as suggested in [CNN Input Size Explained](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/160147) didn’t show any improvement.
2. Using Mixup[5] and MixCut[4] augmentation didn’t work
3. Larger transformers didn’t improve the results. 


## 7. Better “Real World” model

This model can be much simpler if we won’t ensemble and take only one base model, the EfficientNet B5 is probably the best compromise.

In the real world scenario the images from previous years will probably be tagged already. In that case we can use a full transformer with encoder and decoder which will perform a seq2seq operation. 


## 8. Model Execution Time

All numbers refer to training on a system with intel i9-9920, 64GB RAM, Tesla V100 32G GPU.

Training base models ~ 3 - 11 h/fold

Inferencing the train data to prepare the transformer network input 4-14 h (for 3*3 folds, 12 TTA train and 16 TTA test)

Training transformer network 1H/model 

Inferencing base model test data 20- min/model (3*3 folds, 16 TTA)

Inferencing transformer ~ 2min (for 32xTTA


## 9. References


[1] Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin, _Attention Is All You Need_, 2017 [https://arxiv.org/abs/1706.03762v5](https://arxiv.org/abs/1706.03762v5)


[2]  Ross Wightman,  [https://github.com/rwightman/gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)


 [3] Terrance DeVries and Graham W. Taylor,_ Improved Regularization of Convolutional Neural Networks with Cutout_ ,2017, [https://arxiv.org/abs/1708.04552](https://arxiv.org/abs/1708.04552)


[4] Sangdoo Yun and Dongyoon Han and Seong Joon Oh and Sanghyuk Chun and Junsuk Choe and Youngjoon Yoo, _CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features,_ 2019, [https://arxiv.org/abs/1905.04899](https://arxiv.org/abs/1905.04899)


[5] Hongyi Zhang and Moustapha Cisse and Yann N. Dauphin and David Lopez-Paz, _mixup: Beyond Empirical Risk Minimization,_ 2017, [https://arxiv.org/abs/1710.09412v2](https://arxiv.org/abs/1710.09412v2)


[6] Mingxing Tan and Quoc V. Le, _EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks_, 2019, [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)


[7] Nils Gessert, Maximilian Nielsen,Mohsin Shaikh, René Werner, Alexander Schlaefer, _Skin lesion classification using ensembles of multi-resolution EfficientNets with meta data, 2020, [https://doi.org/10.1016/j.mex.2020.100864](https://doi.org/10.1016/j.mex.2020.100864)_
