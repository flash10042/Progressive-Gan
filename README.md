# Progressive-Gan
Keras implementation of Progressive GAN to generate abstract paintings in 128x128 resolution.

# Run models
Firstly you need to create folder 'train' that contains images you want your model to train on. 

To get training data you can run *scrap_data.py* either *scrap_new_data.py* scripts.

- *scrap_data.py* script will get works of the most popular painters from [Art challenge](artchallenge.me). 
- If you want to get dataset of abstact paintings run *scrap_new_data.py* or manually download it 
from [Kaggle](https://www.kaggle.com/flash10042/abstract-paintings-dataset).

Run train.py to create and train your own GAN.

Also you can use pretrained generator and discriminator which are stored in *generator* and *discriminator* folders respectively.

# Results

Example of generated paint:

![Generated example](https://github.com/flash10042/Progressive-Gan/blob/main/generated_images/generated.png)

Some cool paintings generated during training:

![One more 128x128 example](https://github.com/flash10042/Progressive-Gan/blob/main/generated_images/size_(128%2C%20128)_epoch_15_fade_False.png)

![64x64 example](https://github.com/flash10042/Progressive-Gan/blob/main/generated_images/size_(64%2C%2064)_epoch_34_fade_False.png)

At first I was trying to train GAN on Ivan Aivazovsky landscape paintings. Here is what I got after some training:

![Forests](https://github.com/flash10042/Progressive-Gan/blob/main/generated_images/size_(128%2C%20128)_epoch_18_fade_False.png) 


# TODO

Both generator and discriminator requires a serious parameters tuning, so I'm considering changing some parts of train.py script to make it generate better
results. Also I have a bug that prevents me saving discriminator in h5 format. I'm still trying to figure out how to fix it.
