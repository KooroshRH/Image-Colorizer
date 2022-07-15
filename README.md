# Image Colorizer
In this project, we used CycleGAN architecture and CelebA dataset to train a model for translating black and white images into colorized images.  
You can run the notebook in your colab environment and use different datasets as your source and destination domains.

## Cycle loss function
This loss function is the main idea of CycleGAN. In CycleGAN, we want to translate a domain's attributes to another domain and keep the main characteristics from the source domain.  
This loss function returns the difference between translated image and the original one.  

```
def calc_cycle_loss(real_image, cycled_image):
  loss = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return LAMBDA * loss
```

## Example
Here you can see some samples from the model and compare the results to their ground truth.  
> NOTE: Please consider that in this dataset, our target is to colorize the face, the particular part of the subject's skin, hair, and facial hair, not their background in the picture. We can expect good colorization in a different area if we prepare more complete datasets for other purposes.
