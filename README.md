# Conditional GAN

This repository contains a Conditional Generative Adversarial Network (CGAN), which is a variant of the lightweight Deep Convolutional GAN (DCGAN). While the Wasserstein GAN (WGAN) is commonly used in the field of generative models, I aimed to modify the traditional DCGAN architecture to enhance its efficiency and effectiveness. The model successfully generates high-quality images of handwritten digits from the MNIST dataset based on specific labels.

## Overview

Conditional GANs extend the capabilities of traditional GANs by conditioning the generation process on additional information, such as class labels. This allows for the generation of images that correspond to specific categories, making them particularly useful for tasks where control over the output is desired.

## Key Features

- **Conditional Image Generation**: The model generates MNIST digit images conditioned on specified labels, allowing for precise control over the type of digit produced.
- **Modified DCGAN Architecture**: Although typically WGANs are preferred for their stability, this project focuses on a modified DCGAN architecture that shows promising results in image generation.
- **Efficient Training**: The training process utilizes the Adam optimizer and logs results using TensorBoard, enabling effective monitoring of training progress and performance metrics.
- **Image Normalization**: The model applies normalization techniques to enhance the quality of generated images.


## Flexibility with Datasets

While this implementation is demonstrated using the MNIST dataset for handwritten digit generation, the Conditional GAN model can be adapted to train on other image datasets as well. By modifying the input dimensions and ensuring the dataset is properly preprocessed, you can leverage this architecture for various tasks, such as:

- **Fashion MNIST**: For generating clothing images based on specified categories.
- **CIFAR-10**: For generating diverse images of objects across 10 different classes, including animals and vehicles.
- **Custom Datasets**: By providing your own dataset, you can train the model to generate images that align with specific requirements or characteristics relevant to your application.

To adapt the model for different datasets, make sure to:

1. Adjust the input channels and image size in the `Generator` and `Discriminator` classes to match the dataset.
2. Preprocess the dataset accordingly (e.g., normalization, resizing) to ensure it is compatible with the model's architecture.
3. Update the number of classes in the embedding layers to reflect the new dataset.

This flexibility makes the Conditional GAN a versatile tool for a range of image generation tasks across various domains.


## Getting Started

To get started with the project, follow these steps:

### Prerequisites

- Python 3.6 or higher
- PyTorch (1.x)
- torchvision
- Matplotlib
- TensorBoard

### Contributors

B M Manohara @Manohara-AI
