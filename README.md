# Conditional GAN

This repository contains a Conditional Generative Adversarial Network (CGAN), which is a variant of the lightweight Deep Convolutional GAN (DCGAN). While the Wasserstein GAN (WGAN) is commonly used in the field of generative models, I aimed to modify the traditional DCGAN architecture to enhance its efficiency and effectiveness. The model successfully generates high-quality images of handwritten digits from the MNIST dataset based on specific labels.

## Overview

Conditional GANs extend the capabilities of traditional GANs by conditioning the generation process on additional information, such as class labels. This allows for the generation of images that correspond to specific categories, making them particularly useful for tasks where control over the output is desired.

## Key Features

- **Conditional Image Generation**: The model generates MNIST digit images conditioned on specified labels, allowing for precise control over the type of digit produced.
- **Modified DCGAN Architecture**: Although typically WGANs are preferred for their stability, this project focuses on a modified DCGAN architecture that shows promising results in image generation.
- **Efficient Training**: The training process utilizes the Adam optimizer and logs results using TensorBoard, enabling effective monitoring of training progress and performance metrics.
- **Image Normalization**: The model applies normalization techniques to enhance the quality of generated images.

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
