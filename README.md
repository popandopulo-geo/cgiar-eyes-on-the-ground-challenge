# CGIAR "Eyes on the Ground" Challenge Solution

This repository contains my solution for the [CGIAR "Eyes on the Ground" Challenge](https://zindi.africa/competitions/cgiar-eyes-on-the-ground-challenge) hosted on Zindi.

## Competition Overview

The challenge aimed to develop a model capable of identifying and classifying crops from images, contributing to the broader goal of improving agricultural practices and food security. The competition involved using computer vision techniques to analyze images captured in various environments.

## Solution Overview

My solution leverages advanced deep learning techniques for image classification. The approach involved several stages, including data preprocessing, model training, and evaluation.

### Key Components

- **train.py**: This script contains the main code for training the deep learning model on the provided dataset.
- **src/**: Directory containing the core modules for the project:
  - **agents.py**: Manages the different models and their configurations used in the solution.
  - **dataset.py**: Handles data loading, augmentation, and preprocessing tasks.
  - **losses.py**: Custom loss functions used to optimize the model during training.
  - **utils.py**: Utility functions for various tasks, such as metrics calculation, logging, and model evaluation.
- **Notebooks**:
  - **eda.ipynb**: Contains exploratory data analysis performed on the dataset to understand data distribution, class imbalance, and other characteristics.
  - **inference.ipynb**: Notebook used for generating predictions on the test set.
  - **experiments.ipynb**: Documents various experiments conducted during model development, including hyperparameter tuning and architecture testing.


### Model Architecture

- The solution utilizes a convolutional neural network (CNN) architecture, fine-tuned on the competition's dataset.
- Various augmentation techniques were applied to enhance the model's generalization capabilities.
- Custom loss functions were implemented to better address the class imbalance present in the dataset.

### Performance

The final model achieved competitive performance on the test set, securing a **Bronze Medal** in the competition.

## Acknowledgements

I would like to thank CGIAR and Zindi for organizing this impactful competition. Participating in this challenge has been a valuable learning experience in the field of computer vision and its application in agriculture.

## References

- [Zindi Platform](https://zindi.africa/)
- [Competition Page](https://zindi.africa/competitions/cgiar-eyes-on-the-ground-challenge)
- [CGIAR](https://www.cgiar.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
