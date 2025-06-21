# ML Engineer Take-Home Coding Challenge: Sentiment Analysis with Custom Loss Function

## Overview

In this coding challenge, you will build a sentiment analysis model using PyTorch and Hugging Face. You'll implement a custom loss function, create a proper data loading pipeline, and demonstrate your ability to build an end-to-end machine learning solution. This challenge is designed to evaluate your practical ML engineering skills, code organization, and understanding of deep learning concepts.

## Task Description

Develop a sentiment analysis model that classifies movie reviews as positive or negative using the IMDB dataset. The challenge should be completed within 4-6 hours and requires implementing a custom loss function that better handles sentiment classification nuances.

## Requirements

### 1. Data Handling

● Load the IMDB dataset using Hugging Face's datasets library
● Implement a PyTorch Dataset and DataLoader with appropriate preprocessing
● Split the data into training, validation, and test sets

### 2 Model Implementation

● Create a model architecture using PyTorch (LSTM, Transformer-based, etc.)
● You may use a pre-trained model from Hugging Face as a starting point
● Implement model training, evaluation, and prediction functionality

### 3. Custom Loss Function (Required)

● Implement a custom loss function that addresses at least one of these requirements:

- Weighted loss based on review length (longer reviews may have different importance)
- Loss that accounts for prediction confidence (penalize highly confident wrong predictions more)
- Class-weighted loss to handle potential dataset imbalances
  ● Your loss function must subclass torch.nn.Module or be implemented as a standalone function
  ● Include comments explaining the rationale behind your custom loss design

### 4. Training Pipeline

● Implement a training loop with appropriate logging
● Include validation during training to prevent overfitting
● Save and load model checkpoints
● Track and report relevant metrics (accuracy, F1 score, etc.)

### 5. Documentation

Include a well-structured README.md with:
● Project overview
● Installation instructions
● Usage examples
● Explanation of your custom loss function
● Model performance results
● Potential improvements

### 6. Code Organization

● Organize your code in a modular, maintainable structure
● Use proper Git practices (meaningful commits, branches if necessary)
● Include requirements.txt or environment.yml

### Start Code

```
import torch
import torch.nn as nn
class SentimentWeightedLoss(nn.Module):
 """
 Custom loss function for sentiment analysis that weights samples differently
 based on specific criteria.
 """
 def __init__(self, weight_param=1.0):
 super(SentimentWeightedLoss, self).__init__()
 self.weight_param = weight_param
 self.base_loss = nn.BCEWithLogitsLoss(reduction='none')

 def forward(self, predictions, targets, sample_weights=None):
 # Basic loss computation
 base_loss = self.base_loss(predictions, targets)

 # Apply your custom weighting logic here
 # For example, weight by confidence or other metrics

 return weighted_loss
```

### Evaluation Criteria

Your submission will be evaluated on:

1. Code quality and structure
2. Implementation of the custom loss function
3. Training pipeline implementation
4. Model performance
5. Documentation clarity and comprehensiveness
6. Overall solution design

### Bonus Points (Optional)

● Implement a simple web interface for model demos
● Add visualization of model results
● Write unit tests for critical components
● Conduct experiments comparing your custom loss with standard loss functions

### Submission Instructions

1. Create a GitHub repository with your solution
2. Ensure all code is well-commented and follows best practices
3. Include the link to your repository in your submission
4. Make sure the repository includes all required components and documentation

Good luck! This challenge is designed to showcase your ML engineering skills while implementing practical components that would be used in real-world applications.
[Note: Datasets and models referenced are publicly available through the Hugging Face
ecosystem]
