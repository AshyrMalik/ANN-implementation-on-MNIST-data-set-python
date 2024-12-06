

# Neural Network for MNIST Classification

This repository contains PyTorch code for training and evaluating a neural network to classify handwritten digits from the MNIST dataset. The network uses a simple architecture with fully connected layers and ReLU activation, designed for beginner-level deep learning tasks.

---

## Features

- Neural network implemented using **PyTorch**.
- Efficient training using **Adam Optimizer** and **CrossEntropy Loss**.
- Evaluation of training and test accuracy.
- Batch processing with **DataLoader** for better memory management.
- Handles GPU/CPU dynamically for computation.

---

## Requirements

Ensure you have the following libraries installed:

- Python 3.x
- PyTorch
- NumPy

Install the required packages with:

```bash
pip install torch torchvision numpy
```

---

## Code Overview

### 1. Model Architecture
The neural network has the following architecture:
- **Input Layer**: Accepts 784 features (28x28 flattened images).
- **Hidden Layer**: 128 neurons with ReLU activation.
- **Output Layer**: 10 neurons for the 10 classes of digits.

```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

### 2. Training Loop
The model is trained for a specified number of epochs using:
- **CrossEntropyLoss**: For classification tasks.
- **Adam Optimizer**: For efficient parameter updates.
- Batch processing with gradient calculation and weight updates.

### 3. Accuracy Calculation
The training and test accuracy are calculated using:
- Forward pass to obtain predictions.
- Comparison of predicted labels with true labels.

---

## Training and Testing

### Steps:
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Run the script:
   ```bash
   python mnist_nn.py
   ```

### Expected Output
The training and testing loss for each epoch will be displayed, along with the final training and testing accuracy. For example:
```
Epoch 1/10, Train Loss: 0.0059, Test Loss: 0.0858
Epoch 2/10, Train Loss: 0.0047, Test Loss: 0.0789
Epoch 3/10, Train Loss: 0.0045, Test Loss: 0.0911
Epoch 4/10, Train Loss: 0.0049, Test Loss: 0.0886
Epoch 5/10, Train Loss: 0.0043, Test Loss: 0.0941
Epoch 6/10, Train Loss: 0.0028, Test Loss: 0.0940
Epoch 7/10, Train Loss: 0.0025, Test Loss: 0.1039
Epoch 8/10, Train Loss: 0.0050, Test Loss: 0.0994
Epoch 9/10, Train Loss: 0.0047, Test Loss: 0.0969
Epoch 10/10, Train Loss: 0.0019, Test Loss: 0.0921...
Train Accuracy: 98.50%
Test Accuracy: 96.20%
```

---

## Results

- **Training Accuracy**: ~100%
- **Test Accuracy**: ~97.83%
- The model successfully classifies handwritten digits from the MNIST dataset.

---

## Future Improvements

- Add more layers or neurons for better accuracy.
- Experiment with convolutional neural networks (CNNs) for improved performance.
- Implement techniques to reduce overfitting, such as dropout or data augmentation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- Framework: [PyTorch](https://pytorch.org/)

Feel free to fork, improve, and share this project!

--- 

Let me know if you'd like me to include specific sections or improve any part!
