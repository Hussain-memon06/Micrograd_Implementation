# Micrograd_Implementation

This project is my implementation of Andrej Karpathy's *micrograd*, a tiny Autograd engine for building and training a small neural network from scratch. I followed Andrej's tutorial to deepen my understanding of backpropagation, gradient descent, and neural networks, and I've reproduced the core concepts step-by-step.

## Project Overview

In this project, I built a simple feedforward neural network using Python and a custom Autograd engine. This engine allows for automatic differentiation, which is essential for computing gradients during the backpropagation process.

### Key Features

- **Value Class**: Represents a single scalar value and tracks its history of operations for automatic differentiation.
- **Neurons and Layers**: A neural network composed of layers of neurons, where each neuron performs a weighted sum of its inputs followed by a non-linearity (tanh).
- **Backpropagation**: The implementation includes backpropagation through the custom Autograd engine, enabling the network to learn using gradient descent.

## How It Works

1. **Define the network**: A multi-layer perceptron (MLP) is created with configurable inputs and outputs.
2. **Forward Pass**: Given a set of inputs, the network produces a prediction using matrix multiplication and non-linear activation functions.
3. **Loss Calculation**: The loss is computed as the squared difference between the network's predictions and the ground truth.
4. **Backward Pass**: The gradients are calculated automatically through backpropagation, and gradient descent is applied to update the weights of the network.

## Example

Here's an example of training the network on simple inputs and outputs:

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

for k in range(10):
    ypred = [n(x) for x in xs]
    loss = Value(0)
    for ygt, yout in zip(ys, ypred):
        loss += (yout - Value(ygt))**2
    print(f"Epoch {k}, Loss: {loss.data}")
    
    loss.backward()
    for p in n.parameters():
        p.data -= 0.01 * p.grad  # Simple gradient descent
```

### Acknowledgements
Thanks to Andrej Karpathy for his incredible micrograd tutorial, which served as the foundation for this project. 

