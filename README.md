# Smooth ReLU activations

Activations like GELU and Swish require complex hardware implementations to support exponential and logarithmic functions. Further, GELU must be computed numerically or approximated. These properties can make deployment error-prone, expensive, or slow. GELU and Swish are not monotonic (they start by slightly decreasing and then switch to increasing), which may interfere with interpretability (or identifiability), nor do they have a full stop or a clean slope 1 region, properties that simplify implementation and may aid in reproducibility. The Smooth reLU (SmeLU) activation function is designed as a simple function that addresses the concerns with other smooth activations. It connects a 0 slope on the left with a slope 1 line on the right through a quadratic middle region, constraining continuous gradients at the connection points (as an asymmetric version of a Huber loss function). 

SmeLU can be viewed as a convolution of ReLU with a box. It provides a cheap and simple smooth solution that is comparable in reproducibility-accuracy tradeoffs to more computationally expensive and complex smooth activations. The figure below illustrates the transition of the loss (objective) surface as we gradually transition from a non-smooth ReLU to a smoother SmeLU. A transition of width 0 is the basic ReLU function for which the loss objective has many local minima. 

As the transition region widens (SmeLU), the loss surface becomes smoother. If the transition is too wide, i.e., too smooth, the benefit of using a deep network wanes and we approach the linear model solution — the objective surface flattens, potentially losing the ability of the network to express much information.

*As posted in [Reproducibility in Deep Learning and Smooth Activations](https://ai.googleblog.com/2022/04/reproducibility-in-deep-learning-and.html)* by Google Research

## About Repository

This repository contains the implementation of SmeLU activations in CUDA kernels. The implementation supports both single and double precision floating-point numbers Supported by NVIDIA GPUs based on the Volta microarchitecture, such as the Tesla V100, Titan V, and Quadro GV100. Currently, this implementation can be used as a standalone library and can be integrated into existing projects. 

## API Reference

SmeLU CU has two classes: SmeLUf and SmeLUD, which correspond to the float and double precision implementations, respectively. 

Both classes have the following methods:

```python
__init__(self, alpha_value: float = 1.0, size: int = 1)
    Initializes a new instance of the SmeLU activation function.

Parameters:
    alpha_value (float, optional): The value of the alpha parameter. Default is 1.0.
    size (int, optional): The size of the input tensor. Default is 1.

forward(self, input_tensor: List[float]) -> List[float]
    Computes the forward pass of the SmeLU function.

Parameters:
    input_tensor (List[float]): A list of input values.

Returns:
    List[float]: A list of output values.

backward(self, input_tensor: List[float], grad_output: List[float]) -> List[float]
    Computes the backward pass of the SmeLU function.

Parameters:
    input_tensor (List[float]): A list of input values.
    grad_output (List[float]): A list of gradient output values.

Returns:
    List[float]: A list of gradient input values.

```

### Cite

```
Shamir, G., I. et al (2022, February 14). Real World Large Scale Recommendation Systems Reproducibility and Smooth Activations. arXiv.org. https://arxiv.org/abs/2202.06499
 
https://doi.org/10.48550/arXiv.2202.06499
```
