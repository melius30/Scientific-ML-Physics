# Scientific Machine Learning for Physics-Based Modeling

**University of California, Berkeley | Fall 2025**

A collection of scientific machine learning projects developed for
ME 249 in the Department of Mechanical Engineering at UC Berkeley.

The projects explore physics-informed neural networks, deep learning,
numerical methods, genetic algorithms, and engineering optimization.

## Authors

- **Md Elius** — PhD Candidate, Civil & Environmental Engineering, UC Berkeley
- **Parker Yang** — Project Co-author, UC Berkeley

## Course Information

- **Course:** ME 249, Fall 2025
- **Department:** Mechanical Engineering, University of California, Berkeley
- **Instructor:** [Prof. Van P. Carey](https://me.berkeley.edu/people/van-p-carey/)

---

## Projects

### 1. Physics-Informed Neural Networks for Heat Transfer

**Directory:** `Project3_Physics_Informed_Neural_Networks/`

Developed physics-informed neural networks (PINNs) for solving
steady-state heat conduction problems.

**Key features:**
- TensorFlow/Keras neural network implementation
- Custom physics-informed loss functions
- Enforcement of boundary conditions and Laplacian PDE residuals
- Rectangular and hexagonal computational domains
- Comparison against Gauss-Seidel finite-difference solutions
- Evaluation of physics-loss weighting and model convergence

**Reported result:** Minimum training loss approximately 1.0e-4.

**Topics:** PINNs, PDEs, TensorFlow, Keras, numerical simulation.

### 2. Neural Networks and Optimization

**Directory:** `Project2_Neural_Networks_Optimization/`

Investigated neural network training and optimization for
thermodynamic engineering applications.

**Key features:**
- First-principles backpropagation implementation
- Comparison with Keras neural network training
- Hybrid solar-gas turbine performance modeling
- Genetic algorithm optimization using PyGAD
- Comparison of gradient-based and evolutionary optimization

**Topics:** Neural networks, backpropagation, Keras, PyGAD.

### 3. Solar Photovoltaic Performance Prediction

**Directory:** `Project4_Solar_PV_Prediction/`

Applied neural network regression to solar photovoltaic
performance prediction.

**Key features:**
- TensorFlow/Keras regression models
- Three- versus four-hidden-layer architecture comparison
- Training and validation analysis
- Early stopping
- Model performance evaluation

**Reported result:** Best training loss approximately 0.024.

**Topics:** Regression, deep learning, solar energy, model validation.

### 4. Genetic Algorithms for Heat Transfer

**Directory:** `Project1_Genetic_Algorithm_Heat_Transfer/`

Applied genetic algorithms to calibrate empirical
heat-transfer correlations.

**Key features:**
- Parameter optimization
- Genetic algorithm implementation
- Convergence analysis
- RMS and percentage error evaluation
- Comparison of fitted heat-transfer models

**Topics:** Genetic algorithms, optimization, heat transfer.

---

## Additional Machine Learning Notebook

### CNN Classification with Fashion-MNIST

**Notebook:** `CNN_fashion_MNIST.ipynb`

An additional notebook exploring convolutional neural networks
for image classification using the Fashion-MNIST dataset.

This notebook is included separately from the four collaborative
ME 249 course projects.

**Topics:** CNNs, image classification, deep learning.

---

## Technical Skills Demonstrated

| Category | Technologies and Methods |
|----------|--------------------------|
| Programming | Python |
| Deep Learning | TensorFlow, Keras, neural networks, CNNs |
| Scientific ML | Physics-informed neural networks, custom loss functions |
| Numerical Methods | PDEs, finite differences, Gauss-Seidel |
| Optimization | Genetic algorithms, PyGAD, backpropagation |
| Applications | Heat transfer, thermodynamics, solar energy |

## Related Research

Md Elius is a PhD candidate in Civil and Environmental Engineering
at UC Berkeley, conducting research on high-resolution atmospheric
large-eddy simulation, methane transport, Bayesian source inversion,
and scientific machine learning.

Related repositories:

- [ERF GPU Atmospheric Simulation](https://github.com/melius30/ERF-GPU-Savio)
- [FastEddy Fork](https://github.com/melius30/FastEddy-Fork)
- [WRF Fork](https://github.com/melius30/WRF-Fork)

Personal website: https://melius30.github.io

## Acknowledgments

The four ME 249 projects were collaboratively developed by
Md Elius and Parker Yang as part of the Fall 2025 course at
the University of California, Berkeley.

We acknowledge Prof. Van P. Carey for course instruction and
guidance in scientific computing, machine learning, and
engineering applications.

## Notes

These projects represent educational and research-oriented
implementations. Results are specific to the reported experiments
and should not be interpreted as general performance benchmarks.
