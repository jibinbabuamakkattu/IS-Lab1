# IS-Lab1: Perceptron Training & Naive Bayes Classifier

Intelligent Systems. Laboratory Work No. 1.

## Aim
Learn to implement and train a simple linear classifier (Single Perceptron) and compare it with a probabilistic classifier (Naive Bayes).

## Tasks (8 Points)
1. **Perceptron Implementation:**
   Create a program for binary classification using a single Perceptron to distinguish between two objects (Apples and Pears) based on two estimated features:
   * $x_1$: Color (average hue in HSV color space)
   * $x_2$: Roundness metric ($\frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2}$)

   The output calculation is performed using the signum/threshold activation function:
   $$y = \begin{cases} +1, & \text{if } x_1 w_1 + x_2 w_2 + b > 0 \quad (\text{Apple}) \\ -1, & \text{if } x_1 w_1 + x_2 w_2 + b \le 0 \quad (\text{Pear}) \end{cases}$$

2. **Training Algorithm (Delta Rule):**
   Implement the parameter update equations:
   $$w_1(n+1) = w_1(n) + \eta \cdot e(n) \cdot x_1(n)$$
   $$w_2(n+1) = w_2(n) + \eta \cdot e(n) \cdot x_2(n)$$
   $$b(n+1) = b(n) + \eta \cdot e(n)$$
   where:
   * $0 < \eta < 1$ is the learning rate ($\eta = 0.3$).
   * $e(n) = T(n) - y(n)$ is the error between target $T(n)$ and output $y(n)$.

## Additional Task (+2 Points)
Solve the classification task using a **Gaussian Naive Bayes Classifier** and evaluate performance on unseen test images.

## Results

* **Perceptron Test Accuracy:** 100.00% on the 8 unused test images.
* **Naive Bayes Test Accuracy:** 87.50% on the 8 unused test images.

## Suggested Reading
* *Neural Networks and Learning Machines (3rd Edition)*, Simon Haykin, page 54, Table 1.1.
