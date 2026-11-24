import numpy as np
import matplotlib.pyplot as plt

# =========================================================================
# IS-Lab 1: Classification using Single Perceptron & Naive Bayes
# Uses data directly from Data.txt: [Feature1 (Color), Feature2 (Roundness), Class (+1: Apple, -1: Pear)]
# =========================================================================

# 1. LOAD DATA
data = np.loadtxt("Data.txt", delimiter=",")
X_all = data[:, 0:2] # x1 = color, x2 = roundness
T_all = data[:, 2]   # +1 = Apple, -1 = Pear

# As specified in the lab assignment:
# Training set: 3 Apples (indices 0, 1, 2) and 2 Pears (indices 9, 10)
train_indices = [0, 1, 2, 9, 10]
# Test set: remaining 6 Apples and 2 Pears
test_indices = [3, 4, 5, 6, 7, 8, 11, 12]

X_train = X_all[train_indices]
T_train = T_all[train_indices]

X_test = X_all[test_indices]
T_test = T_all[test_indices]

print("==================================================")
print("              PERCEPTRON TRAINING                 ")
print("==================================================")

# 2. INITIALIZE PERCEPTRON PARAMETERS
np.random.seed(42) # Set seed for reproducibility
w1 = np.random.randn()
w2 = np.random.randn()
b  = np.random.randn()
eta = 0.1 # Learning rate

print(f"Initial weights: w1={w1:.4f}, w2={w2:.4f}, bias={b:.4f}")

# 3. TRAINING LOOP
epoch = 0
total_error = 1
max_epochs = 1000

while total_error != 0 and epoch < max_epochs:
    epoch += 1
    total_error = 0
    
    for i in range(len(T_train)):
        x1 = X_train[i, 0]
        x2 = X_train[i, 1]
        target = T_train[i]
        
        # Step A: Forward pass (calculate weighted sum)
        v = x1 * w1 + x2 * w2 + b
        
        # Step B: Activation function (threshold)
        y = 1 if v > 0 else -1
        
        # Step C: Error calculation
        e = target - y
        
        # Step D: Parameter update (Delta rule)
        w1 = w1 + eta * e * x1
        w2 = w2 + eta * e * x2
        b  = b  + eta * e
        
        total_error += abs(e)
        
    print(f"Epoch {epoch:2d}: Total Error = {total_error}")

print("\nPerceptron training finished!")
print(f"Converged in {epoch} epochs.")
print(f"Trained Parameters: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}\n")

# 4. TEST PERCEPTRON ON UNSEEN DATA
print("==================================================")
print("           PERCEPTRON TEST EVALUATION             ")
print("==================================================")
v_test = X_test[:, 0] * w1 + X_test[:, 1] * w2 + b
y_test_perceptron = np.where(v_test > 0, 1, -1)
acc_perceptron = np.mean(y_test_perceptron == T_test) * 100

for idx, (pred, actual) in enumerate(zip(y_test_perceptron, T_test)):
    name = "Apple" if actual == 1 else "Pear "
    status = "CORRECT" if pred == actual else "WRONG"
    print(f"Test sample {idx+1} ({name}): Predicted={pred:+d}, Actual={int(actual):+d} -> {status}")

print(f"\nPerceptron Test Accuracy: {acc_perceptron:.2f}%\n")


# 5. NAIVE BAYES CLASSIFIER (Bonus +2 points)
print("==================================================")
print("          NAIVE BAYES CLASSIFIER (BONUS)          ")
print("==================================================")

# Separate training data by class
X_apple = X_train[T_train == 1]
X_pear  = X_train[T_train == -1]

# Calculate Gaussian parameters (Mean and Variance)
mu_apple = np.mean(X_apple, axis=0)
var_apple = np.var(X_apple, axis=0)

mu_pear = np.mean(X_pear, axis=0)
var_pear = np.var(X_pear, axis=0)

prior_apple = len(X_apple) / len(X_train)
prior_pear  = len(X_pear) / len(X_train)

def gaussian_pdf(x, mean, var):
    return (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-((x - mean)**2) / (2.0 * var))

y_test_nb = []
for i in range(len(X_test)):
    x = X_test[i]
    # Likelihood * Prior
    p_apple = np.prod(gaussian_pdf(x, mu_apple, var_apple)) * prior_apple
    p_pear  = np.prod(gaussian_pdf(x, mu_pear, var_pear)) * prior_pear
    
    pred = 1 if p_apple >= p_pear else -1
    y_test_nb.append(pred)

y_test_nb = np.array(y_test_nb)
acc_nb = np.mean(y_test_nb == T_test) * 100

for idx, (pred, actual) in enumerate(zip(y_test_nb, T_test)):
    name = "Apple" if actual == 1 else "Pear "
    status = "CORRECT" if pred == actual else "WRONG"
    print(f"Test sample {idx+1} ({name}): NB Predicted={pred:+d}, Actual={int(actual):+d} -> {status}")

print(f"\nNaive Bayes Test Accuracy: {acc_nb:.2f}%")
print("==================================================")
