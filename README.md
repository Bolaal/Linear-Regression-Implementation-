# Linear Regression Implementation from Scratch

This Jupyter notebook demonstrates a complete implementation of linear regression using gradient descent algorithm, built from scratch using NumPy and Pandas.

## 📋 Overview

This project implements linear regression to predict profit based on population data. The implementation includes:
- Data preprocessing and visualization
- Cost function calculation
- Gradient descent optimization
- Model training and evaluation
- Results visualization

## 🔧 Requirements

```
numpy
pandas
matplotlib
```

## 📊 Dataset

The notebook uses `dataset.txt` which contains:
- **Population**: City population (in 10,000s)
- **Profit**: Profit from food truck business (in $10,000s)
- **Format**: CSV with 97 data points

Sample data:
```
6.1101,17.592
5.5277,9.1302
8.5186,13.662
```

## 🚀 Implementation Details

### 1. Data Loading and Exploration
- Loads data from `dataset.txt`
- Adds column names: "Population" and "Profit"
- Performs exploratory data analysis
- Visualizes data with scatter plot

### 2. Feature Engineering
- Adds bias term (column of ones) for intercept calculation
- Separates features (X) and target variable (y)
- Converts to NumPy matrices for mathematical operations

### 3. Cost Function
```python
def compute_cost(x, y, theta):
    z = np.power(((x*theta.T)-y), 2)
    return np.sum(z) / (2*len(x))
```
- Implements Mean Squared Error (MSE)
- Measures prediction accuracy

### 4. Gradient Descent Algorithm
```python
def Gradient_Descent(x, y, theta, alpha, iters):
    # Iteratively updates parameters
    # Returns optimized theta and cost history
```

**Hyperparameters:**
- Learning rate (α): 0.01
- Iterations: 10,000

### 5. Model Training Results
- **Initial cost**: 32.07
- **Final cost**: 4.48
- **Optimized parameters**: θ₀ = -3.896, θ₁ = 1.193

## 📈 Visualizations

The notebook generates two key plots:

1. **Prediction vs Training Data**
   - Shows linear regression line overlaid on scatter plot
   - Demonstrates model fit quality

2. **Cost vs Iterations**
   - Shows convergence of gradient descent
   - Validates optimization process

## 🎯 Key Features

- **From-scratch implementation**: No sklearn dependencies
- **Mathematical foundation**: Clear cost function and gradient calculations
- **Visualization**: Comprehensive plotting for analysis
- **Parameter tracking**: Monitors cost reduction over iterations
- **Matrix operations**: Efficient NumPy-based computations

## 📝 Mathematical Formula

The linear regression model predicts:
```
Profit = θ₀ + θ₁ × Population
```

Where:
- θ₀: Intercept (bias term)
- θ₁: Slope (population coefficient)

## 🔍 Results Interpretation

The trained model suggests:
- **Base loss**: ~$38,960 (when population = 0)
- **Profit per 10k population**: ~$11,930
- **Model explains**: Reasonable relationship between city size and food truck profitability

## 💡 Usage

1. Ensure `dataset.txt` is in the same directory
2. Run all cells sequentially in the Jupyter notebook
3. Observe the convergence in the cost plot
4. Analyze the final prediction line fit

## 🎓 Educational Value

This implementation is excellent for understanding:
- Linear regression fundamentals
- Gradient descent optimization
- Cost function minimization
- Feature engineering basics
- Data visualization techniques

Perfect for machine learning beginners who want to understand the mathematics behind linear regression before using higher-level libraries.
