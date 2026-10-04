# Radio Signal Strength Prediction
A machine-learning project for predicting **received radio signal strength** from wireless propagation parameters using Python. The project combines fundamental radio-propagation theory with machine-learning techniques, beginning with a theoretical **Friis free-space propagation model** and progressing to **linear regression implemented from scratch**.

The project is designed as a practical bridge between **wireless communications engineering, mathematical modelling, Python programming, data analysis, and machine learning**.
---

## Project Overview
Received signal strength is a fundamental parameter in the design, analysis, and optimisation of wireless communication systems.

In an ideal free-space environment, received power can be estimated using the Friis transmission equation:

$$
P_{RX}=P_{TX}G_{TX}G_{RX}
\left(\frac{\lambda}{4\pi d}\right)^2
$$

where:
$$
P_{RX}
$$

$$ 
P_{RX}
$$
— received power
* $$ \(P_{TX}\) $$ — transmitted power
* $$ \(G_{TX}\) $$ — transmitter antenna gain
* $$ \(G_{RX}\) $$ — receiver antenna gain
* $$ \(\lambda\) $$ — wavelength
* $$ \(d\) $$ — transmitter-receiver distance

Real wireless environments are more complex because of reflection, diffraction, scattering, multipath propagation, shadowing, terrain, buildings, and other environmental factors.

This project investigates how machine-learning techniques can be used to learn the relationship between propagation parameters and received signal strength.

---

## Objectives

The project aims to:
* Generate or collect radio-propagation data.
* Explore relationships between propagation parameters and received signal strength.
* Implement linear regression from scratch using Python and NumPy.
* Implement gradient descent for model optimisation.
* Train a model to predict received signal strength.
* Evaluate prediction performance using standard regression metrics.
* Compare machine-learning predictions with theoretical propagation calculations.
* Visualise propagation behaviour and model performance.
* Provide a foundation for extending the model to real-world wireless measurements.
---

## Input Features
The initial model considers the following propagation parameters:

| Feature             | Description                     |
| ------------------- | ------------------------------- |
| `distance`          | Transmitter-receiver separation |
| `frequency`         | Operating frequency             |
| `transmitter_power` | Transmitted power               |
| `transmitter_gain`  | Transmitter antenna gain        |
| `receiver_gain`     | Receiver antenna gain           |

### Target Variable
The target variable is:

```text
received_power
```
Depending on the dataset, received power may be represented in watts, milliwatts, or dBm.
---

## Methodology

The project follows this workflow:

```text
Radio Propagation Theory
        │
        ▼
Data Generation / Collection
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Preparation
        │
        ▼
Train/Test Split
        │
        ▼
Linear Regression
        │
        ▼
Gradient Descent
        │
        ▼
Prediction
        │
        ▼
Model Evaluation
        │
        ▼
Visualisation
        │
        ▼
Comparison with Theoretical Model
```

---

## Theoretical Model

The wavelength is calculated using:

$$
\lambda=\frac{c}{f}
$$

where:

* \(c\) is the speed of light.
* \(f\) is the operating frequency.

The Friis equation provides the theoretical free-space reference:

$$
P_{RX}=P_{TX}G_{TX}G_{RX}
\left(\frac{\lambda}{4\pi d}\right)^2
$$

The theoretical prediction is used as a baseline for understanding the machine-learning model.

---

## Linear Regression

The first machine-learning model is deliberately implemented **from scratch** rather than relying immediately on `scikit-learn`.

For a single input feature:

$$
\hat{y}=wx+b
$$

For multiple input features:

$$
\hat{y}=w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

or, in matrix form:

$$
\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}+b
$$

This approach provides a clearer understanding of how regression models operate internally.

---

## Gradient Descent

The model is trained by minimising the Mean Squared Error (MSE):

$$
J(w,b)=
\frac{1}{n}
\sum_{i=1}^{n}
(\hat{y}_i-y_i)^2
$$

The model parameters are updated iteratively:

$$
w=w-\alpha\frac{\partial J}{\partial w}
$$

$$
b=b-\alpha\frac{\partial J}{\partial b}
$$

where:

* \(w\) represents the model weights.
* \(b\) represents the bias.
* \(\alpha\) is the learning rate.

The implementation demonstrates the basic training cycle used in many machine-learning algorithms:

```text
Prediction
    ↓
Calculate Error
    ↓
Calculate Gradients
    ↓
Update Parameters
    ↓
Repeat
```

---

## Model Evaluation

The following regression metrics are used.

### Mean Absolute Error

$$
MAE=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat{y}_i|
$$

### Mean Squared Error

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$

### Root Mean Squared Error

$$
RMSE=\sqrt{MSE}
$$

### Coefficient of Determination

$$
R^2=
1-
\frac{
\sum(y_i-\hat{y}_i)^2
}{
\sum(y_i-\bar{y})^2
}
$$

These metrics provide complementary information about prediction accuracy.

---

## Data

The project can initially use **synthetically generated propagation data** based on the Friis free-space model.

Noise can be introduced into the theoretical measurements to simulate the variability that may occur in practical measurements.

Future versions can incorporate real field measurements.
A suitable dataset structure can be:
```text
distance
frequency
transmitter_power
transmitter_gain
receiver_gain
received_power
```
---

## Project Structure
```text
radio-signal-strength-prediction/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_linear_regression_from_scratch.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── linear_regression.py
│   ├── metrics.py
│   └── visualization.py
│
├── models/
│
├── results/
│   ├── figures/
│   └── reports/
│
└── tests/
    └── test_linear_regression.py
```

---

## Technologies

The initial implementation uses:
* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

Future versions may include:

* Scikit-learn
* SciPy
* XGBoost
* PyTorch
* TensorFlow
* FastAPI
* Django REST Framework

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yuramsoft/radio-signal-strength-prediction.git
cd radio-signal-strength-prediction
```
### 2. Create a virtual environment
Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Then open the notebooks in the `notebooks/` directory.

---

## Example Prediction Workflow

A simplified prediction workflow is:

```python
prediction = model.predict(X_test)
```

The resulting predictions can then be compared with the actual received-power measurements.

---

## Visualisation

The project is intended to generate visualisations including:

* Received power versus distance
* Actual versus predicted received power
* Training loss versus epoch
* Prediction error
* Received power versus frequency
* Log-distance propagation plots

These visualisations help interpret both the physical propagation behaviour and the machine-learning model.

---

## Limitations

The initial linear regression model is intentionally simple.

Real wireless propagation can involve:

* Reflection
* Diffraction
* Scattering
* Multipath propagation
* Shadowing
* Terrain effects
* Building penetration losses
* Antenna orientation
* Frequency-dependent propagation
* Atmospheric effects

Consequently, a simple linear regression model may not adequately represent complex propagation environments.

The purpose of the initial implementation is therefore to establish the machine-learning workflow and provide a foundation for more advanced models.

---

## Future Work

The project can be progressively extended from a simple ML refresher into a more realistic wireless-propagation research project.

Possible developments include:

1. Polynomial regression
2. Ridge regression
3. Lasso regression
4. Decision-tree regression
5. Random Forest regression
6. Gradient Boosting
7. XGBoost
8. Artificial Neural Networks
9. Deep-learning-based propagation prediction
10. Real-world RF measurement datasets
11. Terrain and geographical features
12. TV White Space (TVWS) propagation modelling
13. Spatial/geospatial prediction
14. FastAPI model deployment
15. Django-based web interface

---

## Research and Engineering Applications

Potential applications include:

* Wireless network planning
* RF coverage prediction
* Signal-strength estimation
* Spectrum monitoring
* TV White Space analysis
* Wireless sensor networks
* IoT network planning
* Cellular network optimisation
* Radio propagation research
* Machine-learning-assisted wireless communication design

---

## Learning Outcomes

This project provides practical experience in:

* Python programming
* NumPy-based numerical computing
* Linear algebra
* Statistical modelling
* Regression
* Gradient descent
* Data preprocessing
* Model evaluation
* Data visualisation
* Wireless propagation
* Machine-learning fundamentals

It is particularly intended as a bridge between **engineering theory and practical machine-learning implementation**.

---

## Author

**Ibrahim Mustapha, PhD**

Department of Electrical & Electronic Engineering
University of Maiduguri
Communications Engineering / Wireless Communications
Python & Machine Learning

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This project is primarily an educational and research-oriented implementation. Predictions produced by the initial model should not be treated as a substitute for detailed radio-frequency planning, field measurements, or validated commercial propagation models.
