
# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch

# Convinience functions
def plot_model(model=None):
    # Visualize data
    plt.plot(torch.linspace(0, 1, 1000), ground_truth_function(torch.linspace(0, 1, 1000)), label='Ground truth')
    plt.plot(x_train, y_train, 'ob', label='Train data')
    plt.plot(x_test, y_test, 'xr', label='Test data')
    # Visualize model
    if model is not None:
        plt.plot(torch.linspace(0, 1, 1000), model(torch.linspace(0, 1, 1000)), label=f'Model of degree: {model.degree()}')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    
    plt.show()

# Generate data
n_samples = 11
noise_amplitude = 0.15

def ground_truth_function(x):
    # Generate data of the form sin(2 * Pi * x)
    result = np.sin(2*np.pi*x)
    return result

torch.manual_seed(42)

x_test = torch.linspace(0, 1, n_samples)
y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(n_samples,))
x_train = torch.linspace(0, 1, n_samples)
y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(n_samples,))

# Test plotting
plot_model()
plt.savefig('Initial_data.png')
plt.clf()


# Model fitting

def error_function(model, x_data, y_data):
    y_pred = model(x_data)
    # ---- Fill with the error function from the lecture
    #Mean Squared Error
    error = 0.5*(torch.sum((y_pred-y_data)**2))
    return error

model_degree = 3

model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
train_err = error_function(model, x_train, y_train)
test_err = error_function(model, x_test, y_test)

print(f"{model_degree=},{train_err=}, {test_err=}")


# Result plotting
plot_model(model)
plt.savefig('Initial_fit.png')
plt.clf()

# ---- Continue with the exercises on the degree of the polynomial and the exploration of data size

model_degree = 11
model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
train_err = error_function(model, x_train, y_train)
test_err = error_function(model, x_test, y_test)

print(f"{model_degree=},{train_err=}, {test_err=}")


# Result plotting
plot_model(model)
plt.savefig('Polynomial_M=11_fit.png')
plt.clf()

#----Polynomial degree against the train and test error-----

def rmserror_function(model, x_data, y_data):
    y_pred = model(x_data)
    # ---- Fill with the error function from the lecture
    error = torch.sqrt(torch.sum(torch.square(y_pred-y_data))/len(x_data))
    return error


train_err=[]
test_err=[]
degree=[]
for model_degree in range(0,12):
    model = np.polynomial.Polynomial.fit(x_train, y_train, deg=model_degree)
    degree.append(model_degree)
    train_err.append(rmserror_function(model, x_train, y_train))
    test_err.append(rmserror_function(model, x_test, y_test))
    
plt.plot(degree, train_err, 'bo-', label='Training')
plt.plot(degree, test_err, 'ro-', label='Testing')
plt.xlabel("degree")
plt.ylabel("RMSE")
plt.legend()
plt.show()

#---- Varying sample size----

model_degree = 10
x_train = torch.linspace(0, 1, 10**5)
y_train = ground_truth_function(x_train) + torch.normal(0., noise_amplitude, size=(10**5,))
for N in range(10,10**5,100):
    x_test = torch.linspace(0, 1, N)
    y_test = ground_truth_function(x_test) + torch.normal(0., noise_amplitude, size=(N,))
    model = np.polynomial.Polynomial.fit(x_train[:N], y_train[:N], deg=model_degree)
    train_err = rmserror_function(model, x_train[:N], y_train[:N])
    test_err = rmserror_function(model, x_test, y_test)
    sample_size=N
    #np.isclose(a,b,atol=1e-8) If the absolute difference between a and b is less than or equal to atol, np.isclose returns True. Otherwise, it returns False
    if np.isclose(train_err,test_err):
        print(f"{sample_size=},{train_err=}, {test_err=}")
        break
    

