import matplotlib.pyplot as plt
import math

# Plot the results
def plot_epsilon_real(rapidityVals, epsilonm2New, epsilonm1New, epsilon0New, epsilon1New, epsilon2New):
    plt.figure(figsize=(12, 6))
    plt.plot(rapidityVals, epsilonm2New.real, label='ε-2')
    plt.plot(rapidityVals, epsilonm1New.real, label='ε-1')
    plt.plot(rapidityVals, epsilon0New.real, label='ε0')
    plt.plot(rapidityVals, epsilon1New.real, label='ε1')
    plt.plot(rapidityVals, epsilon2New.real, label='ε2')
    plt.legend()
    plt.xlabel('β')
    plt.ylabel('ε')
    plt.title('Re')
    plt.show()
def plot_epsilon_im(rapidityVals, epsilonm2New, epsilonm1New, epsilon0New, epsilon1New, epsilon2New):
    plt.figure(figsize=(12, 6))
    plt.plot(rapidityVals, epsilonm2New.imag, label='ε-2')
    plt.plot(rapidityVals, epsilonm1New.imag, label='ε-1')
    plt.plot(rapidityVals, epsilon0New.imag, label='ε0')
    plt.plot(rapidityVals, epsilon1New.imag, label='ε1')
    plt.plot(rapidityVals, epsilon2New.imag, label='ε2')
    plt.legend()
    plt.xlabel('β')
    plt.ylabel('ε')
    plt.title('Im')
    plt.show()

#compute ln(1+Y^-1)
def compute_logY0(epsilonm2New, epsilonm1New, epsilon0New, epsilon1New, epsilon2New):
    epsilonfn=0.2*(epsilonm2New.real+epsilonm1New.real+epsilon0New.real+epsilon2New.real+epsilon1New.real)
    Yfun=[math.log(1+math.exp(-x)) for x in epsilonfn]
    return Yfun

#plot ln(1+Y^-1)
def plot_Y0(rapidityVals, Yfun):
    plt.figure(figsize=(12, 6))
    plt.plot(rapidityVals, Yfun, label='ε1')
    plt.legend()
    plt.xlabel('β')
    plt.ylabel('ε')
    plt.title('Re')
    plt.show()