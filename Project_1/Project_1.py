import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(888)
# Parameters
S0 = 100  # Initial stock price
r = 0.05  # Risk-free rate
q = 0.02  # Dividend yield
mu = r - q  # Drift term
sigma = 0.40  # Volatility
T = 0.75  # Time to maturity (years)
dt = 1 / 250  # Time step size
n_steps = round(T / dt)  # Number of time steps
n_simulations = 10000  # Number of simulated paths
discount_factor = np.exp(-r * T) #discount factor


def simu_stock_price():
    # Generate standard normal random variables (epsilon)
    epsilon = np.random.normal(0, 1, (n_simulations, n_steps))
    # Initialize arrays for log price increments and stock price paths
    delta_x = np.zeros((n_simulations, n_steps))  # To store log price increments
    S = np.zeros((n_simulations, n_steps + 1))   # To store stock price paths
    # Set initial stock price
    S[:, 0] = S0
    # Calculate the increments
    for t in range(n_steps):
        delta_x[:, t] = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon[:, t]
    # Calculate stock price paths based on log price increments
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t - 1] * np.exp(delta_x[:, t - 1])  # Use the cumulative formula
    return S

stock_price_set = simu_stock_price()
fin_sp = stock_price_set[:, -1]
S_avg = np.mean(fin_sp)

def group_s_plot(data_set,x_label,ylable,titile):
    sns.histplot(data_set, bins=50, kde=True, color="green", label="Payoff", stat="frequency", alpha=0.6)
    plt.axvline(np.mean(data_set), color="red", linestyle="--", label=f"Average Payoff: {np.mean(data_set):.2f}")
    plt.title(titile)
    plt.xlabel(x_label)
    plt.ylabel(ylable)
    plt.yscale("log")  # Set y-axis to logarithmic scale
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

# Question 1& 2 Plain vanilla European options
def c_van_eur_option():
    payoff_call = np.maximum(fin_sp - 110, 0)
    price_call = np.mean(payoff_call) * discount_factor
    print("Plain Vanilla European Options:")
    print(f"Call Option Price: {price_call:.2f}")
    group_s_plot(payoff_call,"Payoff","Frequency","Group 6 Plain Vanilla European call Option Payoffs")

# c_van_eur_option()

def p_van_eur_option():
    # 计算看涨期权和看跌期权的收益
    payoff_put = np.maximum(90 - fin_sp, 0)
    # calculate price for the option
    price_put = np.mean(payoff_put)* discount_factor
    print("Plain Vanilla European Options:")
    print(f"Put Option Price: {price_put:.2f}")
    group_s_plot(payoff_put, "Payoff", "Frequency", "Group 6 Plain Vanilla European put Option Payoffs")

# p_van_eur_option()

#Question 3,4  Lookback options
def c_lb_option():
    fin_sp = stock_price_set[:, -1]
    S_min = np.min(stock_price_set, axis=1)  # Minimum stock price in each path
    payoff_lookback_call = fin_sp - S_min
    price_lookback_call = np.mean(payoff_lookback_call) * discount_factor

    print("\nLookback Options:")
    print(f"Lookback Call Option Price: {price_lookback_call:.2f}")
    group_s_plot(payoff_lookback_call, "Payoff", "Frequency", "Group 6 Plain Lookback Call Option Payoffs")

# c_lb_option()

def p_lb_option():
    S_max = np.max(stock_price_set, axis=1)  # Maximum stock price in each path
    payoff_lookback_put = S_max - fin_sp
    price_lookback_put = np.mean(payoff_lookback_put) * discount_factor
    print("\nLookback Options:")
    print(f"Lookback Put Option Price: {price_lookback_put:.2f}")
    group_s_plot(payoff_lookback_put, "Payoff", "Frequency", "Group 6 Plain Lookback Put Option Payoffs")

# p_lb_option()

# Asian payoffs
def c_a_option():
    payoff_asian_call = np.maximum(fin_sp - S_avg, 0)
    # Discounted price
    price_asian_call = np.mean(payoff_asian_call) * discount_factor
    print("\nAsian Options:")
    print(f"Asian Call Option Price: {price_asian_call:.2f}")
    group_s_plot(payoff_asian_call, "Payoff", "Frequency", "Group 6 Asian Call Option Payoffs")


# c_a_option()

def p_a_option():
 # Average stock price in each path
    payoff_asian_put = np.maximum(S_avg - fin_sp, 0)
    price_asian_put = np.mean(payoff_asian_put) * discount_factor
    print("\nAsian Options:")
    print(f"Asian Put Option Price: {price_asian_put:.2f}")
# Plot histogram
    group_s_plot(payoff_asian_put, "Payoff", "Frequency", "Group 6 Asian Put Option Payoffs")

p_a_option()




print("Hello World")