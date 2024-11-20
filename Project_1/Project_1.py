import numpy as np
import numpy as np

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

print("Hello World")