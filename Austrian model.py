import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initial conditions
y_0 = 0.02
y_star_0 = 0.02
r_0 = 0.02
r_star_0 = 0.02
pi_0 = 0.02
tilde_r_star_0 = 0.02
pi_star = 0.02
M_0 = 1.0  # Initial value of the financial market

# Parameters
alpha = 4.5  # Increased responsiveness to inflation overshooting
beta = 0.5
lambda_ = 0.7  # Increased learning rate for faster convergence
gamma = 1.0
kappa = 0.5  # Increased sensitivity of inflation to the output gap
delta = 0.1  # Sensitivity of the financial market to interest rate differences

# Simulation parameters
years = 40
y_star_path = [0.02] * 1 + [0.025] * (years - 1)  # y* changes to 2.5% at year 1

# Initialize arrays to store results
y = np.zeros(years)
y_star = np.zeros(years)
r = np.zeros(years)
r_star = np.zeros(years)
pi = np.zeros(years)
tilde_r_star = np.zeros(years)
M = np.zeros(years)

# Set initial values
y[0] = y_0
y_star[0] = y_star_0
r[0] = r_0
r_star[0] = r_star_0
pi[0] = pi_0
tilde_r_star[0] = tilde_r_star_0
M[0] = M_0

# Run simulation
for t in range(1, years):
    y_star[t] = y_star_path[t]
    r_star[t] = gamma * y_star[t]
    tilde_r_star[t] = tilde_r_star[t-1] + lambda_ * (r_star[t-1] - tilde_r_star[t-1])
    r[t] = tilde_r_star[t] + alpha * (pi[t-1] - pi_star)
    y[t] = y_star[t] - beta * (r[t] - r_star[t])
    pi[t] = pi[t-1] + kappa * (y[t] - y_star[t])
    M[t] = M[t-1] + delta * (r_star[t] - r[t])

# Convert to DataFrame and multiply values by 100 to display percentages
df = pd.DataFrame({
    'Year': np.arange(years),
    'y': y * 100,
    'y_star': y_star * 100,
    'r': r * 100,
    'r_star': r_star * 100,
    'tilde_r_star': tilde_r_star * 100,
    'pi': pi * 100,
    'M': M*100
})

# Plot the results up to year 10
plot_years = 10
df_short = df[df['Year'] < plot_years]

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df_short['Year'], df_short['y'], label='y (Aggregate Demand)')
plt.plot(df_short['Year'], df_short['y_star'], label='y* (Potential Output)', linestyle='--')
plt.title('Aggregate Demand and Potential Output')
plt.xlabel('Year')
plt.ylabel('Growth Rate (%)')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(df_short['Year'], df_short['r'], label='r (Interest Rate)')
plt.plot(df_short['Year'], df_short['r_star'], label='r* (Natural Rate)', linestyle='--')
plt.plot(df_short['Year'], df_short['tilde_r_star'], label='~r* (Perceived Natural Rate)', linestyle='-.')
plt.title('Interest Rates')
plt.xlabel('Year')
plt.ylabel('Rate (%)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(df_short['Year'], df_short['pi'], label='π (Inflation)')
plt.axhline(pi_star * 100, color='r', linestyle='--', label='π* (Inflation Target)')
plt.title('Inflation')
plt.xlabel('Year')
plt.ylabel('Rate (%)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(df_short['Year'], df_short['M'], label='M (Financial Market)')
plt.title('Financial Market')
plt.xlabel('Year')
plt.ylabel('Market Value')
plt.legend()

plt.tight_layout()
plt.show()