import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

def pin_likelihood(params, buy_orders, sell_orders):
    alpha, delta, mu, epsilon_b, epsilon_s = params

    lambda_b = alpha * mu + epsilon_b
    lambda_s = alpha * (1 - delta) * mu + epsilon_s
    lambda_0 = (1 - alpha) * mu

    pi_buy = lambda_b / (lambda_b + lambda_s + lambda_0)
    pi_sell = lambda_s / (lambda_b + lambda_s + lambda_0)

    likelihood = (
            np.sum(np.log(pi_buy + 1e-10) * buy_orders) +
            np.sum(np.log(pi_sell + 1e-10) * sell_orders)
    )
    return -likelihood

def estimate_pin(buy_orders, sell_orders, alpha=0.05, delta=0.5, mu=0.5, epsilon_b=0.5, epsilon_s=0.5):
    initial_guess = [alpha, delta, mu, epsilon_b, epsilon_s]
    bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, None), (0.01, None), (0.01, None)]

    options = {'maxiter': 1000, 'disp': False}

    result = minimize(pin_likelihood, initial_guess, args=(buy_orders, sell_orders), bounds=bounds, options=options, method='L-BFGS-B')

    if result.success:
        alpha, delta, mu, epsilon_b, epsilon_s = result.x
        pin = alpha * mu / (alpha * mu + epsilon_b + epsilon_s)
        return pin
    else:
        return 0

def probability_of_informed_trading(buy_orders, sell_orders, alpha=0.05, delta=0.5, mu=0.5, epsilon_b=0.5,
                                    epsilon_s=0.5, window=21):
    buy_orders = buy_orders.reset_index(drop=True)
    sell_orders = sell_orders.reset_index(drop=True)

    n = len(buy_orders)
    pins = []
    for start in tqdm(range(0, n - window + 1)):
        end = start + window
        window_buy_orders = buy_orders[start:end]
        window_sell_orders = sell_orders[start:end]
        pin = estimate_pin(window_buy_orders, window_sell_orders, alpha, delta, mu, epsilon_b, epsilon_s)
        pins.append(pin)

    pins = pd.Series(
        pins,
        index=buy_orders.index[window - 1:],
        name='pin'
    )
    return pins