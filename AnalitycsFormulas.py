from scipy.stats import norm
import math

class AnalyticFormulas:
    @staticmethod
    def blackScholesOptionValue(initialStockValue, riskFreeRate, volatility, optionMaturity, optionStrike):
        """
        Computes the Black-Scholes price of a "European" call option.

        Formula:
            Let:
                S_0 = initial stock price
                K = strike price
                r = risk-free rate
                U+03c = volatility
                T = time to maturity

            We define:
                d1 = [ ln(S_0 / K) + (r + 0.5 U+03c²) T ] / (U+03c √T)
                d2 = d1 - U+03c √T

            Then, the call price is:
                C = S_0 Φ(d1) - K e^{-rT} Φ(d2)

            where Φ(·) is the cumulative distribution function of the standard normal distribution.

        Parameters:
            initialStockValue (float): S_0, initial stock price
            riskFreeRate (float): r, risk-free interest rate
            volatility (float): U+03c, volatility of the underlying
            optionMaturity (float): T, time to maturity (in years)
            optionStrike (float): K, strike price

        Returns:
            float: Call option price according to Black-Scholes formula
        """
        if optionStrike <= 0.0:
            return initialStockValue - optionStrike * math.exp(-riskFreeRate * optionMaturity)

        sqrtT = math.sqrt(optionMaturity)
        d1 = (math.log(initialStockValue / optionStrike) +
              (riskFreeRate + 0.5 * volatility * volatility) * optionMaturity) / (volatility * sqrtT)
        d2 = d1 - volatility * sqrtT

        callPrice = (initialStockValue * norm.cdf(d1)
                     - optionStrike * math.exp(-riskFreeRate * optionMaturity) * norm.cdf(d2))
        return callPrice


