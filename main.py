import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# st.title("Hello World")

class BaseOptions:
    def __init__(self, strike_price, cost):
        self.strike_price = strike_price
        self.cost = cost
        self.payoff = np.array([])
        self.profit = np.array([])

    def draw_payoff(self):
        """
        Draw Payoff line for Long Call option
        """
        if len(self.payoff)==0:
            raise Exception("Payoff and Profit are not calculated for {}".format(self.__repr__()))
        plt.plot(self.stock_xrange, self.payoff, '--', label="Payoff - {}".format(self.__repr__()))

    def draw_profit(self):
        """
        Draw Profit line for Long Call option
        """
        if len(self.profit)==0:
            raise Exception("Payoff and Profit are not calculated for {}".format(self.__repr__()))
        plt.plot(self.stock_xrange, self.profit, '--', label="Profit - {}".format(self.__repr__()))

###############################################################################################################
class LongCall(BaseOptions):
    def __init__(self, strike_price, cost):
        super().__init__(strike_price, cost)
    
    def calc_payoff_profit(self, highest_strike_price):
        self.stock_xrange = np.arange(2*highest_strike_price+1)
        self.payoff = (self.stock_xrange>=self.strike_price) * (self.stock_xrange-self.strike_price)
        self.profit = self.payoff - self.cost

    def __repr__(self):
        return f"<LongCall X=${self.strike_price}>"

###############################################################################################################
class ShortCall(BaseOptions):
    def __init__(self, strike_price, cost):
        super().__init__(strike_price, cost)

    def calc_payoff_profit(self, highest_strike_price):
        self.stock_xrange = np.arange(2*highest_strike_price+1)
        self.payoff = (self.stock_xrange>=self.strike_price) * (self.stock_xrange-self.strike_price) * (-1)
        self.profit = self.payoff + self.cost

    def __repr__(self):
        return f"<ShortCall X=${self.strike_price}>"

###############################################################################################################
class LongPut(BaseOptions):
    def __init__(self, strike_price, cost):
        super().__init__(strike_price, cost)

    def calc_payoff_profit(self, highest_strike_price):
        self.stock_xrange = np.arange(2*highest_strike_price+1)
        self.payoff = (self.stock_xrange<=self.strike_price) * (self.stock_xrange-self.strike_price) * (-1)
        self.profit = self.payoff - self.cost

    def __repr__(self):
        return f"<LongPut X=${self.strike_price}>"

###############################################################################################################
class ShortPut(BaseOptions):
    def __init__(self, strike_price, cost):
        super().__init__(strike_price, cost)

    def calc_payoff_profit(self, highest_strike_price):
        self.stock_xrange = np.arange(2*highest_strike_price+1)
        self.payoff = (self.stock_xrange<=self.strike_price) * (self.stock_xrange-self.strike_price)
        self.profit = self.payoff + self.cost

    def __repr__(self):
        return f"<ShortCall X=${self.strike_price}>"

###############################################################################################################
class StrategyOptions():
    def __init__(self, list_options):
        self.list_options = list_options
        self.max_strike_price = max([op.strike_price for op in list_options])
        self.list_payoff, self.list_profit = [], []

    def draw_payoff(self, plot_individual=False):
        for op in self.list_options:
            op.calc_payoff_profit(self.max_strike_price)
            if plot_individual: op.draw_payoff()
            self.list_payoff.append(op.payoff)
        
        self.total_payoff = np.sum(np.asarray(self.list_payoff), axis=0)
        
        plt.plot(self.total_payoff, label="Total Payoff")
        plt.title("Strategy Options - Payoff")
        plt.legend()
        plt.show()

    def draw_profit(self, plot_individual=False):
        for op in self.list_options:
            op.calc_payoff_profit(self.max_strike_price)
            if plot_individual: op.draw_profit()
            self.list_profit.append(op.profit)
        
        self.total_profit = np.sum(np.asarray(self.list_profit), axis=0)

        plt.plot(self.total_profit, label="Total Profit")
        plt.title("Strategy Options - Profit")
        plt.legend()
        plt.show()

    def __repr__(self):
        return f"<MultipleOptions: {self.list_options}>"

if __name__ == "__main__":
    multipleOptions_plot_individual = True

    # longcall = LongCall(20, 50)
    # shortcall = ShortCall(30, 50)
    # options = StrategyOptions([longcall, shortcall])
    # print(options)
    # options.draw_payoff(plot_individual=multipleOptions_plot_individual)

    # longput = LongPut(30, 30)
    # shortput = ShortPut(10, 10)
    # options = StrategyOptions([longput, shortput])
    # print(options)
    # options.draw_payoff(plot_individual=multipleOptions_plot_individual)

    # Butterfly
    longcall_1 = LongCall(20, 20)
    longcall_3 = LongCall(40, 40)
    shortcall = ShortCall(30, 30)
    options = StrategyOptions([longcall_1, longcall_3, shortcall, shortcall])
    print(options)
    options.draw_payoff(plot_individual=multipleOptions_plot_individual)