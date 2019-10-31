import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# import seaborn as sns
from itertools import count
from Robinhood import Robinhood
import os
from dotenv import load_dotenv
load_dotenv(verbose=True)

st.title("Stock Option Strategies")

class BaseOptions:
    def __init__(self, strike_price, cost):
        self._expand_stock_xrange = 3
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

###############################################################################################################
class LongCall(BaseOptions):
    def __init__(self, strike_price, cost):
        super().__init__(strike_price, cost)
    
    def calc_payoff_profit(self, highest_strike_price):
        self.stock_xrange = np.arange(self._expand_stock_xrange*highest_strike_price+1)
        self.payoff = (self.stock_xrange>=self.strike_price) * (self.stock_xrange-self.strike_price)
        self.profit = self.payoff - self.cost

    def __repr__(self):
        return f"<LongCall X=${self.strike_price}>"

###############################################################################################################
class ShortCall(BaseOptions):
    def __init__(self, strike_price, cost):
        super().__init__(strike_price, cost)

    def calc_payoff_profit(self, highest_strike_price):
        self.stock_xrange = np.arange(self._expand_stock_xrange*highest_strike_price+1)
        self.payoff = (self.stock_xrange>=self.strike_price) * (self.stock_xrange-self.strike_price) * (-1)
        self.profit = self.payoff + self.cost

    def __repr__(self):
        return f"<ShortCall X=${self.strike_price}>"

###############################################################################################################
class LongPut(BaseOptions):
    def __init__(self, strike_price, cost):
        super().__init__(strike_price, cost)

    def calc_payoff_profit(self, highest_strike_price):
        self.stock_xrange = np.arange(self._expand_stock_xrange*highest_strike_price+1)
        self.payoff = (self.stock_xrange<=self.strike_price) * (self.stock_xrange-self.strike_price) * (-1)
        self.profit = self.payoff - self.cost

    def __repr__(self):
        return f"<LongPut X=${self.strike_price}>"

###############################################################################################################
class ShortPut(BaseOptions):
    def __init__(self, strike_price, cost):
        super().__init__(strike_price, cost)

    def calc_payoff_profit(self, highest_strike_price):
        self.stock_xrange = np.arange(self._expand_stock_xrange*highest_strike_price+1)
        self.payoff = (self.stock_xrange<=self.strike_price) * (self.stock_xrange-self.strike_price)
        self.profit = self.payoff + self.cost

    def __repr__(self):
        return f"<ShortCall X=${self.strike_price}>"

###############################################################################################################
class StrategyOptions():
    _ids = count(0)

    def __init__(self, list_options):
        self.id = next(self._ids)
        self.list_options = list_options
        self.max_strike_price = max([op.strike_price for op in list_options])
        self.stock_xrange = np.array([])
        self.list_payoff, self.list_profit = [], []
        self.break_even = np.array([])
        self.max_profit, self.min_profit = 0, 0

    def draw_payoff(self, plot_individual=False, title=None):
        if not title: title = "Payoff & Profit"
        plt.figure(self.id)
        for op in self.list_options:
            op.calc_payoff_profit(self.max_strike_price)
            if len(self.stock_xrange)==0: self.stock_xrange = op.stock_xrange

            if plot_individual: op.draw_payoff()
            self.list_payoff.append(op.payoff)
            self.list_profit.append(op.profit)
        
        self.total_payoff = np.sum(np.asarray(self.list_payoff), axis=0)
        self.total_profit = np.sum(np.asarray(self.list_profit), axis=0)
        self.max_profit, self.min_profit = max(self.total_profit), min(self.total_profit)
        
        plt.plot(self.total_payoff, '-.', label="Total Payoff", linewidth=3.0, color="green")
        plt.plot(self.total_profit, label="Total Profit", linewidth=3.0, color="green")
        plt.title(title)
        plt.legend()
        plt.grid()

        # Move legend table out of the plot
        # Ref: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height*0.15, box.width, box.height*0.85])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

        # plt.show()
        st.pyplot()

    def calc_break_even(self):
        """
        Calculate zero-crossing points for break-even
        This function separates into 2 cases: with/without 0 in total_profit
        """
        break_even_idx = []
        if 0 not in self.total_profit:
            sign = np.sign(self.total_profit)
            sign_product = np.asarray([sign[idx]*sign[idx-1] for idx in range(1, len(sign))])
            for idx in np.where(sign_product == -1.0)[0]:
                x = np.array(self.stock_xrange[idx:idx+2])
                y = np.array(self.total_profit[idx:idx+2])

                # Find line equation: profit = alpha * stock_price + beta
                alpha, beta = np.polyfit(x, y, deg=1)
                self.break_even = np.append(self.break_even, -beta/alpha)
        else:
            for idx, p in enumerate(self.total_profit):
                if p == 0:
                    if idx==0 and self.total_profit[1]!=0:
                        break_even_idx.append(idx)
                    elif idx==len(self.total_profit)-1 and self.total_profit[len(self.total_profit)-2]!=0:
                        break_even_idx.append(idx)
                    elif (0<idx<len(self.total_profit)-1) and\
                        (self.total_profit[idx-1]!=0 and self.total_profit[idx+1]!=0 and self.total_profit[idx-1]*self.total_profit[idx+1]<=0 or\
                            self.total_profit[idx-1]!=0 and self.total_profit[idx+1]==0 or\
                            self.total_profit[idx-1]==0 and self.total_profit[idx+1]!=0):
                        break_even_idx.append(idx)
        
            self.break_even = self.stock_xrange[break_even_idx]

    def display_option_result(self):
        """
        Display option information, i.e. break-even, max profit, max loss, etc.
        Need to convert numpy elements (array, int, ...) to list bc Streamlit can only display list via st.json
        """
        option_json = {"Option": self.__repr__(),
            "Break-even": self.break_even.tolist(),
            "Max profit": self.max_profit.tolist(),
            "Max loss": self.min_profit.tolist()}
        st.json(option_json)

    def __repr__(self):
        return f"<StrategyOptions: {self.id, self.list_options}>"

###############################################################################################################
def read_strategy_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    for _, curr_strategy_df in df.groupby(['Strategy']):
        curr_list_options = []
        curr_title = None
        for index, row in curr_strategy_df.iterrows():
            curr_option = eval("{}(strike_price={}, cost={})".format(row["Option_type"], row["Strike"], row["Cost"]))
            curr_list_options.append(curr_option)

            try: 
                if np.isnan(row["Name"]): pass
            except:
                curr_title = row["Name"]
        
        strategy = StrategyOptions(curr_list_options)
        strategy.draw_payoff(plot_individual=plot_individual, title=curr_title)
        strategy.calc_break_even()
        strategy.display_option_result()

if __name__ == "__main__":
    plot_individual = True
    csv_path = r"BYND.csv"
    read_strategy_from_csv(csv_path)

    # my_trader = Robinhood()
    # logged_in = my_trader.login(username=os.getenv("EMAIL"), password=os.getenv("PASSWORD"), qr_code=os.getenv("QR"))
    # print(logged_in)
    # quote_info = my_trader.quote_data("AMD")
    # print(quote_info)