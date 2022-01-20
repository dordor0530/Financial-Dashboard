# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:07:32 2021

@author: mma
"""

# -*- coding: utf-8 -*-
###############################################################################
# FINANCIAL DASHBOARD 1 - v1
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import yahoo_fin.stock_info as si
import streamlit as st
import numpy as np
import mplfinance as mpf
from dateutil.relativedelta import relativedelta

#==============================================================================
# Setting
#==============================================================================

# Title of the whole dashboard
st.set_page_config(
     page_title="Dorothy's Financial Dashboard"
   )

#==============================================================================
# Defining
#==============================================================================

# def title of each page
def title(page):
        if ticker != '-':
            st.title(page + " for " + ticker)
            st.caption('Currency in USD')
            st.caption('Source: Yahoo! Finance')
        else:
            st.title(page + " for the stock")
            st.caption('Currency in USD')
            st.caption('Source: Yahoo! Finance') 

#==============================================================================
# Disable warning
#==============================================================================

st.set_option('deprecation.showPyplotGlobalUse', False)

#==============================================================================
# Tab 1: Summary
#=============================================================================

class Summary:
    def run(self):
            
        title('Summary')
        
        # Add table to show stock data
        def GetSummary(ticker):
            return si.get_quote_table(ticker, dict_result=False)
           
        if ticker != '-':
            info = GetSummary(ticker)
            info['value'] =info['value'].astype(str)
            info['attribute'] =info['attribute'].astype(str)
            info = info.set_index('attribute')  
            st.dataframe(info, height = 1000)
            
        st.write('Choose a period to view the stock chart')
        
        b1, b2, b3, b4, b5, b6, b7, b8, b9 = st.columns(9)
            
        y = today.year
        
        # 1M    
        if b1.button('1M'):
            if ticker != '-':
                
                start_date_stock_1M = today + relativedelta(months=-1)
                
                graphdata = si.get_data(ticker, start_date = start_date_stock_1M, end_date = today, index_as_date = False, interval = "1d")
                x = graphdata['date']
                y = graphdata['close']
                
                fig,ax = plt.subplots()
                
                ymin = min(graphdata['close'] - min(graphdata['close'])*0.05)
                ymax = max(graphdata['close'] + max(graphdata['close'])*0.05)
                
                ax.set(ylim=(ymin, ymax))

                if graphdata.iloc[-1,4]-graphdata.iloc[0,4]>=0:            
                    ax.fill_between(x, y, facecolor='green')

                    st.pyplot(fig)
                else:
                    ax.fill_between(x, y, facecolor='red')
                    st.pyplot(fig)

        # 3M
        if b2.button('3M'):
            if ticker != '-':
                
                start_date_stock_3M = today + relativedelta(months=-3)
                
                graphdata = si.get_data(ticker, start_date = start_date_stock_3M, end_date = today, index_as_date = False, interval = "1d")
                x = graphdata['date']
                y = graphdata['close']
                fig,ax = plt.subplots()
                                
                ymin = min(graphdata['close'] - min(graphdata['close'])*0.05)
                ymax = max(graphdata['close'] + max(graphdata['close'])*0.05)
                
                ax.set(ylim=(ymin, ymax))
                
                if graphdata.iloc[-1,4]-graphdata.iloc[0,4]>=0:            
                    ax.fill_between(x, y, facecolor='green')
                    st.pyplot(fig)
                else:
                    ax.fill_between(x, y, facecolor='red')
                    st.pyplot(fig)

        # 6M
        if b3.button('6M'):
            if ticker != '-':
                
                start_date_stock_6M = today + relativedelta(months=-6)
                                
                graphdata = si.get_data(ticker, start_date = start_date_stock_6M, end_date = today, index_as_date = False, interval = "1d")
                x = graphdata['date']
                y = graphdata['close']
                fig,ax = plt.subplots()
                                
                ymin = min(graphdata['close'] - min(graphdata['close'])*0.05)
                ymax = max(graphdata['close'] + max(graphdata['close'])*0.05)
                
                ax.set(ylim=(ymin, ymax))
                
                if graphdata.iloc[-1,4]-graphdata.iloc[0,4]>=0:            
                    ax.fill_between(x, y, facecolor='green')
                    st.pyplot(fig)
                else:
                    ax.fill_between(x, y, facecolor='red')
                    st.pyplot(fig)

        # YTD
        if b4.button('YTD'):
            if ticker != '-':
                
                start_date_stock_YTD = date(y,1,1) 
                
                graphdata = si.get_data(ticker, start_date = start_date_stock_YTD, end_date = today, index_as_date = False, interval = "1d")
                x = graphdata['date']
                y = graphdata['close']
                fig,ax = plt.subplots()
                                
                ymin = min(graphdata['close'] - min(graphdata['close'])*0.05)
                ymax = max(graphdata['close'] + max(graphdata['close'])*0.05)
                
                ax.set(ylim=(ymin, ymax))
                
                if graphdata.iloc[-1,4]-graphdata.iloc[0,4]>=0:            
                    ax.fill_between(x, y, facecolor='green')
                    st.pyplot(fig)
                else:
                    ax.fill_between(x, y, facecolor='red')
                    st.pyplot(fig)
        
        # 1Y
        if b5.button('1Y'):
            if ticker != '-':
                    
                start_date_stock_1Y = today + relativedelta(years=-1)
                
                graphdata = si.get_data(ticker, start_date = start_date_stock_1Y, end_date = today, index_as_date = False, interval = "1d")
                x = graphdata['date']
                y = graphdata['close']
                fig,ax = plt.subplots()
                                
                ymin = min(graphdata['close'] - min(graphdata['close'])*0.05)
                ymax = max(graphdata['close'] + max(graphdata['close'])*0.05)
                
                ax.set(ylim=(ymin, ymax))
                
                if graphdata.iloc[-1,4]-graphdata.iloc[0,4]>=0:            
                    ax.fill_between(x, y, facecolor='green')
                    st.pyplot(fig)
                else:
                    ax.fill_between(x, y, facecolor='red')
                    st.pyplot(fig)
                
        # 3Y
        if b6.button('3Y'):
            if ticker != '-':
                
                start_date_stock_3Y = today + relativedelta(years=-3)

                graphdata = si.get_data(ticker, start_date = start_date_stock_3Y, end_date = today, index_as_date = False, interval = "1d")
                x = graphdata['date']
                y = graphdata['close']
                fig,ax = plt.subplots()
                                
                ymin = min(graphdata['close'] - min(graphdata['close'])*0.05)
                ymax = max(graphdata['close'] + max(graphdata['close'])*0.05)
                
                ax.set(ylim=(ymin, ymax))
                
                if graphdata.iloc[-1,4]-graphdata.iloc[0,4]>=0:            
                    ax.fill_between(x, y, facecolor='green')
                    st.pyplot(fig)
                else:
                    ax.fill_between(x, y, facecolor='red')
                    st.pyplot(fig)
        
        # 5Y
        if b7.button('5Y'):
            if ticker != '-':
                
                start_date_stock_5Y = today + relativedelta(years=-5)
                
                graphdata = si.get_data(ticker, start_date = start_date_stock_5Y, end_date = today, index_as_date = False, interval = "1d")
                x = graphdata['date']
                y = graphdata['close']
                fig,ax = plt.subplots()
                                
                ymin = min(graphdata['close'] - min(graphdata['close'])*0.05)
                ymax = max(graphdata['close'] + max(graphdata['close'])*0.05)
                
                ax.set(ylim=(ymin, ymax))
                
                if graphdata.iloc[-1,4]-graphdata.iloc[0,4]>=0:            
                    ax.fill_between(x, y, facecolor='green')
                    st.pyplot(fig)
                else:
                    ax.fill_between(x, y, facecolor='red')
                    st.pyplot(fig)

        # MAX
        if b8.button('MAX'):
            if ticker != '-':        
                graphdata = si.get_data(ticker, end_date = today, index_as_date = False, interval = "1d")
                x = graphdata['date']
                y = graphdata['close']
                fig,ax = plt.subplots()
                                
                ymin = min(graphdata['close'] - min(graphdata['close'])*0.05)
                ymax = max(graphdata['close'] + max(graphdata['close'])*0.05)
                
                ax.set(ylim=(ymin, ymax))
                
                if graphdata.iloc[-1,4]-graphdata.iloc[0,4]>=0:            
                    ax.fill_between(x, y, facecolor='green')
                    st.pyplot(fig)
                else:
                    ax.fill_between(x, y, facecolor='red')
                    st.pyplot(fig)

        # Period Selected
        if b9.button('Period Selected'):
            if ticker != '-':
                try:
                    graphdata = si.get_data(ticker, start_date = start_date_stock, end_date = end_date_stock, index_as_date = False, interval = "1d")
                    x = graphdata['date']
                    y = graphdata['close']
                    fig,ax = plt.subplots()
                                
                    ymin = min(graphdata['close'] - min(graphdata['close'])*0.05)
                    ymax = max(graphdata['close'] + max(graphdata['close'])*0.05)
                
                    ax.set(ylim=(ymin, ymax))
                
                    if graphdata.iloc[-1,4]-graphdata.iloc[0,4]>=0:            
                        ax.fill_between(x, y, facecolor='green')
                        st.pyplot(fig)
                    else:
                        ax.fill_between(x, y, facecolor='red')
                        st.pyplot(fig)
                except KeyError:
                    st.write('Error from the period selected, please choose again.')
                except AssertionError:
                    st.write('Error from the period selected, please choose again.')

#==============================================================================
# Tab 2
#==============================================================================

class Chart:
    def run(self):
        title('Chart')
        st.write('Select the period everytime when you change the time inveral and/or graph type.')
        
        col1, col2 = st.columns(2)
        
        interval_dict = {'Day': '1d', 'Week': '1wk', 'Month': '1mo'}
        graph_dict = {'Line': 'line', 'Candle': 'candle'}
        
        ti = col1.selectbox(
            'Time Interval',
            ('Day', 'Week', 'Month'))
        
        graph = col2.selectbox(
            'Graph Type',
            ('Line','Candle'))
        
        b1, b2, b3, b4, b5, b6, b7, b8, b9 = st.columns(9)
        
        y = today.year

        # 1M    
        if b1.button('1M'):
            if ticker != '-':
                start_date_stock_1M = today + relativedelta(months=-1)
                graphdata = si.get_data(ticker, start_date = start_date_stock_1M, end_date = today, index_as_date = False, interval = interval_dict[ti])
                graphdata = graphdata.set_index('date')
                mpf.plot(graphdata,type=graph_dict[graph],style='yahoo',mav=50,volume=True) 
                st.pyplot()

                    
        # 3M
        if b2.button('3M'):
            if ticker != '-':
                start_date_stock_3M = today + relativedelta(months=-3)
                graphdata = si.get_data(ticker, start_date = start_date_stock_3M, end_date = today, index_as_date = False, interval = interval_dict[ti])
                graphdata = graphdata.set_index('date')
                mpf.plot(graphdata,type=graph_dict[graph],style='yahoo',mav=50,volume=True) 
                st.pyplot()

        # 6M
        if b3.button('6M'):
            if ticker != '-':
                start_date_stock_6M = today + relativedelta(months=-6)             
                graphdata = si.get_data(ticker, start_date = start_date_stock_6M, end_date = today, index_as_date = False, interval = interval_dict[ti])
                graphdata = graphdata.set_index('date')
                mpf.plot(graphdata,type=graph_dict[graph],style='yahoo',mav=50,volume=True) 
                st.pyplot()


        # YTD
        if b4.button('YTD'):
            if ticker != '-':
                start_date_stock_YTD = date(y,1,1) 
                graphdata = si.get_data(ticker, start_date = start_date_stock_YTD, end_date = today, index_as_date = False, interval = interval_dict[ti])
                graphdata = graphdata.set_index('date')
                mpf.plot(graphdata,type=graph_dict[graph],style='yahoo',mav=50,volume=True) 
                st.pyplot()
        
        # 1Y
        if b5.button('1Y'):
            if ticker != '-':
                start_date_stock_1Y = today + relativedelta(years=-1) 
                graphdata = si.get_data(ticker, start_date = start_date_stock_1Y, end_date = today, index_as_date = False, interval = interval_dict[ti])
                graphdata = graphdata.set_index('date')
                mpf.plot(graphdata,type=graph_dict[graph],style='yahoo',mav=50,volume=True) 
                st.pyplot()
                
        # 3Y
        if b6.button('3Y'):
            if ticker != '-':
                start_date_stock_3Y = today + relativedelta(years=-3) 
                graphdata = si.get_data(ticker, start_date = start_date_stock_3Y, end_date = today, index_as_date = False, interval = interval_dict[ti])
                graphdata = graphdata.set_index('date')
                mpf.plot(graphdata,type=graph_dict[graph],style='yahoo',mav=50,volume=True) 
                st.pyplot()
        
        # 5Y
        if b7.button('5Y'):
            if ticker != '-':         
                start_date_stock_5Y = today + relativedelta(years=-5)            
                graphdata = si.get_data(ticker, start_date = start_date_stock_5Y, end_date = today, index_as_date = False, interval = interval_dict[ti])
                graphdata = graphdata.set_index('date')
                mpf.plot(graphdata,type=graph_dict[graph],style='yahoo',mav=50,volume=True) 
                st.pyplot()

        # MAX
        if b8.button('MAX'):
            if ticker != '-':        
                graphdata = si.get_data(ticker, end_date = today, index_as_date = False, interval = interval_dict[ti])
                graphdata = graphdata.set_index('date')
                mpf.plot(graphdata,type=graph_dict[graph],style='yahoo',mav=50,volume=True) 
                st.pyplot()

        # Period Selected
        if b9.button('Period Selected'):
            if ticker != '-':
                try:
                    graphdata = si.get_data(ticker, start_date = start_date_stock, end_date = end_date_stock, index_as_date = False, interval = interval_dict[ti])
                    graphdata = graphdata.set_index('date')
                    mpf.plot(graphdata,type=graph_dict[graph],style='yahoo',mav=50,volume=True) 
                    st.pyplot()
                except KeyError:
                    st.write('Error between time interval and period selected, please choose again.')
                except AssertionError:
                    st.write('Error from the period selected, please choose again.')
                    
#==============================================================================
# Tab 3
#==============================================================================

class Statistics:
    def run(self):    
        title('Statistics')
        
        @st.cache
        def Getstatsvaluation(ticker):
            return si.get_stats_valuation(ticker)
        
        def Getstats(ticker):
            return si.get_stats(ticker)
            
        if ticker != '-':
            # Extract information - Valuation Measures
            info_val = Getstatsvaluation(ticker)
            info_val = info_val.rename(columns={0: 'Valuation Measures',1:'Value'})
            info_val = info_val.set_index('Valuation Measures')
            
            # Extract information - Financial Highlights & Trading Information
            info = Getstats(ticker)
            info = info.set_index('Attribute')
            
            
            fy = info.iloc[29:31,]            
            profitability = info.iloc[31:33,]            
            mgt = info.iloc[33:35,]            
            income_statement = info.iloc[35:43,]            
            bs = info.iloc[43:49,]
            cfs = info.iloc[49:,]
            
            stock_price_history = info.iloc[0:7,]
            share_statistics = info.iloc[7:19,]  
            dividends_splits = info.iloc[19:29,]            
            
            column1, column2 = st.columns([2, 2])            
            
            with column1:
                st.header('Valuation Measures')
                st.table(info_val)
    
                st.header('Financial Highlights')
                st.subheader('Fiscal Year')
                st.table(fy)
    
                st.subheader('Profitability')
                st.table(profitability)
    
                st.subheader('Management Effectiveness')
                st.table(mgt)
    
                st.subheader('Income Statement')
                st.table(income_statement)
    
                st.subheader('Balance Sheet')
                st.table(bs)            
    
                st.subheader('Cash Flow Statement')
                st.table(cfs)

            with column2:
                st.header('Trading Information')            
                st.subheader('Stock Price History')
                st.table(stock_price_history)
           
                st.subheader('Share Statistics')
                st.table(share_statistics)
                
                st.subheader('Dividends & Splits')
                st.table(dividends_splits)

#==============================================================================
# Tab 4
#==============================================================================

class Financials:
    def run(self):
        title('Financials')
        
        # Create period list for annual/quarterly
        period_list = ['Annual','Quarterly']
        # Create dictionary for the annual and quarterly period
        perioddict = {"Annual":True,"Quarterly":False}
        
        # Create statement list
        statement_list = ['Income Statement','Balance Sheet', 'Cash Flow Statement']
        
        # Create selection box for period & statement
        col1, col2 = st.columns(2)
        period = col1.selectbox("Choose a period", period_list)
        statement = col2.selectbox("Choose a statement", statement_list)
   
        # Get the information from the selected statement & period
        if statement == "Income Statement":
            @st.cache
            def GetIS(ticker):
                return si.get_income_statement(ticker, perioddict.get(period))
                
            if ticker != '-':
                info = GetIS(ticker)
                st.table(info)
                    
        elif statement == "Balance Sheet":        
            @st.cache
            def GetBS(ticker):
                return si.get_balance_sheet(ticker, perioddict.get(period))
                
            if ticker != '-':
                info = GetBS(ticker)
                st.table(info)
                
        elif statement == "Cash Flow Statement":      
            @st.cache
            def GetCF(ticker):
                return si.get_cash_flow(ticker, perioddict.get(period))
                
            if ticker != '-':
                info = GetCF(ticker)
                st.table(info)

#==============================================================================
# Tab 5
#==============================================================================

class Analysis:
    def run(self):
        title('Analysis')

        @st.cache
        def Getanalysis(ticker):
            return si.get_analysts_info(ticker)
        
        if ticker != '-':
            analysis = Getanalysis(ticker)

            Earnings_Estimate = analysis['Earnings Estimate'].set_index('Earnings Estimate')
            Revenue_Estimate = analysis['Revenue Estimate'].set_index('Revenue Estimate')
            Earnings_History = analysis['Earnings History'].set_index('Earnings History')
            EPS_Trend = analysis['EPS Trend'].set_index('EPS Trend')
            EPS_revisions = analysis['EPS Revisions'].set_index('EPS Revisions')
            Growth_Estimates = analysis['Growth Estimates'].set_index('Growth Estimates')
            
            st.subheader('Earnings Estimate')
            st.table(Earnings_Estimate)
            
            st.subheader('Revenue Estimate')
            st.table(Revenue_Estimate)
            
            st.subheader('Earnings History')
            st.table(Earnings_History)
            
            st.subheader('EPS Trend')
            st.table(EPS_Trend)
            
            st.subheader('EPS Revisions')
            st.table(EPS_revisions)
            
            st.subheader('Growth Estimates')
            st.table(Growth_Estimates)

#==============================================================================
# Tab 6
#==============================================================================

class Simulation:
    def run(self):
        title('Simulation')
        
        seed = st.number_input('Enter Your Random Number Seed', min_value=0, max_value=None, value=0, step=None, format=None, key=int)
        
        col1, col2 = st.columns(2)
        
        n = col1.selectbox(
            'Number of simulations',
            (200, 500, 1000))
        
        t = col2.selectbox(
            'Predict how many days from today?',
            (30,60,90))
        
        if ticker != '-':
            start_date_sim = date((today.year-1),today.month,today.day)
            SP_data = si.get_data(ticker, start_date = start_date_sim, end_date = today, index_as_date = False, interval = "1d")
            close_price = SP_data['close']
            current_price = SP_data.iloc[-1,4]
            
            daily_return = close_price.pct_change()
            daily_volatility = np.std(daily_return)
            
            # Setup the Monte Carlo simulation
            np.random.seed(seed)
            simulations = n
            time_horizon = t
            
            # Run the simulation
            simulation_df = pd.DataFrame()
            
            for i in range(simulations):
                
                # The list to store the next stock price
                next_price = []
                
                # Create the next stock price
                last_price = SP_data.iloc[-1,4]
                
                for j in range(time_horizon):
                    # Generate the random percentage change around the mean (0) and std (daily_volatility)
                    future_return = np.random.normal(0, daily_volatility)
            
                    # Generate the random future price
                    future_price = last_price * (1 + future_return)
            
                    # Save the price and go next
                    next_price.append(future_price)
                    last_price = future_price
                
                # Store the result of the simulation
                simulation_df[i] = next_price
                
            # Plot the simulation stock price in the future
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 10, forward=True)
            
            plt.plot(simulation_df)
            plt.title('Monte Carlo simulation for ' + ticker + ' stock price in next ' + str(t) + ' days with ' + str(n) + ' simulations')
            plt.xlabel('Day')
            plt.ylabel('Price')
            
            plt.axhline(y=current_price, color='red')
            plt.legend(['Current stock price is: ' + str(np.round(current_price, 2))])
            ax.get_legend().legendHandles[0].set_color('red')
            
            st.pyplot(fig)

            # Get the ending price of the last day of the prediction
            ending_price = simulation_df.iloc[-1:, :].values[0, ]
            # Price at 95% confidence interval
            future_price_95ci = np.percentile(ending_price, 5)
            
            # Value at Risk
            VaR = last_price - future_price_95ci
            st.write('Value at Risk (VaR) at 95% confidence interval is: USD' + str(np.round(VaR, 2)))

#==============================================================================
# Tab 7 Investment Return Comparison
#==============================================================================

class InvestmentReturn:
    def run(self):
        title('Investment Return Compared With S&P 500')
            
        start_date_stock_1Y = today + relativedelta(years=-1)

        if ticker != '-':          
            # Ticker Info
            graphdata = si.get_data(ticker, start_date = start_date_stock_1Y, end_date = today, index_as_date = False, interval = "1d")
            x1 = graphdata['date']
            y1 = graphdata['close']

            # S&P 500 Info
            ref_sp = si.get_data('^GSPC', start_date = start_date_stock_1Y, end_date = today, index_as_date = False, interval = "1d")
            x2 = ref_sp['date']
            times = ref_sp.iloc[0,4]/graphdata.iloc[0,4]
            y2 = ref_sp['close']/times
            
            # Price change           
            ticker_change = (graphdata.iloc[-1,4:5]-graphdata.iloc[0,4:5])/graphdata.iloc[0,4:5]*100
            ticker_change = round(ticker_change.values[0],2)
            ticker_change_text = str(ticker_change) + '%'
            
            ticker_now = round(graphdata.iloc[-1,4:5].values[0],2)
            
            # S&P 500 change
            SP_change = (ref_sp.iloc[-1,4:5]-ref_sp.iloc[0,4:5])/ref_sp.iloc[0,4:5]*100
            SP_change = round(SP_change.values[0],2)
            SP_change_text = str(SP_change) + '%'
            
            SP_now = round(ref_sp.iloc[-1,4:5].values[0],2)
            
            # Show metric
            col1, col2 = st.columns(2)
            col1.metric(label=ticker, value=ticker_now, delta=ticker_change_text)
            col2.metric(label="S&P500", value=SP_now, delta=SP_change_text)

            # Plot graph
            plt.plot(x1, y1, label = ticker, linestyle="-", color = 'orange')         
            plt.plot(x2, y2, label = "S&P500", linestyle="--", color='grey')
            plt.legend([ticker,'S&P500'],loc=2)
            
            st.pyplot()
            st.caption('In order to compare the trend for ' + ticker + ' with S&P500, the base value of S&P500 is reset to the beginning price of ' + ticker)
            st.write('If you invested ' + ticker + ' one year ago and hold until today, you would have ' + str(ticker_change) + '% return, while S&P500 have ' + str(SP_change) + '% return for the same period.')

#==============================================================================
# Sidebar
#==============================================================================

# Add page dictionary
PAGES = {
    "Summary": Summary(),
    "Chart": Chart(),
    "Statistics": Statistics(),
    "Financials": Financials(),
    "Analysis": Analysis(),
    "Monte Carlo Simulation": Simulation(),
    "Investment Return Compared With S&P 500": InvestmentReturn()
    ,
}

def main():
    
    # Add sidebar title
    st.sidebar.title('Choose a stock')
    
    # Get the list of stock tickers from S&P500
    ticker_list = ['-'] + si.tickers_sp500()
    
    # Add a selectbox to the sidebar:
    global ticker
    ticker = st.sidebar.selectbox("Choose a ticker", ticker_list)
    
    # Today
    global today
    today = datetime.today().date()

    global y
    y = today.year

    global m
    m = today.month

    global d
    d = today.day    
    
    # Add select begin-end date
    global start_date_stock, end_date_stock
    col1, col2 = st.sidebar.columns(2)
    start_date_stock = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date_stock = col2.date_input("End date", datetime.today().date())
    
    #Add sidebar title
    st.sidebar.title('Select a page')
    
    # Add selection for pages
    selection = st.sidebar.radio("", list(PAGES.keys()))
    page = PAGES[selection]

    with st.spinner(f'Loading {selection} ...'):
        page.run()
    
if __name__ == "__main__":
    main()

###############################################################################
# END
###############################################################################