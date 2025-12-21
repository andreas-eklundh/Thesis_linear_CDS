import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#### Preliminary investigation. 
data = pd.read_excel("./Data/FullCDSdata19-23.xlsx")
# Function to convert tenors to months to same metric (sortable)

data = data.rename(columns = {'ConvSpreard':'ConvSpread'})

# # Custom conversion function (to months)
# def tenor_to_months(tenor):
#     if tenor.endswith("M"):
#         return int(tenor[:-1])
#     elif tenor.endswith("Y"):
#         return int(tenor[:-1]) * 12
#     else:
#         return float("inf")  # put unknown formats at end

# # Apply the sort
# data["tenor_months"] = data["Tenor"].apply(tenor_to_months)
# data = data.sort_values(by = ['Date','Ticker','tenor_months'])
# # CDS trade differently within the day, converted spread. Just keep
# # The average/aggregate rows in this manner. 
# data_mod = data.groupby(['Date', 'Ticker', 'Tenor']).mean('ConvSpread').reset_index()
# data_mod = data_mod[data_mod['Tenor']!= '0M']
# data_mod['Tenor_plot'] = data_mod["tenor_months"].apply(lambda x: str(x))+str("M = ")+data_mod['Tenor'] 

# Only interested in Tenors in years. 
data_mod = data.groupby(['Date', 'Ticker', 'Tenor']).mean('Par Spread').reset_index()
data_mod = data_mod[data_mod['Tenor'].str.contains('Y')]

tickers = data_mod['Ticker'].unique()

data_mod['Tenor_int'] = data_mod['Tenor'].str.strip('Y').apply(int)

# Loop through each ticker
for ticker in tickers:
    df_ticker = data_mod[data_mod['Ticker'] == ticker]
    # Pivot to make each tenor a column
    df_ticker = df_ticker.sort_values(by = ['Tenor_int'])
    pivot = df_ticker.pivot(index='Date', columns='Tenor_int', values='Par Spread')
    pivot = pivot[[1,2,3,4,5,7,10]]
    # Plot
    pivot.plot(figsize=(12, 6))
    plt.xlabel("Date")
    plt.ylabel("Par Spread")
    plt.legend(loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=7,                       
                fontsize=12,                  
                frameon=True)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    plt.savefig(f"./Spreads_obs/{ticker}_parspread.png")
    plt.close()




## Test of functionality of one reasonably looking CDS.

# TODO: CONSIDER MOVING TO ATTRIBUTE.
# Test on Danske bank
# test_df = data_mod[data_mod['Ticker'] == 'DANBNK']
#test_df = data_mod[(data_mod['Ticker'] == 'DANBNK') & (data_mod['Date'] <= pd.to_datetime('2021-01-01'))]
test_df = data_mod[(data_mod['Ticker'] == 'DANBNK') ]

test_df.to_excel("./Data/test_data.xlsx", index=False) 

## subset more data
# lis tto keep
firms = ['CMZB','DANBNK','MONTE', 'SVSKHB']
sub_df = data_mod[(data_mod['Ticker'].isin(firms)) ]
sub_df.to_excel("./Data/subset_data.xlsx", index=False) 

# Reshape and make into array. 
