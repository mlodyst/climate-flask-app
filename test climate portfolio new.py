# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:02:40 2024

@author: aniac
"""


import os

from openpyxl import load_workbook

from win32com.client import Dispatch

import openpyxl

import pandas as pd

# from pandas_datareader import data as pdr

import numpy as np

# import scipy.stats

from scipy.stats import norm

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import norm

import matplotlib.dates as mdates

from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from scipy.stats import rankdata

import pandas as pd

# from pandas_datareader import data as pdr

import numpy as np

import scipy.stats

from scipy.stats import norm

import matplotlib.pyplot as plt

 

import matplotlib.mlab as mlab

from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import norm

import matplotlib.dates as mdates

from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D

# from pca import pca

import plotly.express as px

from sklearn import datasets

#from plotly.offline import plot

#import plotly.graph_objs as go

import plotly.graph_objects as go


def data_import(path, excel_file_name, sheet_number):

    excel = pd.ExcelFile(path+excel_file_name)

    data = excel.parse(sheet_number)

    data = data.set_index(data.columns[0])

    return (data)
  

def returns(price, type = 'ln'):

    price = price.sort_index()

    if type == 'ln':

        ret = (np.log(price/price.shift(1))).dropna()

    elif type == 'perct_change':

        ret = ((price/price.shift(1))-1).dropna()

    return(ret)


path = r'C:\risk analysis'   #change this path

ex_file_n = r'\input 11182024 test portfolio.xlsx'

prices_daily = data_import(path, ex_file_n, 0)

#w = data_import(path, ex_file_n, 1)['w']

masterdata = data_import(path, ex_file_n, 1)
mean_returns = data_import(path, ex_file_n, 2)
new_deal_attributes = data_import(path, ex_file_n, 3)
new_deal_prices = data_import(path, ex_file_n, 4)
alpha = 0.95 #confidence level

data_type = 'm'

prices_daily_updated = pd.concat([prices_daily, new_deal_prices], axis=1)

new_deal_mean_returns = new_deal_attributes['mean_returns']
new_deal_mean_returns = pd.DataFrame(new_deal_mean_returns)

# Ensure the index of new_deal_mean_returns aligns with new_deal_attributes
new_deal_mean_returns.index = new_deal_attributes.index

# Combine the existing mean_returns with the new deal mean returns
mean_returns_updated = pd.concat([mean_returns, new_deal_mean_returns], axis=0)

new_deal_climate_score = new_deal_attributes['Climate_Score']
new_deal_climate_score = pd.DataFrame(new_deal_climate_score)

# Ensure the index of new_deal_mean_returns aligns with new_deal_attributes
new_deal_climate_score.index = new_deal_attributes.index

# Combine the existing mean_returns with the new deal mean returns
climate_score_updated = pd.concat([masterdata, new_deal_climate_score], axis=0)






if data_type != 'd':

        prices_resample = prices_daily_updated.resample(data_type).last()

else:

        prices_resample = prices_daily_updated

rreturns = returns(prices_resample)
corr_matrix = rreturns.corr()
cov_matrix = rreturns.cov()*12
# mean_returns = rreturns.mean()
mean_returns_updated = mean_returns_updated.squeeze(axis=1)
climate_score_updated = climate_score_updated.squeeze(axis=1)


stock = climate_score_updated.index.tolist() 

#Set the number of iterations to 10000 and define an array to hold the simulation results; initially set to all zeros
num_iterations = 200000
simulation_res = np.zeros((5+len(stock)-1,num_iterations))

for i in range(num_iterations):
#Select random weights and normalize to set the sum to 1
        weights = np.array(np.random.random(len(stock)))
        weights /= np.sum(weights)

#Calculate the return and standard deviation for every step
        portfolio_return = np.sum(mean_returns_updated * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights)))
        portfolio_climate = np.sum(climate_score_updated * weights)

#Store all the results in a defined array
        simulation_res[0,i] = portfolio_return
        simulation_res[1,i] = portfolio_std_dev
        simulation_res[2,i] = portfolio_climate

#Calculate Sharpe ratio and store it in the array
        simulation_res[3,i] = simulation_res[0,i] / simulation_res[1,i]

#Save the weights in the array
        for j in range(len(weights)):
                simulation_res[j+4,i] = weights[j]

sim_frame = pd.DataFrame(simulation_res.T,columns=['ret','stdev','climate_score','sharpe',stock[0],stock[1],stock[2],stock[3],stock[4],stock[5],stock[6],stock[7],stock[8],stock[9],stock[10]])

# Sort sim_frame by Sharpe ratio in descending order
sorted_sim_frame = sim_frame.sort_values(by='sharpe', ascending=False)

# Select the top 10 rows
top_results = sorted_sim_frame.head(10)


plt.scatter(sim_frame.climate_score,sim_frame.sharpe,c=sim_frame.sharpe,cmap='RdYlBu')
plt.ylabel('Sharpe')
plt.xlabel('Climate Score')
plt.title("ESG vs. Sharpe Ratio")
plt.colorbar(label="Sharpe Ratio")
plt.show()

plt.scatter(top_results.climate_score,top_results.sharpe,c=top_results.sharpe,cmap='RdYlBu')
plt.ylabel('Sharpe')
plt.xlabel('Climate Score')
plt.title("ESG vs. Sharpe Ratio")
plt.colorbar(label="Sharpe Ratio")
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(sim_frame.climate_score, sim_frame.sharpe, c=sim_frame.sharpe, cmap='RdYlBu', alpha=0.6, edgecolor='k')
plt.scatter(top_results.climate_score, top_results.sharpe, c='red', label='Top Portfolios', edgecolor='black', s=100)
for i, row in top_results.iterrows():
    plt.annotate(f"Portfolio {i+1}", (row['climate_score'], row['sharpe']), fontsize=9, ha='right')
plt.colorbar(label="Sharpe Ratio")
plt.ylabel('Sharpe Ratio')
plt.xlabel('Climate Score')
plt.title("ESG vs. Sharpe Ratio - Highlighting Top Portfolios")
plt.legend()
plt.show()


x = sim_frame.ret
y = sim_frame.climate_score
z = sim_frame.sharpe

fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=2,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)])

# tight layout

fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis_title='Return',
        yaxis_title='Climate Score',
        zaxis_title='Sharpe Ratio'
    )
)

fig.write_html('first_figure.html', auto_open=True)

