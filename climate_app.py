import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Initialize Flask App
app = Flask(__name__)

# Utility functions (same as the code you provided)
def data_import(path, excel_file_name, sheet_number):
    excel = pd.ExcelFile(path + excel_file_name)
    data = excel.parse(sheet_number)
    data = data.set_index(data.columns[0])
    return data

def returns(price, type='ln'):
    price = price.sort_index()
    if type == 'ln':
        ret = (np.log(price / price.shift(1))).dropna()
    elif type == 'perct_change':
        ret = ((price / price.shift(1)) - 1).dropna()
    return ret

# API endpoint to handle the request from Glideapp
@app.route('/process_deal', methods=['POST'])
def process_deal():
    # Get the data from the POST request
    request_data = request.get_json()

    new_deal_name = request_data.get('deal_name')
    max_size = request_data.get('max_size')

    # Path and Excel file (adjust this path if needed)
    path = r'C:\risk analysis'  
    ex_file_n = r'\input 11182024 test portfolio.xlsx'
    
    # Data import (for simplicity, just using some of the data for this demo)
    prices_daily = data_import(path, ex_file_n, 0)
    mean_returns = data_import(path, ex_file_n, 2)
    new_deal_prices = data_import(path, ex_file_n, 4)
    
    # Data processing (using the provided code)
    prices_daily_updated = pd.concat([prices_daily, new_deal_prices], axis=1)
    new_deal_mean_returns = pd.DataFrame(new_deal_prices.mean(axis=1))  # Sample calculation for new deal
    mean_returns_updated = pd.concat([mean_returns, new_deal_mean_returns], axis=0)
    
    rreturns = returns(prices_daily_updated)
    corr_matrix = rreturns.corr()
    cov_matrix = rreturns.cov() * 12
    
    simulation_res = np.zeros((5 + len(prices_daily.columns) - 1, 200000))
    
    # Simulate portfolio
    for i in range(200000):
        weights = np.random.random(len(prices_daily.columns))
        weights /= np.sum(weights)
        
        portfolio_return = np.sum(mean_returns_updated * weights)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        simulation_res[0, i] = portfolio_return
        simulation_res[1, i] = portfolio_std_dev
        simulation_res[3, i] = portfolio_return / portfolio_std_dev
        
        for j in range(len(weights)):
            simulation_res[j + 4, i] = weights[j]
    
    # Create result dataframe
    sim_frame = pd.DataFrame(simulation_res.T, columns=['ret', 'stdev', 'sharpe'] + list(prices_daily.columns))
    sorted_sim_frame = sim_frame.sort_values(by='sharpe', ascending=False)
    top_results = sorted_sim_frame.head(10)
    
    # Plotting (we'll encode the images to base64 to send them as part of the response)
    def plot_to_base64():
        fig, ax = plt.subplots()
        ax.scatter(sim_frame['stdev'], sim_frame['ret'], c=sim_frame['sharpe'], cmap='RdYlBu')
        ax.set_xlabel('Standard Deviation')
        ax.set_ylabel('Return')
        ax.set_title("Portfolio Simulation")
        
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
    # Create the response data
    response_data = {
        "top_results": top_results.to_dict(),
        "plot": plot_to_base64()
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    # Expose the app to the internet (make sure it's hosted on a public server)
    app.run(host='0.0.0.0', port=5000)
