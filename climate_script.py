from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the Flask app
app = Flask(__name__)

input_file_path = 'input 11182024 test portfolio.xlsx'

# Utility functions from your script
def data_import(path, excel_file_name, sheet_number):
    full_path = os.path.join(path, excel_file_name)
    excel = pd.ExcelFile(full_path)
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

# Main endpoint to execute the script
@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        # Define path to the file
        path = os.getcwd()  # Get the current working directory (root of the project)
        ex_file_n = input_file_path  # Path to the Excel file in the 'data/' directory
               
        # Import data
        prices_daily = data_import(path,ex_file_n, 0)
        masterdata = data_import(path, ex_file_n, 1)
        mean_returns = data_import(path, ex_file_n, 2)
        new_deal_attributes = data_import(path, ex_file_n, 3)
        new_deal_prices = data_import(path, ex_file_n, 4)

        # Data processing
        prices_daily_updated = pd.concat([prices_daily, new_deal_prices], axis=1)
        new_deal_mean_returns = new_deal_attributes['mean_returns']
        new_deal_mean_returns = pd.DataFrame(new_deal_mean_returns)
        new_deal_mean_returns.index = new_deal_attributes.index
        mean_returns_updated = pd.concat([mean_returns, new_deal_mean_returns], axis=0)
        new_deal_climate_score = new_deal_attributes['Climate_Score']
        new_deal_climate_score = pd.DataFrame(new_deal_climate_score)
        new_deal_climate_score.index = new_deal_attributes.index
        climate_score_updated = pd.concat([masterdata, new_deal_climate_score], axis=0)

        # Resample and calculate returns
        prices_resample = prices_daily_updated.resample('M').last()
        rreturns = returns(prices_resample)
        corr_matrix = rreturns.corr()
        cov_matrix = rreturns.cov() * 12
        mean_returns_updated = mean_returns_updated.squeeze(axis=1)
        climate_score_updated = climate_score_updated.squeeze(axis=1)
        stock = climate_score_updated.index.tolist()

        # Monte Carlo simulation
        num_iterations = 50000
        simulation_res = np.zeros((5 + len(stock) - 1, num_iterations))
        for i in range(num_iterations):
            weights = np.array(np.random.random(len(stock)))
            weights /= np.sum(weights)
            portfolio_return = np.sum(mean_returns_updated * weights)
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            portfolio_climate = np.sum(climate_score_updated * weights)
            simulation_res[0, i] = portfolio_return
            simulation_res[1, i] = portfolio_std_dev
            simulation_res[2, i] = portfolio_climate
            simulation_res[3, i] = simulation_res[0, i] / simulation_res[1, i]
            for j in range(len(weights)):
                simulation_res[j + 4, i] = weights[j]

        # Create DataFrame for results
        sim_frame = pd.DataFrame(simulation_res.T, columns=['ret', 'stdev', 'climate_score', 'sharpe'] + stock)
        sorted_sim_frame = sim_frame.sort_values(by='sharpe', ascending=False)
        top_results = sorted_sim_frame.head(5)

        # Return top results as JSON
        return jsonify({
            "message": "Script executed successfully",
            "top_results": top_results.to_dict(orient='records'),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
