import itertools
import random
import time
import datetime
from tqdm import tqdm
import pandas as pd
import openpyxl
from multi_main import RunModel
from multi_data import Dataprep, update_data

hyperparameter_spaces = {#'SVR': {'acquisition_function': ['EX']},}
                         'GP': {'learning_rate': [0.01], 'kernel': ['Matern'], 'lengthscale_prior': [None], 'lengthscale_sigma': [0.2], 'lengthscale_mean': [2.0], 'noise_prior': [None], 'noise_sigma': [0.1], 'noise_mean': [1.1], 'noise_constraint': [1e-3], 'lengthscale_type': ['Single'], 'acquisition_function': ['US'], 'reg_lambda': [0.5]},}
                         #'BNN': {'learning_rate': [0.01], 'hidden_size': [4], 'layer_number': [3], 'prior_sigma': [0.0000001], 'complexity_weight': [0.001], 'acquisition_function': ['US'], 'reg_lambda': [0.05]},}
                         #'DE': {'learning_rate': [0.01], 'hidden_size': [4], 'layer_number': [3], 'ensemble_size': [2], 'acquisition_function': ['US'], 'reg_lambda': [0.05]},}
                         #'MCD': {'learning_rate': [0.01], 'hidden_size': [20], 'layer_number': [5], 'dropout_rate': [0.5], 'acquisition_function': ['US'], 'reg_lambda': [0.05]}}


directory = 'runs' + '_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Directory to save the results
plot = False # Whether to plot the results in the last step
steps = 2 # Number of steps to run the active learning algorithm for
epochs = 100 # Number of epochs to train the model for
num_combinations = 50 # Number of random combinations to generate for each model (random search), set to really high value for grid search

sensors = ['49', '52']#, '59', '60', '164', '1477', '1493', '1509', '1525', '1541', '1563', '2348']

model_performance_list = {}
for model_name, params_space in hyperparameter_spaces.items():
    all_combinations = [dict(zip(params_space, v)) for v in itertools.product(*params_space.values())] # Generate all possible combinations
    if len(all_combinations) > num_combinations:
        all_combinations = random.sample(all_combinations, num_combinations) # Select a random subset
    
    for combination in all_combinations:
        
        data = Dataprep(sensors=sensors, scaling='Standard', initial_samplesize=36)
        print('Data pool:', data.X_pool.shape, data.Y_pool.shape)
        print('Data known:', data.X_selected.shape, data.Y_selected.shape)
        combination_id = f"{model_name}_{'_'.join([''.join([part[0] for part in k.split('_')]) + '_' + str(v) for k, v in combination.items()])}"

        model_performance_list[combination_id] = {'Model': '', 'Sensor': ''}
        model_instances = []
        for index, sensor in enumerate(sensors):
            run_name = f"{sensor}_{combination_id}"
            model = RunModel(model_name=model_name, run_name=run_name, directory=directory, x_total=data.X, y_total=data.Y[:, index], steps=steps, epochs=epochs, **combination)
            model_instances.append(model)
            
            model_performance_list[combination_id]['Model'].append(combination_id)
            model_performance_list[combination_id]['Sensor'].append(sensor)
            print(model_performance_list)

        selected_indices = []
        for step in range(steps):
            print('Step:', step)
            print(selected_indices)
            print(len(set(selected_indices)))

            data.update_data(selected_indices) # Update the known and pool data

            selected_indices = []
            for index, model in enumerate(model_instances):
                topk = int(36/len(model_instances)) # Number of samples to select from the pool data per sensor/model

                # Train the model
                train_model_time = time.time()
                model.train_model(step=step, X_selected=data.X_selected, y_selected=data.Y_selected[:, index])
                tqdm.write(f'---Training time: {time.time() - train_model_time:.2f} seconds')

                # Get the final predictions as if this was the last step
                final_prediction_time = time.time()
                x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, mse, mae, percentage_common, highest_actual_in_top, highest_actual_in_known = model.final_prediction(step=step, X_total=data.X, y_total=data.Y[:, index], X_selected=data.X_selected, topk=topk)
                tqdm.write(f'---Final prediction time: {time.time() - final_prediction_time:.2f} seconds')

                # Evaluate the model on the pool data
                evaluate_pool_data_time = time.time()
                model.evaluate_pool_data(step=step, X_pool=data.X_pool, y_pool=data.Y_pool[:, index]) 
                tqdm.write(f'---Evaluation on pool data time: {time.time() - evaluate_pool_data_time:.2f} seconds')

                # Predict the uncertainty on the pool data
                predict_time = time.time()
                means, stds = model.predict(X_pool=data.X_pool)
                tqdm.write(f'---Prediction time: {time.time() - predict_time:.2f} seconds')

                # Select the next samples from the pool
                acquisition_function_time = time.time()
                selected_indices = model.acquisition_function(means, stds, y_selected=data.Y_selected[:, index], X_Pool=data.X_pool, topk=topk, selected_indices=selected_indices)
                tqdm.write(f'---Acquisition function time: {time.time() - acquisition_function_time:.2f} seconds')
                print('length of selected indices:', selected_indices)
                print('training data (y):', data.Y_selected.shape, data.Y_selected[:5, index])

                # Get the KL divergence
                # get_kl_divergence_time = time.time()
                # kl_divergence_for_plot, x_points, p, q, y_min_extended, y_max_extended, kl_divergence = model.get_kl_divergence()
                # tqdm.write(f'---KL divergence time: {time.time() - get_kl_divergence_time:.2f} seconds')

                # Plot the predictions and selected indices
                print(step+1, model.steps)
                if plot and step+1 == model.steps:
                    plot_time = time.time()
                    model.plot(means=means, stds=stds, selected_indices=selected_indices[-topk:], step=step, x_highest_pred_n=x_highest_pred_n, y_highest_pred_n=y_highest_pred_n, x_highest_actual_n=x_highest_actual_n, y_highest_actual_n=y_highest_actual_n, x_highest_actual_1=x_highest_actual_1, y_highest_actual_1=y_highest_actual_1, X_pool=data.X_pool, y_pool=data.Y_pool[:, index], X_selected=data.X_selected, y_selected=data.Y_selected[:, index])
                    tqdm.write(f'---Plotting time: {time.time() - plot_time:.2f} seconds')

                model_performance_list[combination_id][f'MSE_{step+1}'] = mse
                model_performance_list[combination_id][f'MAE_{step+1}'] = mae
                model_performance_list[combination_id][f'Selected Indices_{step+1}'] = selected_indices
                model_performance_list[combination_id][f'Percentage Common_{step+1}'] = percentage_common
                model_performance_list[combination_id][f'Highest Actual Value in Top Predictions_{step+1}'] = highest_actual_in_top
                model_performance_list[combination_id][f'Highest Actual Value in Known Data_{step+1}'] = highest_actual_in_known
                # aggregated_results[f'KL-Divergence_{step+1}'] = kl_divergence

                #model_performance_list.append(aggregated_results)

                print(selected_indices)
                print(len(set(selected_indices)))
                print(model_performance_list[combination_id])
            print('Model performance list:', model_performance_list)
model_performance_df = pd.DataFrame(model_performance_list)
model_performance_df.to_excel(f'{directory}/{model_name}_performance_{datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")}.xlsx', index=False)
