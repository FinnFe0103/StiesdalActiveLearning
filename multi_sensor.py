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
                         'GP': {'learning_rate': [0.01], 'kernel': ['Matern'], 'lengthscale_prior': [None], 'lengthscale_sigma': [0.2], 'lengthscale_mean': [2.0], 'noise_prior': [None], 'noise_sigma': [0.1], 'noise_mean': [1.1], 'noise_constraint': [1e-3], 'lengthscale_type': ['Single'], 'acquisition_function': ['US'], 'reg_lambda': [0.05]},}
                         #'BNN': {'learning_rate': [0.01], 'hidden_size': [4], 'layer_number': [3], 'prior_sigma': [0.0000001], 'complexity_weight': [0.001], 'acquisition_function': ['US'], 'reg_lambda': [0.05]},}
                         #'DE': {'learning_rate': [0.01], 'hidden_size': [4], 'layer_number': [3], 'ensemble_size': [2], 'acquisition_function': ['US'], 'reg_lambda': [0.05]},}
                         #'MCD': {'learning_rate': [0.01], 'hidden_size': [20], 'layer_number': [5], 'dropout_rate': [0.5], 'acquisition_function': ['US'], 'reg_lambda': [0.05]}}


directory = 'runs' + '_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Directory to save the results
plot = True # Whether to plot the results in the last step
steps = 1 # Number of steps to run the active learning algorithm for
epochs = 100 # Number of epochs to train the model for
number_of_combinations = 50 # Number of random combinations to generate for each model (random search), set to really high value for grid search

sensors = ['49', '52']#, '59', '60', '164', '1477', '1493', '1509', '1525', '1541', '1563', '2348']

def generate_random_combinations(parameter_space, num_combinations):
    # Generate all possible combinations
    all_combinations = [dict(zip(parameter_space, v)) for v in itertools.product(*parameter_space.values())]
    # Select a random subset
    if len(all_combinations) > num_combinations:
        return random.sample(all_combinations, num_combinations)
    return all_combinations

for model_name, params_space in hyperparameter_spaces.items():
    random_combinations = generate_random_combinations(params_space, number_of_combinations) # Generate random combinations for the current model
        
    for combination in random_combinations:
        
        data = Dataprep(sensors=sensors, scaling='Standard', initial_samplesize=36)
        print('Data pool:', data.X_pool.shape, data.Y_pool.shape)
        print('Data known:', data.X_selected.shape, data.Y_selected.shape)

        model_instances = []
        for index, sensor in enumerate(sensors):
            run_name = f"{index}_{sensor}_{model_name}" #_{'_'.join([''.join([part[0] for part in k.split('_')]) + '_' + str(v) for k, v in combination.items()])}"
            model = RunModel(model_name=model_name, run_name=run_name, directory=directory, x_total=data.X, y_total=data.Y[:, index], steps=steps, epochs=epochs, **combination)
            model_instances.append(model) # Store the model instance
        
        # total_selected_indices = []
        # for index, model in enumerate(model_instances):
        #     step = 0
        #     print('----------')
        #     print(model)
        #     print('Run Name:', model.run_name)
        #     print('Model:', model.model)
        #     print('Device', model.device)

        #     topk = int(36/len(model_instances))

        #     # Train the model
        #     train_model_time = time.time()
        #     model.train_model(step=step, X_selected=data.X_selected, y_selected=data.Y_selected[:, index])
        #     tqdm.write(f'---Training time: {time.time() - train_model_time:.2f} seconds')

        #     # Get the final predictions as if this was the last step
        #     final_prediction_time = time.time()
        #     x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, mse, mae, percentage_common, highest_actual_in_top, highest_actual_in_known = model.final_prediction(step=step, X_total=data.X, y_total=data.Y[:, index], X_selected=data.X_selected, topk=topk)
        #     tqdm.write(f'---Final prediction time: {time.time() - final_prediction_time:.2f} seconds')

        #     # Evaluate the model on the pool data
        #     evaluate_pool_data_time = time.time()
        #     model.evaluate_pool_data(step=step, X_pool=data.X_pool, y_pool=data.Y_pool[:, index]) 
        #     tqdm.write(f'---Evaluation on pool data time: {time.time() - evaluate_pool_data_time:.2f} seconds')

        #     # Predict the uncertainty on the pool data
        #     predict_time = time.time()
        #     means, stds = model.predict(X_pool=data.X_pool)
        #     tqdm.write(f'---Prediction time: {time.time() - predict_time:.2f} seconds')

        #     # Select the next samples from the pool
        #     acquisition_function_time = time.time()
        #     selected_indices = model.acquisition_function(means, stds, y_selected=data.Y_selected[:, index], X_Pool=data.X_pool, topk=topk, selected_indices=total_selected_indices)
        #     tqdm.write(f'---Acquisition function time: {time.time() - acquisition_function_time:.2f} seconds')
        #     print('length of selected indices:', selected_indices)
        #     print('training data (y):', data.Y_selected[:, index])

        #     # Get the KL divergence
        #     # get_kl_divergence_time = time.time()
        #     # kl_divergence_for_plot, x_points, p, q, y_min_extended, y_max_extended, kl_divergence = model.get_kl_divergence()
        #     # tqdm.write(f'---KL divergence time: {time.time() - get_kl_divergence_time:.2f} seconds')

        #     # Plot the predictions and selected indices
        #     if plot and step+1 == model.steps:
        #         plot_time = time.time()
        #         model.plot(means=means, stds=stds, selected_indices=selected_indices[-topk:], step=step, x_highest_pred_n=x_highest_pred_n, y_highest_pred_n=y_highest_pred_n, x_highest_actual_n=x_highest_actual_n, y_highest_actual_n=y_highest_actual_n, x_highest_actual_1=x_highest_actual_1, y_highest_actual_1=y_highest_actual_1, X_pool=data.X_pool, y_pool=data.Y_pool[:, index], X_selected=data.X_selected, y_selected=data.Y_selected[:, index])
        #         tqdm.write(f'---Plotting time: {time.time() - plot_time:.2f} seconds')

        print(data.initial_sample)
        selected_indices = [1352, 1412, 1411, 921, 922, 951, 70, 69, 57, 1465, 1614, 1599, 182, 183, 7, 1778, 1601, 1779, 1470, 1471, 1501, 923, 994, 993, 153, 152, 154, 1733, 1748, 71, 1808, 1600, 1777, 1602, 1809, 1571]
        print(len(selected_indices))
        # Update the known and pool data
        data.update_data(selected_indices)
        print(data.X_selected.shape, data.Y_selected.shape, data.X_pool.shape, data.Y_pool.shape)
        print(data.initial_sample)
        print('DATA UPDATED')
        

        #print(total_selected_indices)
        #print(len(set(total_selected_indices)))








# #### GENERATE DIFFERENT COMBINATIONS PER MODEL ####
# with tqdm(total=total_models, desc="Overall progress of models", position=0) as pbar_overall:
#     for model_name, params_space in hyperparameter_spaces.items():
#         random_combinations = generate_random_combinations(params_space, number_of_combinations) # Generate random combinations for the current model
#         model_performance_data = {}  # Store results of all runs for the current model; resets with every model
#         model_instances[model_name] = [] # List to store instances for the current model

#         #### GO THROUGH DIFFERENT COMBINATIONS PER MODEL ####
#         with tqdm(total=len(random_combinations), desc=f"Combinations of {model_name}", leave=False, position=1) as pbar_combinations:
#             for combination in random_combinations:

#                 with tqdm(total=len(sensors), desc="Sensors", leave=False, position=2) as pbar_sensors:
#                     for sensor in sensors:
#                         if sensor not in model_performance_data:
#                             model_performance_data[sensor] = []
#                         # try: # Catch errors with some combinations and save the raised exception
                            
#                         data = Dataprep(sensor=sensor, scaling='Standard', initial_samplesize=36)
#                         print('Data known:', data.data_known.shape)
#                         print('Data pool:', data.data_pool.shape)
#                         run_name = f"{model_name}_{'_'.join([''.join([part[0] for part in k.split('_')]) + '_' + str(v) for k, v in combination.items()])}"
#                         model = RunModel(model_name=model_name, run_name=run_name, directory=directory, data=data, steps=steps, epochs=epochs, **combination)
#                         model_instances[model_name].append(model) # Store the model instance

#                         print(model_instances[model_name])



                        #     aggregated_results = {'Model': run_name}

                        #     #### ACTIVE LEARNING ALGORITHM FOR SPECIFIC COMBINATION ####
                        #     for step in range(model.steps):
                        #         start_step = time.time()

                        #         train_model_time = time.time()
                        #         model.train_model(step) # Train the model
                        #         tqdm.write(f'---Training time: {time.time() - train_model_time:.2f} seconds')

                        #         final_prediction_time = time.time()
                        #         x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, mse, mae, percentage_common, highest_actual_in_top, highest_actual_in_known = model.final_prediction(step) # Get the final predictions as if this was the last step
                        #         tqdm.write(f'---Final prediction time: {time.time() - final_prediction_time:.2f} seconds')

                        #         evaluate_pool_data_time = time.time()
                        #         model.evaluate_pool_data(step) # Evaluate the model on the pool data
                        #         tqdm.write(f'---Evaluation on pool data time: {time.time() - evaluate_pool_data_time:.2f} seconds')

                        #         predict_time = time.time()
                        #         means, stds = model.predict() # Predict the uncertainty on the pool data
                        #         tqdm.write(f'---Prediction time: {time.time() - predict_time:.2f} seconds')

                        #         acquisition_function_time = time.time()
                        #         selected_indices = model.acquisition_function(means, stds) # Select the next samples from the pool
                        #         tqdm.write(f'---Acquisition function time: {time.time() - acquisition_function_time:.2f} seconds')

                        #         get_kl_divergence_time = time.time()
                        #         kl_divergence_for_plot, x_points, p, q, y_min_extended, y_max_extended, kl_divergence = model.get_kl_divergence()
                        #         tqdm.write(f'---KL divergence time: {time.time() - get_kl_divergence_time:.2f} seconds')

                        #         if plot and step+1 == model.steps:
                        #             tqdm.write('Plotting...')
                        #             plot_time = time.time()
                        #             model.plot(means, stds, selected_indices, step, x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, kl_divergence_for_plot, x_points, p, q, y_min_extended, y_max_extended) # Plot the predictions and selected indices
                        #             tqdm.write(f'---Plotting time: {time.time() - plot_time:.2f} seconds')

                        #         update_time = time.time()
                        #         model.update_data(selected_indices) # Update the known and pool data
                        #         tqdm.write(f'---Update data (AL={model.active_learning} - {model.data_pool.shape}, {model.data_known.shape}) time: {time.time() - update_time:.2f} seconds')

                        #         aggregated_results[f'MSE_{step+1}'] = mse
                        #         aggregated_results[f'MAE_{step+1}'] = mae
                        #         aggregated_results[f'Selected Indices_{step+1}'] = selected_indices
                        #         aggregated_results[f'Percentage Common_{step+1}'] = percentage_common
                        #         aggregated_results[f'Highest Actual Value in Top Predictions_{step+1}'] = highest_actual_in_top
                        #         aggregated_results[f'Highest Actual Value in Known Data_{step+1}'] = highest_actual_in_known
                        #         aggregated_results[f'KL-Divergence_{step+1}'] = kl_divergence

                        #         tqdm.write(f'Step: {step+1} of {model.steps} | {time.time() - start_step:.2f} seconds')
                        #         tqdm.write('--------------------------------')

                        #     model_performance_data[sensor_name].append(aggregated_results)

                        # except Exception as e:
                        #     tqdm.write(f"Error with combination {combination}: {e}")
                        #     # Add the failed combination and the error to the DataFrame
                        #     failed_combinations = pd.concat([failed_combinations, pd.DataFrame({'Model': model_name, 'Combination': combination, 'Error': str(e)})], ignore_index=True)
        #                 pbar_sensors.update(1)
        #         pbar_combinations.update(1)
        # pbar_overall.update(1)
    
#         # Save model performance data to an Excel workbook, one worksheet per sensor
#         workbook_path = f'{directory}/{model_name}_performance_{datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")}.xlsx'
#         with pd.ExcelWriter(workbook_path, engine='openpyxl') as writer:
#             for sensor_name, data in model_performance_data.items():
#                 df = pd.DataFrame(data)
#                 worksheet_name = sensor_name[:31]  # Excel worksheet names are limited to 31 characters
#                 df.to_excel(writer, sheet_name=worksheet_name, index=False)

#         pbar_overall.update(1)
#         print(f"{model_name} model performance data saved.")

# failed_combinations.to_excel('runs/failed_combinations.xlsx', index=False)

# exclude_hyperparameters = ['verbose', 'writer', 'data', 'data_known', 'data_pool', 'mse', 'mae', 'device', 'y_total_actual', 'pdf_p', 
#                            'pdf_q', 'likelihood', 'mll', 'validation_size', 'y_total_predicted'] # Paramters not included in the output
# tested_hyperparameters = []
# for model_name, instances in model_instances.items(): # Iterate through each model and its instances to collect hyperparameters
#     for instance in instances:
#         hyperparams = {key: value for key, value in vars(instance).items() if key not in exclude_hyperparameters}
#         hyperparams['model'] = model_name  # Add a column for the model name
#         tested_hyperparameters.append(hyperparams)
# df_hyperparameters = pd.DataFrame(tested_hyperparameters)
# df_hyperparameters.to_excel('runs/tested_hyperparameters.xlsx', index=False)