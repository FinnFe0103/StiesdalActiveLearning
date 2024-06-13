import itertools
import random
import time
import datetime
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from multi_main import RunModel
from multi_data import Dataprep
from tqdm import tqdm

#########################
#### CODE PARAMETERS ####
#########################
directory = 'runs' + '_' + datetime.datetime.now().strftime("%m-%d %H:%M") # Directory to save the results
plot = True # Whether to plot the results in the last step
steps = 20 # Number of steps to run the active learning algorithm for
epochs = 100 # Number of epochs to train the model for
num_combinations = 4000  # Number of random combinations to generate for each model (random search), set to really high value for grid search
samples_per_step = 36

# set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Models and hyperparameters
hyperparameter_spaces = {'GP': {'learning_rate': [0.1], 'kernel': ['Matern'], 'lengthscale_prior': [None], 'lengthscale_sigma': [0.01], 'lengthscale_mean': [1.0], 'noise_prior': [None], 'noise_sigma': [0.1], 'noise_mean': [1.0], 'noise_constraint': [1e-6], 'lengthscale_type': ['ARD'], 'acquisition_function': ['UCB'], 'reg_lambda': [0.001]}, }

# Selected sensors
sensors = ['49', '52', '59', '60', '164', '1477', '1493', '1509', '1525', '1541', '1563', '2348']

# create all combinations of hyperparameters
all_combinations = []
total_models = len(hyperparameter_spaces)
for model_name, params_space in hyperparameter_spaces.items():
    model_combinations = [dict(zip(params_space, v)) for v in itertools.product(*params_space.values())]
    if len(model_combinations) > num_combinations:
        model_combinations = random.sample(model_combinations, num_combinations)
    all_combinations.extend((model_name, combo) for combo in model_combinations)

# Check: Print the total number of models and combinations
print(f"Total models: {total_models}")
print(f"Total combinations: {len(all_combinations)}")
print(all_combinations)

# Function to process each combination
def process_combination(model_name, combination, sensors, steps, epochs, directory, plot):       
    '''
    Function to process each combination of hyperparameters for a model
    
    Parameters:
    model_name: str
        Name of the model
    combination: dict
        Dictionary of hyperparameters for the model
    sensors: list
        List of sensors to use
    steps: int
        Number of steps to run the active learning algorithm for
    epochs: int
        Number of epochs to train the model for
    directory: str
        Directory to save the results
    plot: bool
        Whether to plot the results in the last step
    
    Returns:
    result_list: list
        List of dictionaries containing the results for each step'''

    # Initialize the data
    data = Dataprep(sensors=sensors, scaling='Minmax', initial_samplesize=samples_per_step)
    # Create the unique id for the combination
    combination_id = f"{model_name}_{'_'.join([''.join([part[0] for part in k.split('_')]) + '_' + str(v) for k, v in combination.items()])}"

    # Initialize the models for each sensor
    model_instances = []
    for index, sensor in enumerate(sensors):
        run_name = f"{sensor}_{combination_id}"
        model = RunModel(model_name=model_name, run_name=run_name, directory=directory, steps=steps, epochs=epochs, **combination)
        model_instances.append(model)

    # Initialize the result list and selected indices
    result_list = []
    selected_indices = []
    with tqdm(total=steps, desc="Steps (all sensors)", leave=False, position=2) as pbar_steps:
        
        # Loop through each step
        for step in range(steps):
            # try except block to catch any errors and continue with the next step
            try:
                # Update the known and pool data
                data.update_data(selected_indices)

                # Results per sensor per step
                step_data = []
                # Reset selected indices for each step and after updating the data
                selected_indices = [] 
                for index, model in enumerate(model_instances):
                    
                    tqdm.write(f'------------ Step {step+1} for sensor {model.run_name.split("_")[0]} ------------')

                    # Number of samples to select from the pool data per sensor/model
                    topk = int(samples_per_step/len(model_instances))

                    # Train the model
                    train_model_time = time.time()
                    model.train_model(step=step, X_selected=data.X_selected, y_selected=data.Y_selected[:, index])
                    tqdm.write(f'---Training time: {time.time() - train_model_time:.2f} seconds')

                    # Get the final predictions as if this was the last step
                    final_prediction_time = time.time()
                    x_highest_pred_n, y_highest_pred_n, x_highest_actual_n, y_highest_actual_n, x_highest_actual_1, y_highest_actual_1, mse, mae, percentage_common, index_of_actual_1_in_pred, seen_count, highest_actual_in_top, highest_indices_pred, highest_indices_actual_1 = model.final_prediction(step=step, X_total=data.X, y_total=data.Y[:, index], X_selected=data.X_selected, topk=topk)
                    tqdm.write(f'---Final prediction time: {time.time() - final_prediction_time:.2f} seconds')

                    # Predict the uncertainty on the pool data
                    predict_time = time.time()
                    means, stds = model.predict(X_pool=data.X_pool)
                    tqdm.write(f'---Prediction time: {time.time() - predict_time:.2f} seconds')

                    # Select the next samples from the pool
                    acquisition_function_time = time.time()
                    selected_indices = model.acquisition_function(means, stds, y_selected=data.Y_selected[:, index], X_Pool=data.X_pool, topk=topk, selected_indices=selected_indices)
                    tqdm.write(f'---Acquisition function time: {time.time() - acquisition_function_time:.2f} seconds')

                    # Append the results to the step_data list
                    step_data.append({
                                    'Step': step+1,
                                    'Model': model.run_name.split('_')[1],
                                    'Sensor': model.run_name.split('_')[0],
                                    'Combination': model.run_name.split('_', 2)[2],
                                    'MSE': mse,
                                    'MAE': mae,
                                    'Percentage_common': percentage_common,
                                    'Index of highest simulation': index_of_actual_1_in_pred,
                                    'Simulations seen before': seen_count,
                                    'Highest simulation in pred': highest_actual_in_top,
                                    'highest_indices_pred': highest_indices_pred,
                                    'highest_indices_actual_1': highest_indices_actual_1,
                    })

                    # Plot the results if plot is True and step is a multiple of 5
                    if plot and (step + 1) % 5 == 0:
                        plot_time = time.time()
                        model.plot(means=means, stds=stds, selected_indices=selected_indices[-topk:], step=step, x_highest_pred_n=x_highest_pred_n, y_highest_pred_n=y_highest_pred_n, x_highest_actual_n=x_highest_actual_n, y_highest_actual_n=y_highest_actual_n, x_highest_actual_1=x_highest_actual_1, y_highest_actual_1=y_highest_actual_1, X_pool=data.X_pool, y_pool=data.Y_pool[:, index], X_selected=data.X_selected, y_selected=data.Y_selected[:, index])
                        tqdm.write(f'---Plotting time: {time.time() - plot_time:.2f} seconds')
                    
                    tqdm.write(f'------------ Step {step+1} completed ------------')

            # catch error and save the run that caused the error
            except Exception as e:
                print(f'Error in step {step+1} for sensor {model.run_name.split("_")[0]}: {e}')
                step_data.append({
                                    'Step': step+1,
                                    'Model': model.run_name.split('_')[1],
                                    'Sensor': model.run_name.split('_')[0],
                                    'Combination': model.run_name.split('_', 2)[2],
                                    'MSE': None,
                                    'MAE': None,
                                    'Percentage_common': None,
                                    'Index of highest simulation': None,
                                    'Simulations seen before': None,
                                    'Highest simulation in pred': None,
                                    'highest_indices_pred': None,
                                    'highest_indices_actual_1': None,
                    })
                continue
            # Append the step data to the result list
            result_list.append(step_data)
            pbar_steps.update(1)
        return result_list

# Calculate the start time
start = time.time()

# Initialize the steps_dataframes dictionary to store the results
steps_dataframes = {} 

# Parallelize processing of each combination
results = Parallel(n_jobs=-1)(delayed(process_combination)(model_name, combo, sensors, steps, epochs, directory, plot) for model_name, combo in all_combinations)

# Save the results to an Excel file
excel_filename = f'{directory}/{model_name}_model_results.xlsx'
# Aggregating results into steps_dataframes
for result in results:
    for data in result:
        for row in data:
            step = row['Step']  # Assuming 'Step' is a key in your returned dictionaries
            if step in steps_dataframes:
                steps_dataframes[step].append(row)
            else:
                steps_dataframes[step] = [row]

# Now converting lists of dictionaries to dataframes and writing to Excel
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    for step, data_list in steps_dataframes.items():
        df = pd.DataFrame(data_list)
        sheet_name = f'Step_{step}'
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Calculate the end time and time taken to run the models
end = time.time()
length = end - start

# Print the time taken to run the models
print("It took", length, "seconds to run", total_models, "models for", steps, "steps")