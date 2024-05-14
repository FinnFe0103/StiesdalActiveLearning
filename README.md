# Load Case Selection Algorithm (LCSA)
The *EDA.ipynb* file includes all the exploratory data analysis that is displayed in the data chapter of the thesis.

The folder *Single_sensor* includes the first version of the LCSA that builds on only one metamodel and operates only on one sensor. It was used to develop the methodology.
The file *multi_sensor_parallel.py* is used to initiate the approach, performing the processing parallelized. 
It draws on the file *multi_data.py* that performs the initial sampling as well as the normalization and separation of X and y values.
Further, it uses the *multi_main.py* file that performs the algorithm, including the training of the metamodels, selecting the samples, and evaluation after each exploration step.
The folder Models includes the GP framework, while the SVR is directly implemented into the *multi_main.py* code. 
The *preprocessing.py* file performs the preprocessing of the database and derives the caselist as well as the sensor results.

*GP_postprocessed.xlsx* and *SVR_postprocessed.xlsx* are the outputs of the grid search performed using the *multi_sensor_parallel.py* file.
The files *post-processing.ipynb* and *postpost.ipynb* were used to analyze the results as well as to create the result charts in the result section of the thesis.

The *validation.xlsx* file contains the calculations performed for the savings if the LCSA was implemented on the 4 caselists considered in this thesis.