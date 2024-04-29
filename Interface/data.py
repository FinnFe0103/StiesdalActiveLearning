
import pandas as pd

def import_data():
    sensors = pd.read_csv('C:/Users/finn/Documents/StiesdalActiveLearning/_data/sensors.csv')
    caselist = pd.read_csv('C:/Users/finn/Documents/StiesdalActiveLearning/_data/caselist.csv')
    sim_results = pd.read_csv('C:/Users/finn/Documents/StiesdalActiveLearning/_data/sim_results.csv')

    return sensors, caselist, sim_results