import pandas as pd
import numpy as np
import sqlite3
import argparse

sensor_list = ['foundation_origin xy FloaterOffset [m]',
               'foundation_origin Rxy FloaterTilt [deg]',
               'foundation_origin Rz FloaterYaw [deg]',
               'foundation_origin z FloaterHeave [m]',
               'foundation_origin Mooring GXY Resultant force [kN]',
               'MooringLine1 Effective tension Fairlead [kN]',
               'MooringLine2 Effective tension Fairlead [kN]',
               'MooringLine3 Effective tension Fairlead [kN]',
               'MooringLine4 Effective tension Fairlead [kN]',
               'MooringLine5 Effective tension Fairlead [kN]',
               'GE14-220 GXY acceleration [m/s^2]',
               'CEN_E3 Resultant bending moment ArcLength=2.72 [kN.m]',
]

feature_list = ['WindGeographic',
                'Uhub',
                'WaveGeographic',
                'Hs',
                'YawError',
                'Tp',
                'CurrentGeographic',
                'CurrentSpeed',
]

def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def get_absolute(data, col):
    data_abs = data.copy()
    data_abs[col] = data_abs[col].abs()
    return data_abs

def get_reverse(data, col):
    data_rev = data.copy()
    data_rev[col] = -data_rev[col]
    return data_rev

def create_csvs(db):
    con = sqlite3.connect(db)
    df_sensors = pd.read_sql_query('SELECT * FROM sensors', con)
    df_stats = pd.read_sql_query('SELECT * FROM standardstatistics', con)
    df_simAttr = pd.read_sql_query('SELECT * FROM simulationattributes', con)
    con.close()
    
    # groups of simulations (different seeds)
    groups = df_simAttr.loc[df_simAttr['name'] == 'GroupID'].set_index('simulation_id') #get groupID for each simulation
    groups = df_simAttr.loc[df_simAttr['name'] == 'GroupID'] #get groupID for each simulation
    groups = groups.rename(columns={'value':'GroupID'})[['simulation_id', 'GroupID']].astype(int) #rename and convert to int
    
    # filter simulation results
    sensors = df_sensors[df_sensors['name'].isin(sensor_list)] #filter 2348 sensors for 12
    sim_results = df_stats[df_stats['sensor_id'].isin(sensors['id'].unique())] #filter results of 2348 sensors for 12
    sim_results = sim_results.pivot(index='simulation_id', columns='sensor_id', values='max').sort_index()
    sim_results = sim_results.merge(groups, on='simulation_id').set_index('simulation_id') #join groups with sim_results on simulation_id

    # filter simulation attributes
    df_simAttr_filter = df_simAttr[df_simAttr['name'].isin(feature_list)] #filter 47 simulation attributes for 8
    caselist = df_simAttr_filter.pivot(index='simulation_id', columns='name', values='value') #pivot to get 1 row per simulation
    caselist = caselist.merge(groups, on='simulation_id').set_index('simulation_id') #join groups and caselist on simulation_id

    # find simulations that do not have results and filter them out
    common_indices = caselist.index.intersection(sim_results.index)
    filtered_caselist = caselist.loc[common_indices] # filter the caselist to only include simulations with results
    sim_results = sim_results.loc[common_indices] # just to make sure there are no ouputs without x values
    print(f'Deleted {len(caselist) - len(filtered_caselist)} simulations with missing output data.')
    print(f"Indices: {[index for index in caselist.loc[~caselist.index.isin(common_indices)].index]}")

    # only returning one result per groupID
    mean_results = sim_results.groupby('GroupID').mean().reset_index()
    caselist_unique = caselist.groupby(caselist['GroupID']).first().reset_index()

    # change the data types
    caselist_unique[['CurrentGeographic', 'WaveGeographic', 'WindGeographic', 'YawError', 'GroupID']] = caselist_unique[['CurrentGeographic', 'WaveGeographic', 'WindGeographic', 'YawError', 'GroupID']].astype(int)
    caselist_unique[['Uhub', 'Hs', 'Tp', 'CurrentSpeed']] = caselist_unique[['Uhub', 'Hs', 'Tp', 'CurrentSpeed']].astype(float)

    # encode the cyclical features
    caselist_unique = encode(caselist_unique, 'CurrentGeographic', 360)
    caselist_unique = encode(caselist_unique, 'WaveGeographic', 360)
    caselist_unique = encode(caselist_unique, 'WindGeographic', 360)
    caselist_unique = encode(caselist_unique, 'YawError', 360)

    # drop the original columns
    caselist_unique = caselist_unique.drop(columns=['CurrentGeographic', 'WaveGeographic', 'WindGeographic', 'YawError'])

    mean_results = get_absolute(mean_results, 49) #49: foundation_origin xy FloaterOffset [m]
    mean_results = get_absolute(mean_results, 52) #52: foundation_origin Rxy FloaterTilt [deg]

    # rename the columns of sim_results
    # column_mapping = pd.Series(sensors.name.values,index=sensors.id).to_dict()
    # mean_results = mean_results.rename(columns=column_mapping)

    # drop groupID col from both dataframes
    caselist_unique = caselist_unique.drop(columns=['GroupID'])
    mean_results = mean_results.drop(columns=['GroupID'])

    # save to csv
    caselist_unique.to_csv('_data/caselist.csv', index=False)
    mean_results.to_csv('_data/sim_results.csv', index=False)
    print('Saved caselist and sim_results to csv!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess given database')
    parser.add_argument('-db', '--database', type=str, default='_data/U62_PULSE_simulationstats.db', help='Path to database')
    args = parser.parse_args()
    create_csvs(args.database)