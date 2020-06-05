#!/usr/bin/env python
# coding: utf-8

# ### Imports

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
import cufflinks as cf
from plotly.graph_objs.scatter import Line, Marker
import gpflow
import matplotlib as mpl
import tensorflow as tf
import plotly.offline as pyo
pyo.init_notebook_mode()
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
get_ipython().run_line_magic('matplotlib', 'inline')


class GPModelObject(object):
    ''' Class for the Gaussian Process Model.
    
    Methods
    =======
    normalize_data:
        Does a z-score normalisation on the input vector
    get_data:
        retrieves and prepares the base data set
    kernel_ojects_dic:
        Adds or removes a kernel from the models kernel dictionary
    yearly_diff:
        If the day exists in the previous year, it calculates the YoY difference
    prediction:
       Calculates the predictions of the GP model, creates the columns containing
       the predictions and prints statistical measures of the predictions
    plot_residuals:
        plots the residuals of the predicted values
    plot_act_exp:
        Plots the actuals vs the expected values
    plot_graphs:
        Plots to visualise the seasonal data i.e. boxplots of data
        by day, month, year.
    '''
    
    def __init__(self):
        # kernels is a dictionary that holds all the kernels of the model
        self.kernels = {}

    def normalize_data(self, y):
        ''' Does a z-score normalisation on the input vector '''
        self.y_mean = y.mean()
        self.y_std = y.std()
        return (y - self.y_mean)/self.y_std 

    def get_data(self, testing_cutoff='20180901', exclude_leap=True):
        ''' Retrieves and prepares the base data set
            
            Parameters
            ==========
            testing_cutoff: str (yyyymmdd)
                Date from which the test sample starts.
                With the default date the test sample is a full year.
            
            exclude_leap: boolean
                If True, it excludes the leap day, otherwise it keeps it
            
        '''
        self.testing_cutoff = testing_cutoff
        d = pd.read_excel('thasos_data.xlsx', header=2)

        # Make sure all dates are sequential by 1 day with no gaps
        assert d[['Dates']].diff()[1:].eq('1 days').all()[0] == True

        d['year'] = pd.DatetimeIndex(d['Dates']).year
        d['month'] = pd.DatetimeIndex(d['Dates']).month
        d['day'] = pd.DatetimeIndex(d['Dates']).day
        
        # Exclude leap day for simplicity
        if exclude_leap:
            d = d.iloc[np.where((d['day'] != 29) | (d['month'] != 2))]
            d = d.reset_index(drop=True)

        d['day_of_year'] = pd.DatetimeIndex(d['Dates']).dayofyear + 1
        d['day_of_year_str'] = d['Dates'].dt.strftime("%d-%b")
        d['day_of_week'] = pd.DatetimeIndex(d['Dates']).dayofweek + 1
        q = d['Dates'].shift(365) == d['Dates'] - pd.DateOffset(years=1)
        d.loc[q, 'PX_MID_YD_LOG'] = np.log(d['PX_MID']).diff(365)[q]
        d.loc[q, 'PX_MID_YD'] = d['PX_MID'].diff(365)[q]

        # Split training and test sample
        d['test_sample'] = np.where(d['Dates']>=testing_cutoff, 1, 0)

        d['weekday'] = (~d['day_of_week'].isin([6,7])).astype(int)
        d['weekend'] = (d['day_of_week'].isin([6,7])).astype(int)
        
        # Special fixed days
        special_days = [(1,1),
                        (1,2),
                        (2,2), 
#                         (14,2), 
#                         (29,4), 
#                         (1,7), 
#                         (4,7), 
#                         (5,10), 
                        (22,12), 
                        (23,12), 
                        (24,12), 
                        (25,12), 
                        (26,12), 
                        (27,12), 
                        (28,12), 
                        (29,12), 
                        (30,12), 
                        (31,12)]

        for i in range(len(special_days)):
            q = 'sd_f_{}_{}'.format(special_days[i][0], special_days[i][1])
            d.loc[:, q] = np.float32((d['month'] == special_days[i][1]) 
                                     & (d['day'] == special_days[i][0]))

        # Special variable days

        # Thanks giving
        for i in range(len(d['year'].unique())):
            q = np.where((d['year'] == d['year'].unique()[i]) & (d['month'] == 11) & (d['day_of_week'] == 4))
            if len(q[0]) > 3:
                d.loc[q[0][3],'sd_v_thanksg'] = 1
                d.loc[q[0][3] + 1,'sd_v_thanksg_plus_one'] = 1

        # Labor day
        for i in range(len(d['year'].unique())):
            q = np.where((d['year'] == d['year'].unique()[i]) & (d['month'] == 9) & (d['day_of_week'] == 1))
            if len(q[0]) > 0:
                d.loc[q[0][0], 'sd_v_labor'] = 1
                d.loc[q[0][0] + 1, 'sd_v_labor_plus_one'] = 1

        # Memorial Day
        for i in range(len(d['year'].unique())):
            q = np.where((d['year'] == d['year'].unique()[i]) & (d['month'] == 5) & (d['day_of_week'] == 1))
            if len(q[0]) > 0:  
                d.loc[q[0].max(), 'sd_v_memorial'] = 1

        # Easter
        easter_days = ['20150405', '20160327', '20170416', '20180401', 
                       '20190421', '20200412', '20210404', '20220417', 
                       '20230409', '20240331', '20250420'] 

        d.loc[d['Dates'].isin(easter_days), 'sd_v_easter'] = 1

        q = ['sd_v_thanksg', 'sd_v_thanksg_plus_one', 'sd_v_labor', 'sd_v_labor_plus_one', 'sd_v_memorial', 'sd_v_easter']

        for col in q:
            d[col].fillna(0, inplace=True)
        
        # All data
        self.data = d
        d['x'] = d.index
        d['y'] = d['PX_MID']
        
        # Training data
        self.data_d = d[d['test_sample'] == 0]
        self.data_d['yn'] = self.normalize_data(self.data_d['y'])

        # Test data
        self.data_t = d[d['test_sample'] == 1]
        self.data_t['yn'] = (self.data_t['y'] - self.y_mean)/self.y_std 

        # Get numerical only cols to input to the GP model
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        
        # Training numerical only data
        self.data_d_num = self.data_d.select_dtypes(include=numerics)
        
        # Testing numerical only data
        self.data_t_num = self.data_t.select_dtypes(include=numerics)

    def kernel_ojects_dic(self, kname='dum', ktype='x',
                          kobject='x', add=True, all=False):
        ''' Adds or removes a kernel from the models kernel dictionary
            
            Parameters
            ==========
            kname: str
                Chosen Name of the kernel i.e. k1, k2, k3 etc.
            
            ktype: str
                Description of the kernel i.e. Periodic
            
            kobject: str
                The gpflow object of the kernel
                
            add: boolean
                If true it adds the kernel with the kname,
                otherwise it removes it.
                
            all: boolean
                if True it removes all kernels
        '''
        
        if add is True:
            self.kernels.update({kname: {'type' : ktype, 'obj': kobject}})
        else:
            if all:
                self.kernels = {}
            else:
                del self.kernels[kname]

    def yearly_diff(self, data, datecol, col, newcol):
        '''If the day exists in the previous year (i.e. not leap day),
            it calculates the YoY difference
            
            Parameters
            ==========
            data: pandas dataframe
                Pandas dataframe that contains the
                column to be differenced
            
            datecol: str
                column containing the date to be used for differening
            
            col: str
                Name of column for yearly difference
            
            newcol: str
                Name of the resulting column
            
        '''
        
        for i in range(data.shape[0]):
            if i in data.index:
                q = np.where((data[datecol] == data[datecol][i] - pd.DateOffset(years=1)))
                if len(q[0]) > 0 and q[0][0] in data.index: 
                    data.loc[i, newcol] = data.loc[i, col] - data.loc[q[0][0], col]  

    def prediction(self):
        '''Calculates the predictions of the GP model, creates the columns containing
            the predictions and prints statistical measures of the predictions
        '''
        # Prediction of training data
        m, s = self.m.predict_y(self.m.X.value)
        self.data_d['yn_pred'] = m
        self.data_d['yn_var'] = s
        
        # Denormalise
        self.data_d['y_pred'] = m * self.y_std + self.y_mean
        self.data_d['y_std'] = np.sqrt(s * self.y_std**2)
        
        # Mean square error of training data
        self.mse_d = round(((self.data_d['y_pred']-self.data_d['y'])**2).mean(), 3)
        
        # Prediction of test data
        m, s = self.m.predict_y(self.data_t_num.values)
        self.data_t['yn_pred'] = m
        self.data_t['yn_var'] = s
        
        # Denormalise
        self.data_t['y_pred'] = m * self.y_std + self.y_mean 
        self.data_t['y_std'] = np.sqrt(s * self.y_std**2)
        
        # Mean square error of test data
        self.mse_t = round(((self.data_t['y_pred']-self.data_t['y'])**2).mean(), 3)
        
        print('MSE training: ' + str(self.mse_d))
        print('MSE testing: ' + str(self.mse_t))
        print('Prior Log Likelihood: ' + str(self.m.compute_log_prior()))
        print('Log Likelihood: ' + str(self.m.compute_log_likelihood()))
        
    def plot_residuals(self, data, title):
        ''' Plots the residuals of the predicted values
            
            Parameters
            ==========
            data: pandas dataframe
                Dataframe that contains the actual (y) and predicted (y_pred) values.
        '''
        
        data.iplot(data = [dict(y=data['y'] - data['y_pred'], 
                                x=data['Dates'],
                                name='Actual',
                                mode='markers',
                                marker = Marker(color='Black',
                                            size=3,
                                            opacity=1),
                                color='Gray',
                                showlegend=False),
                          ], 
                   title = title,
                   dimensions=(700, 400), 
                   vspan=dict(x0=datetime.datetime.strptime(self.testing_cutoff, "%Y%m%d"),
                              x1= data['Dates'].max(),
                              color='Black',
                              width=0,
                             fill= False,
                             opacity = 0.1)
                  )
        
        
    def plot_act_exp(self, data, title):
        ''' Plots the actuals vs the expected values
            
            Parameters
            ==========
            data: pandas dataframe
                Dataframe that contains the actual (y) and
                the predicted (y_pred) values with the standard deviation y_std
                
        '''
        
        data.iplot(data = [dict(y=data['y_pred'] - 1.96 * data['y_std'], 
                                x=data['Dates'],
                                name='Predicted - 1.96 σ',
                                line=Line(color='Gray',
                                          width=0),
                                showlegend=False
                              ),
                           
                           dict(y=data['y_pred'] + 1.96 * data['y_std'], 
                                x=data['Dates'],
                                name='Predicted + 1.96 σ',
                                fill='tonexty',
                                line=Line(color='Gray',
                                          width=0),
                                showlegend=False
                              ),
                           dict(y=data['y'], 
                                x=data['Dates'],
                                name='Actual',
                                mode='markers',
                                marker = Marker(color='Black',
                                            size=3,
                                            opacity=1),
                                color='Gray',
                                showlegend=False),
                                
                           dict(y=data['y_pred'], 
                                x=data['Dates'],
                                name='Predicted',
                                line=Line(color='Blue',
                                         width=0.8),
                                showlegend=False,
                                opacity =0.5),
                          ], 
                   title = title,
                   dimensions=(700, 400), 
                   vspan=dict(x0=datetime.datetime.strptime(self.testing_cutoff, "%Y%m%d"),
                              x1= data['Dates'].max(),
                              color='Black',
                              width=0,
                             fill= False,
                             opacity = 0.1)
                  )
        
    def plot_graphs(self, data, col):
        ''' Plots to visualise the seasonal data i.e. boxplots of data
            by day, month, year.
            
            Parameters
            ==========
            data: pandas dataframe
                Pandas dataframe that contains the
                necessary columns i.e year, month, day, day_of_week etc.
            
            col: str
                column containing the values to be used for the plots
        '''
        
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 16), sharey=False)
        fig.tight_layout(h_pad=7)
        
        data.boxplot(column=col, by='year', rot=0, grid=True, ax=axes.flatten()[0]);
        plt.suptitle("");
        axes[0,0].title.set_text('Footfall by Year')
        axes[0,0].set_ylim(500, 8000)
            
        data.boxplot(column=col, by='month', rot=90, grid=True, ax=axes.flatten()[1]);
        plt.suptitle("");
        axes[0,1].title.set_text('Footfall by Month')
        axes[0,1].set_ylim(500, 8000)


        data.boxplot(column=col, by='day', rot=90, grid=True, ax=axes.flatten()[2]);
        plt.suptitle("");
        axes[1,0].title.set_text('Footfall by Day')
        axes[1,0].set_ylim(500, 8000)
        
        data.boxplot(column=col, by='day_of_week', rot=90, grid=True, ax=axes.flatten()[3]);
        plt.suptitle("");
        axes[1,1].title.set_text('Footfall by Day of Week')
        axes[1,1].set_ylim(500, 8000)


        temp_plot = data.groupby([ 'year', 'day_of_week'])[[col]].mean()
        temp_plot = temp_plot.groupby([ 'year'])[[col]].transform(lambda x: (x - x[0])).unstack().stack(dropna=False)
        temp_plot.groupby('year').plot(kind='line', ax=axes.flatten()[4]);
        
        labels = data['year'].unique()
        axes[2,0].legend(labels, loc='upper left');

        axes[2,0].title.set_text('Day of Week variation by Year')
        axes[2,0].set_xlabel('day_of_week');
        axes[2,0].set_ylim([temp_plot[col].min() - 50, temp_plot[col].max() + 50])

        plt.xticks(np.arange(0, 7, 1))
        axes[2,0].set_xticklabels(np.arange(1, 8, 1)); 

        temp_plot = data.groupby([ 'year', 'month'])[[col]].mean()
        temp_plot = temp_plot.groupby([ 'year'])[[col]].transform(lambda x: (x - x[0])).unstack().stack(dropna=False)
        temp_plot.groupby('year').plot(kind='line', ax=axes.flatten()[5]);
        
        labels = data['year'].unique()
        axes[2,1].legend(labels, loc='upper left');

        axes[2,1].title.set_text('Month variation by Year')
        axes[2,1].set_xlabel('month');
        axes[2,1].set_ylim([temp_plot[col].min() - 50, temp_plot[col].max() + 50])

        plt.xticks(np.arange(0, 12, 1))
        axes[2,1].set_xticklabels(np.arange(1, 13, 1));
