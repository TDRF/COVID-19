import lightgbm as lgb
import pandas as pd
import numpy as np
import sklearn
import datetime
import gc


goldstandard=pd.read_csv('./data/goldstandard.csv')


person=pd.read_csv('./data/person.csv')
#person['race']=person['race_concept_id'].apply(lambda x: 'White' if x==8527 else ('Asian' if x==8515 else ('Black or African American' if x==8516 else 'Unknown')))
person['race']=person['race_concept_id'].apply(lambda x: 1 if x==8527 else (2 if x==8515 else (3 if x==8516 else 0)))
#person['ethnicity']=person['ethnicity_concept_id'].apply(lambda x: 'Not Hispanic or Latino' if x==38003564 else 'Hispanic or Latino')
person['ethnicity']=person['ethnicity_concept_id'].apply(lambda x: 0 if x==38003564 else 1)
person['gender']=person['gender_source_value'].apply(lambda x: 0 if x=='F' else 1)
person['age']=2020-person['year_of_birth']
person_feature=person[['person_id','age','race','ethnicity','gender']]
person_feature=person_feature.set_index('person_id',drop=True)
del person
gc.collect()




visit_occurrence=pd.read_csv('./data/visit_occurrence.csv')
visit_feature=pd.DataFrame(visit_occurrence['person_id'].value_counts()).reset_index().rename(columns={'person_id':'visit times','index':'person_id'})
visit_feature=pd.merge(pd.DataFrame(person_feature.index),visit_feature,how='outer',on='person_id')
visit_feature=visit_feature.fillna(0)
visit_feature=visit_feature.set_index('person_id',drop=True)
del visit_occurrence
gc.collect()

condition_occurrence=pd.read_csv('./data/condition_occurrence.csv')
#condition_occurrence=pd.merge(condition_occurrence, goldstandard, on='person_id', how='outer')
#condition_dict=data_dict[data_dict['table']=='condition_occurrence'][['concept_id','concept_name']]
#condition_dict=condition_dict.rename(columns={'concept_id':'condition_concept_id'})
#condition_occurrence=pd.merge(condition_occurrence,condition_dict,on='condition_concept_id',how='outer')
#condition_occurrence=condition_occurrence.sort_values(by='person_id',ascending=True)
#condition_occurrence['type']=condition_occurrence['condition_type_concept_id'].apply(lambda x: 'EHR billing diagnosis' if x==32019 else ('EHR encounter diagnosis' if x==32020 else('Observation recorded from EHR' if x==43542353 else 'EHR Chief Complaint' )))
#condition_occurrence['condition_status']=condition_occurrence['condition_status_concept_id'].apply(lambda x: 'Admitting diagnosis' if x==4203942 else ('Final diagnosis' if x== 4230359 else'Preliminary diagnosis' ))

condition_occurrence['condition_start_date']=pd.to_datetime(condition_occurrence['condition_start_date'])
condition_recent=condition_occurrence[condition_occurrence['condition_start_date']>datetime.datetime(2019,4,1)]

a=pd.DataFrame(condition_recent.groupby(['person_id','condition_concept_id'])['condition_concept_id'].count()).rename(columns={'condition_concept_id':'y/n'}).reset_index()
condition_feature=a.pivot(index='person_id',columns='condition_concept_id',values='y/n')
condition_feature[condition_feature>0]=1
condition_feature=condition_feature.fillna(0)
del condition_occurrence
gc.collect()
#del condition_dict
#gc.collect()





measurement=pd.read_csv('./data/measurement.csv')
#measure_dict=data_dict[data_dict['table']=='measurement'][['concept_id','concept_name']]
#measure_dict=measure_dict.rename(columns={'concept_id':'measurement_concept_id'})
#measurement=pd.merge(measurement,measure_dict,on='measurement_concept_id',how='outer')
#measurement['type']=measurement['measurement_type_concept_id'].apply(lambda x: 'Lab result' if x==44818702 else 'Vital signs')
#measurement['operator']=measurement['operator_concept_id'].apply(lambda x: '=' if x==4172703 else ('>' if x==4172704 else ('<' if x==4171756 else '')))
#measurement=measurement.sort_values(by='person_id', ascending=True)
measurement['measurement_date']=pd.to_datetime(measurement['measurement_date'])
measurement_recent=measurement[measurement['measurement_date']>datetime.datetime(2019,4,1)]

b=pd.DataFrame(measurement_recent.groupby(['person_id','measurement_concept_id'])['value_as_number'].mean()).reset_index()
measurement_feature=b.pivot(index='person_id',columns='measurement_concept_id',values='value_as_number')

del measurement
gc.collect()
del measurement_recent
gc.collect()
#del measure_dict
#gc.collect()

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
mis=missing_values_table(measurement_feature)
mea_col=mis[mis['% of Total Values']<65].index
measurement_feature=measurement_feature[mea_col]
"""
mea_index=measurement_feature.index

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights="distance")
X=np.array(measurement_feature)
X_new=imputer.fit_transform(X)
mea_feature=pd.DataFrame(X_new)
mea_feature.index=mea_index
mea_feature.columns=mea_col


procedure_occurrence=pd.read_csv('D:/graduate/biostat/duke_project/synthetic_data/procedure_occurrence.csv')
procedure_occurrence['type']=procedure_occurrence['procedure_type_concept_id'].apply(lambda x: 'Hospitalization Cost Record' if x==257 else 'Inferred from claim')
procedure_dict=data_dict[data_dict['table']=='procedure_occurrence'][['concept_id','concept_name']]
procedure_dict=procedure_dict.rename(columns={'concept_id':'procedure_concept_id'})
procedure_occurrence=pd.merge(procedure_occurrence,procedure_dict,on='procedure_concept_id',how='outer')
c=pd.DataFrame(procedure_occurrence.groupby(['person_id','concept_name'])['concept_name'].count()).rename(columns={'concept_name':'count'}).reset_index()

procedure_feature=c.pivot(index='person_id',columns='concept_name',values='count')
procedure_feature[procedure_feature>0]=1
procedure_feature=procedure_feature.fillna(0)

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.95 * (1 - .95)))
p_f=np.array(procedure_feature)
p_f_new=sel.fit_transform(p_f[0:,1:])
"""

device_exposure=pd.read_csv('./data/device_exposure.csv')
#device_exposure['concept']=device_exposure['device_concept_id'].apply(lambda x: 'Ventilator' if x==45768197 else '')


d=pd.DataFrame(device_exposure.groupby(['person_id','device_concept_id'])['device_concept_id'].count()).rename(columns={'device_concept_id':'y/n'}).reset_index()

device_feature=d.pivot(index='person_id',columns='device_concept_id',values='y/n')

device_feature[device_feature>0]=1
del device_exposure
gc.collect()

observation=pd.read_csv('./data/observation.csv')
#observation_dict=data_dict[data_dict['table']=='observation'][['concept_id','concept_name']]
#observation_dict=observation_dict.rename(columns={'concept_id':'observation_concept_id'})
#observation=pd.merge(observation,observation_dict,on='observation_concept_id',how='outer')
#observation['type']=observation['observation_type_concept_id'].apply(lambda x: 'Observation recorded from EHR' if x==38000280 else 'Vital signs')
#observation=observation.sort_values(by='person_id', ascending=True)

observation['observation_datetime']=pd.to_datetime(observation['observation_datetime'])
observation_oxygen=pd.DataFrame(observation[(observation['observation_concept_id']==4196147) & (observation['observation_datetime']>datetime.datetime(2019,4,1))].groupby(['person_id'])['value_as_number'].mean()).rename(columns={'value_as_number':'Peripheral oxygen saturation'})
observation_alcohol=pd.DataFrame(observation[observation['observation_concept_id']==37208405].sort_values(['person_id','observation_datetime'],ascending=False).groupby(['person_id'])['value_as_string'].first()).rename(columns={'value_as_string':'History of alcohol use'})
observation_alcohol['History of alcohol use']=observation_alcohol['History of alcohol use'].map({'Yes':2,'No':1,'Never':0})
observation_tobacco=pd.DataFrame(observation[observation['observation_concept_id']==4005823].sort_values(['person_id','observation_datetime'],ascending=False).groupby(['person_id'])['value_as_string'].first()).rename(columns={'value_as_string':'Tobacco user'})
observation_tobacco['Tobacco user']=observation_tobacco['Tobacco user'].map({'Yes':1,'Never':0})
#observation_feature=observation_oxygen.join(observation_alcohol).join(observation_tobacco)
del observation
gc.collect()
#del observation_dict
#gc.collect()



drug_exposure=pd.read_csv('./data/drug_exposure.csv')
#drug_dict=data_dict[data_dict['table']=='drug_exposure'][['concept_id','concept_name']]
#drug_dict=drug_dict.rename(columns={'concept_id':'drug_concept_id'})
#drug_exposure=pd.merge(drug_exposure,drug_dict,on='drug_concept_id',how='outer')
#drug_exposure['drug_type']=drug_exposure['drug_type_concept_id'].apply(lambda x: 'Prescription written' if x==38000177 else('Physician administered drug (identified from EHR observation)' if x==43542358 else 'Medication list entry'))
drug_exposure['drug_exposure_start_date']=pd.to_datetime(drug_exposure['drug_exposure_start_date'])
drug_recent=drug_exposure[drug_exposure['drug_exposure_start_date']>datetime.datetime(2019,4,1)]

f=pd.DataFrame(drug_recent.groupby(['person_id','drug_concept_id'])['quantity'].mean()).rename(columns={'drug_concept_id':'quantity'}).reset_index()

drug_feature=f.pivot(index='person_id',columns='drug_concept_id',values='quantity')
drug_feature=drug_feature.fillna(0)
#del drug_dict
#gc.collect()
del drug_exposure
gc.collect()


status=goldstandard.set_index('person_id',drop=True)
#feature=person_feature.join(visit_feature).join(condition_feature).join(measurement_feature).join(device_feature).join(observation_feature).join(drug_feature)
#feature=feature.fillna(0)


cate=person_feature.join(condition_feature).join(observation_alcohol).join(observation_tobacco).join(device_feature)
del condition_feature
gc.collect()
del observation_alcohol
gc.collect()
del observation_tobacco
gc.collect()
del device_feature
gc.collect()


from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="most_frequent")
cat_index=cate.index
cat_col=cate.columns
cat=imp.fit_transform(cate)
cat=pd.DataFrame(cat)
cat.index=cat_index
cat.columns=cat_col


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X=np.array(cat)
y=np.array(status['status'].astype('int'))
X=X[:,4:cat.shape[1]].astype('int')

model=SelectKBest(chi2,k=100)
X_new=model.fit_transform(X,y)
scores = model.scores_
p_values=model.pvalues_
indices=np.argsort(scores)[::-1]
cat_feature=cat.iloc[:,indices[0:100]+4]


conti=person_feature.join(visit_feature).join(observation_oxygen).join(drug_feature).join(measurement_feature)
del visit_feature
gc.collect()
del observation_oxygen
gc.collect()
del drug_feature
gc.collect()
del measurement_feature
gc.collect()



conti=conti.iloc[:,4:conti.shape[1]]
conti['visit times']=conti['visit times'].fillna(0)
conti['Peripheral oxygen saturation']=conti['Peripheral oxygen saturation'].fillna(conti['Peripheral oxygen saturation'].mean())

cont=conti.fillna(conti.mean())
"""
cont_index=conti.index
cont_col=conti.columns
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights="distance")
X=np.array(conti)
X_new=imputer.fit_transform(X)
cont=pd.DataFrame(X_new)
cont.index=cont_index
cont.columns=cont_col

cont.to_csv('D:/graduate/biostat/duke_project/synthetic_data_07-06-2020/release_07-06-2020/training/conti.csv')
"""

fea_new=cat_feature.join(cont)


#fea_new=fea_new.set_index('person_id',drop=True)

fea_new=fea_new.dropna(axis=0,how='all')
d=fea_new.join(status)
status=pd.DataFrame(d['status'])

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

X=np.array(fea_new)
y=np.array(status['status'].astype('int'))
clf = ExtraTreesClassifier(n_estimators=300)
clf = clf.fit(X, y)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
model = SelectFromModel(clf, prefit=True)
X_new=model.transform(X)
X_new.shape
fea=fea_new.iloc[:,indices[0:200]]
fea=fea.join(person_feature[['age','race','ethnicity','gender']])

fea.to_csv('./scratch/train_features.csv')

def cross_validate_lightgbm(train_data, train_output,  
                            n_folds, param_grid,  
                            type_dict,  
                            fixed_param_dict = {'objective': 'binary:logistic', 'eval_metric': ['auc']},  
                            metric_func_dict = {'auc': sklearn.metrics.roc_auc_score},  
                            other_metrics_dict = None, keep_data = True, **kwargs): 
 
 
     """  
     Perform k-fold cross-validation with xgboost hyperparameters 
     Get the average performance across folds and save all of the results 
     for easier calibration (Platt Scaling/Isotonic Regression) 
12  
13     Parameters 
14     --------------- 
15     train_data (pd.DataFrame or np.array): 
16         A matrix that contains 1 row per observation and 1 column per feature 
17  
18     train_output (pd.DataFrame or pd.Series or np.array): 
19         An array-like that contains the outcome of interest as a binary  
20         indicator 
21  
22     param_grid (OrderedDict):  
23         An Ordered Dict where the keys are the parameter of interest, 
24         and the value is a list containing all possible parameter settings that 
25         need to be tested. The reason this parameter is an ordered dict is so  
26         that the inner loop can keep track of which parameter is being set. In  
27         python 3, this should not be an issue, since dictionaries have implicit  
28         orderings when calling .keys(), but to be safe, an Ordered Dict is  
29         required. 
30  
31     type_dict (Dict):  
32         A dictionary whose keys are the same as param_grid, and the values are 
33         either int or float (the base python functions). These are used to  
34         coerce the parameters downstream 
35  
36     metric_func_dict (Dict):  
37         key: the name of the metric as a string 
38         value: A function that takes in arguments (y_true, y_pred) and computes some  
39         metric to be used to select the best cross-validated parameters. 
40         Default is sklearn.metrics.roc_auc_score 
41  
42     other_metrics_dict (Dict): 
43         A dictionary with the same structure as `metric_func_dict`. These metrics will  
44         not be used to determine the best parameters for the model. 
45  
46     **kwargs:  
47         Each key is an argument to the xgboost model that has  
48         only 1 value. These values will be passed every time the xgboost model is  
49         run. `objective` and `eval_metric` are two parameters that need to be set. 
50      
51     Returns 
52     -------- 
53     A tuple consisting of: 
54  
55     results_dict 
56     best_settings 
57     final_model_uncalibrated 
58     keep_dict 
59     """ 
 
 
     # Set up indices to keep track of training and validation folds 
     indices = np.arange(0, train_data.shape[0]) 
     indices = np.random.permutation(indices) 
     indices_list = np.array_split(indices, n_folds) 
 
 
     # Build up an OrderedDict to save results 
     results_dict = {} 
     for item in param_grid: 
         results_dict[item] = [] 
         results_dict['best_iteration'] = [] 
      
     for key in metric_func_dict: 
         results_dict[key] = [] 
     if other_metrics_dict: 
         for key in other_metrics_dict: 
             results_dict[key] = [] 
      
     # Initialize the data to keep 
     if keep_data: 
         keep_dict = {'true': [], 'pred': []} 
 

     # Build up the expanded grid of parameter values 
     expanded_grid = np.array(np.meshgrid(*param_grid.values())).T.reshape(-1, len(param_grid)) 
      
     # Implement Cross-validation 
     for ind, fold in enumerate(indices_list): 
         validation_fold = train_data.iloc[fold, :] 
         training_indices = np.concatenate([indices_list[f] for f in range(0, len(indices_list)) if f != ind]) 
         training_fold = train_data.iloc[training_indices, :] 
 
 
         validation_fold_output = train_output.iloc[fold] 
         train_fold_output = train_output.iloc[training_indices] 
 
 
         ## Train the model with parameters 
         ## For each fold, fit all of the models with all parameter settings 
         ## Store the results in another dictionary with the same keys 
         ## as the result 
 
 
         for setting in range(0, expanded_grid.shape[0]): 
             # Create the current setting parameter dict 
             current_parameter_dict = {} 
             for index, (key, value) in enumerate(param_grid.items()): 
                 current_parameter_dict[key] = type_dict[key](expanded_grid[setting][index]) 
                 results_dict[key].append(current_parameter_dict[key]) 
             current_parameter_dict.update(fixed_param_dict) 
 
 
             X_train = lgb.Dataset(training_fold, label = train_fold_output) 
             X_test = lgb.Dataset(validation_fold, label = validation_fold_output) 
 
 
             temp_model = lgb.train(current_parameter_dict, X_train, valid_sets = [X_train,X_test], **kwargs) 
             # Now that the model is fit, evaluate the metric 
             X_test=validation_fold
             temp_pred = temp_model.predict(X_test) 
              
             # Compute the metric of interest: Default is AUC 
             # Append result 
             for key in metric_func_dict: 
                 fold_result = metric_func_dict[key](validation_fold_output, temp_pred) 
                 results_dict[key].append(fold_result) 
              
             if other_metrics_dict: 
                 for key, func in other_metrics_dict.items(): 
                     results_dict[key].append(func(validation_fold_output, temp_pred)) 
             # Append best_iteration 
             results_dict['best_iteration'].append(temp_model.best_iteration) 
     def _find_best_settings(_result_dict, _param_dict): 
         """ 
         Now, we want to find the best settings of the hyperparameters given by results_dict 
         We want the highest value of the metric in metric_func_dict and to return 
129         the elements of param_grid that correspond to that value. 
130  
131         """ 
         for key in metric_func_dict: 
             max_index = _result_dict[key].index(max(_result_dict[key])) 
          
         final_setting_dict = {} 
         for key in _param_dict: 
             final_setting_dict[key] = _result_dict[key][max_index] 
         # Add the best iteration (with early_stopping_rounds provided) 
         final_setting_dict['best_iteration'] = _result_dict['best_iteration'][max_index] 
         return final_setting_dict 
 

     best_settings = _find_best_settings(results_dict, param_grid) 
     best_settings.update(fixed_param_dict) 
     number_boost_rounds = best_settings.pop('best_iteration') 
 

     if keep_data: 
         for ind, fold in enumerate(indices_list): 
             validation_fold = train_data.iloc[fold, :] 
             training_indices = np.concatenate([indices_list[f] for f in range(0, len(indices_list)) if f != ind]) 
             training_fold = train_data.iloc[training_indices, :] 
 
 
             validation_fold_output = train_output.iloc[fold] 
             train_fold_output = train_output.iloc[training_indices] 
 
 
             X_train = lgb.Dataset(training_fold, label = train_fold_output) 
             X_test = lgb.Dataset(validation_fold, label = validation_fold_output) 
 
 
             temp_model = lgb.train(best_settings, X_train, num_boost_round = number_boost_rounds, early_stopping_rounds = None) 
              
             # Now that the model is fit, evaluate the metric 
             X_test=validation_fold
             temp_pred = temp_model.predict(X_test) 
              
             keep_dict['true'] += list(validation_fold_output.values) 
             keep_dict['pred'] += list(temp_pred) 
 
 
     # Retrain model with best settings ====================== 
     train_df = lgb.Dataset(train_data, label = train_output) 
 
 
     final_model_uncalibrated = lgb.train(best_settings, train_df, num_boost_round = number_boost_rounds, early_stopping_rounds = None) 
 
     
     return results_dict, best_settings, final_model_uncalibrated, keep_dict 

train_data=fea
train_output=status
data=train_data.join(train_output)
train_data=data.drop(['status'],axis=1).astype('float')
train_output=data['status'].astype('int')
param_grid={
    'max_depth':[6,12],
     'num_leaves':[30,100],
     'learning_rate':[0.01,0.05],
     'feature_fraction':[0.8,0.7]}



fixed_param={
    'objective':'binary',
    'metric':['auc'],
    'bagging_fraction': 0.95,
    'min_data_in_leaf':200,
    'bagging_freq':5,
    'pos_bagging_fraction':1,
    'neg_bagging_fraction':0.5,
    
    }

type_dict={
    'max_depth':int,
     'num_leaves':int,
     'learning_rate':float,
     'feature_fraction':float}


results_dict, best_settings, final_model_uncalibrated, keep_dict=cross_validate_lightgbm(train_data, train_output,  
                            5, param_grid,  
                            type_dict,  
                            fixed_param_dict = fixed_param,  
                            metric_func_dict = {'auc': sklearn.metrics.roc_auc_score},  
                            other_metrics_dict = None, keep_data = True, num_boost_round=100,early_stopping_rounds=20)

final_model_uncalibrated.save_model('./model/lightgbm.txt')
print("Training stage finished", flush = True)