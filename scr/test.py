# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:36:41 2020

@author: asus
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
import sklearn
import datetime
import gc



data_dict=pd.read_csv('./data/data_dictionary.csv')



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
condition_dict=data_dict[data_dict['table']=='condition_occurrence'][['concept_id','concept_name']]
condition_dict=condition_dict.rename(columns={'concept_id':'condition_concept_id'})
condition_occurrence=pd.merge(condition_occurrence,condition_dict,on='condition_concept_id',how='outer')
#condition_occurrence=condition_occurrence.sort_values(by='person_id',ascending=True)
#condition_occurrence['type']=condition_occurrence['condition_type_concept_id'].apply(lambda x: 'EHR billing diagnosis' if x==32019 else ('EHR encounter diagnosis' if x==32020 else('Observation recorded from EHR' if x==43542353 else 'EHR Chief Complaint' )))
#condition_occurrence['condition_status']=condition_occurrence['condition_status_concept_id'].apply(lambda x: 'Admitting diagnosis' if x==4203942 else ('Final diagnosis' if x== 4230359 else'Preliminary diagnosis' ))

condition_occurrence['condition_start_date']=pd.to_datetime(condition_occurrence['condition_start_date'])
condition_recent=condition_occurrence[condition_occurrence['condition_start_date']>datetime.datetime(2019,4,1)]

a=pd.DataFrame(condition_recent.groupby(['person_id','concept_name'])['concept_name'].count()).rename(columns={'concept_name':'y/n'}).reset_index()
condition_feature=a.pivot(index='person_id',columns='concept_name',values='y/n')
condition_feature[condition_feature>0]=1
condition_feature=condition_feature.fillna(0)
del condition_occurrence
gc.collect()
del condition_dict
gc.collect()





measurement=pd.read_csv('./data/measurement.csv')
measure_dict=data_dict[data_dict['table']=='measurement'][['concept_id','concept_name']]
measure_dict=measure_dict.rename(columns={'concept_id':'measurement_concept_id'})
measurement=pd.merge(measurement,measure_dict,on='measurement_concept_id',how='outer')
#measurement['type']=measurement['measurement_type_concept_id'].apply(lambda x: 'Lab result' if x==44818702 else 'Vital signs')
#measurement['operator']=measurement['operator_concept_id'].apply(lambda x: '=' if x==4172703 else ('>' if x==4172704 else ('<' if x==4171756 else '')))
#measurement=measurement.sort_values(by='person_id', ascending=True)
measurement['measurement_date']=pd.to_datetime(measurement['measurement_date'])
measurement_recent=measurement[measurement['measurement_date']>datetime.datetime(2019,4,1)]

b=pd.DataFrame(measurement_recent.groupby(['person_id','concept_name'])['value_as_number'].mean()).reset_index()
measurement_feature=b.pivot(index='person_id',columns='concept_name',values='value_as_number')

del measurement
gc.collect()
del measurement_recent
gc.collect()
del measure_dict
gc.collect()

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
device_exposure['concept']=device_exposure['device_concept_id'].apply(lambda x: 'Ventilator' if x==45768197 else '')


d=pd.DataFrame(device_exposure.groupby(['person_id','concept'])['concept'].count()).rename(columns={'concept':'y/n'}).reset_index()

device_feature=d.pivot(index='person_id',columns='concept',values='y/n')

device_feature[device_feature>0]=1
del device_exposure
gc.collect()

observation=pd.read_csv('./data/observation.csv')
observation_dict=data_dict[data_dict['table']=='observation'][['concept_id','concept_name']]
observation_dict=observation_dict.rename(columns={'concept_id':'observation_concept_id'})
observation=pd.merge(observation,observation_dict,on='observation_concept_id',how='outer')
#observation['type']=observation['observation_type_concept_id'].apply(lambda x: 'Observation recorded from EHR' if x==38000280 else 'Vital signs')
#observation=observation.sort_values(by='person_id', ascending=True)

observation['observation_datetime']=pd.to_datetime(observation['observation_datetime'])
observation_oxygen=pd.DataFrame(observation[(observation['concept_name']=='Peripheral oxygen saturation') & (observation['observation_datetime']>datetime.datetime(2019,4,1))].groupby(['person_id'])['value_as_number'].mean()).rename(columns={'value_as_number':'Peripheral oxygen saturation'})
observation_alcohol=pd.DataFrame(observation[observation['concept_name']=='History of alcohol use'].sort_values(['person_id','observation_datetime'],ascending=False).groupby(['person_id'])['value_as_string'].first()).rename(columns={'value_as_string':'History of alcohol use'})
observation_alcohol['History of alcohol use']=observation_alcohol['History of alcohol use'].map({'Yes':2,'No':1,'Never':0})
observation_tobacco=pd.DataFrame(observation[observation['concept_name']=='Tobacco user'].sort_values(['person_id','observation_datetime'],ascending=False).groupby(['person_id'])['value_as_string'].first()).rename(columns={'value_as_string':'Tobacco user'})
observation_tobacco['Tobacco user']=observation_tobacco['Tobacco user'].map({'Yes':1,'Never':0})
observation_feature=observation_oxygen.join(observation_alcohol).join(observation_tobacco)
del observation
gc.collect()
del observation_dict
gc.collect()



drug_exposure=pd.read_csv('./data/drug_exposure.csv')
drug_dict=data_dict[data_dict['table']=='drug_exposure'][['concept_id','concept_name']]
drug_dict=drug_dict.rename(columns={'concept_id':'drug_concept_id'})
drug_exposure=pd.merge(drug_exposure,drug_dict,on='drug_concept_id',how='outer')
#drug_exposure['drug_type']=drug_exposure['drug_type_concept_id'].apply(lambda x: 'Prescription written' if x==38000177 else('Physician administered drug (identified from EHR observation)' if x==43542358 else 'Medication list entry'))
drug_exposure['drug_exposure_start_date']=pd.to_datetime(drug_exposure['drug_exposure_start_date'])
drug_recent=drug_exposure[drug_exposure['drug_exposure_start_date']>datetime.datetime(2019,4,1)]

f=pd.DataFrame(drug_recent.groupby(['person_id','concept_name'])['quantity'].mean()).rename(columns={'concept_name':'quantity'}).reset_index()

drug_feature=f.pivot(index='person_id',columns='concept_name',values='quantity')
drug_feature=drug_feature.fillna(0)
del drug_dict
gc.collect()
del drug_exposure
gc.collect()

feature=person_feature.join(visit_feature).join(condition_feature).join(measurement_feature).join(device_feature).join(observation_feature).join(drug_feature)


train_data=pd.read_csv('./scratch/train_features.csv')
train_col=train_data.columns
feature_col=feature.columns
diff_in_train=list(set(train_col).difference(set(feature_col)))
same_in_train=list(set(train_col).intersection(set(feature_col)))

test_data=feature[same_in_train].join(train_data[diff_in_train])

final_model = lgb.Booster(model_file='./model/lightgbm.txt')
predict=final_model.predict(test_data)
predict=pd.DataFrame(person_feature.index).join(pd.DataFrame(predict))
predict.columns=['person_id','score']
predict.to_csv('./output/predictions.csv')
