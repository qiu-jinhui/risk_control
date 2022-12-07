import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import metrics,cross_validation
from matplotlib import pyplot as plt



def Missingrate_Column(df, col):
    '''
    :param df: 数据集
    :param col:需要判断缺失率的特征
    :return:缺失率
    '''
    missing_records = df[col].map(lambda x: int(x!=x))
    return missing_records.mean()


#######################
####  1，读取数据  #####
#######################
data = pd.read_csv('/Users/Code/Data Collections/all/application_train_small.csv', header=0)
cash_loan_data = data[data['NAME_CONTRACT_TYPE'] == 'Cash loans']
selected_features = ['CODE_GENDER','FLAG_OWN_CAR','LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',
                    'FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','WEEKDAY_APPR_PROCESS_START',
                    'AMT_ANNUITY','AMT_GOODS_PRICE','NAME_TYPE_SUITE','NAME_INCOME_TYPE','OCCUPATION_TYPE',
                    'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','REGION_POPULATION_RELATIVE',
                    'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','OWN_CAR_AGE','FLAG_MOBIL',
                    'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL',
                    'CNT_FAM_MEMBERS','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY',
                    'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION',
                    'LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY']

all_data = cash_loan_data[['TARGET']+selected_features]

train_data, test_data = train_test_split(all_data, test_size=0.3)

#########################
####  2，数据预处理  #####
#########################

all_columns = list(train_data.columns)
all_columns.remove('TARGET')

#查看每个字段的缺失率
column_missingrate = {col: Missingrate_Column(train_data, col) for col in all_columns}
column_MR_df = pd.DataFrame.from_dict(column_missingrate, orient='index')
column_MR_df.columns = ['missing_rate']
column_MR_df_sorted = column_MR_df.sort_values(by='missing_rate', ascending=False)
columns_with_missing = column_MR_df_sorted[column_MR_df_sorted['missing_rate']>0]

'''
                 missing_rate
OWN_CAR_AGE          0.662521
OCCUPATION_TYPE      0.318319
NAME_TYPE_SUITE      0.003441
AMT_ANNUITY          0.000032
'''
#注意到，变量OWN_CAR_AGE和FLAG_OWN_CAR有对应关系：当FLAG_OWN_CAR='Y'时，OWN_CAR_AGE无缺失，否则OWN_CAR_AGE为有缺失
#这种缺失机制属于随机缺失。
#此外，对于非缺失的OWN_CAR_AGE，我们发现有异常值，例如0, 1，2等，无法判断该变量的含义，建议将其删除
selected_features.remove('OWN_CAR_AGE')
del train_data['OWN_CAR_AGE']
#变量OCCUPATION_TYPE和NAME_TYPE_SUITE属于类别型变量，可用哑变量进行编码
categorical_features = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                        'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE']
train_data_2 = pd.get_dummies(data=train_data, columns=categorical_features)

#删除AMT_ANNUITY缺失的样本
train_data_2 = train_data_2[~train_data_2['AMT_ANNUITY'].isna()]

#######################
####  3，特征衍生  #####
#######################
train_data_2['credit_to_income'] = train_data_2.apply(lambda x: x['AMT_CREDIT']/x['AMT_INCOME_TOTAL'],axis=1)
train_data_2['annuity_to_income'] = train_data_2.apply(lambda x: x['AMT_ANNUITY']/x['AMT_INCOME_TOTAL'],axis=1)
train_data_2['price_to_income'] = train_data_2.apply(lambda x: x['AMT_GOODS_PRICE']/x['AMT_INCOME_TOTAL'],axis=1)

#有四个与时长相关的变量DAYS_BIRTH，DAYS_EMPLOYED，DAYS_REGISTRATION	，DAYS_ID_PUBLISH中带有负号，不清楚具体的含义。
#我们在案例中仍然保留4个变量，但是建议在真实场景中获得字段的真实含义


##########################
####  4，构建随机森林  #####
##########################
#使用默认参数进行建模
all_features = list(train_data_2.columns)
all_features.remove('TARGET')
X, y = train_data_2[all_features], train_data_2['TARGET']

RFC = RandomForestClassifier(oob_score=True)
RFC.fit(X,y)
print(RFC.oob_score_)
y_predprob = RFC.predict_proba(X)[:,1]
result = pd.DataFrame({'real':y,'pred':y_predprob})
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))
ROC_AUC(result, 'pred', 'real')
feature_importance = pd.DataFrame({'feature':all_features,'importance':RFC.feature_importances_})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)



#参数调整
#1，调整n_estimators
param_test1 = {'n_estimators':range(10,151,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(),param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
best_n_estimators_1 = gsearch1.best_params_['n_estimators']  #140

param_test1 = {'n_estimators':range(131,150)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(),param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
best_n_estimators = gsearch1.best_params_['n_estimators']  #134


#2，对决策树最大深度max_depth,内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf进行网格搜索
param_test2 = {'max_depth':range(5,21), 'min_samples_split':range(20,81,10), 'min_samples_leaf':range(5,21,5)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= best_n_estimators),param_grid = param_test2, scoring='roc_auc',cv=5)
gsearch2.fit(X,y)
best_max_depth, best_min_samples_split, best_min_samples_leaf = gsearch2.best_params_['max_depth'],gsearch2.best_params_['min_samples_split'],gsearch2.best_params_['min_samples_leaf']


#3，对max_features进行调优
param_test3 ={'max_features':['sqrt','log2']}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= best_n_estimators,
                                                           max_depth = best_max_depth,
                                                           min_samples_split = best_min_sample_split,
                                                           min_samples_leaf = best_min_samples_leaf),
                        param_grid = param_test3, scoring='roc_auc',cv=5)
gsearch3.fit(X,y)
best_max_features = gsearch3.best_params_['max_features']

RFC_2 = RandomForestClassifier(oob_score=True, n_estimators= best_n_estimators,
                            max_depth = best_max_depth,min_samples_split = best_min_sample_split,
                            min_samples_leaf = best_min_samples_leaf,max_features = best_max_features)
RFC_2.fit(X,y)
print(RFC_2.oob_score_)
y_predprob = RFC_2.predict_proba(X)[:,1]
result = pd.DataFrame({'real':y,'pred':y_predprob})
#print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))
ROC_AUC(result, 'pred', 'real')


#特征重要性评估
fi = RFC_2.feature_importances_
fi = sorted(fi, reverse=True)
plt.bar(list(range(len(fi))), fi)
plt.title('feature importance')
