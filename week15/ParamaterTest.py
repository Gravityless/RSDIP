import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('sample_training.csv')
feature = data.iloc[:, 2:5]
label = data.iloc[:, 5]

model = RandomForestClassifier()
n_estimators = [1, 10, 100, 500, 1000, 1500]
max_features = [None, 'sqrt', 1]

##将需要训练的参数转为字典形式
para_grid = dict(n_estimators = n_estimators, max_features = max_features)
#将训练样本进行十折交叉验证（样本分成十份，九份训练，一份检验）
Kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 1)

grid_search = GridSearchCV(model, para_grid, cv = Kfold)

#将所有参数可能性组合全部放入模型去运行，获得各自结果
grid_result = grid_search.fit(feature, label)

print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with: %r" % (mean, param))

print('done')