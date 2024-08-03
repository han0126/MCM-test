#import "../template/template.typ": *

#let code1 = ```python
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family']='Microsoft YaHei'
file_path = r"C:\Users\典总\Desktop\train.xlsx"
data = pd.read_excel(file_path)
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
fmt='.3f',linewidths=.5)
plt.title('相关矩阵热图')
plt.show()
sns.pairplot(data)
plt.suptitle('散点图矩阵', y=1.02)
plt.show()
```

#let code2 = ```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('E:/python/train.csv', encoding='GBK')
columns_to_analyze=['季风强度', '地形排水', '河流管理', '森林砍伐',
'城市化', '气候变化', '大坝质量', '淤积', '农业实践', '侵蚀', '无效防
灾','排水系统', '海岸脆弱性', '滑坡', '流域', '基础设施恶化', '人口得
分', '湿地损失', '规划不足', '政策因素', '洪水概率']
sc = StandardScaler()
df_sc = sc.fit_transform(df[columns_to_analyze])
kmeans = KMeans(n_clusters=3, random_state=1,
n_init=10).fit(df_sc)
df['risk_cluster'] = kmeans.labels_
cluster_features = df.groupby('risk_cluster')
[columns_to_analyze].mean()
cluster_centers = kmeans.cluster_centers_
cluster_centers = sc.inverse_transform(cluster_centers)
cluster_probabilities = df.groupby('risk_cluster')['洪水概率'].
mean()
sorted_clusters = np.argsort(cluster_probabilities)
risk_levels = ['C', 'B', 'A']
select_features_count = 4
for i, cluster_idx in enumerate(sorted_clusters):
select_features =
cluster_features.columns[np.argsort(cluster_features.loc[cluste
r_idx].values)[-select_features_count:]]
print(f"风险等级:{risk_levels[i]}")
print(cluster_features.loc[cluster_idx,
select_features].to_string(), "\n")
X = df[columns_to_analyze]
y = df['risk_cluster']
model_train = xgb.DMatrix(X, label=y)
params = {
'max_depth': 3,
'eta': 0.1,
'objective': 'multi:softmax',
'num_class': 3,
'eval_metric': 'mlogloss'
}
bst = xgb.train(params, model_train, num_boost_round=100)
predict = xgb.DMatrix(X)
y_predict = bst.predict(predict)
df['predicted_risk_cluster'] = y_predict
accuracy = accuracy_score(df['risk_cluster'],
df['predicted_risk_cluster'])
print(f"模型准确率:{accuracy:.3%}")
```

#let code3 = ```python
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('E:/python/9.csv', encoding='GBK')
columns_to_analyze = ['季风强度', '地形排水', '河流管理', '气候变化',
'大坝质量', '淤积', '基础设施恶化', '人口得分', '洪水概率']
X = df[columns_to_analyze[:-1]]
y = df['洪水概率']
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = MLPRegressor(hidden_layer_sizes=(200, 100, 50),
activation='tanh',
learning_rate_init=0.01, max_iter=500,
alpha=0.0001, random_state=1, early_stopping=True)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
error_rate = mean_absolute_percentage_error(y_test, y_predict)
accuracy = 100-error_rate
print(f"准确率为：{accuracy:.2f}%")
```

#let code4 = ```python
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('E:/python/5.csv', encoding='GBK')
columns_to_analyze = ['季风强度', '地形排水', '河流管理', '大坝质量',
'基础设施恶化', '洪水概率']
X = df[columns_to_analyze[:-1]]
y = df['洪水概率']
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = MLPRegressor(hidden_layer_sizes=(200, 100, 50),
activation='tanh',
learning_rate_init=0.01, max_iter=500,
alpha=0.0001, random_state=1,
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
error_rate = mean_absolute_percentage_error(y_test, y_predict)
accuracy = 100-error_rate
print(f"准确率为：{accuracy:.2f}%")
```

#let code5 = ```python
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('E:/python/5.csv', encoding='GBK')
columns_to_analyze = ['季风强度', '地形排水', '河流管理', '大坝质量',
'基础设施恶化', '洪水概率']
X = df[columns_to_analyze[:-1]]
y = df['洪水概率']
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = MLPRegressor(hidden_layer_sizes=(200, 100, 50),
activation='tanh',
learning_rate_init=0.01, max_iter=500,
alpha=0.0001, random_state=1, early_stopping=True)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
error_rate = mean_absolute_percentage_error(y_test, y_predict)
accuracy = 100-error_rate
print(f"准确率为：{accuracy:.2f}%")
new_data = pd.read_csv('E:/python/test.csv', encoding='GBK')
new_data = new_data[columns_to_analyze[:-1]]
new_data_scaled = scaler.transform(new_data)
new_predictions = model.predict(new_data_scaled)
new_data['洪水概率'] = new_predictions
new_data.to_csv('E:/python/predict.csv', index=False)
```

#let code6 = ```matlab
clc,clear,close all
data = table2array(readtable("predict.csv",'Range'  ,'F2:F1048576'));
[h_ks,p_ks]=kstest(data);
disp(h_ks);
disp(p_ks);
```

#let codeZip=(
  "问题一相关矩阵热图":code1,
  "问题二聚类及预警模型构建":code2,
  "问题三洪水概率预测模型构建":code3,
  "问题三简化模型指标为5":code4,
  "问题四概率预测":code5,
  "正态分布检验":code6
)

= 附录
#for (desc,code) in codeZip{
    codeAppendix(
        code,
        caption: desc
    )
    v(2em)
}