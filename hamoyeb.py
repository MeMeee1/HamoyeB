import pandas as pd, numpy as np, seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

df=pd.read_csv('C:/Users/HP/Downloads/energydata_complete.csv')
#print(df.head())
linear_model = df[['T2','T6']].sample(15,random_state=2)
sns.regplot(x='T2',y='T6',data=linear_model)

##r_squared = np.corrcoef(df.T6,df.T2)
##r_s=r_squared[0,1]
##r2 = r_s**2
##print(round(r2,2))

scaler = MinMaxScaler()

droped_df = df.drop(columns=['date','lights'])
normalised_df = pd.DataFrame(scaler.fit_transform(droped_df),columns=droped_df.columns)
print(normalised_df.columns)

target_variable = normalised_df.Appliances
print(target_variable)
x_train,x_test,y_train,y_test = train_test_split(droped_df,target_variable,test_size=0.33, random_state=42)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
predicted_values=linear_model.predict(x_test)

mae = mean_absolute_error(y_test,predicted_values)

print(round(mae,2))

rmse = np.sqrt(mean_squared_error(y_test,predicted_values))
print(round(rmse,2))

r2_score = r2_score(y_test, predicted_values)
print(round(r2_score,2))

mse = mean_squared_error(y_test,predicted_values)
rmse = math.sqrt(mse)
print(round(rmse,3))
