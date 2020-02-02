import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import itertools
import datetime as dt
import statsmodels.api as sm
from datetime import timedelta,time,date
from datetime import datetime
import julian
import numpy as rng
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



df=pd.read_excel('WFM_Sample_Data_-_2016_2017.xlsx',index_col=0, parse_dates=True)

X=np.array(df.index.to_julian_date()).reshape((-1, 1))     # date_time transformed to julian to perform regression
Y=df['total_calls']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)     # Arbitrary value 0.3
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)
plt.scatter(X_test,Y_test,color='gray',s=0.5)
plt.plot(X_test,Y_pred,color='red',linewidth=1)

rng = pd.date_range(str(df.index[-1]+timedelta(minutes=15)), str(df.index[-1]+timedelta(weeks=6)), freq='15T')

rng=rng.to_julian_date()

data=[]     # List to store the predicted number of calls
val=[]      # List to store each 15 minute period of the 6 weeks

for i in rng:
    future_calls = round(regressor.predict(np.array([i]).reshape(-1,1))[0])      # For each 15 minute period of the 6 weeks is calculated the predicted value for the number of calls
    data.append(future_calls)
    val.append(i)

plt.plot(val,data,color='green',linewidth=1)
plt.ylabel("Calls count")
plt.xlabel("Date")     # Date should be in datetime format, not in julian
plt.show()
