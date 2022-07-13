import pandas as pd
import numpy as np

data = pd.read_csv("odev_tenis.csv")



from sklearn import preprocessing

wheather = data.iloc[:,:1].values
le = preprocessing.LabelEncoder()

wheather[:,0] = le.fit_transform(data.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
wheather = ohe.fit_transform(wheather).toarray()



windy = data.iloc[:,-2:-1].values

le = preprocessing.LabelEncoder()

windy[:,-1] = le.fit_transform(data.iloc[:,-2:-1])


ohe = preprocessing.OneHotEncoder()
windy = ohe.fit_transform(windy).toarray()


play = data.iloc[:,-1:].values

le = preprocessing.LabelEncoder()

play[:,-1] = le.fit_transform(data.iloc[:,-1])


ohe = preprocessing.OneHotEncoder()
play = ohe.fit_transform(play).toarray()


outlook =pd.DataFrame(data=wheather,index=range(14),columns=["Overcast","Rainy","Sunny"])
windy_or_not = pd.DataFrame(data=windy,index=range(14),columns=["Not Windy","Windy"])
play_or_not = pd.DataFrame(data=play,index=range(14),columns=["Don't Play","Play"])


result1= pd.concat([outlook,data.iloc[:,1:3]],axis=1)
result2 = pd.concat([result1,windy_or_not.iloc[:,1:2]],axis=1)
result3 = pd.concat([result2,play_or_not.iloc[:,1:2]],axis=1)
# Training part

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(result2,play_or_not.iloc[:,1:2],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)



import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int),values=result2,axis = 1)

X_list = result2.iloc[:,[0,1,2,3,4,5]].values

model =sm.OLS(result3.iloc[:,-1].values,X_list).fit()

#print(model.summary())

#Temperature has big P value. So that, it needs to be removed from table

X = np.append(arr=np.ones((14,1)).astype(int),values=result2,axis = 1)

X_list = result2.iloc[:,[0,1,2,4,5]].values

model =sm.OLS(result3.iloc[:,-1].values,X_list).fit()

print(model.summary())


