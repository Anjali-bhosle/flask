#import libraries

import pandas as pd
import pickle



dataset = pd.read_csv(r"D:\Data\Internship\data_hotel.csv")

dataset.fillna(dataset.median(),inplace=True)

dataset["Check_in_date"]=pd.to_datetime(dataset["Check_in_date"])

dataset["check_in_month"]=dataset["Check_in_date"].dt.month

dataset["check_in_year"]=dataset["Check_in_date"].dt.year

dataset["check_in_day"]=dataset["Check_in_date"].dt.day

dataset["Check_out_date"]=pd.to_datetime(dataset["Check_out_date"])

dataset["check_out_year"]=dataset["Check_out_date"].dt.year

dataset["check_out_month"]=dataset["Check_out_date"].dt.month

dataset["check_out_day"]=dataset["Check_out_date"].dt.day

dataset.drop(["Check_in_date","Check_out_date"],axis=1,inplace=True)




#Encoding
dataset=pd.get_dummies(dataset, drop_first=True)#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.


x = dataset.iloc[:, 0:22]
y = dataset.iloc[:, 22]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


from catboost import CatBoostRegressor
cbr=CatBoostRegressor(learning_rate=0.5)
cbr.fit(x_train, y_train)


# Saving model to disk


pickle.dump(cbr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
