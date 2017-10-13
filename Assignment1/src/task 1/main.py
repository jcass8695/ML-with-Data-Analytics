import xlrd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime


workbook = xlrd.open_workbook('C:/Users/user/Documents/College Stuff/SS/Machine Learning with Media Applications/Assignment 1/Assignment1/src/Task 1/New_York_Taxi_Data_Formatted.xls', on_demand = True)
worksheet = workbook.sheet_by_name('New_York_Taxi_Data')

cell = worksheet.cell(2,2)

Trip_Duration_Train = np.zeros((1000,1))
Pickup_Day_Train = np.zeros((1000))
Pickup_Time_Train = np.zeros((1000))
Vendor_ID_Train = np.zeros((1000))


Trip_Duration_Test = np.zeros((1000,1))
Pickup_Time_Test = np.zeros((1000))
Pickup_Day_Test = np.zeros((1000))


for i in range(1, 1001):

    ##Extracting Duration
    Trip_Duration_Train[i-1][0] = worksheet.cell(i,10).value
    
    if(worksheet.cell(i+1000, 10).value > 5000):
        Trip_Duration_Test[i-1][0] = 5000
    else:
        Trip_Duration_Test[i-1][0] = worksheet.cell(i+1000, 10).value

    #Extracting Date and Time
    Train_Cell = worksheet.cell(i, 2).value
    year_train, month_train, day_train, hour_train, minute_train, second_train = xlrd.xldate_as_tuple(Train_Cell, workbook.datemode)
    py_date_train = datetime(year_train, month_train, day_train, hour_train, minute_train, second_train)
    
    Test_Cell = worksheet.cell(i+1000, 2).value
    year_test, month_test, day_test, hour_test, minute_test, second_test = xlrd.xldate_as_tuple(Train_Cell, workbook.datemode)
    py_date_test = datetime(year_test, month_test, day_test, hour_test, minute_test, second_test)

    Pickup_Day_Train[i-1] = py_date_train.weekday()
    Pickup_Day_Test[i-1] = py_date_test.weekday()
    Pickup_Time_Train[i-1] = py_date_train.hour
    Pickup_Time_Test[i-1] = py_date_test.hour

    #Extracting vendor id
    #Vendor_ID_Train

X_train = np.zeros((1000, 2))
X_train[:,0] = Pickup_Day_Train
X_train[:,1] = Pickup_Time_Train

X_test = np.zeros((1000, 2))
X_test[:,0] = Pickup_Day_Test
X_test[:,1] = Pickup_Time_Test

regr = linear_model.LinearRegression(fit_intercept = False)
#regr.fit(Pickup_Time_Train, Trip_Duration_Train)
regr.fit(X_train, Trip_Duration_Train)

#Trip_Duration_Predict = regr.predict(Pickup_Time_Test)
Trip_Duration_Predict = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
#print("Mean squared error: %.2f"
#      % mean_squared_error(Trip_Duration_Test, Trip_Duration_Predict))
## Explained variance score: 1 is perfect prediction
#print('Variance score: %.2f' % r2_score(Trip_Duration_Test, Trip_Duration_Predict))

#plt.scatter(Pickup_Time_Test, Trip_Duration_Test,  color='black')
#plt.plot(Pickup_Time_Test, Trip_Duration_Predict, color='blue', linewidth=3)

#plt.scatter(Pickup_Day_Test, Trip_Duration_Test,  color='black')
#plt.plot(Pickup_Day_Test, Trip_Duration_Predict, color='blue', linewidth=3)

#plt.xticks(())
#plt.yticks(())

#plt.show()
print('end')
