
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
import pandas as pd
import numpy as np
# Create your views here.
from Remote_User.models import ClientRegister_Model,Bigmart_model,detection_values_model


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            Bigmart_model.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = Bigmart_model.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_values_model.objects.values('names').annotate(dcount=Avg('MSE'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_values_model.objects.values('names').annotate(dcount=Avg('RMSE'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def Find_Big_Mart_Sale_Predicted_Details(request):

    obj =Bigmart_model.objects.all()
    return render(request, 'SProvider/Find_Big_Mart_Sale_Predicted_Details.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_values_model.objects.values('names').annotate(dcount=Avg('MAE'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Sales_Predictions_Results.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Bigmart_model.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, my_row.Item_Identifier, font_style)
        ws.write(row_num, 1, my_row.Outlet_Identifier, font_style)
        ws.write(row_num, 2, my_row.Item_Outlet_Sales, font_style)

    wb.save(response)
    return response

def train_model(request):
    obj=''
    detection_values_model.objects.all().delete()
    train = pd.read_csv("Train.csv")
    test = pd.read_csv("Test.csv")
    test1 = test.copy()
    train.shape, test.shape
    train.head()
    test["Outlet_Size"].unique()
    train.nunique()
    test.nunique()
    train.isna().sum()
    map1 = {"Small": 1, "Medium": 2, "High": 3}
    train["Outlet_Size"] = train["Outlet_Size"].map(map1)
    train["Item_Weight"] = train["Item_Weight"].fillna(train.Item_Weight.mean())
    train["Outlet_Size"] = train["Outlet_Size"].fillna(train["Outlet_Size"].median())
    train.isna().sum()
    map1 = {"Small": 1, "Medium": 2, "High": 3}
    test["Outlet_Size"] = test["Outlet_Size"].map(map1)
    test["Item_Weight"] = test["Item_Weight"].fillna(test.Item_Weight.mean())
    test["Outlet_Size"] = test["Outlet_Size"].fillna(test["Outlet_Size"].median())
    train.head()
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['figure.figsize'] = (10, 10)
    plt.hist(train["Item_Outlet_Sales"], bins=100)
    # plt.show()

    # plt.rcParams['figure.figsize'] = (10,10)
    plt.hist(train["Item_MRP"], alpha=0.3, bins=150)
    # plt.show()
    plt.rcParams['figure.figsize'] = (5, 5)
    plt.hist(train["Item_MRP"], alpha=0.3, bins=150)
    # plt.show()
    sns.countplot(train["Outlet_Location_Type"], palette='dark')
    # plt.show()
    sns.countplot(train["Outlet_Size"], palette='dark')
    # plt.show()
    sns.countplot(train["Outlet_Type"], palette='RdYlGn')
    plt.xticks(rotation=90)
    # plt.show()
    sns.violinplot(x=train["Outlet_Size"], y=train["Item_Outlet_Sales"], hue=train["Outlet_Size"], palette="Reds")
    plt.legend()
    # plt.show()
    train.drop(labels=["Outlet_Establishment_Year"], inplace=True, axis=1)
    test.drop(labels=["Outlet_Establishment_Year"], inplace=True, axis=1)
    feat = ['Outlet_Size', 'Outlet_Type', 'Outlet_Location_Type', 'Item_Fat_Content', "Item_Type"]
    X = pd.get_dummies(train[feat])
    train = pd.concat([train, X], axis=1)
    train.head()
    feat = ['Outlet_Size', 'Outlet_Type', 'Outlet_Location_Type', 'Item_Fat_Content', "Item_Type"]
    X1 = pd.get_dummies(test[feat])
    test = pd.concat([test, X1], axis=1)
    train.drop(labels=["Outlet_Size", 'Outlet_Location_Type', "Outlet_Type", 'Item_Fat_Content', 'Outlet_Identifier',
                       'Item_Identifier', "Item_Type"], axis=1, inplace=True)
    test.drop(labels=["Outlet_Size", 'Outlet_Location_Type', "Outlet_Type", 'Item_Fat_Content', 'Outlet_Identifier',
                      'Item_Identifier', "Item_Type"], axis=1, inplace=True)
    X_train = train.drop(labels=["Item_Outlet_Sales"], axis=1)
    y_train = train["Item_Outlet_Sales"]
    X_train.shape, y_train.shape
    train.head()
    y_train.head()
    from sklearn import preprocessing

    x = X_train.values  # returns a numpy array
    test_s = test.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled_train = min_max_scaler.fit_transform(x)
    x_scaled_test = min_max_scaler.fit_transform(test_s)
    df_train = pd.DataFrame(x_scaled_train)
    df_test = pd.DataFrame(x_scaled_test)
    df_train.head()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.4)

    # Linear Regression ****************
    print("Linear Regression")

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    from sklearn.linear_model import Ridge
    model2 = Ridge()
    model2.fit(X_train, y_train)
    preds2 = model2.predict(X_test)

    from sklearn import metrics

    print("**********************Linear Regression Model Results**********************")
    print("MAE:", metrics.mean_absolute_error(y_test, preds))
    print('MSE:', metrics.mean_squared_error(y_test, preds))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds)))
    detection_values_model.objects.create(names="Linear Regression Model Results", MAE=metrics.mean_absolute_error(y_test, preds),MSE=metrics.mean_squared_error(y_test, preds),RMSE=np.sqrt(metrics.mean_squared_error(y_test, preds)))

    print("**********************Ridge Regression Model Results**********************")
    print("MAE:", metrics.mean_absolute_error(y_test, preds2))
    print('MSE:', metrics.mean_squared_error(y_test, preds2))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, preds2)))
    detection_values_model.objects.create(names="Ridge Regression Model Results",
                                          MAE=metrics.mean_absolute_error(y_test, preds2),
                                          MSE=metrics.mean_squared_error(y_test, preds2),
                                          RMSE=np.sqrt(metrics.mean_squared_error(y_test, preds2)))
    predictions = model.predict(df_test)
    final = pd.DataFrame({"Item_Identifier": test1["Item_Identifier"], "Outlet_Identifier": test1["Outlet_Identifier"],
                          "Item_Outlet_Sales": abs(predictions)})
    final.head()
    final.to_excel('Predictions_Results.xlsx', index=False, header=True)
    print("PREDICTED RESULTS DOWNLOADED")
    obj = detection_values_model.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj})














