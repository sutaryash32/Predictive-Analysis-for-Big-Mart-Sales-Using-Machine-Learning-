from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class Bigmart_model(models.Model):

    Item_Identifier=models.CharField(max_length=300)
    Outlet_Identifier=models.CharField(max_length=300)
    Item_Outlet_Sales=models.CharField(max_length=300)

class detection_values_model(models.Model):

    names = models.CharField(max_length=300)
    MAE= models.CharField(max_length=300)
    MSE= models.CharField(max_length=300)
    RMSE= models.CharField(max_length=300)



