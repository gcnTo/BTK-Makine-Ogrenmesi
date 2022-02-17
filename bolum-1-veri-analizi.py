# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:32:48 2022

@author: Togrul
"""

# Kütüphanelerin Çağrılması
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Verisetinin Yüklenmesi
veriler = pd.read_csv("eksikveriler.csv")

# Yaşın Ayrıştırılması ve Eksik Verilerin Doldurulması
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
yas = veriler.iloc[:,1:4].values
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])

# Ülkelerin Ayrıştırılması
ulke = veriler.iloc[:,0:1].values
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

# Cinsiyetlerin Ayrıştırılması
cinsiyet = veriler.iloc[:,-1].values

# Verilerin Ölçeklenmesi Adımında str --> int dönüşümü gerçekleşemediği için
# kadın yani k --> 1 ve erkek yani e --> 0 dönüşümü yapan for döngüsünü yazdım
# Problemi görmek için for döngüsünü silip programı çalıştırmayı deneyebilirsiniz
#   Alternatif bir çözüm olarak da cinsiyet verisine ait kod satırları kullanım
# dışı bırakılabilir.

cinsiyet_to_int = copy.copy(cinsiyet) # Orjinal cinsiyet verilerinin saklanması için
                                      # oluşturulmuş kopya.

for i in range(cinsiyet_to_int.size):
    if(cinsiyet_to_int[i] == "k"):
        cinsiyet_to_int[i] = 1
    else:
        cinsiyet_to_int[i] = 0

# Sonuçların DataFrame Haline Dönüştürülmesi
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ["fr","tr","us"])
sonuc2 = pd.DataFrame(data=yas, index = range(22), columns = ["boy","kilo","yas"])
sonuc3 = pd.DataFrame(data = cinsiyet_to_int, index = range(22), columns = ["cinsiyet"])

# Sonuçların Birleştirilmesi
s = pd.concat([sonuc,sonuc2], axis=1)
s = pd.concat([s, sonuc3], axis=1)

# Verinin(Sonuçların) Bölünmesi
x_train, x_test, y_train, y_test = train_test_split(s,sonuc2,test_size=0.33,random_state=0)

# Verilerin(Sonuçların) Ölçeklenmesi
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
