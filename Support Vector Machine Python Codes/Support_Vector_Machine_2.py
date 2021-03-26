#Gerçekleştireceğimiz analizler için kullanacağımız kütüphaneleri sırası ile projemize dahil edelim
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
#Modeli oluşturmak için üzerinde çalışacağımız Iris Veri Seti’ni proje içerisine aktaralım
iris = datasets.load_iris()
#Sepal Uzunluk ve Sepal Genişlik üzerinden türler arasındaki korelasyonları gözlemlemek için
#keşif verisi adımlarımızı sırası ile gerçekleştirelim
def visualize_sepal_data():
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.xlabel(‘Sepal length’)
plt.ylabel(‘Sepal width’)
plt.title(‘Sepal Width & Length’)
plt.show()
visualize_sepal_data()
#Sepal Uzunluk ve Sepal Genişlik üzerinde gerçekleştirdiğimiz türler arasındaki korelasyonları gözlemlemek için 
#keşif verisi adımlarını şimdi Petal Uzunluk ve Petal Genişlik için gerçekleştirelim
def visualize_petal_data():
iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel(‘Petal length’)
plt.ylabel(‘Petal width’)
plt.title(‘Petal Width & Length’)
plt.show()
visualize_petal_data()
#Çiçeğin ait olduğu sınıfın türünü tahmin etmek için ilk iki özelliği (Sepal Uzunluk/Genişlik) kullanarak bir DVM / SVM modeli oluşturalım.
#(Petal Uzunluk/Genişlik alternatif olarak kullanılabilir.)
X = iris.data[:, :2]
y = iris.target
#Çeşitli çekirdekler kullanarak DVM / SVM karar sınırlarını çizmek için bir model ağacı oluşturalım
def plotSVM(title):
x_min, x_max = X[:, 0].min() -1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() — 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z,alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel(“Sepal length”)
plt.ylabel(“Sepal width”)
plt.xlim(xx.min(), xx.max())
plt.title(title)
plt.show()
#Oluşturduğumuz yapımız için Lineer ve Lineer Olmayan (Polinomal ve Gauss) modellemeler üzerinden çekirdek işlemleri gerçekleştirelim
kernels = [“linear”, “rbf”, “poly”]
for kernel in kernels:
svc = svm.SVC(kernel=kernel).fit(X, y)
plotSVM(“kernel=” + str(kernel))
#Farklı Gama (Gamma : γ) değerleri (0.1, 1, 10, 100) üzerinden çeşitli çekirdekleri gözlemleyerek hiperparametre ayarı oluşturalım
gammas = [0.1, 1, 10, 100]
for gamma in gammas:
svc = svm.SVC(kernel=’rbf’, gamma=gamma).fit(X, y)
plotSVM(‘gamma=’ + str(gamma))
#(Hata Değeri / Ceza Değeri) parametresi üzerinde belirli değerler (0.1, 1, 10, 100, 1000) belirleyerek gözlemde bulunalım.
#Sorunsuz bir karar sınırı ile eğitim noktalarının doğru şekilde sınıflandırılması arasındaki dengeyi kontrol eder.
cs = [0.1, 1, 10, 100, 1000]
for c in cs:
svc = svm.SVC(kernel=’rbf’, C=c).fit(X, y)
plotSVM(‘C=’ + str(c))

"""
Kaynakça:
https://miracozturk.com/python-ile-siniflandirma-analizleri-destek-vektor-makinasi-dvm/
"""