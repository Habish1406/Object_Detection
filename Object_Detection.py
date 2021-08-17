import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import cv2

name={0: 'car',1:'flower'}
with_mask=np.load('car.npy')
without_mask=np.load('flower.npy')
with_mask.resize(200,50*50*3)
without_mask.resize(200,50*50*3)
x=np.r_[with_mask,without_mask]
y=np.zeros(x.shape[0])
y[200:]=1.0
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
pca= PCA(n_components=3)
x_train=pca.fit_transform(x_train)
x_test=pca.fit_transform(x_test)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
svm=SVC()
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
haar_data = cv2.CascadeClassifier(r'C:\Users\habis\PycharmProjects\pythonProject2\cascade.xml')
capture =cv2.VideoCapture(0)
while True:
 flag, img= capture.read()
 if flag:
     faces=haar_data.detectMultiScale(img)
     for x,y,w,h in faces:
         cv2.rectangle(img,(x,y),(x+w , y+h),(225,0,0),4)
         face=img[y:y+h,x:x+w,:]
         face=cv2.resize(face,(50,50))
         face=face.reshape(1,-1)
         pred=svm.predict(face)
         n=name[int(pred)]
         print(n)
     cv2.imshow('result',img)
     if cv2.waitKey(2) == 27 :
         break
capture.release()
cv2.destroyAllWindows()