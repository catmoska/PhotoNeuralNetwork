import cv2
import numpy as np
import os
import random
from functionss import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
from tensorflow import keras as k
from keras.layers import Dense
from datetime import datetime





start_time = datetime.now()
model = k.Sequential()
model.add(Dense(units=12288, activation='relu'))
model.add(Dense(units=1536, activation='sigmoid'))
model.add(Dense(units=1536, activation='sigmoid'))
model.add(Dense(units=768, activation='linear'))

model.add(Dense(units=30, activation='linear'))

model.add(Dense(units=768, activation='linear'))
model.add(Dense(units=1536, activation='sigmoid'))
model.add(Dense(units=1536, activation='sigmoid'))
model.add(Dense(units=12288, activation='relu'))

model = k.models.load_model("ttt")
# model.compile(loss='mse', optimizer="sgd", metrics=["accuracy"])




print(datetime.now() - start_time)
print("начало")
ii = 1
oo = 0
for i in range(ii,117-1):
    img = cv2.imread("kOn/"+str(i+1+20)+".jpg")

    # u = np.resizeMas(nrevracenDla(razdel64(img),64,64),(12289,))
    # g = model.predict(u)
    # v = nrevracenS(g,64,64)
    # cv2.imwrite(os.path.join('w.png'), v)

    N = 20000

    s = trasformerirovatFotoXAUTOMAX(img,N)
    for o in range(oo,s):
        start_time = datetime.now()
        print("первий",i+1)
        print("второй",str(o+1)+"/"+str(s))
        p = trasformerirovatFotoX(img,N,o*N)[0]
        x = nrevracenDlaMas(p,64,64)
        model.fit(x=x, y=x, epochs=1, validation_split=0.2)
        # print("177/250 [====================>.........] - ETA: 46s - loss: 23824.1250 - accuracy: 3.5311e-04")
        model.save("ttt")
        print("повтор")
        print(datetime.now() - start_time)
        print()





# img1 = cv2.imread("2.png")
# razdel64(img1)
# v = np.zeros((3,3))
# if v[0]==v[1]:
#     print("d")

# y = np.random.random((10,3))
#
# print(v[0:5,0:5])

# u = v + y
# print(u)




# img1 = cv2.imread("2.png")
#
# print(rasformerirovatFoto(img1,100,5000))





#
# p= imgM.shape[0]
# for i in range(p):
#     if i%100==0:
#         print(i)
#     cv2.imwrite(os.path.join("s/"+str(i)+'w.png'), imgM[i])


# img1 = cv2.imread("2.png")
# mas, W,H = i(img1)
# img1 = i2(mas, H,W)



# cv2.imwrite(os.path.join('waka2.png'), img1)

# img1 = cv2.imread("2.png")
# img2 = cv2.imread("3.png")
# img3 = cv2.imread("4.png")
#
#
#
# f = img1-img3



# w = 1000
# h = 1000
#
# f = np.zeros((w,h,3),dtype=np.int32)
#
# for x in range(w):
#     for y in range(h):
#         f[x, y] = np.array([random.randint(0,20), (x*10)//100, (y*10)//100])*random.randint(2,5)
#
# for x in range(w):
#     for y in range(h):
#         f[x, y] += np.array([(255-y)//50, ((255-x)*10)//100, 255//2])


# for x in range(w):
#     for y in range(h):
#         for z in range(3):
#             if f[x, y, z] == 255:
#                 f[x, y, z] = 255

# cv2.imwrite(os.path.join('waka2.png'), f)









# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# import matplotlib.pyplot as plt
# from tensorflow import keras as k
# from keras.layers import Dense





# print("1")
# c = np.random.random((500,))*2000
# f = np.array(c*1.8+32)
# # print(c)
# # print(f)
# print("2")
# model = k.Sequential()
#
# model.add(Dense(units=1, input_shape=(1,), activation='linear'))
#
# model.compile(loss='mean_squared_error', optimizer=k.optimizers.Adam(0.01))
#
# print("3")
# history = model.fit(c, f, epochs=10000)
#
# print("4")
#
# res = 100
# print(model.predict([res]))
# print(res*1.8+32)
# print(model.get_weights())
#
#
# plt.plot(history.history['loss'][300:500])
# plt.grid(True)
# plt.show()














# c = np.array([-40, -10, 0, 8, 15, 22, 38])
# f = np.array([-40, 14, 32, 46, 59, 72, 100])
#
# model = k.Sequential()
# model.add(Dense(units=1, input_shape=(1,), activation='linear'))
# model.compile(loss='mean_squared_error', optimizer=k.optimizers.Adam(0.1))
#
# history = model.fit(c, f, epochs=500, verbose=0)
# print("Обучение завершено")
#
# print(model.predict([100]))
# print(model.get_weights())
#
# plt.plot(history.history['loss'])
# plt.grid(True)
# plt.show()











