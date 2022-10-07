from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

import keras
from numpy import load

# import sklearn
from keras import layers

import numpy as np

from keras.layers import Dense

from sklearn import metrics

from sklearn import preprocessing
from scipy.signal import resample
#%% defination varibles

Hangover = [0 for k in range(30)]
When_Harry_Met_Sally = [0 for k in range(30)]

European_Travel_Skills = [0 for k in range(30)]
Matcha_The_way_of_Tea = [0 for k in range(30)]

Relaxing_Music_with_Beach = [0 for k in range(30)]
Natural_World_Zambezi = [0 for k in range(30)]

Shutter = [0 for k in range(30)]
Mama = [0 for k in range(30)]


trainset_ = [[] for k in range(30)]
testset_ = [[] for k in range(30)]

X = []
y = []

#%% load dataset

our_Features = load('subFeatures.npy',allow_pickle=True)
valence_arousal = load('data.npy')

#%% spilit movies

for i in range(30):

    Hangover[i] = [j[4461:8161] for j in valence_arousal[i][:]]
    When_Harry_Met_Sally[i] = [j[526:699] for j in valence_arousal[i][:]]
    
    European_Travel_Skills[i] = [j[16425:18805] for j in valence_arousal[i][:]]
    Matcha_The_way_of_Tea[i] = [j[1058:1218] for j in valence_arousal[i][:]]
    
    Relaxing_Music_with_Beach[i] = [j[26809:29709] for j in valence_arousal[i][:]]
    Natural_World_Zambezi[i] = [j[1605:1752] for j in valence_arousal[i][:]]
    
    Shutter[i] = [j[37453:41393] for j in valence_arousal[i][:]]
    Mama[i] = [j[2189:2333] for j in valence_arousal[i][:]]

for i in range(30):
    temp_train1 = []
    temp_train2 = []
    temp_train3 = []
    temp_train4 = []
    for j in range(4):
        aa = our_Features[i][j][0]
        temp_train1.append(resample(aa[221000:406000], 200000, axis = 0))
        temp_train2.append(resample(aa[819000:938000], 200000, axis = 0))
        temp_train3.append(resample(aa[1338000:1483000], 200000, axis = 0))
        temp_train4.append(resample(aa[1870000:2067000], 200000, axis = 0))
       
    trainset_[i].append(temp_train1)
    trainset_[i].append(temp_train2)
    trainset_[i].append(temp_train3)
    trainset_[i].append(temp_train4)

#%% Building Trainset

X = []
y = []
temp_sig = [[] for k in range(4)]
temp_y = []
len_movie = 0
for i in range(30):
    for j in range(4):
        len_movie = len(trainset_[i][j][0])
        if len_movie == 185000: # 3700:
            for k in range(4):
                temp_f = []
                for l in range(len_movie):
                    temp_f.append(trainset_[i][j][k][l]) 
                    temp_sig[k].append(trainset_[i][j][k][l])
                
                    
            for l in range(len_movie):
                temp_y.append(0)#Hangover[i][0][l]) #valence
                
                y.append(temp_y)
                
                X.append(temp_f)
        elif len_movie == 119000: #2380:
            for k in range(4):
                temp_f = []
                for l in range(len_movie):
                    
                    temp_f.append(trainset_[i][j][k][l]) 
                    temp_sig[k].append(trainset_[i][j][k][l])
    
            for l in range(len_movie):
                temp_y.append(1)#European_Travel_Skills[i][0][l])
                
                y.append(temp_y)
                
                X.append(temp_f)
        elif len_movie == 145000: #2900:
            for k in range(4):
                temp_f = []
                for l in range(len_movie):
                    
                    temp_f.append(trainset_[i][j][k][l]) 
                    temp_sig[k].append(trainset_[i][j][k][l])
    
            for l in range(len_movie):
                temp_y.append(2)#Relaxing_Music_with_Beach[i][0][l])
                
                y.append(temp_y)
                
                X.append(temp_f)
        elif len_movie == 197000: #3940:
            for k in range(4):
                temp_f = []
                for l in range(len_movie):
                    
                    temp_f.append(trainset_[i][j][k][l]) 
                    temp_sig[k].append(trainset_[i][j][k][l])
    
            for l in range(len_movie):
                temp_y.append(3)#Shutter[i][0][l])
                
                y.append(temp_y)
                
                X.append(temp_f)

X = np.array(temp_sig)
y = np.array(temp_y)
x_train = X.transpose()
y_train = y
y_train[np.where(np.isnan(y))] = 0 #[0,0]

#%% normalize

X_normalized = preprocessing.normalize(x_train)

#%%  define the keras model
model = keras.Sequential()
model.add(Dense(8, kernel_initializer='uniform', input_shape=(,), activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))#activation='sigmoid'))


opt = keras.optimizers.Adam(learning_rate=0.001)

# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])#, optimizer='sgd')#, metrics=['mse'])


#%% fit the keras model on the dataset
history = model.fit(X[0], y_train, epochs=20, batch_size=1000)

#%% evaluate the keras model

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


#%% make probability predictions with the model

predictions = model.predict(x_train)
predictions = np.argmax(predictions,axis=1)
metrics.accuracy_score(y_train, predictions)

#%% plot loss

aa1 = history.history['loss']










