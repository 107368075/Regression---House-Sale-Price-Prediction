# Regression---House-Sale-Price-Prediction

作業要求
========

    使用回歸模型做房價預測:
    1.用train.csv跟valid.csv訓練模型（一行是一筆房屋交易資料的紀錄，包括id, price與19種房屋參數）
    2.將test.csv中的每一筆房屋參數，輸入訓練好的模型，預測其房價
    3.將預測結果上傳（從“Submit Predictions”連結）
    4.看系統幫你算出來的Mean Abslute Error（MAE，就是跟實際房價差多少，取絕對值）分數夠不夠好？
    5.嘗試改進預測模型

工作環境
========

    1.Ubuntu
    2.Python3.5
    3.Tensorflow
    4.Keras
    
程式流程
========

    1.宣告和定義  
    
    import pandas as pd
    import numpy as np
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.callbacks import ModelCheckpoint
    from sklearn.preprocessing import scale

    2.讀檔(將不必要的資料刪除)
    
    d1= pd.read_csv('/home/scarlet/ML/train-v3.csv')
    X_train = d1.drop(['price','id','sale_day'],axis=1).values
    Y_train = d1['price'].values
    #data_1 = pd.read_csv(d1)
    #dataset_1 = data_1.values
    #X_train = dataset_1[:,2:22]
    #Y_train = dataset_1[:,1]

    3.確認model有訓練到

    d2 = pd.read_csv('/home/scarlet/ML/valid-v3.csv')
    X_valid = d2.drop(['price','id','sale_day'],axis=1).values
    Y_valid = d2['price'].values
    #data_2 = pd.read_csv(d2)
    #dataset_2 = data_2.values
    #X_valid = dataset_2[:,2:22]
    #Y_valid = dataset_2[:,1]
    
    4.test資料丟進model，得price

    d3 = pd.read_csv('/home/scarlet/ML/test-v3.csv')
    X_test = d3.drop(['id','sale_day'],axis=1).values
    #data_3 = pd.read_csv(d3)
    #dataset_3 = data_3.values
    #X_test = dataset_3[:,1:21]
    
    5.正規化

    def normalize(train, vaild, test):
        tmp = train
        mean, std = tmp.mean(axis=0), tmp.std(axis=0)
        train = (train - mean)/std
        vaild = (vaild - mean)/std
        test = (test - mean)/std
        return train,vaild,test
    X_train,X_valid,X_test = normalize(X_train,X_valid,X_test)
    
    6.建立model

    model = Sequential()
    model.add(Dense(32,input_dim=20,kernel_initializer='normal',activation='relu'))
    model.add(Dense(64,kernel_initializer='normal',activation='relu'))
    model.add(Dense(128,kernel_initializer='normal',activation='relu'))
    model.add(Dense(128,kernel_initializer='normal',activation='relu'))
    model.add(Dense(20,kernel_initializer='normal',activation='relu'))
    model.add(Dense(32,kernel_initializer='normal',activation='relu'))
    model.add(Dense(128,kernel_initializer='normal',activation='relu'))
    model.add(Dense(20,kernel_initializer='normal',activation='linear'))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss = 'MAE', optimizer='adam')
    
    8.將最好的model保留(這裡留valid loss最小的)

    checkpoint = ModelCheckpoint(filepath='weight.best.hdf5', verbose=1, save_best_only=True)
    model.fit(X_train, Y_train,batch_size=10, nb_epoch=200, verbose=0, validation_data=(X_valid,Y_valid), callbacks=[checkpoint])
    
    9.存檔

    Y_predict = model.predict(X_test)
    np.savetxt('/home/scarlet/ML/result1.csv',Y_predict,delimiter=',')

程式Running過程
===============
    
    暫時無法顯示

Kaggle排名
==========

    暫時無法顯示
