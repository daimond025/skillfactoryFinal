import datetime
import os
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib

from lightautoml.tasks import Task
from lightautoml.automl.presets.tabular_presets import TabularAutoML

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from sklearn.metrics import  accuracy_score


dirname = os.path.dirname(__file__)
fileModel =  os.path.join( dirname , 'data/model/model.pkl')
USER_CLICK = 0
USER_BUY = 1

fileUser = os.path.join( dirname , 'data/user.pkl')
fileItems =  os.path.join( dirname ,'data/item.pkl')

data = pd.read_pickle(fileUser)
items = pd.read_pickle(fileItems)

userList  = list(data['userID'].unique())
itemList  = list(data['itemID'].unique())

def saveModel():
    data_log = pd.read_pickle(fileUser)
    dateEnd = datetime.datetime(year=2015, month=7, day=2, hour=0, minute=0, second=0)

    dataTrain = data_log[data_log['date'] < dateEnd]

    # резделим классы пропорционально
    trainBuyShape = dataTrain[dataTrain['rating'] == USER_BUY].copy()
    trainUserClick = dataTrain[dataTrain['rating'] == USER_CLICK].sample(n=trainBuyShape.shape[0], random_state=15, replace=True).copy()
    dataTrain = pd.concat([trainBuyShape, trainUserClick], ignore_index=True, axis=0).copy()

    task = Task('binary', metric='accuracy')
    roles = {'target': 'rating','drop': ['date']}

    automl = TabularAutoML(task=task,timeout=800, cpu_limit=4,
                           general_params={'use_algos': [['linear_l2', 'lgb', 'lgb_tuned', 'cb']]})

    automl.fit_predict(dataTrain, roles=roles)
    joblib.dump(automl, fileModel)

def loadModel():
    if not os.path.exists(fileModel):
        saveModel()
    return joblib.load(fileModel)

model  = loadModel()

def getItemsUser( iserId : int):

    dateNow = datetime.datetime.now()

    def get_quarter(month):
        if month in [1, 2, 3]:
            return 1
        elif month in [4, 5, 6]:
            return 2
        elif month in [7, 8, 9]:
            return 3
        else:
            return 4
    collect = pd.DataFrame()
    collect['itemID'] = itemList
    collect['userID'] = iserId
    collect['hour'] = dateNow.hour
    collect['dayofweek'] = dateNow.weekday()
    collect['quarter'] = get_quarter(dateNow.month)
    collect['month'] = dateNow.month
    collect['year'] = dateNow.year
    collect['dayofyear'] = dateNow.timetuple().tm_yday
    collect['dayofmonth'] = int(dateNow.strftime("%d"))

    y_pred = model.predict(collect)
    y_pred = y_pred.data[:, 0].copy()
    y_pred = [USER_BUY if x >= 0.5 else USER_CLICK for x in y_pred]

    collect["predict"] = y_pred
    collect = collect[collect['predict'] == USER_BUY].copy()
    offerItems = []
    if not collect.empty:
        offerItems = collect['itemID'].values

    return offerItems

def getMetrics():
    dateEnd = datetime.datetime(year=2015, month=7, day=2, hour=0, minute=0, second=0)
    dataTrain = data[data['date'] < dateEnd]
    dataTest = data[data['date'] >= dateEnd]

    trainBuyShape = dataTrain[dataTrain['rating'] == USER_BUY].copy()
    trainUserClick = dataTrain[dataTrain['rating'] == USER_CLICK].sample(n=trainBuyShape.shape[0], random_state=15, replace=True).copy()
    dataTrain = pd.concat([trainBuyShape, trainUserClick], ignore_index=True, axis=0).copy()

    y_pred = model.predict(dataTest)
    y_pred = y_pred.data[:, 0].copy()
    y_pred = [ USER_BUY if x >= 0.5 else USER_CLICK for x in y_pred ]

    X_test = dataTest.drop(['date', 'rating', ], axis=1).copy()
    y_test = dataTest['rating'].copy()

    collect = X_test
    collect['ratingPred'] = y_pred
    collect['rating'] = y_test

    acc =  accuracy_score(collect['rating'], collect['ratingPred'])

    user_inter = defaultdict(list)
    for userId in userList:
        useItem = collect[collect['userID'] == userId].copy()

        if not useItem.empty:
            for index, row in useItem.iterrows():
                user_inter[userId].append([int(row['itemID']), int(row['rating']), int(row['ratingPred'])])

    collect = pd.DataFrame()
    for uid, user_dict in user_inter.items():
        user_dict.sort(key=lambda x: x[2], reverse=True)
        n_rec_k = sum((est >= USER_BUY) for (_,  _, est) in user_dict[:3])
        n_rel_and_rec_k = sum(
            ((true_r >= USER_BUY) and (est >= USER_BUY))
            for (_,true_r, est, ) in user_dict[:3]
        )
        resitionBuy = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        add = {}
        add['userId'] = uid
        add['Precision@3'] = resitionBuy
        collect = collect.append(add, ignore_index=True)

    return  acc , collect['Precision@3'].mean()


# getItemsUser(iserId =257597 )
# getMetrics()

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}", response_class=JSONResponse)
def read_item(item_id: int):
    answer = {}
    answer['code'] = 1
    answer['error'] = ''
    answer['items'] = ''
    if not item_id in userList:
        answer['code'] = 0
        answer['error'] = "Not found UserId"
        return answer
    else:
        items = getItemsUser(iserId = item_id)

        if len(items) < 3:
            answer['items'] = items
        else:
            itemID =  [str(item) for item in random.choices(items, k=3)]
            answer['items'] = ','.join(itemID)
        return answer

@app.get("/metrics/")
def read_metrics():
    acc , presitiopn = getMetrics()
    return {"accuracy_score": acc,"presitiopn": presitiopn,}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)







