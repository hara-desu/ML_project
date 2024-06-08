# -*- coding: utf-8 -*-

"""# 모델 불러오기"""
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# 서버 관리용 fastapi 의존 라이브러리
import uvicorn

# fast api 라이브러리
from fastapi import FastAPI

# 인터페이스 데이터 관리를 위한 라이브러리
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
origins = ["*"]

app = FastAPI(title="ML API")

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

with open("./mlcore.dump","rb") as fr:
    loadedRef = pickle.load(fr)

class InDataset(BaseModel):
    year : int
    region : int
    family_member : int
    gender : int
    year_born : int
    education_level : int
    marriage : int
    religion : int

loadedModel = load_model("./model_b_weights.h5")

@app.post("/predict", status_code=200)
async def predictDl(x:InDataset):
    try:
        # 화면입력데이터 변수 할당
        year = x.year
        region = x.region
        family_member = x.family_member
        gender = x.gender
        year_born = x.year_born
        education_level = x.education_level
        marriage = x.marriage
        religion = x.religion
        print("Predicting...")
        # 예측을 위한 데이터셋 생성
        testData = pd.DataFrame([[
            year,
            region,
            family_member,
            gender,
            year_born,
            education_level,
            marriage,
            religion
        ]])
        # 예측
        predictValue = loadedModel.predict(testData)
        for _ in predictValue[0]:
            if _ == np.argmax(predictValue):
                result = "0: Income less than average of 3441"
            else:
                result = "1: Income more than average of 3441"
                break
    except Exception as e:
        print(e)
        result = {"prediction":"00"}
    return result

@app.get("/")
async def root():
    return {"message":"server is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9999, log_level="debug",
                proxy_headers=True, reload=True)

