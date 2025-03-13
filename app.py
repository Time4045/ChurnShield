from fastapi import UploadFile, File, FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import catboost
import time
import json
import python_multipart
from sklearn.preprocessing import LabelEncoder

with open('catboost_model.pkl', 'rb') as fl:
    loaded_model = pickle.load(fl)

app = FastAPI()

class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    # Проверяем, что файл имеет правильный тип


    try:
        # Считываем JSON-файл в DataFrame
        df = pd.read_json(file.file, orient='records')
    except Exception as e:
        return {"error": f"Failed to process JSON file: {str(e)}"}

    # Сохрание id и кодирование категориальных фичей
    cust_id = df['customerID']
    df.drop('customerID', axis=1, inplace=True)
    for col in df.select_dtypes('object'):
        df[col] = LabelEncoder().fit_transform(df[col])

    #Предсказание и score отношения к каждому из классу
    predictions = loaded_model.predict(df)
    probabilities = loaded_model.predict_proba(df)
    result = []
    for i in range(len(predictions)):
        #Формирование результата
        result.append({
            'customerID': cust_id[i],
            'prediction': 'high_risk' if predictions[i] == 1 else 'low_risk',
            'churn_probability': float(max(probabilities[i]))
        })

    #Сохранение в JSON файл
    try:
        timestamp = int(time.time())
        filename = f"prediction_results_{timestamp}.json"

        # Сохраняем результат в файл
        with open(filename, "w") as outfile:
            json.dump({"results": result}, outfile, indent=4)

        return {
            "status": "success",
            "message": f"Predictions saved to {filename}",
            "results": result  # Оставляем возврат данных для API
        }
    except Exception as e:
        return {"error": f"Failed to save results: {str(e)}"}


