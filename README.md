# ChurnShield: Predict and Prevent Customer Loss 🛡️🚀

## Описание проекта  
ChurnShield — это ML-сервис, который предсказывает отток клиентов (churn) на основе их данных.  
**Цель:** Помочь компаниям выявлять клиентов с высоким риском ухода и сохранять их.  
**Стек:** FastAPI, CatBoost, Docker, Pandas.  

---

## Фичи 📊  
Модель анализирует следующие признаки клиентов:  

### **Демография** 👥  
- `customerID` — уникальный идентификатор клиента 🆔  
- `gender` — пол (Male/Female) 👱♀️👱♂️  
- `SeniorCitizen` — является ли клиент пенсионером (0/1) 👴  

### **Услуги и контракты** 📞💼  
- `PhoneService` — подключена ли телефонная связь (0/1) 📟  
- `MultipleLines` — наличие нескольких телефонных линий (0/1/2) 📞×2  
- `InternetService` — тип интернет-услуг (DSL, Fiber optic, No) 🌐  
- `Contract` — срок контракта (Month-to-month, One year, Two year) 📝  
- `PaperlessBilling` — безбумажный биллинг (0/1) 📄🚫  
- `PaymentMethod` — способ оплаты (Electronic check, Mailed check, Bank transfer, Credit card) 💳  

### **Дополнительные услуги** 🛠️📺  
- `OnlineSecurity` — онлайн-безопасность (0/1) 🔒  
- `OnlineBackup` — онлайн-бэкап данных (0/1) 💾  
- `DeviceProtection` — защита устройства (0/1) 📱🛡️  
- `TechSupport` — техническая поддержка (0/1) 🛠️  
- `StreamingTV` — стриминговое ТВ (0/1) 📺  
- `StreamingMovies` — стриминг фильмов (0/1) 🎬  

### **Финансы** 💸  
- `MonthlyCharges` — ежемесячные платежи (например, 79.85) 💵  
- `TotalCharges` — общая сумма платежей за всё время (например, 930.25) 💰  

### **Семейное положение** 👨👩👧👦  
- `Partner` — наличие партнера/супруга (0/1) 💍  
- `Dependents` — наличие иждивенцев (0/1) 👶  

### **Лояльность** ⏳  
- `tenure` — срок пользования услугами (в месяцах, например, 12) ⏳  

---

### **Примечания**  
- **Кодирование категорий:**  
  Категориальные признаки (например, `InternetService`, `Contract`) автоматически кодируются в числа с помощью `LabelEncoder`.  
- **Пропущенные значения:**  
  Модель ожидает, что все данные заполнены (без NaN).  

---

## REST API 🌐  
### **Эндпоинт:** `/predict_file`  
**Метод:** POST  
**Описание:** Загружает JSON-файл с данными клиентов, выполняет предсказание и сохраняет результат.  

#### Пример запроса (curl):  
```bash
curl -X POST "http://localhost:8000/predict_file" \
-H "Content-Type: multipart/form-data" \
-F "file=@input.json"
```

#### Пример ответа:  
```json
{
  "status": "success",
  "message": "Predictions saved to prediction_results_1620000000.json",
  "results": [
    {
      "customerID": "12345",
      "prediction": "high_risk",
      "churn_probability": 0.85
    }
  ]
}
```

---

## Как запустить локально 🖥️  
### 1. Установите зависимости  
```bash
pip install -r requirements.txt
```

### 2. Запустите сервер  
```bash
uvicorn app:app --reload
```

### 3. Или используйте Docker 🐳  
```bash
# Сборка образа
docker build -t churnshield .

# Запуск контейнера
docker run -d -p 8000:8000 -v $(pwd)/output:/app churnshield
```

---

## Пример JSON для тестирования 📁  
Используйте уже готовый `input.json` в корне проекта или же создайте файл `data.json` вида:  
```json
[
  {
    "customerID": "12345",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "MultipleLines": 0,
    "InternetService": 1,
    "OnlineSecurity": 0,
    "TechSupport": 0,
    "Contract": 1,
    "PaperlessBilling": 1,
    "PaymentMethod": 2,
    "MonthlyCharges": 79.85,
    "TotalCharges": 930.25
  }
]
```

---

## Где сохраняются результаты? 💾  
- Результаты сохраняются в файл `prediction_results_*.json` внутри контейнера.  
- Чтобы получить их на локальной машине, используйте Docker volume:  
  ```bash
  docker run -v $(pwd)/output:/app ...
  ```
  Файлы появятся в папке `output` на вашем компьютере.  

---
## Лицензия 📜

MIT License
---

**Note:** Для работы нужна предобученная CatBoost-модель (`catboost_model.pkl`) в корне проекта.
