# 🚦 Smart Traffic Management System (AI + YOLO)

## 📌 Overview

Smart Traffic Management System is a **full-stack AI-based application** that analyzes traffic from images and videos in real time. The system uses **YOLOv8** for vehicle detection, performs **lane-wise analysis**, predicts future traffic using machine learning, and dynamically controls traffic signals.

The goal is to reduce congestion, improve traffic flow, and support intelligent transportation systems.

---

## 🚀 Tech Stack

### Frontend

* React.js
* Vite
* Tailwind CSS
* Axios

### Backend

* Python (FastAPI / Flask)

### AI / Computer Vision

* YOLOv8 (Ultralytics)
* OpenCV

### Machine Learning

* scikit-learn (Random Forest)
* TensorFlow (LSTM - optional)

### Database

* SQLite / MongoDB

---

## 🎯 Core Features

### 1. Input System

* Upload traffic images (JPG, PNG)
* Upload traffic videos (MP4)
* Display uploaded media

---

### 2. Vehicle Detection (YOLO)

* Detect:

  * Car 🚗
  * Bike 🏍️
  * Bus 🚌
  * Truck 🚛
* Bounding boxes + confidence score

---

### 3. Lane-wise Detection

* Divide frame into 3 lanes:

  * Left
  * Center
  * Right
* Assign vehicles to lanes
* Display lane-wise counts

---

### 4. Vehicle Counting

* Total vehicle count
* Lane-wise vehicle count
* Real-time updates

---

### 5. Timestamp Tracking

* Track traffic per second
* Store:

  * Time
  * Vehicle count
  * Lane data

Example:
10:01 → 20 vehicles
10:02 → 35 vehicles

---

### 6. Traffic Level Classification

* Low → < 15
* Medium → 15–40
* High → > 40

---

### 7. Traffic Signal Decision

* High traffic → longer green
* Low traffic → shorter green

Example:

* Lane 1 → 60 sec
* Lane 2 → 20 sec

---

### 8. Emergency Vehicle Detection 🚑

* Detect ambulance
* Override signals
* Give instant priority

---

### 9. AI Traffic Prediction

* Predict next 5 minutes traffic
* Models:

  * Random Forest
  * LSTM (optional)

---

### 10. Smart Signal Control

* Adjust signal timing based on prediction

---

### 11. Peak Traffic Detection

* Detect highest traffic time

Example:
Peak → 10:05 → 50 vehicles

---

### 12. Dashboard

* Live video/image
* Detection output
* Lane-wise counts
* Traffic level
* Signal timing
* Prediction graphs

---

### 13. Data Storage

Store:

* Traffic timestamps
* Vehicle counts
* Predictions
* Peak data

---

## 🖥️ Frontend Pages

* Landing Page
* Upload Page
* Live Detection Page
* Dashboard Page
* Analytics Page

---

## 🔗 Backend API Routes

* `POST /upload`
* `POST /detect`
* `GET /traffic-data`
* `GET /prediction`
* `GET /signal-decision`

---

## ⚙️ System Flow

```
Input (Image/Video)
        ↓
YOLO Detection
        ↓
Lane Segmentation
        ↓
Vehicle Counting
        ↓
Timestamp Storage
        ↓
Emergency Detection
        ↓
Traffic Prediction
        ↓
Signal Decision
        ↓
Dashboard Output
```

---

## 🛠️ Installation & Setup

### 🔹 Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend URL: http://localhost:5000

---

### 🔹 Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: http://localhost:5173

---

## 📂 Project Structure

```
/frontend
/backend
/models
/data
/uploads
```

---

## 📈 Future Enhancements

* Multi-camera support
* Cloud deployment
* IoT traffic signal integration
* Advanced deep learning models
* Smart city integration

---

## 👨‍💻 Author

**Daddala Hruthik**
B.Tech Information Technology

---

## 📜 License

This project is for academic and research purposes.

