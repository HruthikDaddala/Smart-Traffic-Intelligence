from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from ai_engine import TrafficAI
from ml_predictor import TrafficPredictor
import threading
import time

# --- CONFIGURATION ---
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./traffic_system.db"
Base = declarative_base()

class TrafficRecord(Base):
    __tablename__ = "traffic_records"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_vehicles = Column(Integer)
    lane1_count = Column(Integer)
    lane2_count = Column(Integer)
    lane3_count = Column(Integer)
    emergency_detected = Column(String) # "Yes" or "No"

# Create engine and tables
from sqlalchemy import create_engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# --- MODELS & ENGINES ---
ai_engine = TrafficAI()
ml_predictor = TrafficPredictor()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- SCHEMAS ---
class TrafficData(BaseModel):
    timestamp: str
    total_vehicles: int
    lane1_count: int
    lane2_count: int
    lane3_count: int
    traffic_level: str
    signal_times: dict
    emergency: bool

# --- ROUTES ---
@app.get("/")
async def root():
    return {"message": "Smart Traffic Management System API"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # If video, start background processing
    if file.filename.endswith(('.mp4', '.avi', '.mov')):
        threading.Thread(target=process_video_background, args=(file_path,)).start()
        
    return {"filename": file.filename, "path": file_path, "status": "Uploaded and processing started"}

@app.get("/traffic-data")
async def get_traffic_data(db: Session = Depends(get_db)):
    records = db.query(TrafficRecord).order_by(TrafficRecord.timestamp.desc()).limit(100).all()
    return records

@app.get("/prediction")
async def get_prediction(db: Session = Depends(get_db)):
    # Fetch historical data for training
    historical = db.query(TrafficRecord).order_by(TrafficRecord.timestamp.desc()).limit(50).all()
    data = [{"timestamp": r.timestamp, "total_vehicles": r.total_vehicles} for r in historical]
    
    if len(data) > 10:
        ml_predictor.train(data)
    
    preds = ml_predictor.predict_next_5_minutes()
    return {"predictions": preds}

@app.get("/signal-decision")
async def get_signal_decision(db: Session = Depends(get_db)):
    latest = db.query(TrafficRecord).order_by(TrafficRecord.timestamp.desc()).first()
    if not latest:
        return {"lane1": 30, "lane2": 30, "lane3": 30, "emergency": False}
    
    counts = [latest.lane1_count, latest.lane2_count, latest.lane3_count]
    times = calculate_signal_times(counts)
    
    # Emergency Override
    if latest.emergency_detected == "Yes":
        # Give immediate green (say 120s) to the lane with emergency
        # For simplicity, we'll just set all to high priority if emergency detected
        return {"lane1": 120, "lane2": 120, "lane3": 120, "emergency": True, "message": "Emergency Vehicle Detected - Priority Given"}
        
    return {**times, "emergency": False}

def process_video_background(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    db = SessionLocal()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 1 second (approx)
        if count % int(fps) == 0:
            results = ai_engine.process_frame(frame)
            
            new_record = TrafficRecord(
                total_vehicles=results["total_count"],
                lane1_count=results["lane_counts"][0],
                lane2_count=results["lane_counts"][1],
                lane3_count=results["lane_counts"][2],
                emergency_detected="Yes" if results["emergency_detected"] else "No"
            )
            db.add(new_record)
            db.commit()
            
        count += 1
    
    cap.release()
    db.close()

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Placeholder for signal logic
def calculate_signal_times(counts):
    # Logic: High traffic -> longer green time
    # Low: 20s, Medium: 40s, High: 60s
    times = {}
    for i, count in enumerate(counts):
        if count < 15:
            times[f"lane{i+1}"] = 20
        elif count < 40:
            times[f"lane{i+1}"] = 40
        else:
            times[f"lane{i+1}"] = 60
    return times

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
