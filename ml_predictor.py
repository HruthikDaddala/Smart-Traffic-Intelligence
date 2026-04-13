import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

class TrafficPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.is_trained = False

    def prepare_data(self, historical_data):
        # historical_data should be a list of dicts: [{"timestamp": dt, "count": int}, ...]
        if not historical_data or len(historical_data) < 10:
            return None, None

        df = pd.DataFrame(historical_data)
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        
        # Features: hour, minute, second
        X = df[['hour', 'minute', 'second']]
        y = df['total_vehicles']
        return X, y

    def train(self, historical_data):
        X, y = self.prepare_data(historical_data)
        if X is not None:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        return False

    def predict_next_5_minutes(self):
        if not self.is_trained:
            # Return some mock data if not enough history
            return [20, 25, 30, 28, 26]

        now = datetime.now()
        predictions = []
        for i in range(1, 6):
            future_time = now + timedelta(minutes=i)
            feat = [[future_time.hour, future_time.minute, future_time.second]]
            pred = self.model.predict(feat)[0]
            predictions.append(int(pred))
        return predictions
