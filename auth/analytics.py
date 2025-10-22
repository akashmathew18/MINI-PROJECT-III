import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any

class AnalyticsManager:
    def __init__(self, analytics_file: str = "auth/analytics.json"):
        self.analytics_file = analytics_file
        self.ensure_file()
        self.load_data()
    
    def ensure_file(self):
        """Create analytics file if it doesn't exist"""
        os.makedirs("auth", exist_ok=True)
        if not os.path.exists(self.analytics_file):
            with open(self.analytics_file, 'w') as f:
                json.dump({
                    "script_analyses": [],
                    "ml_trainings": [],
                    "genre_predictions": [],
                    "user_activities": []
                }, f)
    
    def load_data(self):
        """Load analytics data"""
        with open(self.analytics_file, 'r') as f:
            self.data = json.load(f)
    
    def save_data(self):
        """Save analytics data"""
        with open(self.analytics_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def log_script_analysis(self, username: str, filename: str, file_size: int, 
                           characters_count: int, locations_count: int, genre: str, 
                           summary_length: int, processing_time: float):
        """Log script analysis activity"""
        activity = {
            "username": username,
            "filename": filename,
            "file_size": file_size,
            "characters_count": characters_count,
            "locations_count": locations_count,
            "genre": genre,
            "summary_length": summary_length,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        self.data["script_analyses"].append(activity)
        self.save_data()
    
    def log_ml_training(self, username: str, dataset_size: int, epochs: int, 
                       model_type: str, training_time: float, accuracy: float = None):
        """Log ML training activity"""
        activity = {
            "username": username,
            "dataset_size": dataset_size,
            "epochs": epochs,
            "model_type": model_type,
            "training_time": training_time,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat()
        }
        self.data["ml_trainings"].append(activity)
        self.save_data()
    
    def log_genre_prediction(self, username: str, text_length: int, predicted_genre: str, 
                           processing_time: float, model_used: str = "keyword"):
        """Log genre prediction activity"""
        activity = {
            "username": username,
            "text_length": text_length,
            "predicted_genre": predicted_genre,
            "processing_time": processing_time,
            "model_used": model_used,
            "timestamp": datetime.now().isoformat()
        }
        self.data["genre_predictions"].append(activity)
        self.save_data()
    
    def log_user_activity(self, username: str, activity_type: str, details: str = ""):
        """Log general user activity"""
        activity = {
            "username": username,
            "activity_type": activity_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.data["user_activities"].append(activity)
        self.save_data()
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get key metrics for dashboard"""
        now = datetime.now()
        today = now.date()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        # Filter data by time periods
        recent_analyses = [a for a in self.data["script_analyses"] 
                          if datetime.fromisoformat(a["timestamp"]) > week_ago]
        recent_trainings = [a for a in self.data["ml_trainings"] 
                           if datetime.fromisoformat(a["timestamp"]) > week_ago]
        recent_predictions = [a for a in self.data["genre_predictions"] 
                             if datetime.fromisoformat(a["timestamp"]) > week_ago]
        
        # Calculate metrics
        total_analyses = len(self.data["script_analyses"])
        total_trainings = len(self.data["ml_trainings"])
        total_predictions = len(self.data["genre_predictions"])
        
        analyses_this_week = len(recent_analyses)
        trainings_this_week = len(recent_trainings)
        predictions_this_week = len(recent_predictions)
        
        # Average processing times
        avg_analysis_time = sum(a["processing_time"] for a in recent_analyses) / max(len(recent_analyses), 1)
        avg_training_time = sum(a["training_time"] for a in recent_trainings) / max(len(recent_trainings), 1)
        avg_prediction_time = sum(a["processing_time"] for a in recent_predictions) / max(len(recent_predictions), 1)
        
        # Genre distribution
        genre_counts = {}
        for analysis in self.data["script_analyses"]:
            genre = analysis["genre"]
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # User activity
        unique_users = set()
        for activity_list in [self.data["script_analyses"], self.data["ml_trainings"], 
                             self.data["genre_predictions"]]:
            for activity in activity_list:
                unique_users.add(activity["username"])
        
        return {
            "total_analyses": total_analyses,
            "total_trainings": total_trainings,
            "total_predictions": total_predictions,
            "analyses_this_week": analyses_this_week,
            "trainings_this_week": trainings_this_week,
            "predictions_this_week": predictions_this_week,
            "avg_analysis_time": round(avg_analysis_time, 2),
            "avg_training_time": round(avg_training_time, 2),
            "avg_prediction_time": round(avg_prediction_time, 2),
            "genre_distribution": genre_counts,
            "unique_users": len(unique_users),
            "recent_activities": self.get_recent_activities(limit=10)
        }
    
    def get_recent_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent activities across all types"""
        all_activities = []
        
        for activity in self.data["script_analyses"]:
            all_activities.append({
                "type": "Script Analysis",
                "username": activity["username"],
                "details": f"Analyzed {activity['filename']} ({activity['genre']})",
                "timestamp": activity["timestamp"]
            })
        
        for activity in self.data["ml_trainings"]:
            all_activities.append({
                "type": "ML Training",
                "username": activity["username"],
                "details": f"Trained {activity['model_type']} model ({activity['epochs']} epochs)",
                "timestamp": activity["timestamp"]
            })
        
        for activity in self.data["genre_predictions"]:
            all_activities.append({
                "type": "Genre Prediction",
                "username": activity["username"],
                "details": f"Predicted {activity['predicted_genre']} genre",
                "timestamp": activity["timestamp"]
            })
        
        # Sort by timestamp and return most recent
        all_activities.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_activities[:limit]
    
    def get_user_analytics(self, username: str) -> Dict[str, Any]:
        """Get analytics for specific user"""
        user_analyses = [a for a in self.data["script_analyses"] if a["username"] == username]
        user_trainings = [a for a in self.data["ml_trainings"] if a["username"] == username]
        user_predictions = [a for a in self.data["genre_predictions"] if a["username"] == username]
        
        return {
            "script_analyses": len(user_analyses),
            "ml_trainings": len(user_trainings),
            "genre_predictions": len(user_predictions),
            "total_files_analyzed": len(set(a["filename"] for a in user_analyses)),
            "favorite_genre": max(set(a["genre"] for a in user_analyses), key=user_analyses.count) if user_analyses else "None",
            "total_processing_time": sum(a["processing_time"] for a in user_analyses + user_predictions),
            "recent_activities": [a for a in self.get_recent_activities(limit=20) if a["username"] == username]
        }

# Initialize global analytics manager
analytics_manager = AnalyticsManager()
