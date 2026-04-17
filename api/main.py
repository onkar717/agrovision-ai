"""
FastAPI Application for Pest Detection System

RESTful API endpoints for:
- Real-time sensor data retrieval
- Pest predictions
- NDVI analysis
- Irrigation control
- WebSocket for real-time updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from loguru import logger
import sqlite3
import json
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, setup_logging
from pest_model.inference import PestInferenceEngine
from ndvi_analysis.compute import NDVICompute
from ndvi_analysis.historical_comparison import NDVIComparison


# Load configuration
config = load_config()
setup_logging(config)

# Initialize FastAPI app
app = FastAPI(
    title="Pest Detection API",
    description="AI-powered pest prediction and vegetation health monitoring",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api'].get('cors_origins', ['*']),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for dashboard
app.mount("/static", StaticFiles(directory="dashboard"), name="static")

# Global instances
db_path = config['database'].get('sqlite_path', 'data/pest_detection.db')
try:
    inference_engine = PestInferenceEngine(
        model_path=config['model']['model_save_path'],
        db_path=db_path
    )
    ndvi_computer = NDVICompute()
    ndvi_comparison = NDVIComparison()
    logger.info("Inference engine and NDVI modules initialized")
except Exception as e:
    logger.warning(f"Could not initialize inference engine: {e}. Some endpoints may not work.")
    inference_engine = None
    ndvi_computer = NDVICompute()
    ndvi_comparison = NDVIComparison()


# Pydantic Models
class SensorReading(BaseModel):
    device_id: str
    crop_type: str
    temperature: float
    humidity: float
    timestamp: Optional[float] = None

class IrrigationCommand(BaseModel):
    device_id: str
    crop_type: str
    command: str  # "activate" or "deactivate"
    duration_minutes: Optional[int] = None


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")

manager = ConnectionManager()


# Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard"""
    with open("dashboard/index.html", "r") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "inference_engine": "available" if inference_engine else "unavailable"
    }


@app.get("/api/sensors/{device_id}/latest")
async def get_latest_sensor_reading(device_id: str):
    """Get latest sensor reading for a device"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM sensor_readings
        WHERE device_id = ?
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (device_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        raise HTTPException(status_code=404, detail="Device not found or no data available")
    
    return dict(row)


@app.get("/api/sensors/{device_id}/history")
async def get_sensor_history(
    device_id: str,
    hours: int = Query(default=24, ge=1, le=168)
):
    """Get historical sensor readings"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cutoff_time = (datetime.now() - timedelta(hours=hours)).timestamp()
    
    cursor.execute('''
        SELECT * FROM sensor_readings
        WHERE device_id = ? AND timestamp >= ?
        ORDER BY timestamp ASC
    ''', (device_id, cutoff_time))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


@app.get("/api/sensors/all")
async def get_all_devices():
    """List all registered devices"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT DISTINCT device_id, crop_type,
               MAX(timestamp) as last_reading_time
        FROM sensor_readings
        GROUP BY device_id
        ORDER BY last_reading_time DESC
    ''')
    
    devices = []
    for row in cursor.fetchall():
        devices.append({
            'device_id': row[0],
            'crop_type': row[1],
            'last_reading_time': row[2],
            'last_reading_datetime': datetime.fromtimestamp(row[2]).isoformat()
        })
    
    conn.close()
    
    return devices


@app.post("/api/predict/pest")
async def predict_pest_risk(device_id: str, crop_type: Optional[str] = None, pest_type: Optional[str] = None):
    """Run pest prediction for a device"""
    if not inference_engine:
        raise HTTPException(status_code=503, detail="Inference engine not available. Train model first.")
    
    prediction = inference_engine.predict(
        device_id=device_id,
        crop_type=crop_type,
        pest_type=pest_type
    )
    
    if prediction is None:
        raise HTTPException(status_code=400, detail="Insufficient data for prediction")
    
    # Broadcast to WebSocket clients
    await manager.broadcast({
        'type': 'pest_prediction',
        'data': prediction
    })
    
    return prediction


@app.get("/api/alerts/{crop_type}")
async def get_active_alerts(crop_type: str):
    """Get active pest alerts for a crop type"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get predictions from last 24 hours with high risk
    cutoff_time = (datetime.now() - timedelta(hours=24)).timestamp()
    
    cursor.execute('''
        SELECT * FROM pest_predictions
        WHERE crop_type = ? AND prediction_timestamp >= ?
              AND risk_level IN ('High', 'Medium')
        ORDER BY growth_probability DESC
    ''', (crop_type, cutoff_time))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


@app.get("/api/ndvi/{plot_id}/current")
async def get_current_ndvi(plot_id: str):
    """Get current NDVI for a plot"""
    # For demo, simulate healthy crop
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    readings = ndvi_computer.simulate_ndvi_timeseries(
        plot_id=plot_id,
        start_date=start_date,
        end_date=end_date,
        crop_health="healthy"
    )
    
    return {
        'plot_id': plot_id,
        'current_ndvi': readings[-1]['ndvi'],
        'timestamp': readings[-1]['timestamp'],
        'date': readings[-1]['date']
    }


@app.get("/api/ndvi/{plot_id}/compare")
async def compare_ndvi_to_historical(plot_id: str, days: int = Query(default=30, ge=7, le=90)):
    """Compare current NDVI to historical baseline"""
    # Generate current readings
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    current_readings = ndvi_computer.simulate_ndvi_timeseries(
        plot_id=plot_id,
        start_date=start_date,
        end_date=end_date,
        crop_health="healthy"
    )
    
    # Compare to historical
    comparison = ndvi_comparison.compare_current_to_historical(
        plot_id=plot_id,
        current_readings=current_readings,
        historical_years=3
    )
    
    return comparison


@app.post("/api/irrigation/{device_id}/activate")
async def activate_irrigation(command: IrrigationCommand):
    """Activate irrigation valve"""
    # In production, this would publish MQTT command
    # For demo, we'll just return success
    
    message = {
        'device_id': command.device_id,
        'crop_type': command.crop_type,
        'command': 'activate',
        'duration_minutes': command.duration_minutes or 30,
        'timestamp': datetime.now().isoformat(),
        'status': 'commanded'
    }
    
    logger.info(f"Irrigation activation commanded for {command.device_id}")
    
    return message


@app.post("/api/irrigation/{device_id}/deactivate")
async def deactivate_irrigation(device_id: str, crop_type: str):
    """Deactivate irrigation valve"""
    message = {
        'device_id': device_id,
        'crop_type': crop_type,
        'command': 'deactivate',
        'timestamp': datetime.now().isoformat(),
        'status': 'commanded'
    }
    
    logger.info(f"Irrigation deactivation commanded for {device_id}")
    
    return message


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket endpoint for real-time sensor data"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for demo
            await websocket.send_json({'status': 'connected', 'message': 'Real-time telemetry active'})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Background task to simulate live data updates (for demo)
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI server started")
    logger.info(f"API running at http://localhost:{config['api']['port']}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port'],
        log_level="info"
    )
