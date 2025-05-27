#!/usr/bin/env python3
"""
Enhanced ATLAS Backend Server with Complete Feature Set
Integrates all enhanced systems: consciousness monitoring, human features,
secure code execution, model switching, and advanced learning
"""

import os
import asyncio
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Enhanced ATLAS imports
try:
    from enhanced_atlas_system import EnhancedAtlasSystem, ModelType
    from enhanced_consciousness_monitor import EnhancedConsciousnessMonitor
    from enhanced_human_features import EnhancedHumanFeaturesSystem
    from secure_code_executor import SecureCodeExecutor, SecurityPolicy
except ImportError as e:
    print(f"Enhanced ATLAS import error: {e}")
    print("Some enhanced features may not be available")
    EnhancedAtlasSystem = None
    ModelType = None
    EnhancedConsciousnessMonitor = None
    EnhancedHumanFeaturesSystem = None
    SecureCodeExecutor = None

# Database
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv
from pathlib import Path
import psutil

# Load environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


# Enhanced Pydantic Models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    include_consciousness: bool = True
    context: Optional[Dict[str, Any]] = None
    preferred_model: Optional[str] = None
    auto_model_selection: Optional[bool] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    model_used: str
    consciousness_level: float
    consciousness_state: Optional[Dict[str, Any]] = None
    human_enhancements_active: Dict[str, bool]
    system_coherence: float
    response_time: float
    learning_session_id: Optional[str] = None
    timestamp: datetime


class EnhancedCodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"
    session_id: Optional[str] = None
    timeout: Optional[int] = Field(None, ge=1, le=60)
    custom_security_policy: Optional[Dict[str, Any]] = None


class EnhancedCodeExecutionResponse(BaseModel):
    output: str
    error: Optional[str] = None
    execution_time: float
    memory_usage: int
    exit_code: int
    timeout: bool
    session_id: str
    language: str
    security_violation: bool
    timestamp: datetime


class ModelSwitchRequest(BaseModel):
    model_type: str
    force_load: bool = False


class ConsciousnessQuery(BaseModel):
    include_history: bool = False
    include_dreams: bool = False
    history_limit: int = Field(50, ge=1, le=1000)


class LearningFeedback(BaseModel):
    session_id: str
    user_satisfaction: float = Field(..., ge=0.0, le=1.0)
    response_quality: float = Field(..., ge=0.0, le=1.0)
    helpfulness: float = Field(..., ge=0.0, le=1.0)
    feedback_text: Optional[str] = None


class SystemStatus(BaseModel):
    status: str
    initialization_complete: bool
    current_model: Optional[str]
    available_models: List[str]
    consciousness_monitor: Dict[str, Any]
    human_features: Dict[str, Any]
    learning_system: Dict[str, Any]
    active_sessions: int
    active_learning_sessions: int
    uptime: float
    system_metrics: Dict[str, float]


class StreamConfig(BaseModel):
    duration: int = Field(30, ge=1, le=300)
    include_consciousness: bool = True
    include_dreams: bool = False
    include_human_features: bool = True
    update_interval: float = Field(1.0, ge=0.1, le=10.0)


# Global instances
atlas_system: Optional[EnhancedAtlasSystem] = None
db_client: Optional[AsyncIOMotorClient] = None
database = None

# Active connections and sessions
active_sessions: Dict[str, Dict] = {}
websocket_connections: Dict[str, WebSocket] = {}
background_tasks_active: Dict[str, bool] = {}

# System status
system_start_time = datetime.now()
system_status = {
    "initialization_complete": False,
    "models_loaded": [],
    "consciousness_active": False,
    "human_features_active": False,
    "secure_execution_active": False
}


async def initialize_enhanced_atlas():
    """Initialize the enhanced ATLAS system"""
    global atlas_system, system_status
    
    try:
        logger.info("🚀 Initializing Enhanced ATLAS System...")
        
        if EnhancedAtlasSystem is None:
            logger.error("Enhanced ATLAS System not available")
            return
        
        # Configuration
        config = {
            "hidden_dim": 512,
            "i2c_units": 8,
            "consciousness_threshold": 0.3,
            "auto_model_switching": True,
            "enable_dreaming": True,
            "enable_multimodal": True,
            "learning_enabled": True,
            "max_context_length": 4096,
            "temperature": 0.7,
            "do_sample": True
        }
        
        # Initialize ATLAS system
        atlas_system = EnhancedAtlasSystem(config)
        
        # Load initial model (small one first)
        await atlas_system.initialize([ModelType.QWEN_SMALL])
        
        system_status["models_loaded"] = [model.value for model in atlas_system.initialized_models]
        system_status["consciousness_active"] = True
        system_status["human_features_active"] = True
        system_status["secure_execution_active"] = True
        system_status["initialization_complete"] = True
        
        logger.info("✅ Enhanced ATLAS System fully initialized!")
        
        # Start background tasks
        asyncio.create_task(consciousness_monitoring_task())
        asyncio.create_task(learning_analytics_task())
        
    except Exception as e:
        logger.error(f"❌ Error initializing Enhanced ATLAS: {str(e)}")
        # Don't raise to allow server to start for debugging


async def initialize_database():
    """Initialize MongoDB connection"""
    global db_client, database
    
    try:
        mongo_url = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
        db_name = os.getenv('DB_NAME', 'atlas_enhanced_database')
        
        db_client = AsyncIOMotorClient(mongo_url)
        database = db_client[db_name]
        
        # Test connection
        await db_client.admin.command('ping')
        logger.info("✅ Enhanced database connection established")
        
        # Create indexes
        await database.conversations.create_index("session_id")
        await database.conversations.create_index("user_id")
        await database.memories.create_index("timestamp")
        await database.consciousness_logs.create_index("timestamp")
        await database.code_executions.create_index("session_id")
        await database.learning_sessions.create_index("session_id")
        await database.user_feedback.create_index("session_id")
        
    except Exception as e:
        logger.error(f"❌ Enhanced database initialization error: {str(e)}")
        raise


async def consciousness_monitoring_task():
    """Background task for consciousness monitoring"""
    while True:
        try:
            if atlas_system and atlas_system.consciousness_monitor:
                # Log consciousness state
                current_state = atlas_system.consciousness_monitor.get_current_state()
                
                if database:
                    await database.consciousness_logs.insert_one({
                        "timestamp": datetime.now(),
                        "consciousness_level": current_state.get("consciousness_level", 0.0),
                        "dream_active": current_state.get("dream_active", False),
                        "lucid_state": current_state.get("lucid_state", False),
                        "system_coherence": current_state.get("consciousness_patterns", {})
                    })
                
            await asyncio.sleep(60)  # Log every minute
            
        except Exception as e:
            logger.error(f"Consciousness monitoring task error: {e}")
            await asyncio.sleep(60)


async def learning_analytics_task():
    """Background task for learning analytics"""
    while True:
        try:
            if atlas_system and atlas_system.learning_system:
                # Get learning analytics
                analytics = atlas_system.learning_system.get_learning_analytics()
                
                if database:
                    await database.learning_analytics.insert_one({
                        "timestamp": datetime.now(),
                        "analytics": analytics
                    })
                
            await asyncio.sleep(300)  # Log every 5 minutes
            
        except Exception as e:
            logger.error(f"Learning analytics task error: {e}")
            await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await initialize_database()
    # Initialize ATLAS in background to avoid blocking startup
    asyncio.create_task(initialize_enhanced_atlas())
    yield
    # Shutdown
    if db_client:
        db_client.close()
    
    # Cleanup background tasks
    for task_name, active in background_tasks_active.items():
        if active:
            logger.info(f"Stopping background task: {task_name}")


# Create FastAPI app
app = FastAPI(
    title="Enhanced ATLAS System API",
    description="Advanced Thinking and Learning AI System with Enhanced Consciousness Monitoring and Human-like Features",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Optional authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional authentication (can be disabled for development)"""
    # For now, return None (no authentication required)
    # Can be enhanced with JWT validation later
    return None


@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "enhanced_features": {
            "consciousness_monitoring": system_status["consciousness_active"],
            "human_features": system_status["human_features_active"],
            "secure_execution": system_status["secure_execution_active"],
            "model_switching": len(system_status["models_loaded"]) > 1
        }
    }


@app.get("/api/status", response_model=SystemStatus)
async def get_enhanced_system_status():
    """Get comprehensive enhanced system status"""
    
    uptime = (datetime.now() - system_start_time).total_seconds()
    
    # Get system metrics
    cpu_usage = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    
    # Try to get GPU usage
    gpu_usage = None
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = gpus[0].load * 100
    except:
        pass
    
    # Get ATLAS system status
    atlas_status = {}
    if atlas_system:
        atlas_status = atlas_system.get_system_status()
    
    return SystemStatus(
        status="running" if system_status["initialization_complete"] else "initializing",
        initialization_complete=system_status["initialization_complete"],
        current_model=atlas_status.get("current_model"),
        available_models=atlas_status.get("available_models", []),
        consciousness_monitor=atlas_status.get("consciousness_monitor", {}),
        human_features=atlas_status.get("human_features", {}),
        learning_system=atlas_status.get("learning_system", {}),
        active_sessions=len(active_sessions),
        active_learning_sessions=atlas_status.get("active_learning_sessions", 0),
        uptime=uptime,
        system_metrics={
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_usage": gpu_usage or 0.0,
            "disk_usage": psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0.0
        }
    )


@app.post("/api/chat", response_model=ChatResponse)
async def enhanced_chat_with_atlas(
    request: ChatMessage,
    current_user=Depends(get_current_user)
):
    """Enhanced chat with ATLAS system"""
    
    if not atlas_system or not system_status["initialization_complete"]:
        raise HTTPException(status_code=503, detail="Enhanced ATLAS system not ready")
    
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "default_user"
    
    try:
        # Handle model preference
        if request.preferred_model:
            try:
                preferred_model_type = ModelType(request.preferred_model)
                if preferred_model_type in atlas_system.initialized_models:
                    await atlas_system.switch_model(preferred_model_type)
            except ValueError:
                logger.warning(f"Invalid model type: {request.preferred_model}")
        
        # Get or create session
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                "created": datetime.now(),
                "user_id": user_id,
                "conversation_history": [],
                "learning_data": []
            }
        
        session = active_sessions[session_id]
        
        # Generate enhanced response
        response_data = await atlas_system.generate_response(
            message=request.message,
            session_id=session_id,
            user_id=user_id,
            context=request.context or {},
            include_consciousness=request.include_consciousness,
            auto_model_selection=request.auto_model_selection
        )
        
        # Store conversation in database
        conversation_entry = {
            "session_id": session_id,
            "user_id": user_id,
            "user_message": request.message,
            "atlas_response": response_data["response"],
            "model_used": response_data["model_used"],
            "consciousness_level": response_data["consciousness_level"],
            "consciousness_state": response_data.get("consciousness_state"),
            "human_enhancements": response_data.get("human_enhancements_active", {}),
            "system_coherence": response_data.get("system_coherence", 0.0),
            "response_time": response_data["response_time"],
            "timestamp": datetime.now()
        }
        
        if database:
            await database.conversations.insert_one(conversation_entry)
        
        # Update session
        session["conversation_history"].append(conversation_entry)
        
        return ChatResponse(
            response=response_data["response"],
            session_id=session_id,
            model_used=response_data["model_used"],
            consciousness_level=response_data["consciousness_level"],
            consciousness_state=response_data.get("consciousness_state"),
            human_enhancements_active=response_data.get("human_enhancements_active", {}),
            system_coherence=response_data.get("system_coherence", 0.0),
            response_time=response_data["response_time"],
            learning_session_id=response_data.get("learning_session_id"),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/api/code/execute", response_model=EnhancedCodeExecutionResponse)
async def execute_code_enhanced(
    request: EnhancedCodeExecutionRequest,
    current_user=Depends(get_current_user)
):
    """Execute code with enhanced security"""
    
    if not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Custom security policy if provided
        custom_policy = None
        if request.custom_security_policy:
            from secure_code_executor import SecurityPolicy
            custom_policy = SecurityPolicy(**request.custom_security_policy)
        
        # Execute code
        result = await atlas_system.execute_code(
            code=request.code,
            language=request.language,
            session_id=session_id,
            timeout=request.timeout
        )
        
        # Store execution in database
        if database:
            execution_entry = {
                "session_id": session_id,
                "code": request.code,
                "language": request.language,
                "output": result["output"],
                "error": result.get("error"),
                "execution_time": result["execution_time"],
                "memory_usage": result.get("memory_usage", 0),
                "security_violation": result.get("security_violation", False),
                "timestamp": datetime.now()
            }
            await database.code_executions.insert_one(execution_entry)
        
        return EnhancedCodeExecutionResponse(
            output=result["output"],
            error=result.get("error"),
            execution_time=result["execution_time"],
            memory_usage=result.get("memory_usage", 0),
            exit_code=result.get("exit_code", 0),
            timeout=result.get("timeout", False),
            session_id=session_id,
            language=result["language"],
            security_violation=result.get("security_violation", False),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Code execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Code execution error: {str(e)}")


@app.post("/api/model/switch")
async def switch_model(
    request: ModelSwitchRequest,
    current_user=Depends(get_current_user)
):
    """Switch between available models"""
    
    if not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    try:
        model_type = ModelType(request.model_type)
        success = await atlas_system.switch_model(model_type, request.force_load)
        
        if success:
            return {
                "message": f"Successfully switched to {model_type.value}",
                "current_model": model_type.value,
                "available_models": [m.value for m in atlas_system.initialized_models]
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to switch to {model_type.value}")
            
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model switch error: {str(e)}")


@app.get("/api/consciousness", response_model=Dict[str, Any])
async def get_consciousness_state(
    query: ConsciousnessQuery = Depends(),
    current_user=Depends(get_current_user)
):
    """Get detailed consciousness state and analytics"""
    
    if not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    try:
        result = atlas_system.get_consciousness_analytics()
        
        if not query.include_history:
            result.pop("consciousness_history", None)
        elif query.history_limit != 50:
            history = result.get("consciousness_history", [])
            result["consciousness_history"] = history[-query.history_limit:]
        
        if not query.include_dreams:
            result.pop("dream_states", None)
        
        return result
        
    except Exception as e:
        logger.error(f"Consciousness query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Consciousness query error: {str(e)}")


@app.get("/api/human-features")
async def get_human_features_state(current_user=Depends(get_current_user)):
    """Get human features analytics"""
    
    if not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    try:
        return atlas_system.get_human_features_analytics()
    except Exception as e:
        logger.error(f"Human features query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Human features query error: {str(e)}")


@app.post("/api/feedback")
async def submit_learning_feedback(
    feedback: LearningFeedback,
    current_user=Depends(get_current_user)
):
    """Submit user feedback for learning system"""
    
    if not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    try:
        # Store feedback in database
        if database:
            feedback_entry = {
                "session_id": feedback.session_id,
                "user_satisfaction": feedback.user_satisfaction,
                "response_quality": feedback.response_quality,
                "helpfulness": feedback.helpfulness,
                "feedback_text": feedback.feedback_text,
                "timestamp": datetime.now()
            }
            await database.user_feedback.insert_one(feedback_entry)
        
        # Update learning system if session is active
        if feedback.session_id in atlas_system.active_learning_sessions:
            learning_session_id = atlas_system.active_learning_sessions[feedback.session_id]
            # Calculate overall satisfaction
            overall_satisfaction = (feedback.user_satisfaction + feedback.response_quality + feedback.helpfulness) / 3
            
            # This would normally update the learning session, but we'll simulate it
            logger.info(f"Received feedback for session {feedback.session_id}: satisfaction={overall_satisfaction:.2f}")
        
        return {"message": "Feedback received successfully", "session_id": feedback.session_id}
        
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")


@app.get("/api/analytics/learning")
async def get_learning_analytics(current_user=Depends(get_current_user)):
    """Get learning system analytics"""
    
    if not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    try:
        return atlas_system.learning_system.get_learning_analytics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning analytics error: {str(e)}")


@app.get("/api/sessions")
async def get_active_sessions(current_user=Depends(get_current_user)):
    """Get list of active sessions with enhanced information"""
    
    sessions_info = []
    for session_id, session_data in active_sessions.items():
        sessions_info.append({
            "session_id": session_id,
            "user_id": session_data.get("user_id", "unknown"),
            "created": session_data["created"].isoformat(),
            "message_count": len(session_data.get("conversation_history", [])),
            "last_activity": session_data.get("conversation_history", [{}])[-1].get("timestamp", session_data["created"]).isoformat() if session_data.get("conversation_history") else session_data["created"].isoformat()
        })
    
    return {"active_sessions": sessions_info, "total": len(sessions_info)}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, current_user=Depends(get_current_user)):
    """Delete a session and end learning session"""
    
    if session_id in active_sessions:
        # End learning session if active
        if atlas_system and session_id in atlas_system.active_learning_sessions:
            await atlas_system.end_session(session_id)
        
        del active_sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


# Enhanced WebSocket endpoint for real-time streaming
@app.websocket("/api/ws/stream/{session_id}")
async def enhanced_websocket_stream(websocket: WebSocket, session_id: str):
    """Enhanced WebSocket endpoint for real-time ATLAS streaming"""
    await websocket.accept()
    websocket_connections[session_id] = websocket
    
    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message_type = message_data.get("type")
            
            if message_type == "start_stream":
                await start_enhanced_consciousness_stream(websocket, session_id, message_data)
            elif message_type == "chat":
                await handle_enhanced_websocket_chat(websocket, session_id, message_data)
            elif message_type == "get_consciousness":
                await send_consciousness_update(websocket, session_id)
            elif message_type == "get_human_features":
                await send_human_features_update(websocket, session_id)
            elif message_type == "stop_stream":
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))
    finally:
        if session_id in websocket_connections:
            del websocket_connections[session_id]


async def start_enhanced_consciousness_stream(websocket: WebSocket, session_id: str, config: Dict):
    """Start enhanced consciousness and system streaming"""
    
    if not atlas_system:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "ATLAS system not ready"
        }))
        return
    
    duration = config.get("duration", 30)
    update_interval = config.get("update_interval", 1.0)
    include_dreams = config.get("include_dreams", False)
    include_human_features = config.get("include_human_features", True)
    
    end_time = datetime.now() + timedelta(seconds=duration)
    
    await websocket.send_text(json.dumps({
        "type": "stream_started",
        "duration": duration,
        "session_id": session_id,
        "features": {
            "consciousness": True,
            "dreams": include_dreams,
            "human_features": include_human_features
        }
    }))
    
    try:
        while datetime.now() < end_time:
            # Get consciousness state
            consciousness_data = atlas_system.get_consciousness_analytics()
            
            # Get human features state if requested
            human_features_data = None
            if include_human_features:
                human_features_data = atlas_system.get_human_features_analytics()
            
            # Get system metrics
            system_metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send comprehensive stream update
            stream_data = {
                "type": "enhanced_stream_update",
                "consciousness": consciousness_data,
                "human_features": human_features_data,
                "system_metrics": system_metrics,
                "remaining_time": (end_time - datetime.now()).total_seconds(),
                "current_model": atlas_system.current_model_type.value if atlas_system.current_model_type else None
            }
            
            await websocket.send_text(json.dumps(stream_data))
            await asyncio.sleep(update_interval)
            
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Enhanced stream error: {str(e)}"
        }))


async def handle_enhanced_websocket_chat(websocket: WebSocket, session_id: str, message_data: Dict):
    """Handle enhanced chat through WebSocket"""
    
    if not atlas_system:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "ATLAS system not ready"
        }))
        return
    
    try:
        response_data = await atlas_system.generate_response(
            message=message_data["message"],
            session_id=session_id,
            user_id=message_data.get("user_id", "websocket_user"),
            context=message_data.get("context", {}),
            include_consciousness=message_data.get("include_consciousness", True),
            auto_model_selection=message_data.get("auto_model_selection", True)
        )
        
        await websocket.send_text(json.dumps({
            "type": "enhanced_chat_response",
            "response": response_data["response"],
            "model_used": response_data["model_used"],
            "consciousness_level": response_data["consciousness_level"],
            "consciousness_state": response_data.get("consciousness_state"),
            "human_enhancements": response_data.get("human_enhancements_active", {}),
            "system_coherence": response_data.get("system_coherence", 0.0),
            "response_time": response_data["response_time"],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }))
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Enhanced chat error: {str(e)}"
        }))


async def send_consciousness_update(websocket: WebSocket, session_id: str):
    """Send consciousness state update"""
    
    if not atlas_system:
        return
    
    try:
        consciousness_data = atlas_system.get_consciousness_analytics()
        await websocket.send_text(json.dumps({
            "type": "consciousness_update",
            "data": consciousness_data,
            "timestamp": datetime.now().isoformat()
        }))
    except Exception as e:
        logger.error(f"Consciousness update error: {e}")


async def send_human_features_update(websocket: WebSocket, session_id: str):
    """Send human features state update"""
    
    if not atlas_system:
        return
    
    try:
        human_features_data = atlas_system.get_human_features_analytics()
        await websocket.send_text(json.dumps({
            "type": "human_features_update",
            "data": human_features_data,
            "timestamp": datetime.now().isoformat()
        }))
    except Exception as e:
        logger.error(f"Human features update error: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "enhanced_server:app",
        host="0.0.0.0", 
        port=8001,
        reload=True,
        log_level="info"
    )
