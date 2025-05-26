"""
ATLAS Backend Server - Unified Multi-Modal System
Integrates unified ATLAS consciousness monitoring system with FastAPI backend
"""

import os
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Database
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv
from pathlib import Path

# ATLAS unified system imports
try:
    import sys
    import os
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    
    import unified_atlas_system
    from unified_atlas_system import UnifiedATLASSystem, ATLASConfig, create_atlas_system
    print("✅ Unified ATLAS system imported successfully")
    ATLAS_AVAILABLE = True
except ImportError as e:
    print(f"❌ ATLAS import error: {e}")
    print("Running without ATLAS features")
    UnifiedATLASSystem = None
    ATLASConfig = None
    create_atlas_system = None
    ATLAS_AVAILABLE = False
except Exception as e:
    print(f"❌ ATLAS initialization error: {e}")
    print("Running without ATLAS features")
    UnifiedATLASSystem = None
    ATLASConfig = None
    create_atlas_system = None
    ATLAS_AVAILABLE = False

# Load environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')


# Pydantic Models for API
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_consciousness: bool = True
    modality: str = "text"
    creativity_level: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    consciousness_level: Optional[float] = None
    memory_stored: Optional[bool] = None
    timestamp: datetime
    modality: str = "text"
    processing_time: Optional[float] = None

class SystemStatus(BaseModel):
    status: str
    model_loaded: bool
    consciousness_active: bool
    memory_count: int
    uptime: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    capabilities: List[str] = []

class CodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"
    session_id: Optional[str] = None
    creative: bool = False

class CodeExecutionResponse(BaseModel):
    output: str
    error: Optional[str] = None
    execution_time: float
    session_id: str
    type: str = "code_execution"

class ImaginationRequest(BaseModel):
    prompt: str
    modality: str = "text"  # text, image, audio, multimodal
    creativity_level: Optional[float] = 0.8
    cross_modal: bool = False
    session_id: Optional[str] = None

class ImaginationResponse(BaseModel):
    content: Any
    modality: str
    creativity_score: float
    session_id: str
    timestamp: datetime
    processing_time: float

class MemoryQuery(BaseModel):
    query: str
    limit: int = 10

class StreamConfig(BaseModel):
    duration: int = 30  # seconds
    include_consciousness: bool = True
    update_interval: float = 1.0
    modality: str = "multimodal"


# Global instances
atlas_system: Optional[UnifiedATLASSystem] = None
db_client: Optional[AsyncIOMotorClient] = None
database = None

# Active sessions and connections
active_sessions: Dict[str, Dict] = {}
websocket_connections: Dict[str, WebSocket] = {}

# System status
system_start_time = datetime.now()
system_status = {
    "model_loaded": False,
    "consciousness_active": False,
    "initialization_complete": False
}


async def initialize_atlas():
    """Initialize unified ATLAS system"""
    global atlas_system, system_status
    
    if not ATLAS_AVAILABLE:
        print("⚠️ ATLAS system not available - running in basic mode")
        return
    
    try:
        print("🚀 Initializing Unified ATLAS System...")
        
        # Create ATLAS configuration for multi-modal system
        config = ATLASConfig(
            language_model="Qwen/Qwen2.5-0.5B",
            consciousness_dim=512,
            i2c_units=16,  # Increased for multi-modal
            learning_rate=0.001,
            imagination_creativity=0.8,
            cross_modal_attention=True,
            temporal_memory=50,
            memory_capacity=10000,
            consolidation_interval=100
        )
        
        # Initialize unified ATLAS system
        atlas_system = create_atlas_system(config)
        await atlas_system.initialize()
        
        system_status["model_loaded"] = atlas_system.initialized
        system_status["consciousness_active"] = True
        system_status["initialization_complete"] = True
        
        print("🎉 Unified ATLAS System initialized successfully!")
        print(f"   System ID: {atlas_system.system_id}")
        print(f"   Multi-modal imagination: ✅")
        print(f"   Advanced learning: ✅")
        print(f"   Consciousness monitoring: ✅")
        print(f"   Available modalities: {', '.join(['text', 'image', 'audio', 'code', 'imagination', 'multimodal'])}")
        
    except Exception as e:
        print(f"❌ Error initializing ATLAS: {str(e)}")
        system_status["initialization_complete"] = False


async def initialize_database():
    """Initialize MongoDB connection"""
    global db_client, database
    
    try:
        mongo_url = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
        db_name = os.getenv('DB_NAME', 'atlas_unified_database')
        
        db_client = AsyncIOMotorClient(mongo_url)
        database = db_client[db_name]
        
        # Test connection
        await db_client.admin.command('ping')
        print("✅ Database connection established")
        
        # Create indexes for unified system
        await database.conversations.create_index("session_id")
        await database.memories.create_index("timestamp")
        await database.consciousness_logs.create_index("timestamp")
        await database.imagination_creations.create_index("modality")
        await database.learning_patterns.create_index("interaction_count")
        
    except Exception as e:
        print(f"❌ Database initialization error: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await initialize_database()
    await initialize_atlas()
    yield
    # Shutdown
    if db_client:
        db_client.close()


# Create FastAPI app
app = FastAPI(
    title="ATLAS Unified System API",
    description="Advanced Thinking and Learning AI System with Multi-Modal Imagination and Consciousness Monitoring",
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


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "atlas_available": ATLAS_AVAILABLE,
        "version": "2.0.0"
    }


@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    import psutil
    
    uptime = datetime.now() - system_start_time
    
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
    
    # Get memory count from database
    memory_count = 0
    if database is not None:
        try:
            memory_count = await database.memories.count_documents({})
        except:
            pass
    
    # Get capabilities
    capabilities = ["basic_chat", "health_monitoring"]
    if ATLAS_AVAILABLE and atlas_system:
        capabilities.extend([
            "multi_modal_imagination",
            "consciousness_monitoring", 
            "advanced_learning",
            "cross_modal_creativity",
            "text_generation",
            "image_generation",
            "audio_generation", 
            "code_execution",
            "memory_search",
            "real_time_streaming"
        ])
    
    return SystemStatus(
        status="running" if system_status["initialization_complete"] else "initializing",
        model_loaded=system_status["model_loaded"],
        consciousness_active=system_status["consciousness_active"],
        memory_count=memory_count,
        uptime=str(uptime),
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        gpu_usage=gpu_usage,
        capabilities=capabilities
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_atlas(request: ChatMessage):
    """Chat with unified ATLAS system"""
    if not ATLAS_AVAILABLE or not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not available")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Use unified ATLAS system for processing
        result = await atlas_system.process_request(
            request=request.message,
            modality=request.modality,
            session_id=session_id,
            include_consciousness=request.include_consciousness,
            creativity_level=request.creativity_level,
            learning_enabled=True
        )
        
        # Store conversation in database
        if database is not None:
            conversation_entry = {
                "session_id": session_id,
                "user_message": request.message,
                "atlas_response": result.get("content", ""),
                "modality": request.modality,
                "consciousness_level": result.get("consciousness_level"),
                "creativity_score": request.creativity_level,
                "processing_time": result.get("processing_time"),
                "timestamp": datetime.now(),
                "learning_applied": result.get("learning", {})
            }
            await database.conversations.insert_one(conversation_entry)
        
        return ChatResponse(
            response=result.get("content", ""),
            session_id=session_id,
            consciousness_level=result.get("consciousness_level"),
            memory_stored=result.get("learning", {}).get("memory_stored", False),
            timestamp=datetime.now(),
            modality=request.modality,
            processing_time=result.get("processing_time")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/api/imagination/generate", response_model=ImaginationResponse)
async def generate_imagination(request: ImaginationRequest):
    """Generate creative content using multi-modal imagination"""
    if not ATLAS_AVAILABLE or not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS imagination system not available")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Use unified ATLAS system for imagination
        result = await atlas_system.process_request(
            request=request.prompt,
            modality="imagination" if request.modality == "text" else request.modality,
            session_id=session_id,
            include_consciousness=True,
            creativity_level=request.creativity_level,
            learning_enabled=True
        )
        
        # Store imagination creation in database
        if database is not None:
            creation_entry = {
                "session_id": session_id,
                "prompt": request.prompt,
                "modality": request.modality,
                "content": result.get("content"),
                "creativity_score": result.get("creativity_score", request.creativity_level),
                "consciousness_level": result.get("consciousness_level"),
                "cross_modal": request.cross_modal,
                "timestamp": datetime.now(),
                "processing_time": result.get("processing_time")
            }
            await database.imagination_creations.insert_one(creation_entry)
        
        return ImaginationResponse(
            content=result.get("content"),
            modality=request.modality,
            creativity_score=result.get("creativity_score", request.creativity_level),
            session_id=session_id,
            timestamp=datetime.now(),
            processing_time=result.get("processing_time", 0.0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Imagination error: {str(e)}")


@app.post("/api/code/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """Execute code with unified ATLAS system"""
    if not ATLAS_AVAILABLE or not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS code execution not available")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Determine modality based on creative flag
        modality = "imagination" if request.creative else "code"
        
        result = await atlas_system.process_request(
            request=request.code,
            modality=modality,
            session_id=session_id,
            include_consciousness=True,
            learning_enabled=True
        )
        
        # Store execution in database
        if database is not None:
            execution_entry = {
                "session_id": session_id,
                "code": request.code,
                "language": request.language,
                "creative": request.creative,
                "output": result.get("content", ""),
                "error": result.get("error"),
                "execution_time": result.get("processing_time", 0.0),
                "consciousness_level": result.get("consciousness_level"),
                "timestamp": datetime.now()
            }
            await database.code_executions.insert_one(execution_entry)
        
        return CodeExecutionResponse(
            output=result.get("content", ""),
            error=result.get("error"),
            execution_time=result.get("processing_time", 0.0),
            session_id=session_id,
            type=result.get("type", "code_execution")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code execution error: {str(e)}")


@app.get("/api/consciousness/current")
async def get_consciousness_state():
    """Get current consciousness state from unified system"""
    if not ATLAS_AVAILABLE or not atlas_system:
        return {
            "consciousness_level": 0.0,
            "i2c_activations": [],
            "attention_patterns": {},
            "timestamp": datetime.now().isoformat(),
            "error": "ATLAS system not available"
        }
    
    try:
        system_state = atlas_system.get_system_state()
        consciousness_state = system_state.get("consciousness", {})
        
        return {
            "consciousness_level": consciousness_state.get("global_consciousness", 0.0),
            "modality_states": consciousness_state.get("modality_states", {}),
            "i2c_activations": consciousness_state.get("i2c_activations", []),
            "attention_patterns": consciousness_state.get("attention_patterns", {}),
            "cross_modal_integration": consciousness_state.get("cross_modal_integration", False),
            "timestamp": consciousness_state.get("timestamp", datetime.now().isoformat())
        }
    except Exception as e:
        return {
            "consciousness_level": 0.0,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/learning/state")
async def get_learning_state():
    """Get current learning state from unified system"""
    if not ATLAS_AVAILABLE or not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS learning system not available")
    
    try:
        system_state = atlas_system.get_system_state()
        return system_state.get("learning", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning state error: {str(e)}")


@app.get("/api/imagination/state")
async def get_imagination_state():
    """Get current imagination system state"""
    if not ATLAS_AVAILABLE or not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS imagination system not available")
    
    try:
        system_state = atlas_system.get_system_state()
        return system_state.get("imagination", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Imagination state error: {str(e)}")


@app.get("/api/memory/search")
async def search_memories(query: str, limit: int = 10):
    """Search stored memories"""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Search across all collections
        results = {
            "conversations": [],
            "imagination_creations": [],
            "learning_patterns": []
        }
        
        # Search conversations
        conv_cursor = database.conversations.find(
            {"$or": [
                {"user_message": {"$regex": query, "$options": "i"}},
                {"atlas_response": {"$regex": query, "$options": "i"}}
            ]}
        ).limit(limit//3).sort("timestamp", -1)
        
        async for conv in conv_cursor:
            conv["_id"] = str(conv["_id"])
            results["conversations"].append(conv)
        
        # Search imagination creations
        imag_cursor = database.imagination_creations.find(
            {"prompt": {"$regex": query, "$options": "i"}}
        ).limit(limit//3).sort("timestamp", -1)
        
        async for imag in imag_cursor:
            imag["_id"] = str(imag["_id"])
            results["imagination_creations"].append(imag)
        
        return {
            "query": query,
            "results": results,
            "total_count": len(results["conversations"]) + len(results["imagination_creations"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search error: {str(e)}")


@app.get("/api/sessions")
async def get_active_sessions():
    """Get list of active sessions"""
    if not ATLAS_AVAILABLE or not atlas_system:
        return {"active_sessions": [], "total": 0}
    
    system_state = atlas_system.get_system_state()
    sessions_info = []
    
    for session_id in atlas_system.sessions.keys():
        session_data = atlas_system.sessions[session_id]
        sessions_info.append({
            "session_id": session_id,
            "created": datetime.fromtimestamp(session_data["created"]).isoformat(),
            "interactions": len(session_data["interactions"])
        })
    
    return {"active_sessions": sessions_info, "total": len(sessions_info)}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if ATLAS_AVAILABLE and atlas_system and session_id in atlas_system.sessions:
        del atlas_system.sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


# WebSocket endpoint for real-time streaming
@app.websocket("/api/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time ATLAS streaming"""
    await websocket.accept()
    websocket_connections[session_id] = websocket
    
    try:
        while True:
            # Wait for client message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "start_stream":
                # Start unified consciousness stream
                await start_unified_stream(websocket, session_id, message_data)
            elif message_data.get("type") == "chat":
                # Handle chat message through unified system
                await handle_websocket_chat(websocket, session_id, message_data)
            elif message_data.get("type") == "stop_stream":
                # Stop any active streams
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        if session_id in websocket_connections:
            del websocket_connections[session_id]


async def start_unified_stream(websocket: WebSocket, session_id: str, config: Dict):
    """Start streaming unified ATLAS data"""
    if not ATLAS_AVAILABLE or not atlas_system:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "ATLAS unified system not ready"
        }))
        return
    
    duration = config.get("duration", 30)
    update_interval = config.get("update_interval", 1.0)
    end_time = datetime.now() + timedelta(seconds=duration)
    
    await websocket.send_text(json.dumps({
        "type": "stream_started",
        "duration": duration,
        "session_id": session_id,
        "unified_system": True
    }))
    
    try:
        while datetime.now() < end_time:
            # Get unified system state
            system_state = atlas_system.get_system_state()
            
            # Get system metrics
            import psutil
            system_metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send stream update with unified data
            stream_data = {
                "type": "unified_stream_update",
                "consciousness": system_state.get("consciousness", {}),
                "learning": system_state.get("learning", {}),
                "imagination": system_state.get("imagination", {}),
                "system_metrics": system_metrics,
                "capabilities": system_state.get("capabilities", {}),
                "remaining_time": (end_time - datetime.now()).total_seconds()
            }
            
            await websocket.send_text(json.dumps(stream_data))
            await asyncio.sleep(update_interval)
            
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unified stream error: {str(e)}"
        }))


async def handle_websocket_chat(websocket: WebSocket, session_id: str, message_data: Dict):
    """Handle chat through WebSocket using unified system"""
    if not ATLAS_AVAILABLE or not atlas_system:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "ATLAS unified system not ready"
        }))
        return
    
    try:
        # Process through unified system
        result = await atlas_system.process_request(
            request=message_data["message"],
            modality=message_data.get("modality", "text"),
            session_id=session_id,
            include_consciousness=message_data.get("include_consciousness", True),
            creativity_level=message_data.get("creativity_level"),
            learning_enabled=True
        )
        
        await websocket.send_text(json.dumps({
            "type": "unified_chat_response",
            "response": result.get("content", ""),
            "consciousness_level": result.get("consciousness_level"),
            "creativity_score": result.get("creativity_score"),
            "learning_applied": result.get("learning", {}),
            "modality": message_data.get("modality", "text"),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }))
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unified chat error: {str(e)}"
        }))


if __name__ == "__main__":
    uvicorn.run(
        "unified_server:app",
        host="0.0.0.0", 
        port=8001,
        reload=True,
        log_level="info"
    )