"""
Enhanced ATLAS Backend Server
Integrates Enhanced ATLAS consciousness monitoring system with FastAPI backend
Uses the enhanced system with model switching, advanced learning, and complete human features
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

# Enhanced ATLAS imports
try:
    from enhanced_atlas_system import EnhancedAtlasSystem, ModelType
    from enhanced_consciousness_monitor import EnhancedConsciousnessMonitor
    from enhanced_human_features import EnhancedHumanFeaturesSystem
    from secure_code_executor import SecureCodeExecutor
    # Fallback to original for compatibility
    AtlasQwenSystem = EnhancedAtlasSystem
    ConsciousnessMonitor = EnhancedConsciousnessMonitor
    HumanEnhancementModule = EnhancedHumanFeaturesSystem
    CodeExecutor = SecureCodeExecutor
    ENHANCED_ATLAS_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced ATLAS import error: {e}")
    print("Falling back to original ATLAS features")
    ENHANCED_ATLAS_AVAILABLE = False
    try:
        from atlas_qwen_system import AtlasQwenSystem
        # Try to import consciousness monitor
        try:
            from consciousness_monitor import ConsciousnessMonitor  
        except ImportError:
            # Use the simple one from atlas_qwen_system
            from atlas_qwen_system import SimpleConsciousnessMonitor as ConsciousnessMonitor
        
        # Try to import human enhancements
        try:
            from human_enhancements import HumanEnhancementModule
        except ImportError:
            from atlas_qwen_system import SimpleHumanEnhancements as HumanEnhancementModule
        
        # Try to import code executor
        try:
            from code_executor import CodeExecutor
        except ImportError:
            from atlas_qwen_system import SimpleCodeExecutor as CodeExecutor
            
        ModelType = None
    except ImportError as e2:
        print(f"Original ATLAS import error: {e2}")
        print("Some ATLAS features may not be available")
        AtlasQwenSystem = None
        ConsciousnessMonitor = None
        HumanEnhancementModule = None
        CodeExecutor = None
        ModelType = None

# Database
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv
from pathlib import Path

# Load environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')


# Pydantic Models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_consciousness: bool = True

class ChatResponse(BaseModel):
    response: str
    session_id: str
    consciousness_level: Optional[float] = None
    memory_stored: Optional[bool] = None
    timestamp: datetime

class SystemStatus(BaseModel):
    status: str
    model_loaded: bool
    consciousness_active: bool
    memory_count: int
    uptime: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None

class CodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"
    session_id: Optional[str] = None

class CodeExecutionResponse(BaseModel):
    output: str
    error: Optional[str] = None
    execution_time: float
    session_id: str

class MemoryQuery(BaseModel):
    query: str
    limit: int = 10

class StreamConfig(BaseModel):
    duration: int = 30  # seconds
    include_consciousness: bool = True
    update_interval: float = 1.0


# Global instances
atlas_system: Optional[AtlasQwenSystem] = None
consciousness_monitor: Optional[ConsciousnessMonitor] = None
human_enhancements: Optional[HumanEnhancementModule] = None
code_executor: Optional[CodeExecutor] = None
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
    """Initialize Enhanced ATLAS system components"""
    global atlas_system, consciousness_monitor, human_enhancements, code_executor
    global system_status
    
    try:
        print("🚀 Initializing Enhanced ATLAS system...")
        
        # Check if enhanced system is available
        if 'EnhancedAtlasSystem' in globals() and EnhancedAtlasSystem is not None:
            print("📡 Loading Enhanced ATLAS with model switching capabilities...")
            
            # Enhanced system configuration
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
            
            # Initialize enhanced ATLAS system
            atlas_system = EnhancedAtlasSystem(config)
            if ModelType:
                await atlas_system.initialize([ModelType.QWEN_SMALL])  # Start with small model
            else:
                await atlas_system.initialize()
            
            # The consciousness monitor and human features are part of the enhanced system
            consciousness_monitor = atlas_system.consciousness_monitor
            human_enhancements = atlas_system.human_features
            code_executor = atlas_system.code_executor
            
            system_status["model_loaded"] = True
            system_status["consciousness_active"] = True
            system_status["initialization_complete"] = True
            
            print("✅ Enhanced ATLAS system initialized with advanced features!")
            
        elif AtlasQwenSystem is not None:
            print("📡 Loading Original ATLAS system...")
            
            # Simplified configuration for original system
            config = {
                "model_name": "Qwen/Qwen2.5-0.5B",
                "max_length": 512,
                "device": "auto",
                "torch_dtype": "float16",
                "temperature": 0.7,
                "do_sample": True,
                "consciousness": {
                    "hidden_dim": 256,
                    "i2c_units": 4
                },
                "human_enhancements": {
                    "enable_all": True
                }
            }
            
            # Initialize original ATLAS system with mock if needed
            print("📡 Setting up simplified ATLAS...")
            try:
                atlas_system = AtlasQwenSystem(config)
                if hasattr(atlas_system, 'initialize'):
                    await atlas_system.initialize()
                system_status["model_loaded"] = True
                print("✅ ATLAS model loaded successfully")
            except Exception as e:
                print(f"⚠️ Model loading failed ({e}), using mock system")
                atlas_system = None
                system_status["model_loaded"] = False
            
            # Initialize consciousness monitor
            print("🧠 Initializing consciousness monitor...")
            if ConsciousnessMonitor:
                consciousness_monitor = ConsciousnessMonitor(
                    hidden_dim=config["consciousness"]["hidden_dim"],
                    i2c_units=config["consciousness"]["i2c_units"]
                )
                system_status["consciousness_active"] = True
                print("✅ Consciousness monitor initialized")
            else:
                consciousness_monitor = None
                system_status["consciousness_active"] = False
                print("⚠️ Consciousness monitor not available")
            
            # Initialize human enhancements
            print("👤 Loading human enhancement modules...")
            if HumanEnhancementModule:
                human_enhancements = HumanEnhancementModule()
                print("✅ Human enhancements loaded")
            else:
                human_enhancements = None
                print("⚠️ Human enhancements not available")
            
            # Initialize code executor
            print("💻 Setting up code execution environment...")
            if CodeExecutor:
                code_executor = CodeExecutor()
                print("✅ Code executor ready")
            else:
                code_executor = None
                print("⚠️ Code executor not available")
            
            system_status["initialization_complete"] = True
            print("🎉 Original ATLAS system initialization complete!")
            
        else:
            print("⚠️ No ATLAS system available, running in mock mode")
            atlas_system = None
            consciousness_monitor = None
            human_enhancements = None
            code_executor = None
            system_status["model_loaded"] = False
            system_status["consciousness_active"] = False
            system_status["initialization_complete"] = True
        
    except Exception as e:
        print(f"❌ Error initializing ATLAS: {str(e)}")
        # Set up mock system
        print("🔧 Setting up mock ATLAS system for development...")
        atlas_system = None
        consciousness_monitor = None
        human_enhancements = None
        code_executor = None
        system_status["model_loaded"] = False
        system_status["consciousness_active"] = False
        system_status["initialization_complete"] = True


async def initialize_database():
    """Initialize MongoDB connection"""
    global db_client, database
    
    try:
        mongo_url = os.getenv('MONGO_URL', 'mongodb://localhost:27017')
        db_name = os.getenv('DB_NAME', 'atlas_database')
        
        db_client = AsyncIOMotorClient(mongo_url)
        database = db_client[db_name]
        
        # Test connection
        await db_client.admin.command('ping')
        print("✅ Database connection established")
        
        # Create indexes
        await database.conversations.create_index("session_id")
        await database.memories.create_index("timestamp")
        await database.consciousness_logs.create_index("timestamp")
        
    except Exception as e:
        print(f"❌ Database initialization error: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await initialize_database()
    # Initialize ATLAS in background to avoid blocking startup
    asyncio.create_task(initialize_atlas())
    yield
    # Shutdown
    if db_client:
        db_client.close()


# Create FastAPI app
app = FastAPI(
    title="Enhanced ATLAS System API",
    description="Advanced Thinking and Learning AI System with Enhanced Consciousness Monitoring, Model Switching, and Complete Human-like Features",
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
    """Enhanced health check endpoint"""
    enhanced_features = {
        "consciousness_monitoring": system_status.get("consciousness_active", False),
        "human_features": system_status.get("initialization_complete", False),
        "secure_execution": system_status.get("initialization_complete", False),
        "model_switching": False
    }
    
    # Check if enhanced system is available
    if hasattr(atlas_system, 'initialized_models'):
        enhanced_features["model_switching"] = len(atlas_system.initialized_models) > 1
    
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "enhanced_features": enhanced_features
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
    
    return SystemStatus(
        status="running" if system_status["initialization_complete"] else "initializing",
        model_loaded=system_status["model_loaded"],
        consciousness_active=system_status["consciousness_active"],
        memory_count=memory_count,
        uptime=str(uptime),
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        gpu_usage=gpu_usage
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_atlas(request: ChatMessage):
    """Chat with ATLAS system"""
    if not atlas_system or not system_status["initialization_complete"]:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Get or create session
        if session_id not in active_sessions:
            active_sessions[session_id] = {
                "created": datetime.now(),
                "conversation_history": []
            }
        
        session = active_sessions[session_id]
        
        # Generate response with ATLAS
        response_data = await atlas_system.generate_response(
            message=request.message,
            session_id=session_id,
            include_consciousness=request.include_consciousness
        )
        
        # Store conversation in database
        conversation_entry = {
            "session_id": session_id,
            "user_message": request.message,
            "atlas_response": response_data["response"],
            "consciousness_level": response_data.get("consciousness_level"),
            "timestamp": datetime.now(),
            "memory_stored": response_data.get("memory_stored", False)
        }
        
        if database is not None:
            await database.conversations.insert_one(conversation_entry)
        
        # Update session
        session["conversation_history"].append(conversation_entry)
        
        return ChatResponse(
            response=response_data["response"],
            session_id=session_id,
            consciousness_level=response_data.get("consciousness_level"),
            memory_stored=response_data.get("memory_stored"),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/api/code/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """Execute code with ATLAS code executor"""
    if not code_executor:
        raise HTTPException(status_code=503, detail="Code executor not ready")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        result = await code_executor.execute_code(
            code=request.code,
            language=request.language
        )
        
        # Store execution in database
        if database is not None:
            execution_entry = {
                "session_id": session_id,
                "code": request.code,
                "language": request.language,
                "output": result["output"],
                "error": result.get("error"),
                "execution_time": result["execution_time"],
                "timestamp": datetime.now()
            }
            await database.code_executions.insert_one(execution_entry)
        
        return CodeExecutionResponse(
            output=result["output"],
            error=result.get("error"),
            execution_time=result["execution_time"],
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code execution error: {str(e)}")


@app.post("/api/model/switch")
async def switch_model(model_type: str, force_load: bool = False):
    """Switch between available models (Enhanced ATLAS only)"""
    
    if not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    # Check if enhanced system with model switching is available
    if not hasattr(atlas_system, 'switch_model') or ModelType is None:
        raise HTTPException(status_code=501, detail="Model switching not available in current ATLAS configuration")
    
    try:
        model_enum = ModelType(model_type)
        success = await atlas_system.switch_model(model_enum, force_load)
        
        if success:
            return {
                "message": f"Successfully switched to {model_type}",
                "current_model": model_type,
                "available_models": [m.value for m in atlas_system.initialized_models]
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to switch to {model_type}")
            
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}. Available: {[m.value for m in ModelType] if ModelType else []}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model switch error: {str(e)}")


@app.get("/api/consciousness/detailed")
async def get_detailed_consciousness():
    """Get detailed consciousness analytics (Enhanced ATLAS only)"""
    
    if not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    if not hasattr(atlas_system, 'get_consciousness_analytics'):
        raise HTTPException(status_code=501, detail="Detailed consciousness analytics not available")
    
    try:
        return atlas_system.get_consciousness_analytics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consciousness analytics error: {str(e)}")


@app.get("/api/human-features")
async def get_human_features():
    """Get human features analytics (Enhanced ATLAS only)"""
    
    if not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    if not hasattr(atlas_system, 'get_human_features_analytics'):
        raise HTTPException(status_code=501, detail="Human features analytics not available")
    
    try:
        return atlas_system.get_human_features_analytics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Human features analytics error: {str(e)}")


@app.get("/api/learning/analytics")
async def get_learning_analytics():
    """Get learning system analytics (Enhanced ATLAS only)"""
    
    if not atlas_system:
        raise HTTPException(status_code=503, detail="ATLAS system not ready")
    
    if not hasattr(atlas_system, 'learning_system'):
        raise HTTPException(status_code=501, detail="Learning analytics not available")
    
    try:
        return atlas_system.learning_system.get_learning_analytics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning analytics error: {str(e)}")


@app.get("/api/memory/search")
async def search_memories(query: str, limit: int = 10):
    """Search stored memories"""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Try text search first, fallback to regex search if no text index
        try:
            cursor = database.memories.find(
                {"$text": {"$search": query}}
            ).limit(limit).sort("timestamp", -1)
            
            memories = []
            async for memory in cursor:
                memory["_id"] = str(memory["_id"])
                memories.append(memory)
                
        except Exception as text_search_error:
            # Fallback to regex search if text index doesn't exist
            cursor = database.memories.find(
                {"$or": [
                    {"content": {"$regex": query, "$options": "i"}},
                    {"title": {"$regex": query, "$options": "i"}}
                ]}
            ).limit(limit).sort("timestamp", -1)
            
            memories = []
            async for memory in cursor:
                memory["_id"] = str(memory["_id"])
                memories.append(memory)
        
        return {"memories": memories, "count": len(memories)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search error: {str(e)}")


@app.get("/api/consciousness/current")
async def get_consciousness_state():
    """Get current consciousness state"""
    if not consciousness_monitor:
        raise HTTPException(status_code=503, detail="Consciousness monitor not ready")
    
    try:
        state = consciousness_monitor.get_current_state()
        return {
            "consciousness_level": state.get("consciousness_level", 0.0),
            "i2c_activations": state.get("i2c_activations", []),
            "attention_patterns": state.get("attention_patterns", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "consciousness_level": 0.0,
            "i2c_activations": [],
            "attention_patterns": {},
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


@app.get("/api/sessions")
async def get_active_sessions():
    """Get list of active sessions"""
    sessions_info = []
    for session_id, session_data in active_sessions.items():
        sessions_info.append({
            "session_id": session_id,
            "created": session_data["created"].isoformat(),
            "message_count": len(session_data["conversation_history"])
        })
    
    return {"active_sessions": sessions_info, "total": len(sessions_info)}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in active_sessions:
        del active_sessions[session_id]
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
                # Start consciousness stream
                await start_consciousness_stream(websocket, session_id, message_data)
            elif message_data.get("type") == "chat":
                # Handle chat message
                await handle_websocket_chat(websocket, session_id, message_data)
            elif message_data.get("type") == "stop_stream":
                # Stop any active streams
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        if session_id in websocket_connections:
            del websocket_connections[session_id]


async def start_consciousness_stream(websocket: WebSocket, session_id: str, config: Dict):
    """Start streaming consciousness data"""
    if not consciousness_monitor or not atlas_system:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "ATLAS system not ready"
        }))
        return
    
    duration = config.get("duration", 30)
    update_interval = config.get("update_interval", 1.0)
    end_time = datetime.now() + timedelta(seconds=duration)
    
    await websocket.send_text(json.dumps({
        "type": "stream_started",
        "duration": duration,
        "session_id": session_id
    }))
    
    try:
        while datetime.now() < end_time:
            # Get consciousness state
            consciousness_state = consciousness_monitor.get_current_state()
            
            # Get system metrics
            import psutil
            system_metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send stream update
            stream_data = {
                "type": "stream_update",
                "consciousness": consciousness_state,
                "system_metrics": system_metrics,
                "remaining_time": (end_time - datetime.now()).total_seconds()
            }
            
            await websocket.send_text(json.dumps(stream_data))
            await asyncio.sleep(update_interval)
            
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Stream error: {str(e)}"
        }))


async def handle_websocket_chat(websocket: WebSocket, session_id: str, message_data: Dict):
    """Handle chat through WebSocket"""
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
            include_consciousness=message_data.get("include_consciousness", True)
        )
        
        await websocket.send_text(json.dumps({
            "type": "chat_response",
            "response": response_data["response"],
            "consciousness_level": response_data.get("consciousness_level"),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }))
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Chat error: {str(e)}"
        }))


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0", 
        port=8001,
        reload=True,
        log_level="info"
    )
