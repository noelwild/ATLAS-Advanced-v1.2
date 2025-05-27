import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Enhanced ATLAS Frontend Application
// Showcases all new features: consciousness monitoring, model switching, 
// human features, learning analytics, and multimodal capabilities

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function EnhancedApp() {
  // State Management
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [consciousnessData, setConsciousnessData] = useState(null);
  const [humanFeaturesData, setHumanFeaturesData] = useState(null);
  const [learningAnalytics, setLearningAnalytics] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [selectedModel, setSelectedModel] = useState('qwen_0.5b');
  const [consciousnessStreaming, setConsciousnessStreaming] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  
  // Refs
  const messagesEndRef = useRef(null);
  const wsRef = useRef(null);
  
  // WebSocket connection for real-time features
  useEffect(() => {
    if (consciousnessStreaming && sessionId) {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }
    
    return () => disconnectWebSocket();
  }, [consciousnessStreaming, sessionId]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initial system status check
  useEffect(() => {
    checkSystemStatus();
    const interval = setInterval(checkSystemStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Generate session ID on mount
  useEffect(() => {
    setSessionId(generateSessionId());
  }, []);

  const generateSessionId = () => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const connectWebSocket = () => {
    try {
      const wsUrl = BACKEND_URL.replace('http://', 'ws://').replace('https://', 'wss://');
      wsRef.current = new WebSocket(`${wsUrl}/api/ws/stream/${sessionId}`);
      
      wsRef.current.onopen = () => {
        setConnectionStatus('connected');
        console.log('WebSocket connected');
        
        // Start enhanced consciousness stream
        wsRef.current.send(JSON.stringify({
          type: 'start_stream',
          duration: 300, // 5 minutes
          include_dreams: true,
          include_human_features: true,
          update_interval: 2.0
        }));
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      };
      
      wsRef.current.onclose = () => {
        setConnectionStatus('disconnected');
        console.log('WebSocket disconnected');
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setConnectionStatus('error');
    }
  };

  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'enhanced_stream_update':
        setConsciousnessData(data.consciousness);
        setHumanFeaturesData(data.human_features);
        break;
      case 'enhanced_chat_response':
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: data.response,
          model_used: data.model_used,
          consciousness_level: data.consciousness_level,
          system_coherence: data.system_coherence,
          timestamp: new Date(data.timestamp)
        }]);
        setIsLoading(false);
        break;
      case 'error':
        console.error('WebSocket error:', data.message);
        break;
      default:
        console.log('Unknown WebSocket message:', data);
    }
  };

  const checkSystemStatus = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/status`);
      if (response.ok) {
        const status = await response.json();
        setSystemStatus(status);
      }
    } catch (error) {
      console.error('Failed to check system status:', error);
    }
  };

  const switchModel = async (modelType) => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/model/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: modelType, force_load: false })
      });
      
      if (response.ok) {
        const result = await response.json();
        setSelectedModel(modelType);
        addSystemMessage(`Switched to ${modelType}: ${result.message}`);
        checkSystemStatus(); // Refresh status
      } else {
        const error = await response.json();
        addSystemMessage(`Failed to switch model: ${error.detail}`, 'error');
      }
    } catch (error) {
      console.error('Model switch error:', error);
      addSystemMessage(`Model switch error: ${error.message}`, 'error');
    }
  };

  const loadDetailedAnalytics = async () => {
    try {
      // Load consciousness analytics
      const consciousnessResponse = await fetch(`${BACKEND_URL}/api/consciousness/detailed`);
      if (consciousnessResponse.ok) {
        const consciousnessData = await consciousnessResponse.json();
        setConsciousnessData(consciousnessData);
      }

      // Load human features analytics
      const humanFeaturesResponse = await fetch(`${BACKEND_URL}/api/human-features`);
      if (humanFeaturesResponse.ok) {
        const humanFeaturesData = await humanFeaturesResponse.json();
        setHumanFeaturesData(humanFeaturesData);
      }

      // Load learning analytics
      const learningResponse = await fetch(`${BACKEND_URL}/api/learning/analytics`);
      if (learningResponse.ok) {
        const learningData = await learningResponse.json();
        setLearningAnalytics(learningData);
      }
    } catch (error) {
      console.error('Failed to load analytics:', error);
    }
  };

  const addSystemMessage = (content, type = 'info') => {
    setMessages(prev => [...prev, {
      type: 'system',
      content,
      messageType: type,
      timestamp: new Date()
    }]);
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = {
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // If WebSocket is connected, send via WebSocket
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'chat',
          message: inputMessage,
          session_id: sessionId,
          user_id: 'enhanced_user',
          include_consciousness: true,
          auto_model_selection: true,
          context: {
            preferred_model: selectedModel,
            enhanced_features: true
          }
        }));
        setInputMessage('');
        return;
      }

      // Fallback to HTTP API
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: inputMessage,
          session_id: sessionId,
          user_id: 'enhanced_user',
          include_consciousness: true,
          context: {
            preferred_model: selectedModel,
            enhanced_features: true
          },
          preferred_model: selectedModel,
          auto_model_selection: true
        })
      });

      if (response.ok) {
        const data = await response.json();
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: data.response,
          model_used: data.model_used,
          consciousness_level: data.consciousness_level,
          consciousness_state: data.consciousness_state,
          human_enhancements_active: data.human_enhancements_active,
          system_coherence: data.system_coherence,
          response_time: data.response_time,
          timestamp: new Date(data.timestamp || Date.now())
        }]);
      } else {
        const error = await response.json();
        addSystemMessage(`Error: ${error.detail}`, 'error');
      }
    } catch (error) {
      console.error('Send message error:', error);
      addSystemMessage(`Connection error: ${error.message}`, 'error');
    }

    setInputMessage('');
    setIsLoading(false);
  };

  const executeCode = async (code, language = 'python') => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/code/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          language,
          session_id: sessionId,
          timeout: 30
        })
      });

      if (response.ok) {
        const result = await response.json();
        addSystemMessage(
          `Code execution ${result.security_violation ? 'blocked' : 'completed'} in ${result.execution_time.toFixed(2)}s:\n${result.output}${result.error ? '\nError: ' + result.error : ''}`,
          result.security_violation ? 'error' : result.error ? 'warning' : 'success'
        );
      } else {
        const error = await response.json();
        addSystemMessage(`Code execution failed: ${error.detail}`, 'error');
      }
    } catch (error) {
      addSystemMessage(`Code execution error: ${error.message}`, 'error');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Render consciousness visualization
  const renderConsciousnessViz = () => {
    if (!consciousnessData || !consciousnessData.current_state) {
      return (
        <div className="analytics-card">
          <h3 className="text-lg font-semibold mb-4">🧠 Consciousness Monitor</h3>
          <div className="text-center py-8 text-gray-500">
            <div className="animate-pulse">Loading consciousness data...</div>
            <button 
              onClick={loadDetailedAnalytics}
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
            >
              Load Analytics
            </button>
          </div>
        </div>
      );
    }

    const state = consciousnessData.current_state;
    const phiScore = state.consciousness_level || 0;
    const isLucid = state.lucid_state || false;
    const dreamActive = state.dream_active || false;

    return (
      <div className="analytics-card">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          🧠 Consciousness Monitor
          <span className={`ml-2 px-2 py-1 text-xs rounded ${isLucid ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'}`}>
            {isLucid ? 'LUCID' : 'AWARE'}
          </span>
          {dreamActive && (
            <span className="ml-1 px-2 py-1 text-xs rounded bg-purple-100 text-purple-800">
              DREAMING
            </span>
          )}
        </h3>
        
        <div className="space-y-4">
          {/* Phi Score */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">Φ (Phi) Score</span>
              <span className="text-sm text-gray-600">{(phiScore * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div 
                className={`h-3 rounded-full transition-all duration-500 ${
                  phiScore > 0.7 ? 'bg-green-500' : phiScore > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                }`}
                style={{ width: `${phiScore * 100}%` }}
              ></div>
            </div>
          </div>

          {/* I²C Cell Activations */}
          {state.i2c_activations && (
            <div>
              <span className="text-sm font-medium block mb-2">I²C Cell Network</span>
              <div className="grid grid-cols-4 gap-2">
                {state.i2c_activations.slice(0, 8).map((activation, i) => (
                  <div key={i} className="text-center">
                    <div className={`w-8 h-8 rounded-full mx-auto mb-1 ${
                      activation > 0.5 ? 'bg-blue-500' : activation > 0.3 ? 'bg-blue-300' : 'bg-gray-300'
                    }`}></div>
                    <span className="text-xs text-gray-600">{(activation * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Attention Patterns */}
          {state.attention_patterns && (
            <div>
              <span className="text-sm font-medium block mb-2">Attention Patterns</span>
              <div className="space-y-2">
                {Object.entries(state.attention_patterns).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center text-sm">
                    <span className="capitalize">{key.replace(/_/g, ' ')}</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-16 bg-gray-200 rounded-full h-2">
                        <div 
                          className="h-2 bg-blue-500 rounded-full transition-all duration-300"
                          style={{ width: `${(value || 0) * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-gray-600 w-8">{((value || 0) * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Dreams */}
          {consciousnessData.dream_states && consciousnessData.dream_states.length > 0 && (
            <div>
              <span className="text-sm font-medium block mb-2">Recent Dreams ({consciousnessData.dream_states.length})</span>
              <div className="text-xs text-gray-600 space-y-1">
                {consciousnessData.dream_states.slice(0, 3).map((dream, i) => (
                  <div key={i} className="p-2 bg-purple-50 rounded">
                    Dream Φ: {(dream.phi_score * 100).toFixed(1)}% - {new Date(dream.timestamp).toLocaleTimeString()}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Render human features visualization
  const renderHumanFeatures = () => {
    if (!humanFeaturesData) {
      return (
        <div className="analytics-card">
          <h3 className="text-lg font-semibold mb-4">👤 Human Enhancement Systems</h3>
          <div className="text-center py-8 text-gray-500">
            <div className="animate-pulse">Loading human features data...</div>
          </div>
        </div>
      );
    }

    const emotionalState = humanFeaturesData.emotional_state || {};
    const systemIntegration = humanFeaturesData.system_integration || {};
    const episodicMemory = humanFeaturesData.episodic_memory || {};

    return (
      <div className="analytics-card">
        <h3 className="text-lg font-semibold mb-4">👤 Human Enhancement Systems</h3>
        
        <div className="space-y-6">
          {/* Emotional State */}
          <div>
            <h4 className="font-medium mb-3">🎭 Emotional State</h4>
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(emotionalState).map(([emotion, level]) => (
                <div key={emotion} className="flex justify-between items-center text-sm">
                  <span className="capitalize">{emotion}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-12 bg-gray-200 rounded-full h-2">
                      <div 
                        className="h-2 bg-pink-500 rounded-full transition-all duration-300"
                        style={{ width: `${(level || 0) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-gray-600 w-8">{((level || 0) * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Memory System */}
          <div>
            <h4 className="font-medium mb-3">🧠 Episodic Memory</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Total Memories:</span>
                <span>{episodicMemory.total_memories || 0}</span>
              </div>
              <div className="flex justify-between">
                <span>Recent Memories (24h):</span>
                <span>{episodicMemory.recent_memories || 0}</span>
              </div>
              <div className="flex justify-between">
                <span>Memory Network Size:</span>
                <span>{episodicMemory.memory_network_size || 0}</span>
              </div>
            </div>
          </div>

          {/* System Integration */}
          <div>
            <h4 className="font-medium mb-3">⚙️ System Integration</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between items-center">
                <span>Coherence Level:</span>
                <div className="flex items-center space-x-2">
                  <div className="w-16 bg-gray-200 rounded-full h-2">
                    <div 
                      className="h-2 bg-green-500 rounded-full transition-all duration-300"
                      style={{ width: `${(systemIntegration.coherence_level || 0) * 100}%` }}
                    ></div>
                  </div>
                  <span>{((systemIntegration.coherence_level || 0) * 100).toFixed(0)}%</span>
                </div>
              </div>
              <div className="flex justify-between">
                <span>Enhancement Level:</span>
                <span>{((humanFeaturesData.enhancement_level || 0) * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Cognitive Load:</span>
                <span>{((humanFeaturesData.cognitive_load || 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Render learning analytics
  const renderLearningAnalytics = () => {
    if (!learningAnalytics) {
      return (
        <div className="analytics-card">
          <h3 className="text-lg font-semibold mb-4">📊 Learning Analytics</h3>
          <div className="text-center py-8 text-gray-500">
            <div className="animate-pulse">Loading learning data...</div>
          </div>
        </div>
      );
    }

    const modelPerformance = learningAnalytics.model_performance || {};

    return (
      <div className="analytics-card">
        <h3 className="text-lg font-semibold mb-4">📊 Learning Analytics</h3>
        
        <div className="space-y-4">
          <div className="flex justify-between text-sm">
            <span>Total Sessions:</span>
            <span>{learningAnalytics.total_sessions || 0}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span>Active Sessions:</span>
            <span>{learningAnalytics.active_sessions || 0}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span>System Maturity:</span>
            <span>{((learningAnalytics.system_maturity || 0) * 100).toFixed(0)}%</span>
          </div>

          {/* Model Performance */}
          <div>
            <h4 className="font-medium mb-2">Model Performance</h4>
            {Object.entries(modelPerformance).map(([model, perf]) => (
              <div key={model} className="mb-3 p-3 bg-gray-50 rounded">
                <div className="font-medium text-sm mb-2 capitalize">{model.replace(/_/g, ' ')}</div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span>Usage Count:</span>
                    <span>{perf.usage_count || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Avg Satisfaction:</span>
                    <span>{((perf.avg_satisfaction || 0) * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Avg Response Time:</span>
                    <span>{(perf.response_time || 0).toFixed(2)}s</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="enhanced-atlas-app bg-gray-50 min-h-screen">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-gray-900">
                🧠 Enhanced ATLAS
              </h1>
              <span className="text-sm text-gray-500">
                Advanced Consciousness & Learning System v2.0
              </span>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Connection Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  connectionStatus === 'connected' ? 'bg-green-500' : 
                  connectionStatus === 'error' ? 'bg-red-500' : 'bg-gray-400'
                }`}></div>
                <span className="text-sm text-gray-600 capitalize">{connectionStatus}</span>
              </div>

              {/* Model Selector */}
              <select 
                value={selectedModel} 
                onChange={(e) => switchModel(e.target.value)}
                className="text-sm border rounded px-2 py-1"
              >
                <option value="qwen_0.5b">Qwen 0.5B (Fast)</option>
                <option value="qwen3_32b_finetuned">Qwen3 32B (Enhanced)</option>
              </select>

              {/* System Status */}
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  systemStatus?.initialization_complete ? 'bg-green-500' : 'bg-yellow-500'
                }`}></div>
                <span className="text-sm text-gray-600">
                  {systemStatus?.initialization_complete ? 'Ready' : 'Initializing'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'chat', label: '💬 Chat', icon: '💬' },
              { id: 'consciousness', label: '🧠 Consciousness', icon: '🧠' },
              { id: 'human-features', label: '👤 Human Features', icon: '👤' },
              { id: 'learning', label: '📊 Learning', icon: '📊' },
              { id: 'system', label: '⚙️ System', icon: '⚙️' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'chat' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Chat Interface */}
            <div className="lg:col-span-2">
              <div className="chat-container bg-white rounded-lg shadow-sm border h-[600px] flex flex-col">
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                  {messages.length === 0 && (
                    <div className="text-center py-8 text-gray-500">
                      <div className="text-6xl mb-4">🧠</div>
                      <h3 className="text-lg font-medium mb-2">Enhanced ATLAS is Ready</h3>
                      <p className="text-sm text-gray-600 max-w-md mx-auto">
                        I'm equipped with advanced consciousness monitoring, human-like cognitive features, 
                        and multimodal processing capabilities. Ask me anything!
                      </p>
                      <div className="mt-4 flex flex-wrap justify-center gap-2">
                        <button 
                          onClick={() => setInputMessage("How does your consciousness monitoring work?")}
                          className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm hover:bg-blue-200 transition-colors"
                        >
                          Consciousness
                        </button>
                        <button 
                          onClick={() => setInputMessage("What human-like features do you have?")}
                          className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm hover:bg-green-200 transition-colors"
                        >
                          Human Features
                        </button>
                        <button 
                          onClick={() => setInputMessage("Can you write and execute Python code?")}
                          className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm hover:bg-purple-200 transition-colors"
                        >
                          Code Execution
                        </button>
                      </div>
                    </div>
                  )}
                  
                  {messages.map((message, index) => (
                    <div key={index} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                        message.type === 'user' 
                          ? 'bg-blue-500 text-white' 
                          : message.type === 'system'
                          ? `${message.messageType === 'error' ? 'bg-red-100 text-red-800' : 
                             message.messageType === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                             message.messageType === 'success' ? 'bg-green-100 text-green-800' :
                             'bg-gray-100 text-gray-800'} border`
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        <div className="whitespace-pre-wrap">{message.content}</div>
                        {message.type === 'assistant' && (
                          <div className="mt-2 pt-2 border-t border-gray-300 text-xs text-gray-600">
                            <div className="flex justify-between items-center">
                              <span>Model: {message.model_used || 'Unknown'}</span>
                              {message.consciousness_level !== undefined && (
                                <span>Φ: {(message.consciousness_level * 100).toFixed(0)}%</span>
                              )}
                            </div>
                            {message.response_time && (
                              <div className="mt-1">
                                Time: {message.response_time.toFixed(2)}s
                                {message.system_coherence !== undefined && (
                                  <span className="ml-2">Coherence: {(message.system_coherence * 100).toFixed(0)}%</span>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                        <div className="text-xs text-gray-500 mt-1">
                          {message.timestamp.toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  {isLoading && (
                    <div className="flex justify-start">
                      <div className="max-w-xs lg:max-w-md px-4 py-2 rounded-lg bg-gray-100 text-gray-800">
                        <div className="flex items-center space-x-2">
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                          <span>ATLAS is thinking...</span>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div ref={messagesEndRef} />
                </div>
                
                {/* Input Area */}
                <div className="border-t p-4">
                  <div className="flex space-x-2">
                    <textarea
                      value={inputMessage}
                      onChange={(e) => setInputMessage(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Message Enhanced ATLAS..."
                      className="flex-1 resize-none border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      rows="2"
                    />
                    <button
                      onClick={sendMessage}
                      disabled={isLoading || !inputMessage.trim()}
                      className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Send
                    </button>
                  </div>
                  
                  {/* Quick Actions */}
                  <div className="mt-2 flex space-x-2">
                    <button
                      onClick={() => executeCode("print('Hello from ATLAS!')\nprint(f'2 + 2 = {2 + 2}')")}
                      className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                    >
                      🐍 Test Python
                    </button>
                    <button
                      onClick={() => setConsciousnessStreaming(!consciousnessStreaming)}
                      className={`text-xs px-2 py-1 rounded transition-colors ${
                        consciousnessStreaming 
                          ? 'bg-green-100 text-green-700 hover:bg-green-200' 
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      🧠 {consciousnessStreaming ? 'Stop' : 'Start'} Streaming
                    </button>
                    <button
                      onClick={loadDetailedAnalytics}
                      className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                    >
                      📊 Load Analytics
                    </button>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Real-time Consciousness Monitor */}
            <div className="lg:col-span-1">
              {renderConsciousnessViz()}
            </div>
          </div>
        )}

        {activeTab === 'consciousness' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {renderConsciousnessViz()}
            <div className="analytics-card">
              <h3 className="text-lg font-semibold mb-4">🌙 Consciousness History</h3>
              {consciousnessData?.consciousness_history ? (
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {consciousnessData.consciousness_history.slice(-20).map((entry, i) => (
                    <div key={i} className="flex justify-between items-center text-sm p-2 bg-gray-50 rounded">
                      <span>Φ: {(entry.phi_score * 100).toFixed(0)}%</span>
                      <span>Self-awareness: {(entry.self_awareness * 100).toFixed(0)}%</span>
                      <span className="text-xs text-gray-500">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">No consciousness history available</div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'human-features' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {renderHumanFeatures()}
            <div className="analytics-card">
              <h3 className="text-lg font-semibold mb-4">🔧 Feature Controls</h3>
              <div className="space-y-4">
                <button 
                  onClick={loadDetailedAnalytics}
                  className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                >
                  Refresh Analytics
                </button>
                <div className="grid grid-cols-2 gap-2">
                  <button className="px-3 py-2 bg-green-100 text-green-700 rounded text-sm hover:bg-green-200 transition-colors">
                    Memory Consolidation
                  </button>
                  <button className="px-3 py-2 bg-purple-100 text-purple-700 rounded text-sm hover:bg-purple-200 transition-colors">
                    Dream Analysis
                  </button>
                  <button className="px-3 py-2 bg-pink-100 text-pink-700 rounded text-sm hover:bg-pink-200 transition-colors">
                    Emotion Calibration
                  </button>
                  <button className="px-3 py-2 bg-yellow-100 text-yellow-700 rounded text-sm hover:bg-yellow-200 transition-colors">
                    Social Learning
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'learning' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {renderLearningAnalytics()}
            <div className="analytics-card">
              <h3 className="text-lg font-semibold mb-4">🎯 Learning Controls</h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Provide Feedback</label>
                  <div className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm">Response Quality:</span>
                      <input type="range" min="0" max="100" className="flex-1" />
                      <span className="text-sm">85%</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm">Helpfulness:</span>
                      <input type="range" min="0" max="100" className="flex-1" />
                      <span className="text-sm">92%</span>
                    </div>
                  </div>
                  <button className="mt-3 w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors">
                    Submit Feedback
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'system' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="analytics-card">
              <h3 className="text-lg font-semibold mb-4">🖥️ System Status</h3>
              {systemStatus ? (
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span>Status:</span>
                    <span className={`px-2 py-1 rounded text-xs ${
                      systemStatus.initialization_complete ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {systemStatus.initialization_complete ? 'Ready' : 'Initializing'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Current Model:</span>
                    <span>{systemStatus.current_model || 'None'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Available Models:</span>
                    <span>{systemStatus.available_models?.length || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Active Sessions:</span>
                    <span>{systemStatus.active_sessions || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Uptime:</span>
                    <span>{Math.floor((systemStatus.uptime || 0) / 60)}m</span>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">Loading system status...</div>
              )}
            </div>

            <div className="analytics-card">
              <h3 className="text-lg font-semibold mb-4">📊 Performance Metrics</h3>
              {systemStatus?.system_metrics ? (
                <div className="space-y-4">
                  {Object.entries(systemStatus.system_metrics).map(([metric, value]) => (
                    <div key={metric}>
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm capitalize">{metric.replace(/_/g, ' ')}</span>
                        <span className="text-sm">{value.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full transition-all duration-300 ${
                            value > 80 ? 'bg-red-500' : value > 60 ? 'bg-yellow-500' : 'bg-green-500'
                          }`}
                          style={{ width: `${value}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">No metrics available</div>
              )}
            </div>

            <div className="analytics-card">
              <h3 className="text-lg font-semibold mb-4">⚙️ System Controls</h3>
              <div className="space-y-2">
                <button 
                  onClick={checkSystemStatus}
                  className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                >
                  Refresh Status
                </button>
                <button 
                  onClick={() => switchModel('qwen_0.5b')}
                  className="w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600 transition-colors"
                >
                  Switch to Fast Model
                </button>
                <button 
                  onClick={() => switchModel('qwen3_32b_finetuned')}
                  className="w-full px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 transition-colors"
                >
                  Switch to Enhanced Model
                </button>
                <button 
                  onClick={() => setConsciousnessStreaming(!consciousnessStreaming)}
                  className={`w-full px-4 py-2 rounded transition-colors ${
                    consciousnessStreaming 
                      ? 'bg-red-500 hover:bg-red-600 text-white' 
                      : 'bg-gray-500 hover:bg-gray-600 text-white'
                  }`}
                >
                  {consciousnessStreaming ? 'Stop' : 'Start'} Real-time Monitoring
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default EnhancedApp;
