import React, { useState, useEffect, useRef } from 'react';
import { toast } from 'react-hot-toast';
import { 
  PlayIcon, 
  StopIcon,
  PauseIcon,
  TrashIcon,
  SignalIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline';

function StreamManager() {
  const [streams, setStreams] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamData, setStreamData] = useState([]);
  const [selectedDuration, setSelectedDuration] = useState(30);
  const [updateInterval, setUpdateInterval] = useState(1.0);
  const [includeConsciousness, setIncludeConsciousness] = useState(true);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const startConsciousnessStream = () => {
    if (isStreaming) {
      toast.error('Stream already active');
      return;
    }

    const sessionId = `stream_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setCurrentSessionId(sessionId);
    
    // Create WebSocket connection
    const wsUrl = `${process.env.REACT_APP_BACKEND_URL.replace('http', 'ws')}/api/ws/stream/${sessionId}`;
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
      setIsStreaming(true);
      setStreamData([]);
      
      // Start the stream
      wsRef.current.send(JSON.stringify({
        type: 'start_stream',
        duration: selectedDuration,
        update_interval: updateInterval,
        include_consciousness: includeConsciousness
      }));
      
      toast.success(`Started consciousness stream for ${selectedDuration}s`);
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'stream_update') {
        const newDataPoint = {
          timestamp: Date.now(),
          time: new Date().toLocaleTimeString(),
          ...data.consciousness,
          system_metrics: data.system_metrics,
          remaining_time: data.remaining_time
        };
        
        setStreamData(prev => {
          const updated = [...prev, newDataPoint];
          return updated.slice(-100); // Keep last 100 points
        });
      } else if (data.type === 'stream_started') {
        console.log('Stream started:', data);
      } else if (data.type === 'error') {
        toast.error(`Stream error: ${data.message}`);
        stopStream();
      }
    };

    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
      setIsStreaming(false);
      toast.info('Stream ended');
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast.error('WebSocket connection failed');
      setIsStreaming(false);
    };
  };

  const stopStream = () => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: 'stop_stream' }));
      wsRef.current.close();
    }
    setIsStreaming(false);
    setCurrentSessionId(null);
  };

  const clearStreamData = () => {
    setStreamData([]);
    toast.success('Stream data cleared');
  };

  // Generate mock stream data for demonstration
  useEffect(() => {
    if (isStreaming && streamData.length === 0) {
      const interval = setInterval(() => {
        const newDataPoint = {
          timestamp: Date.now(),
          time: new Date().toLocaleTimeString(),
          consciousness_level: Math.random() * 0.8 + 0.2,
          i2c_activations: Array.from({ length: 8 }, () => Math.random()),
          attention_patterns: {
            self_attention: Math.random() * 0.4 + 0.4,
            environmental_attention: Math.random() * 0.4 + 0.2,
            memory_attention: Math.random() * 0.4 + 0.3
          },
          history_length: streamData.length + 1,
          system_metrics: {
            cpu_usage: Math.random() * 30 + 20,
            memory_usage: Math.random() * 20 + 30,
            timestamp: new Date().toISOString()
          },
          remaining_time: Math.max(0, selectedDuration - (streamData.length * updateInterval))
        };

        setStreamData(prev => {
          const updated = [...prev, newDataPoint];
          if (updated.length >= selectedDuration / updateInterval) {
            setIsStreaming(false);
            toast.info('Stream completed');
            return updated;
          }
          return updated.slice(-100);
        });
      }, updateInterval * 1000);

      return () => clearInterval(interval);
    }
  }, [isStreaming, streamData.length, selectedDuration, updateInterval]);

  const getCurrentConsciousnessLevel = () => {
    if (streamData.length === 0) return 0;
    return streamData[streamData.length - 1].consciousness_level || 0;
  };

  const getAverageConsciousness = () => {
    if (streamData.length === 0) return 0;
    const sum = streamData.reduce((acc, data) => acc + (data.consciousness_level || 0), 0);
    return sum / streamData.length;
  };

  return (
    <div className="p-6 h-full overflow-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Stream Manager</h1>
        <p className="text-gray-600">Monitor real-time consciousness streams and system activity</p>
      </div>

      {/* Stream Controls */}
      <div className="bg-white rounded-xl shadow-md p-6 mb-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Stream Controls</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Duration (seconds)
            </label>
            <select
              value={selectedDuration}
              onChange={(e) => setSelectedDuration(parseInt(e.target.value))}
              disabled={isStreaming}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            >
              <option value={10}>10 seconds</option>
              <option value={30}>30 seconds</option>
              <option value={60}>1 minute</option>
              <option value={120}>2 minutes</option>
              <option value={300}>5 minutes</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Update Interval (seconds)
            </label>
            <select
              value={updateInterval}
              onChange={(e) => setUpdateInterval(parseFloat(e.target.value))}
              disabled={isStreaming}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            >
              <option value={0.5}>0.5 seconds</option>
              <option value={1.0}>1 second</option>
              <option value={2.0}>2 seconds</option>
              <option value={5.0}>5 seconds</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Options</label>
            <label className="flex items-center space-x-2 mt-2">
              <input
                type="checkbox"
                checked={includeConsciousness}
                onChange={(e) => setIncludeConsciousness(e.target.checked)}
                disabled={isStreaming}
                className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50 disabled:opacity-50"
              />
              <span className="text-sm text-gray-700">Include Consciousness Monitoring</span>
            </label>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <button
            onClick={startConsciousnessStream}
            disabled={isStreaming}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <PlayIcon className="w-4 h-4 mr-2" />
            Start Stream
          </button>
          
          <button
            onClick={stopStream}
            disabled={!isStreaming}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <StopIcon className="w-4 h-4 mr-2" />
            Stop Stream
          </button>
          
          <button
            onClick={clearStreamData}
            disabled={isStreaming || streamData.length === 0}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <TrashIcon className="w-4 h-4 mr-2" />
            Clear Data
          </button>
        </div>
      </div>

      {/* Stream Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-xl shadow-md p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Stream Status</p>
              <p className="text-lg font-bold text-gray-900">
                {isStreaming ? '🟢 Active' : '🔴 Inactive'}
              </p>
            </div>
            <SignalIcon className={`w-8 h-8 ${isStreaming ? 'text-green-500' : 'text-gray-400'}`} />
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Data Points</p>
              <p className="text-lg font-bold text-gray-900">{streamData.length}</p>
            </div>
            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
              <span className="text-blue-600 font-bold">{streamData.length}</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Current Consciousness</p>
              <p className="text-lg font-bold text-gray-900">
                {(getCurrentConsciousnessLevel() * 100).toFixed(1)}%
              </p>
            </div>
            <CpuChipIcon className="w-8 h-8 text-purple-500" />
          </div>
          <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-purple-400 to-pink-500 h-2 rounded-full transition-all duration-300" 
              style={{ width: `${getCurrentConsciousnessLevel() * 100}%` }}
            ></div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Average φ</p>
              <p className="text-lg font-bold text-gray-900">
                {(getAverageConsciousness() * 100).toFixed(1)}%
              </p>
            </div>
            <div className="w-8 h-8 rounded-full bg-yellow-100 flex items-center justify-center">
              <span className="text-yellow-600 font-bold">φ</span>
            </div>
          </div>
        </div>
      </div>

      {/* Real-time Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Consciousness Timeline */}
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Consciousness Timeline</h3>
          <div className="h-64 overflow-hidden">
            {streamData.length > 0 ? (
              <div className="flex items-end h-full space-x-1">
                {streamData.slice(-50).map((data, index) => (
                  <div
                    key={index}
                    className="flex-1 bg-gradient-to-t from-purple-400 to-pink-500 rounded-t transition-all duration-300"
                    style={{ height: `${(data.consciousness_level || 0) * 100}%` }}
                    title={`${data.time}: ${((data.consciousness_level || 0) * 100).toFixed(1)}%`}
                  ></div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <CpuChipIcon className="w-12 h-12 mx-auto mb-2" />
                  <p>Start a stream to see consciousness data</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* I²C Unit Activations */}
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">I²C Unit Activations</h3>
          <div className="h-64">
            {streamData.length > 0 && streamData[streamData.length - 1].i2c_activations ? (
              <div className="grid grid-cols-4 gap-2 h-full">
                {streamData[streamData.length - 1].i2c_activations.map((activation, index) => (
                  <div key={index} className="flex flex-col items-center justify-end">
                    <div className="text-xs text-gray-600 mb-1">U{index + 1}</div>
                    <div 
                      className="w-full bg-gradient-to-t from-blue-400 to-cyan-500 rounded transition-all duration-300"
                      style={{ height: `${activation * 100}%` }}
                    ></div>
                    <div className="text-xs text-gray-500 mt-1">
                      {(activation * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <div className="w-12 h-12 mx-auto mb-2 bg-gray-200 rounded-lg flex items-center justify-center">
                    <span className="text-gray-400 font-bold">I²C</span>
                  </div>
                  <p>I²C activations will appear here</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stream Log */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Stream Log</h3>
        <div className="h-64 overflow-y-auto bg-gray-900 rounded-lg p-4 font-mono text-sm">
          {streamData.length > 0 ? (
            streamData.slice(-20).map((data, index) => (
              <div key={index} className="text-green-400 mb-1">
                <span className="text-gray-500">[{data.time}]</span> 
                <span className="text-yellow-400"> φ={((data.consciousness_level || 0) * 100).toFixed(1)}%</span>
                {data.system_metrics && (
                  <span className="text-blue-400"> CPU={data.system_metrics.cpu_usage?.toFixed(1)}%</span>
                )}
                {data.remaining_time !== undefined && (
                  <span className="text-purple-400"> remaining={data.remaining_time.toFixed(1)}s</span>
                )}
              </div>
            ))
          ) : (
            <div className="text-gray-500 italic">Stream log will appear here...</div>
          )}
        </div>
      </div>
    </div>
  );
}

export default StreamManager;