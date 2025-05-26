import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { 
  CpuChipIcon, 
  CircleStackIcon, 
  BoltIcon,
  ClockIcon 
} from '@heroicons/react/24/outline';

function SystemMonitor({ systemStatus }) {
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [consciousnessState, setConsciousnessState] = useState(null);
  const [sessions, setSessions] = useState([]);

  useEffect(() => {
    // Fetch consciousness state
    const fetchConsciousnessState = async () => {
      try {
        const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/consciousness/current`);
        const data = await response.json();
        setConsciousnessState(data);
      } catch (error) {
        console.error('Failed to fetch consciousness state:', error);
      }
    };

    // Fetch active sessions
    const fetchSessions = async () => {
      try {
        const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/sessions`);
        const data = await response.json();
        setSessions(data.active_sessions || []);
      } catch (error) {
        console.error('Failed to fetch sessions:', error);
      }
    };

    fetchConsciousnessState();
    fetchSessions();

    const interval = setInterval(() => {
      fetchConsciousnessState();
      fetchSessions();
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (systemStatus) {
      const newMetric = {
        time: new Date().toLocaleTimeString(),
        cpu: systemStatus.cpu_usage,
        memory: systemStatus.memory_usage,
        gpu: systemStatus.gpu_usage || 0,
        consciousness: consciousnessState?.consciousness_level * 100 || 0
      };

      setMetricsHistory(prev => {
        const updated = [...prev, newMetric];
        return updated.slice(-20); // Keep last 20 points
      });
    }
  }, [systemStatus, consciousnessState]);

  const formatTime = (seconds) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs}h ${mins}m ${secs}s`;
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">System Monitor</h1>
        <p className="text-gray-600">Real-time monitoring of ATLAS system metrics</p>
      </div>

      {/* System Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">CPU Usage</p>
              <p className="text-2xl font-bold text-gray-900">
                {systemStatus ? `${systemStatus.cpu_usage.toFixed(1)}%` : '0%'}
              </p>
            </div>
            <div className="p-3 bg-blue-100 rounded-lg">
              <CpuChipIcon className="w-6 h-6 text-blue-600" />
            </div>
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${systemStatus?.cpu_usage || 0}%` }}
              ></div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Memory Usage</p>
              <p className="text-2xl font-bold text-gray-900">
                {systemStatus ? `${systemStatus.memory_usage.toFixed(1)}%` : '0%'}
              </p>
            </div>
            <div className="p-3 bg-green-100 rounded-lg">
              <CircleStackIcon className="w-6 h-6 text-green-600" />
            </div>
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-green-500 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${systemStatus?.memory_usage || 0}%` }}
              ></div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">GPU Usage</p>
              <p className="text-2xl font-bold text-gray-900">
                {systemStatus?.gpu_usage ? `${systemStatus.gpu_usage.toFixed(1)}%` : 'N/A'}
              </p>
            </div>
            <div className="p-3 bg-purple-100 rounded-lg">
              <BoltIcon className="w-6 h-6 text-purple-600" />
            </div>
          </div>
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-purple-500 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${systemStatus?.gpu_usage || 0}%` }}
              ></div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Uptime</p>
              <p className="text-lg font-bold text-gray-900">
                {systemStatus ? systemStatus.uptime : '0:00:00'}
              </p>
            </div>
            <div className="p-3 bg-orange-100 rounded-lg">
              <ClockIcon className="w-6 h-6 text-orange-600" />
            </div>
          </div>
          <div className="mt-4">
            <p className="text-sm text-gray-600">
              Status: <span className="font-medium">{systemStatus?.status || 'Unknown'}</span>
            </p>
          </div>
        </div>
      </div>

      {/* Metrics Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Metrics Over Time</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metricsHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line type="monotone" dataKey="cpu" stroke="#3B82F6" strokeWidth={2} name="CPU %" />
                <Line type="monotone" dataKey="memory" stroke="#10B981" strokeWidth={2} name="Memory %" />
                <Line type="monotone" dataKey="gpu" stroke="#8B5CF6" strokeWidth={2} name="GPU %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Consciousness Level</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metricsHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="consciousness" 
                  stroke="#F59E0B" 
                  strokeWidth={3} 
                  name="Consciousness %" 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Consciousness State Details */}
      {consciousnessState && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Consciousness State Details</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="text-md font-medium text-gray-700 mb-3">I²C Unit Activations</h4>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={consciousnessState.i2c_activations?.map((activation, index) => ({
                    unit: `Unit ${index + 1}`,
                    activation: activation * 100
                  })) || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="unit" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="activation" fill="#F59E0B" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div>
              <h4 className="text-md font-medium text-gray-700 mb-3">Attention Patterns</h4>
              <div className="space-y-4">
                {consciousnessState.attention_patterns && Object.entries(consciousnessState.attention_patterns).map(([pattern, value]) => (
                  <div key={pattern}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="capitalize">{pattern.replace('_', ' ')}</span>
                      <span>{(value * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-yellow-400 to-orange-500 h-2 rounded-full transition-all duration-300" 
                        style={{ width: `${value * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 p-4 bg-yellow-50 rounded-lg">
                <div className="text-sm text-gray-600">
                  <div>Current Level: <span className="font-medium">{(consciousnessState.consciousness_level * 100).toFixed(2)}%</span></div>
                  <div>History Length: <span className="font-medium">{consciousnessState.history_length}</span></div>
                  <div>Last Update: <span className="font-medium">{consciousnessState.timestamp}</span></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Active Sessions */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Active Sessions</h3>
        {sessions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Session ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Messages
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {sessions.map((session, index) => (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {session.session_id.substring(0, 20)}...
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(session.created).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {session.message_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-gray-500">No active sessions</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default SystemMonitor;