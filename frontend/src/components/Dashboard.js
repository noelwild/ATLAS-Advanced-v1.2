import React, { useState, useEffect } from 'react';
import { 
  ChartBarIcon, 
  ChatBubbleLeftRightIcon, 
  CpuChipIcon,
  CommandLineIcon,
  BookOpenIcon,
  PlayCircleIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

function Dashboard({ systemStatus }) {
  const [consciousnessData, setConsciousnessData] = useState([]);
  const [recentActivity, setRecentActivity] = useState([]);

  useEffect(() => {
    // Simulate consciousness data
    const interval = setInterval(() => {
      const newPoint = {
        time: new Date().toLocaleTimeString(),
        consciousness: Math.random() * 0.8 + 0.2,
        i2c_avg: Math.random() * 0.6 + 0.3
      };
      
      setConsciousnessData(prev => {
        const updated = [...prev, newPoint];
        return updated.slice(-20); // Keep last 20 points
      });
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const quickActions = [
    {
      title: 'Start Chat',
      description: 'Interact with ATLAS consciousness',
      icon: ChatBubbleLeftRightIcon,
      color: 'bg-blue-500',
      href: '/chat'
    },
    {
      title: 'System Monitor',
      description: 'View real-time system metrics',
      icon: ChartBarIcon,
      color: 'bg-green-500',
      href: '/monitor'
    },
    {
      title: 'Execute Code',
      description: 'Run code with ATLAS',
      icon: CommandLineIcon,
      color: 'bg-purple-500',
      href: '/code'
    },
    {
      title: 'Explore Memory',
      description: 'Browse stored memories',
      icon: BookOpenIcon,
      color: 'bg-orange-500',
      href: '/memory'
    },
    {
      title: 'Manage Streams',
      description: 'Control consciousness streams',
      icon: PlayCircleIcon,
      color: 'bg-red-500',
      href: '/streams'
    },
    {
      title: 'Run Tests',
      description: 'Execute system tests',
      icon: CpuChipIcon,
      color: 'bg-indigo-500',
      href: '/test'
    }
  ];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-700 rounded-2xl p-8 text-white">
        <h1 className="text-3xl font-bold mb-2">ATLAS Dashboard</h1>
        <p className="text-blue-100 text-lg">
          Advanced Thinking and Learning AI System - Consciousness Monitoring
        </p>
        
        {systemStatus && (
          <div className="mt-6 grid grid-cols-3 gap-4">
            <div className="bg-white/10 rounded-lg p-4">
              <div className="text-2xl font-bold">
                {systemStatus.model_loaded ? '✅' : '⏳'}
              </div>
              <div className="text-sm text-blue-100">Model Status</div>
            </div>
            <div className="bg-white/10 rounded-lg p-4">
              <div className="text-2xl font-bold">
                {systemStatus.consciousness_active ? '🧠' : '💤'}
              </div>
              <div className="text-sm text-blue-100">Consciousness</div>
            </div>
            <div className="bg-white/10 rounded-lg p-4">
              <div className="text-2xl font-bold">{systemStatus.memory_count}</div>
              <div className="text-sm text-blue-100">Memories</div>
            </div>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {quickActions.map((action, index) => (
            <a
              key={index}
              href={action.href}
              className="block p-6 bg-white rounded-xl shadow-md hover:shadow-lg transition-shadow border border-gray-200 hover:border-gray-300"
            >
              <div className="flex items-center space-x-4">
                <div className={`p-3 rounded-lg ${action.color}`}>
                  <action.icon className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{action.title}</h3>
                  <p className="text-sm text-gray-600">{action.description}</p>
                </div>
              </div>
            </a>
          ))}
        </div>
      </div>

      {/* Consciousness Monitor */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Consciousness Level Over Time
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={consciousnessData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="consciousness" 
                  stroke="#3B82F6" 
                  strokeWidth={2}
                  name="Consciousness"
                />
                <Line 
                  type="monotone" 
                  dataKey="i2c_avg" 
                  stroke="#10B981" 
                  strokeWidth={2}
                  name="I²C Average"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            System Metrics
          </h3>
          {systemStatus ? (
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm">
                  <span>CPU Usage</span>
                  <span>{systemStatus.cpu_usage.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div 
                    className="bg-blue-500 h-2 rounded-full" 
                    style={{ width: `${systemStatus.cpu_usage}%` }}
                  ></div>
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm">
                  <span>Memory Usage</span>
                  <span>{systemStatus.memory_usage.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div 
                    className="bg-green-500 h-2 rounded-full" 
                    style={{ width: `${systemStatus.memory_usage}%` }}
                  ></div>
                </div>
              </div>

              {systemStatus.gpu_usage && (
                <div>
                  <div className="flex justify-between text-sm">
                    <span>GPU Usage</span>
                    <span>{systemStatus.gpu_usage.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                    <div 
                      className="bg-purple-500 h-2 rounded-full" 
                      style={{ width: `${systemStatus.gpu_usage}%` }}
                    ></div>
                  </div>
                </div>
              )}

              <div className="pt-4 border-t border-gray-200">
                <div className="text-sm text-gray-600 space-y-1">
                  <div>Status: <span className="font-medium">{systemStatus.status}</span></div>
                  <div>Uptime: <span className="font-medium">{systemStatus.uptime}</span></div>
                  <div>Stored Memories: <span className="font-medium">{systemStatus.memory_count}</span></div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-32">
              <div className="text-gray-500">Loading system metrics...</div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
        <div className="space-y-3">
          <div className="flex items-center space-x-3 p-3 bg-blue-50 rounded-lg">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <div className="flex-1">
              <div className="text-sm font-medium text-gray-900">
                ATLAS System Initialized
              </div>
              <div className="text-xs text-gray-500">
                Consciousness monitoring active
              </div>
            </div>
            <div className="text-xs text-gray-500">
              {systemStatus ? systemStatus.uptime : 'now'}
            </div>
          </div>
          
          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <div className="flex-1">
              <div className="text-sm font-medium text-gray-900">
                Model Loaded Successfully
              </div>
              <div className="text-xs text-gray-500">
                Qwen 0.5B model ready for inference
              </div>
            </div>
            <div className="text-xs text-gray-500">recently</div>
          </div>

          <div className="flex items-center space-x-3 p-3 bg-purple-50 rounded-lg">
            <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
            <div className="flex-1">
              <div className="text-sm font-medium text-gray-900">
                Database Connected
              </div>
              <div className="text-xs text-gray-500">
                MongoDB connection established
              </div>
            </div>
            <div className="text-xs text-gray-500">recently</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;