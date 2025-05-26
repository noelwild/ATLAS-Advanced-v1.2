import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { 
  ChartBarIcon, 
  ChatBubbleLeftRightIcon, 
  CpuChipIcon, 
  CommandLineIcon,
  BookOpenIcon,
  PlayCircleIcon,
  HomeIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline';

// Import components
import Dashboard from './components/Dashboard';
import ChatInterface from './components/ChatInterface';
import SystemMonitor from './components/SystemMonitor';
import CodeExecutor from './components/CodeExecutor';
import MemoryExplorer from './components/MemoryExplorer';
import StreamManager from './components/StreamManager';
import TestRunner from './components/TestRunner';
import Settings from './components/Settings';

import './App.css';

function App() {
  const [systemStatus, setSystemStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/status`);
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <Router>
      <div className="flex h-screen bg-gray-100">
        <Toaster position="top-right" />
        
        {/* Sidebar */}
        <Sidebar systemStatus={systemStatus} />
        
        {/* Main Content */}
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Dashboard systemStatus={systemStatus} />} />
            <Route path="/chat" element={<ChatInterface />} />
            <Route path="/monitor" element={<SystemMonitor systemStatus={systemStatus} />} />
            <Route path="/code" element={<CodeExecutor />} />
            <Route path="/memory" element={<MemoryExplorer />} />
            <Route path="/streams" element={<StreamManager />} />
            <Route path="/test" element={<TestRunner />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

function Sidebar({ systemStatus }) {
  const location = useLocation();
  
  const navigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Chat Interface', href: '/chat', icon: ChatBubbleLeftRightIcon },
    { name: 'System Monitor', href: '/monitor', icon: ChartBarIcon },
    { name: 'Code Executor', href: '/code', icon: CommandLineIcon },
    { name: 'Memory Explorer', href: '/memory', icon: BookOpenIcon },
    { name: 'Stream Manager', href: '/streams', icon: PlayCircleIcon },
    { name: 'Test Runner', href: '/test', icon: CpuChipIcon },
    { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
  ];

  const getStatusColor = () => {
    if (!systemStatus) return 'bg-gray-500';
    if (systemStatus.status === 'running' && systemStatus.model_loaded) return 'bg-green-500';
    if (systemStatus.status === 'initializing') return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="w-64 bg-white shadow-lg">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <CpuChipIcon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-900">ATLAS</h1>
            <p className="text-sm text-gray-500">Consciousness Monitor</p>
          </div>
        </div>
        
        {/* Status Indicator */}
        <div className="mt-4 flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${getStatusColor()}`}></div>
          <span className="text-sm text-gray-600">
            {systemStatus ? systemStatus.status : 'Connecting...'}
          </span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="mt-6">
        <div className="px-3">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`
                  group flex items-center px-3 py-2 text-sm font-medium rounded-md mb-1
                  ${isActive 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
                  }
                `}
              >
                <item.icon 
                  className={`
                    mr-3 h-5 w-5
                    ${isActive ? 'text-blue-500' : 'text-gray-400'}
                  `} 
                />
                {item.name}
              </Link>
            );
          })}
        </div>
      </nav>

      {/* System Info */}
      {systemStatus && (
        <div className="absolute bottom-0 w-64 p-4 border-t border-gray-200 bg-white">
          <div className="text-xs text-gray-500 space-y-1">
            <div>Uptime: {systemStatus.uptime}</div>
            <div>CPU: {systemStatus.cpu_usage.toFixed(1)}%</div>
            <div>Memory: {systemStatus.memory_usage.toFixed(1)}%</div>
            {systemStatus.gpu_usage && (
              <div>GPU: {systemStatus.gpu_usage.toFixed(1)}%</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;