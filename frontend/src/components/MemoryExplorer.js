import React, { useState, useEffect } from 'react';
import { toast } from 'react-hot-toast';
import { 
  MagnifyingGlassIcon, 
  BookOpenIcon,
  ClockIcon,
  HashtagIcon 
} from '@heroicons/react/24/outline';

function MemoryExplorer() {
  const [memories, setMemories] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [searchLimit, setSearchLimit] = useState(10);
  const [selectedMemory, setSelectedMemory] = useState(null);

  const searchMemories = async () => {
    if (!searchQuery.trim()) {
      toast.error('Please enter a search query');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(
        `${process.env.REACT_APP_BACKEND_URL}/api/memory/search?query=${encodeURIComponent(searchQuery)}&limit=${searchLimit}`
      );
      
      const data = await response.json();
      
      if (response.ok) {
        setMemories(data.memories || []);
        toast.success(`Found ${data.count || 0} memories`);
      } else {
        throw new Error(data.detail || 'Failed to search memories');
      }
    } catch (error) {
      console.error('Error searching memories:', error);
      toast.error(`Search failed: ${error.message}`);
      setMemories([]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      searchMemories();
    }
  };

  // Mock data for demonstration
  const mockMemories = [
    {
      _id: '1',
      content: 'User asked about consciousness and AI awareness. Discussed the nature of consciousness monitoring and I²C cells.',
      type: 'conversation',
      timestamp: new Date().toISOString(),
      metadata: {
        session_id: 'session_123',
        consciousness_level: 0.7,
        keywords: ['consciousness', 'AI', 'awareness']
      }
    },
    {
      _id: '2',
      content: 'Executed Python code for calculating fibonacci sequence. Code ran successfully with 89ms execution time.',
      type: 'code_execution',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      metadata: {
        language: 'python',
        execution_time: 0.089,
        success: true
      }
    },
    {
      _id: '3',
      content: 'User explored system monitoring features. Showed interest in real-time consciousness tracking.',
      type: 'interaction',
      timestamp: new Date(Date.now() - 7200000).toISOString(),
      metadata: {
        page: 'system_monitor',
        consciousness_level: 0.6,
        duration: 300
      }
    }
  ];

  useEffect(() => {
    // If no real memories, show mock data
    if (memories.length === 0 && !loading && !searchQuery) {
      setMemories(mockMemories);
    }
  }, [memories.length, loading, searchQuery]);

  const getMemoryTypeColor = (type) => {
    switch (type) {
      case 'conversation': return 'bg-blue-100 text-blue-800';
      case 'code_execution': return 'bg-green-100 text-green-800';
      case 'interaction': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getMemoryIcon = (type) => {
    switch (type) {
      case 'conversation': return '💬';
      case 'code_execution': return '💻';
      case 'interaction': return '🔗';
      default: return '📝';
    }
  };

  return (
    <div className="p-6 h-full overflow-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Memory Explorer</h1>
        <p className="text-gray-600">Search and explore ATLAS stored memories</p>
      </div>

      {/* Search Interface */}
      <div className="bg-white rounded-xl shadow-md p-6 mb-6">
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <div className="relative">
              <MagnifyingGlassIcon className="w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Search memories by content, keywords, or type..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>
          
          <select
            value={searchLimit}
            onChange={(e) => setSearchLimit(parseInt(e.target.value))}
            className="border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value={5}>5 results</option>
            <option value={10}>10 results</option>
            <option value={25}>25 results</option>
            <option value={50}>50 results</option>
          </select>
          
          <button
            onClick={searchMemories}
            disabled={loading || !searchQuery.trim()}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <div className="w-4 h-4 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <MagnifyingGlassIcon className="w-4 h-4 mr-2" />
            )}
            Search
          </button>
        </div>
      </div>

      {/* Results */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Memory List */}
        <div className="bg-white rounded-xl shadow-md">
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">
              Memory Results ({memories.length})
            </h3>
          </div>
          
          <div className="max-h-96 overflow-y-auto">
            {memories.length === 0 && !loading ? (
              <div className="p-8 text-center">
                <BookOpenIcon className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No memories found</h3>
                <p className="text-gray-600">
                  {searchQuery ? 'Try a different search query' : 'Search for memories to explore stored data'}
                </p>
              </div>
            ) : (
              <div className="p-4 space-y-3">
                {memories.map((memory) => (
                  <div
                    key={memory._id}
                    onClick={() => setSelectedMemory(memory)}
                    className={`p-4 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors ${
                      selectedMemory?._id === memory._id ? 'ring-2 ring-blue-500 border-blue-300' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-lg">{getMemoryIcon(memory.type)}</span>
                        <span className={`text-xs px-2 py-1 rounded-full ${getMemoryTypeColor(memory.type)}`}>
                          {memory.type}
                        </span>
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(memory.timestamp).toLocaleString()}
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-900 line-clamp-3">
                      {memory.content.length > 150 
                        ? memory.content.substring(0, 150) + '...'
                        : memory.content
                      }
                    </p>
                    
                    {memory.metadata && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {memory.metadata.keywords?.slice(0, 3).map((keyword, index) => (
                          <span key={index} className="text-xs px-2 py-1 bg-gray-100 text-gray-600 rounded">
                            #{keyword}
                          </span>
                        ))}
                        {memory.metadata.consciousness_level && (
                          <span className="text-xs px-2 py-1 bg-yellow-100 text-yellow-800 rounded">
                            φ {(memory.metadata.consciousness_level * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Memory Details */}
        <div className="bg-white rounded-xl shadow-md">
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Memory Details</h3>
          </div>
          
          <div className="p-4">
            {selectedMemory ? (
              <div className="space-y-4">
                {/* Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="text-2xl">{getMemoryIcon(selectedMemory.type)}</span>
                    <span className={`text-sm px-3 py-1 rounded-full ${getMemoryTypeColor(selectedMemory.type)}`}>
                      {selectedMemory.type}
                    </span>
                  </div>
                  <div className="flex items-center text-sm text-gray-500">
                    <ClockIcon className="w-4 h-4 mr-1" />
                    {new Date(selectedMemory.timestamp).toLocaleString()}
                  </div>
                </div>

                {/* Content */}
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Content</h4>
                  <div className="p-3 bg-gray-50 rounded-lg text-sm text-gray-900">
                    {selectedMemory.content}
                  </div>
                </div>

                {/* Metadata */}
                {selectedMemory.metadata && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Metadata</h4>
                    <div className="space-y-2">
                      {Object.entries(selectedMemory.metadata).map(([key, value]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span className="text-gray-600 capitalize">{key.replace('_', ' ')}:</span>
                          <span className="text-gray-900 font-medium">
                            {typeof value === 'boolean' 
                              ? (value ? 'Yes' : 'No')
                              : typeof value === 'number'
                              ? value.toFixed(3)
                              : Array.isArray(value)
                              ? value.join(', ')
                              : value
                            }
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Keywords */}
                {selectedMemory.metadata?.keywords && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Keywords</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedMemory.metadata.keywords.map((keyword, index) => (
                        <span key={index} className="inline-flex items-center text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
                          <HashtagIcon className="w-3 h-3 mr-1" />
                          {keyword}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Consciousness Level */}
                {selectedMemory.metadata?.consciousness_level && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Consciousness Level</h4>
                    <div className="flex items-center space-x-3">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-yellow-400 to-orange-500 h-2 rounded-full" 
                          style={{ width: `${selectedMemory.metadata.consciousness_level * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-gray-900">
                        {(selectedMemory.metadata.consciousness_level * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <BookOpenIcon className="w-12 h-12 mb-4" />
                <p>Select a memory to view details</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Memory Statistics */}
      <div className="mt-6 bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Memory Statistics</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{memories.length}</div>
            <div className="text-sm text-blue-800">Total Memories</div>
          </div>
          
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {memories.filter(m => m.type === 'conversation').length}
            </div>
            <div className="text-sm text-green-800">Conversations</div>
          </div>
          
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">
              {memories.filter(m => m.type === 'code_execution').length}
            </div>
            <div className="text-sm text-purple-800">Code Executions</div>
          </div>
          
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <div className="text-2xl font-bold text-orange-600">
              {memories.filter(m => m.metadata?.consciousness_level > 0.5).length}
            </div>
            <div className="text-sm text-orange-800">High Consciousness</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MemoryExplorer;