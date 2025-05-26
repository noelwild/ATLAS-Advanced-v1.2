import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { 
  PlayIcon, 
  TrashIcon,
  ClockIcon,
  ExclamationTriangleIcon 
} from '@heroicons/react/24/outline';

function CodeExecutor() {
  const [code, setCode] = useState('# Enter your Python code here\nprint("Hello from ATLAS!")\n\n# Simple calculations\nresult = 2 + 2\nprint(f"2 + 2 = {result}")');
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [executionTime, setExecutionTime] = useState(null);
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('python');
  const [executionHistory, setExecutionHistory] = useState([]);

  const executeCode = async () => {
    if (!code.trim()) {
      toast.error('Please enter some code to execute');
      return;
    }

    setLoading(true);
    setOutput('');
    setError('');
    setExecutionTime(null);

    try {
      const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/code/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code: code,
          language: language,
          session_id: `code_session_${Date.now()}`
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setOutput(data.output || '');
        setError(data.error || '');
        setExecutionTime(data.execution_time);
        
        // Add to history
        const historyEntry = {
          id: Date.now(),
          code: code,
          output: data.output || '',
          error: data.error || '',
          executionTime: data.execution_time,
          timestamp: new Date().toLocaleString(),
          language: language
        };
        
        setExecutionHistory(prev => [historyEntry, ...prev.slice(0, 9)]); // Keep last 10
        
        if (data.error) {
          toast.error('Code executed with errors');
        } else {
          toast.success('Code executed successfully');
        }
      } else {
        throw new Error(data.detail || 'Failed to execute code');
      }
    } catch (error) {
      console.error('Error executing code:', error);
      setError(error.message);
      toast.error(`Execution failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const clearCode = () => {
    setCode('');
    setOutput('');
    setError('');
    setExecutionTime(null);
    toast.success('Code cleared');
  };

  const clearHistory = () => {
    setExecutionHistory([]);
    toast.success('History cleared');
  };

  const loadFromHistory = (historyItem) => {
    setCode(historyItem.code);
    setOutput(historyItem.output);
    setError(historyItem.error);
    setExecutionTime(historyItem.executionTime);
    setLanguage(historyItem.language);
    toast.success('Code loaded from history');
  };

  return (
    <div className="p-6 h-full overflow-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Code Executor</h1>
        <p className="text-gray-600">Execute code with ATLAS consciousness monitoring</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
        {/* Code Input */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-xl shadow-md h-full flex flex-col">
            {/* Code Editor Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <div className="flex items-center space-x-4">
                <h3 className="text-lg font-semibold text-gray-900">Code Editor</h3>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="python">Python</option>
                  <option value="javascript">JavaScript</option>
                  <option value="bash">Bash</option>
                </select>
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={clearCode}
                  className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
                >
                  <TrashIcon className="w-4 h-4 mr-2" />
                  Clear
                </button>
                <button
                  onClick={executeCode}
                  disabled={loading || !code.trim()}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <div className="w-4 h-4 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  ) : (
                    <PlayIcon className="w-4 h-4 mr-2" />
                  )}
                  {loading ? 'Executing...' : 'Execute'}
                </button>
              </div>
            </div>

            {/* Code Textarea */}
            <div className="flex-1 p-4">
              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="Enter your code here..."
                className="w-full h-full resize-none border border-gray-300 rounded-lg p-4 font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                style={{ minHeight: '400px' }}
              />
            </div>

            {/* Output Section */}
            <div className="border-t border-gray-200">
              {/* Output Header */}
              <div className="flex items-center justify-between p-4 bg-gray-50">
                <h4 className="text-md font-semibold text-gray-900">Output</h4>
                {executionTime !== null && (
                  <div className="flex items-center text-sm text-gray-600">
                    <ClockIcon className="w-4 h-4 mr-1" />
                    {(executionTime * 1000).toFixed(2)} ms
                  </div>
                )}
              </div>

              {/* Output Content */}
              <div className="p-4 bg-gray-900 text-white min-h-32 max-h-48 overflow-auto">
                {error ? (
                  <div className="text-red-400">
                    <div className="flex items-center mb-2">
                      <ExclamationTriangleIcon className="w-4 h-4 mr-2" />
                      Error:
                    </div>
                    <pre className="whitespace-pre-wrap text-sm">{error}</pre>
                  </div>
                ) : output ? (
                  <pre className="whitespace-pre-wrap text-sm text-green-400">{output}</pre>
                ) : (
                  <div className="text-gray-500 italic">Output will appear here...</div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Execution History */}
        <div className="bg-white rounded-xl shadow-md h-full flex flex-col">
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Execution History</h3>
            {executionHistory.length > 0 && (
              <button
                onClick={clearHistory}
                className="text-sm text-gray-600 hover:text-gray-900"
              >
                Clear
              </button>
            )}
          </div>
          
          <div className="flex-1 overflow-auto p-4">
            {executionHistory.length === 0 ? (
              <div className="text-center text-gray-500 mt-8">
                <PlayIcon className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>No executions yet</p>
                <p className="text-sm">Execute some code to see history</p>
              </div>
            ) : (
              <div className="space-y-3">
                {executionHistory.map((item) => (
                  <div
                    key={item.id}
                    onClick={() => loadFromHistory(item)}
                    className="border border-gray-200 rounded-lg p-3 cursor-pointer hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs text-gray-500">{item.timestamp}</span>
                      <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
                        {item.language}
                      </span>
                    </div>
                    
                    <div className="mb-2">
                      <SyntaxHighlighter
                        language={item.language}
                        style={tomorrow}
                        customStyle={{
                          fontSize: '10px',
                          margin: 0,
                          padding: '8px',
                          borderRadius: '4px',
                          maxHeight: '60px',
                          overflow: 'hidden'
                        }}
                      >
                        {item.code.length > 100 ? item.code.substring(0, 100) + '...' : item.code}
                      </SyntaxHighlighter>
                    </div>
                    
                    <div className="flex items-center justify-between text-xs">
                      <span className={`flex items-center ${item.error ? 'text-red-600' : 'text-green-600'}`}>
                        {item.error ? '❌ Error' : '✅ Success'}
                      </span>
                      <span className="text-gray-500">
                        {(item.executionTime * 1000).toFixed(1)}ms
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Examples */}
      <div className="mt-6 bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Examples</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button
            onClick={() => setCode('# Basic arithmetic\nprint("2 + 2 =", 2 + 2)\nprint("10 * 5 =", 10 * 5)\nprint("15 / 3 =", 15 / 3)')}
            className="text-left p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <h4 className="font-medium text-gray-900 mb-1">Basic Math</h4>
            <p className="text-sm text-gray-600">Simple arithmetic operations</p>
          </button>
          
          <button
            onClick={() => setCode('# List operations\nnumbers = [1, 2, 3, 4, 5]\nprint("Original list:", numbers)\nprint("Squared:", [x**2 for x in numbers])\nprint("Sum:", sum(numbers))')}
            className="text-left p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <h4 className="font-medium text-gray-900 mb-1">List Operations</h4>
            <p className="text-sm text-gray-600">Working with Python lists</p>
          </button>
          
          <button
            onClick={() => setCode('# String manipulation\ntext = "Hello, ATLAS!"\nprint("Original:", text)\nprint("Uppercase:", text.upper())\nprint("Length:", len(text))\nprint("Words:", text.split())')}
            className="text-left p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <h4 className="font-medium text-gray-900 mb-1">String Functions</h4>
            <p className="text-sm text-gray-600">String manipulation examples</p>
          </button>
        </div>
      </div>
    </div>
  );
}

export default CodeExecutor;