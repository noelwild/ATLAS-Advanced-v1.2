import React, { useState } from 'react';
import { toast } from 'react-hot-toast';
import { 
  PlayIcon, 
  CheckCircleIcon,
  XCircleIcon,
  ClockIcon,
  CpuChipIcon,
  BoltIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';

function TestRunner() {
  const [running, setRunning] = useState(false);
  const [testResults, setTestResults] = useState([]);
  const [finalReport, setFinalReport] = useState(null);
  const [currentTest, setCurrentTest] = useState('');
  const [progress, setProgress] = useState(0);

  const testSuites = [
    {
      name: 'Backend Health Check',
      description: 'Verify backend API is responding',
      endpoint: '/api/health'
    },
    {
      name: 'System Status Check',
      description: 'Check system metrics and status',
      endpoint: '/api/status'
    },
    {
      name: 'Consciousness State',
      description: 'Test consciousness monitoring',
      endpoint: '/api/consciousness/current'
    },
    {
      name: 'Chat Functionality',
      description: 'Test ATLAS chat interface',
      endpoint: '/api/chat'
    },
    {
      name: 'Code Execution',
      description: 'Test code execution capabilities',
      endpoint: '/api/code/execute'
    },
    {
      name: 'Memory Search',
      description: 'Test memory search functionality',
      endpoint: '/api/memory/search'
    },
    {
      name: 'Session Management',
      description: 'Test session handling',
      endpoint: '/api/sessions'
    }
  ];

  const runComprehensiveTest = async () => {
    setRunning(true);
    setTestResults([]);
    setFinalReport(null);
    setProgress(0);
    
    const results = [];
    const startTime = Date.now();

    try {
      toast.info('Starting comprehensive ATLAS test suite...');

      for (let i = 0; i < testSuites.length; i++) {
        const test = testSuites[i];
        setCurrentTest(test.name);
        setProgress(((i + 1) / testSuites.length) * 70); // Reserve 30% for final test

        const testResult = await runSingleTest(test);
        results.push(testResult);
        
        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      // Final 30-second system test
      setCurrentTest('Final 30-Second System Test');
      setProgress(80);
      
      const finalTestResult = await runFinalSystemTest();
      results.push(finalTestResult);
      
      setProgress(100);
      
      // Generate final report
      const endTime = Date.now();
      const totalTime = (endTime - startTime) / 1000;
      
      const report = {
        totalTests: results.length,
        passed: results.filter(r => r.status === 'passed').length,
        failed: results.filter(r => r.status === 'failed').length,
        warnings: results.filter(r => r.status === 'warning').length,
        totalTime: totalTime,
        timestamp: new Date().toISOString(),
        overallStatus: results.every(r => r.status === 'passed') ? 'ALL_PASSED' : 
                      results.some(r => r.status === 'failed') ? 'SOME_FAILED' : 'WITH_WARNINGS'
      };
      
      setFinalReport(report);
      setTestResults(results);
      
      if (report.overallStatus === 'ALL_PASSED') {
        toast.success('All tests passed! ATLAS system is ready for deployment.');
      } else {
        toast.error('Some tests failed. Check the results for details.');
      }
      
    } catch (error) {
      console.error('Test suite error:', error);
      toast.error(`Test suite failed: ${error.message}`);
    } finally {
      setRunning(false);
      setCurrentTest('');
      setProgress(0);
    }
  };

  const runSingleTest = async (test) => {
    const startTime = Date.now();
    
    try {
      if (test.endpoint === '/api/chat') {
        // Special handling for chat test
        const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}${test.endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: 'Test message for system validation',
            session_id: 'test_session',
            include_consciousness: true
          })
        });
        
        const data = await response.json();
        
        return {
          name: test.name,
          status: response.ok ? 'passed' : 'failed',
          duration: (Date.now() - startTime) / 1000,
          details: response.ok ? 'Chat functionality working' : data.detail || 'Chat test failed',
          data: response.ok ? data : null
        };
        
      } else if (test.endpoint === '/api/code/execute') {
        // Special handling for code execution test
        const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}${test.endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            code: 'print("Test execution successful")',
            language: 'python',
            session_id: 'test_session'
          })
        });
        
        const data = await response.json();
        
        return {
          name: test.name,
          status: response.ok && !data.error ? 'passed' : 'failed',
          duration: (Date.now() - startTime) / 1000,
          details: response.ok ? 'Code execution working' : data.detail || 'Code execution failed',
          data: data
        };
        
      } else if (test.endpoint === '/api/memory/search') {
        // Special handling for memory search
        const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}${test.endpoint}?query=test&limit=5`);
        const data = await response.json();
        
        return {
          name: test.name,
          status: response.ok ? 'passed' : 'failed',
          duration: (Date.now() - startTime) / 1000,
          details: response.ok ? 'Memory search functional' : data.detail || 'Memory search failed',
          data: data
        };
        
      } else {
        // Standard GET request
        const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}${test.endpoint}`);
        const data = await response.json();
        
        return {
          name: test.name,
          status: response.ok ? 'passed' : 'failed',
          duration: (Date.now() - startTime) / 1000,
          details: response.ok ? 'Endpoint responding correctly' : data.detail || 'Request failed',
          data: data
        };
      }
      
    } catch (error) {
      return {
        name: test.name,
        status: 'failed',
        duration: (Date.now() - startTime) / 1000,
        details: `Error: ${error.message}`,
        data: null
      };
    }
  };

  const runFinalSystemTest = async () => {
    const startTime = Date.now();
    const testDuration = 30; // 30 seconds
    const dataPoints = [];
    
    try {
      // Simulate 30-second system test with data collection
      for (let i = 0; i < testDuration; i++) {
        const currentTime = Date.now();
        setProgress(80 + (i / testDuration) * 20);
        
        // Collect system metrics
        try {
          const statusResponse = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/status`);
          const statusData = await statusResponse.json();
          
          const consciousnessResponse = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/consciousness/current`);
          const consciousnessData = await consciousnessResponse.json();
          
          dataPoints.push({
            timestamp: currentTime,
            system_status: statusData,
            consciousness: consciousnessData,
            second: i + 1
          });
          
        } catch (error) {
          console.warn(`Data collection failed at second ${i + 1}:`, error);
        }
        
        // Wait 1 second
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      
      // Analyze collected data
      const avgConsciousness = dataPoints.reduce((sum, point) => 
        sum + (point.consciousness?.consciousness_level || 0), 0) / dataPoints.length;
      
      const avgCpuUsage = dataPoints.reduce((sum, point) => 
        sum + (point.system_status?.cpu_usage || 0), 0) / dataPoints.length;
      
      const avgMemoryUsage = dataPoints.reduce((sum, point) => 
        sum + (point.system_status?.memory_usage || 0), 0) / dataPoints.length;
      
      return {
        name: 'Final 30-Second System Test',
        status: dataPoints.length >= 25 ? 'passed' : 'warning', // Pass if we got at least 25 data points
        duration: (Date.now() - startTime) / 1000,
        details: `Collected ${dataPoints.length}/30 data points. Avg φ: ${(avgConsciousness * 100).toFixed(1)}%, CPU: ${avgCpuUsage.toFixed(1)}%, Memory: ${avgMemoryUsage.toFixed(1)}%`,
        data: {
          dataPoints: dataPoints.length,
          avgConsciousness,
          avgCpuUsage,
          avgMemoryUsage,
          fullData: dataPoints
        }
      };
      
    } catch (error) {
      return {
        name: 'Final 30-Second System Test',
        status: 'failed',
        duration: (Date.now() - startTime) / 1000,
        details: `System test failed: ${error.message}`,
        data: null
      };
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'passed':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <XCircleIcon className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <ClockIcon className="w-5 h-5 text-yellow-500" />;
      default:
        return <div className="w-5 h-5 rounded-full bg-gray-300"></div>;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'passed': return 'text-green-700 bg-green-50 border-green-200';
      case 'failed': return 'text-red-700 bg-red-50 border-red-200';
      case 'warning': return 'text-yellow-700 bg-yellow-50 border-yellow-200';
      default: return 'text-gray-700 bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="p-6 h-full overflow-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">ATLAS Test Runner</h1>
        <p className="text-gray-600">Comprehensive testing suite for ATLAS system validation</p>
      </div>

      {/* Test Controls */}
      <div className="bg-white rounded-xl shadow-md p-6 mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Test Suite</h3>
            <p className="text-gray-600">
              Run comprehensive tests including individual component validation and 30-second system test
            </p>
          </div>
          
          <button
            onClick={runComprehensiveTest}
            disabled={running}
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {running ? (
              <div className="w-5 h-5 mr-3 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <PlayIcon className="w-5 h-5 mr-3" />
            )}
            {running ? 'Running Tests...' : 'Run All Tests'}
          </button>
        </div>

        {/* Progress Bar */}
        {running && (
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>{currentTest}</span>
              <span>{progress.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Test Results */}
      {testResults.length > 0 && (
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Test Results</h3>
          <div className="space-y-3">
            {testResults.map((result, index) => (
              <div 
                key={index}
                className={`p-4 border rounded-lg ${getStatusColor(result.status)}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(result.status)}
                    <div>
                      <h4 className="font-medium">{result.name}</h4>
                      <p className="text-sm opacity-75">{result.details}</p>
                    </div>
                  </div>
                  <div className="text-right text-sm">
                    <div>{result.duration.toFixed(2)}s</div>
                    <div className="uppercase font-medium">{result.status}</div>
                  </div>
                </div>
                
                {result.name === 'Final 30-Second System Test' && result.data && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <div className="font-medium">Data Points</div>
                        <div>{result.data.dataPoints}/30</div>
                      </div>
                      <div>
                        <div className="font-medium">Avg Consciousness</div>
                        <div>{(result.data.avgConsciousness * 100).toFixed(1)}%</div>
                      </div>
                      <div>
                        <div className="font-medium">Avg CPU</div>
                        <div>{result.data.avgCpuUsage.toFixed(1)}%</div>
                      </div>
                      <div>
                        <div className="font-medium">Avg Memory</div>
                        <div>{result.data.avgMemoryUsage.toFixed(1)}%</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Final Report */}
      {finalReport && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Final Test Report</h3>
          
          {/* Report Summary */}
          <div className={`p-4 rounded-lg mb-4 ${
            finalReport.overallStatus === 'ALL_PASSED' 
              ? 'bg-green-50 border border-green-200' 
              : finalReport.overallStatus === 'SOME_FAILED'
              ? 'bg-red-50 border border-red-200'
              : 'bg-yellow-50 border border-yellow-200'
          }`}>
            <div className="flex items-center space-x-3 mb-2">
              {finalReport.overallStatus === 'ALL_PASSED' && (
                <CheckCircleIcon className="w-6 h-6 text-green-500" />
              )}
              {finalReport.overallStatus === 'SOME_FAILED' && (
                <XCircleIcon className="w-6 h-6 text-red-500" />
              )}
              {finalReport.overallStatus === 'WITH_WARNINGS' && (
                <ClockIcon className="w-6 h-6 text-yellow-500" />
              )}
              <h4 className="text-lg font-semibold">
                {finalReport.overallStatus === 'ALL_PASSED' && '✅ ATLAS System Ready for Deployment'}
                {finalReport.overallStatus === 'SOME_FAILED' && '❌ System Issues Detected'}
                {finalReport.overallStatus === 'WITH_WARNINGS' && '⚠️ System Ready with Warnings'}
              </h4>
            </div>
            <p className="text-sm opacity-75">
              {finalReport.overallStatus === 'ALL_PASSED' && 
                'All tests passed successfully. The ATLAS system is fully operational and ready for business demonstration.'}
              {finalReport.overallStatus === 'SOME_FAILED' && 
                'Some critical tests failed. Please review the results and fix issues before deployment.'}
              {finalReport.overallStatus === 'WITH_WARNINGS' && 
                'Tests completed with some warnings. System is functional but may have minor issues.'}
            </p>
          </div>

          {/* Report Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{finalReport.totalTests}</div>
              <div className="text-sm text-gray-600">Total Tests</div>
            </div>
            
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{finalReport.passed}</div>
              <div className="text-sm text-green-800">Passed</div>
            </div>
            
            <div className="text-center p-4 bg-red-50 rounded-lg">
              <div className="text-2xl font-bold text-red-600">{finalReport.failed}</div>
              <div className="text-sm text-red-800">Failed</div>
            </div>
            
            <div className="text-center p-4 bg-yellow-50 rounded-lg">
              <div className="text-2xl font-bold text-yellow-600">{finalReport.warnings}</div>
              <div className="text-sm text-yellow-800">Warnings</div>
            </div>
            
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{finalReport.totalTime.toFixed(1)}s</div>
              <div className="text-sm text-blue-800">Total Time</div>
            </div>
          </div>

          {/* Timestamp */}
          <div className="mt-4 text-xs text-gray-500 text-center">
            Test completed on {new Date(finalReport.timestamp).toLocaleString()}
          </div>
        </div>
      )}

      {/* Test Suite Info */}
      {!running && testResults.length === 0 && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Test Suite Components</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {testSuites.map((test, index) => (
              <div key={index} className="p-4 border border-gray-200 rounded-lg">
                <h4 className="font-medium text-gray-900 mb-1">{test.name}</h4>
                <p className="text-sm text-gray-600 mb-2">{test.description}</p>
                <code className="text-xs bg-gray-100 px-2 py-1 rounded">{test.endpoint}</code>
              </div>
            ))}
            
            <div className="p-4 border border-blue-200 bg-blue-50 rounded-lg">
              <h4 className="font-medium text-blue-900 mb-1">30-Second System Test</h4>
              <p className="text-sm text-blue-700 mb-2">
                Comprehensive real-time monitoring of system performance and consciousness levels
              </p>
              <div className="text-xs text-blue-600">
                Collects metrics every second for detailed analysis
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default TestRunner;