#!/usr/bin/env python3
"""
Comprehensive ATLAS Testing Script
This script performs a 30-second comprehensive test of the ATLAS system
"""

import asyncio
import aiohttp
import json
import time
import sys
from datetime import datetime
from typing import Dict, List

class ATLASTestSuite:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.session = None
        self.test_results = []
        self.consciousness_metrics = []
        self.chat_sessions = []
        
    async def initialize(self):
        """Initialize the test session"""
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            
    async def get_system_status(self):
        """Get current system status"""
        try:
            async with self.session.get(f"{self.base_url}/api/status") as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}
            
    async def get_consciousness_metrics(self):
        """Get consciousness monitoring data"""
        try:
            async with self.session.get(f"{self.base_url}/api/consciousness/detailed") as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    # Try basic consciousness endpoint
                    async with self.session.get(f"{self.base_url}/api/consciousness/current") as resp2:
                        return await resp2.json()
        except Exception as e:
            return {"error": str(e)}
            
    async def get_human_features(self):
        """Get human-like features status"""
        try:
            async with self.session.get(f"{self.base_url}/api/human-features") as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}
            
    async def send_chat_message(self, message: str, session_id: str = "test_session"):
        """Send a chat message and measure response"""
        try:
            start_time = time.time()
            payload = {
                "message": message,
                "session_id": session_id,
                "use_consciousness": True,
                "use_human_features": True
            }
            async with self.session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                response = await resp.json()
                end_time = time.time()
                response["response_time"] = end_time - start_time
                return response
        except Exception as e:
            return {"error": str(e), "response_time": 0}
            
    async def execute_code(self, code: str):
        """Test code execution capabilities"""
        try:
            start_time = time.time()
            payload = {"code": code, "session_id": "test_session"}
            async with self.session.post(f"{self.base_url}/api/code/execute", json=payload) as resp:
                response = await resp.json()
                end_time = time.time()
                response["execution_time"] = end_time - start_time
                return response
        except Exception as e:
            return {"error": str(e), "execution_time": 0}
            
    async def run_comprehensive_test(self, duration_seconds=30):
        """Run comprehensive test for specified duration"""
        print(f"🚀 Starting ATLAS Comprehensive Test - Duration: {duration_seconds}s")
        print("=" * 60)
        
        start_time = time.time()
        test_count = 0
        
        # Initial status check
        print("📊 Initial System Status:")
        initial_status = await self.get_system_status()
        print(json.dumps(initial_status, indent=2))
        
        while time.time() - start_time < duration_seconds:
            test_count += 1
            current_time = time.time() - start_time
            
            print(f"\n⏱️  Test Cycle {test_count} - Time: {current_time:.1f}s")
            
            # 1. System Status Check
            status = await self.get_system_status()
            
            # 2. Consciousness Metrics
            consciousness = await self.get_consciousness_metrics()
            self.consciousness_metrics.append({
                "timestamp": datetime.now().isoformat(),
                "metrics": consciousness
            })
            
            # 3. Human Features Check
            human_features = await self.get_human_features()
            
            # 4. Chat Interaction Test
            chat_messages = [
                "Hello ATLAS, how are you functioning today?",
                "Can you explain your consciousness monitoring?",
                "What human-like features do you have?",
                "Perform a self-assessment of your capabilities",
                "Tell me about your current emotional state"
            ]
            
            message = chat_messages[test_count % len(chat_messages)]
            chat_response = await self.send_chat_message(message, f"session_{test_count}")
            self.chat_sessions.append({
                "cycle": test_count,
                "message": message,
                "response": chat_response
            })
            
            # 5. Code Execution Test (every 3rd cycle)
            code_result = None
            if test_count % 3 == 0:
                test_code = f"""
import time
import random

# Test computation
result = sum(range(100))
current_time = time.time()
random_value = random.randint(1, 100)

print(f"Computation result: {{result}}")
print(f"Current timestamp: {{current_time}}")
print(f"Random value: {{random_value}}")
print("ATLAS code execution test successful!")
"""
                code_result = await self.execute_code(test_code)
            
            # Store test results
            test_result = {
                "cycle": test_count,
                "timestamp": datetime.now().isoformat(),
                "elapsed_time": current_time,
                "system_status": status,
                "consciousness_metrics": consciousness,
                "human_features": human_features,
                "chat_response": chat_response,
                "code_execution": code_result
            }
            self.test_results.append(test_result)
            
            # Print quick summary
            print(f"   📡 Status: {status.get('status', 'unknown')}")
            print(f"   🧠 Model Loaded: {status.get('model_loaded', False)}")
            print(f"   💭 Consciousness: {status.get('consciousness_active', False)}")
            print(f"   💬 Chat Response Time: {chat_response.get('response_time', 0):.3f}s")
            if code_result:
                print(f"   🔧 Code Execution: {code_result.get('success', False)}")
            
            # Brief pause before next cycle
            await asyncio.sleep(2)
            
        print(f"\n✅ Test completed! Total cycles: {test_count}")
        return self.generate_report()
        
    def generate_report(self):
        """Generate comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results to report"}
            
        report = {
            "test_summary": {
                "total_cycles": len(self.test_results),
                "duration": self.test_results[-1]["elapsed_time"],
                "start_time": self.test_results[0]["timestamp"],
                "end_time": self.test_results[-1]["timestamp"]
            },
            "system_performance": self._analyze_system_performance(),
            "consciousness_analysis": self._analyze_consciousness_metrics(),
            "chat_performance": self._analyze_chat_performance(),
            "code_execution_analysis": self._analyze_code_execution(),
            "human_features_status": self._analyze_human_features(),
            "detailed_results": self.test_results
        }
        
        return report
        
    def _analyze_system_performance(self):
        """Analyze system performance metrics"""
        cpu_usages = [r["system_status"].get("cpu_usage", 0) for r in self.test_results if "system_status" in r]
        memory_usages = [r["system_status"].get("memory_usage", 0) for r in self.test_results if "system_status" in r]
        
        return {
            "cpu_usage": {
                "min": min(cpu_usages) if cpu_usages else 0,
                "max": max(cpu_usages) if cpu_usages else 0,
                "avg": sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
            },
            "memory_usage": {
                "min": min(memory_usages) if memory_usages else 0,
                "max": max(memory_usages) if memory_usages else 0,
                "avg": sum(memory_usages) / len(memory_usages) if memory_usages else 0
            }
        }
        
    def _analyze_consciousness_metrics(self):
        """Analyze consciousness monitoring data"""
        return {
            "total_measurements": len(self.consciousness_metrics),
            "active_sessions": len([c for c in self.consciousness_metrics if c.get("metrics", {}).get("consciousness_active", False)]),
            "latest_metrics": self.consciousness_metrics[-1] if self.consciousness_metrics else None
        }
        
    def _analyze_chat_performance(self):
        """Analyze chat interaction performance"""
        response_times = [s["response"]["response_time"] for s in self.chat_sessions if "response_time" in s["response"]]
        successful_chats = [s for s in self.chat_sessions if "error" not in s["response"]]
        
        return {
            "total_interactions": len(self.chat_sessions),
            "successful_interactions": len(successful_chats),
            "success_rate": len(successful_chats) / len(self.chat_sessions) if self.chat_sessions else 0,
            "response_time_stats": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "avg": sum(response_times) / len(response_times) if response_times else 0
            }
        }
        
    def _analyze_code_execution(self):
        """Analyze code execution capabilities"""
        code_results = [r["code_execution"] for r in self.test_results if r.get("code_execution")]
        successful_executions = [r for r in code_results if r.get("success", False)]
        
        return {
            "total_executions": len(code_results),
            "successful_executions": len(successful_executions),
            "success_rate": len(successful_executions) / len(code_results) if code_results else 0
        }
        
    def _analyze_human_features(self):
        """Analyze human-like features"""
        human_features_data = [r["human_features"] for r in self.test_results if "human_features" in r and "error" not in r["human_features"]]
        
        if not human_features_data:
            return {"status": "No human features data available"}
            
        latest_features = human_features_data[-1] if human_features_data else {}
        return {
            "available_features": latest_features,
            "feature_count": len(latest_features) if isinstance(latest_features, dict) else 0
        }

async def main():
    """Main test function"""
    tester = ATLASTestSuite()
    
    try:
        await tester.initialize()
        
        # Run the comprehensive test
        report = await tester.run_comprehensive_test(30)  # 30 seconds
        
        # Print detailed report
        print("\n" + "=" * 80)
        print("📋 ATLAS COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        # Print summary
        summary = report["test_summary"]
        print(f"🕐 Test Duration: {summary['duration']:.1f} seconds")
        print(f"🔄 Total Test Cycles: {summary['total_cycles']}")
        
        # System Performance
        perf = report["system_performance"]
        print(f"\n📊 SYSTEM PERFORMANCE:")
        print(f"   CPU Usage: {perf['cpu_usage']['avg']:.1f}% (min: {perf['cpu_usage']['min']:.1f}%, max: {perf['cpu_usage']['max']:.1f}%)")
        print(f"   Memory Usage: {perf['memory_usage']['avg']:.1f}% (min: {perf['memory_usage']['min']:.1f}%, max: {perf['memory_usage']['max']:.1f}%)")
        
        # Chat Performance
        chat = report["chat_performance"]
        print(f"\n💬 CHAT PERFORMANCE:")
        print(f"   Total Interactions: {chat['total_interactions']}")
        print(f"   Success Rate: {chat['success_rate']:.1%}")
        print(f"   Avg Response Time: {chat['response_time_stats']['avg']:.3f}s")
        
        # Code Execution
        code = report["code_execution_analysis"]
        print(f"\n🔧 CODE EXECUTION:")
        print(f"   Total Executions: {code['total_executions']}")
        print(f"   Success Rate: {code['success_rate']:.1%}")
        
        # Consciousness Analysis
        consciousness = report["consciousness_analysis"]
        print(f"\n🧠 CONSCIOUSNESS MONITORING:")
        print(f"   Total Measurements: {consciousness['total_measurements']}")
        print(f"   Active Sessions: {consciousness['active_sessions']}")
        
        # Human Features
        human = report["human_features_status"]
        print(f"\n👤 HUMAN-LIKE FEATURES:")
        if isinstance(human.get("available_features"), dict):
            for feature, status in human["available_features"].items():
                print(f"   {feature}: {status}")
        else:
            print(f"   Status: {human.get('status', 'Unknown')}")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"/app/atlas_test_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        return report
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {"error": str(e)}
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())