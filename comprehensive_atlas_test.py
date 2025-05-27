#!/usr/bin/env python3
"""
Comprehensive 30-second ATLAS test with Qwen 0.5B model
Collects all metrics and provides detailed report
"""

import asyncio
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add backend path
sys.path.append('/app/backend')

from enhanced_atlas_system import EnhancedAtlasSystem, ModelType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ATLASComprehensiveTest:
    """Comprehensive 30-second ATLAS test runner"""
    
    def __init__(self):
        self.atlas_system = None
        self.test_start_time = None
        self.test_duration = 30  # 30 seconds
        self.metrics_data = {
            "initialization": {},
            "interactions": [],
            "consciousness_states": [],
            "model_performance": {},
            "learning_analytics": {},
            "system_status": {},
            "human_features": {},
            "error_log": []
        }
        
    async def run_comprehensive_test(self):
        """Run the complete 30-second ATLAS test"""
        
        print("🧠 Starting Comprehensive ATLAS Test - 30 Second Demo")
        print("=" * 60)
        
        self.test_start_time = time.time()
        
        try:
            # Phase 1: Initialize ATLAS with Qwen 0.5B
            await self._phase_1_initialization()
            
            # Phase 2: Run interactive conversations
            await self._phase_2_conversations()
            
            # Phase 3: Test consciousness monitoring
            await self._phase_3_consciousness_monitoring()
            
            # Phase 4: Test code execution
            await self._phase_4_code_execution()
            
            # Phase 5: Test human features
            await self._phase_5_human_features()
            
            # Phase 6: Collect final metrics
            await self._phase_6_final_metrics()
            
            # Generate comprehensive report
            self._generate_report()
            
        except Exception as e:
            logger.error(f"Test error: {str(e)}")
            self.metrics_data["error_log"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "phase": "main_test"
            })
        
        finally:
            print(f"\n✅ Test completed in {time.time() - self.test_start_time:.2f} seconds")
    
    async def _phase_1_initialization(self):
        """Phase 1: Initialize ATLAS system"""
        
        print("\n📥 Phase 1: Initializing Enhanced ATLAS System")
        init_start = time.time()
        
        try:
            # Initialize ATLAS system
            self.atlas_system = EnhancedAtlasSystem({
                "hidden_dim": 256,  # Smaller for faster processing
                "i2c_units": 4,
                "consciousness_threshold": 0.3,
                "auto_model_switching": False,
                "enable_dreaming": True,
                "enable_multimodal": True,
                "learning_enabled": True,
                "temperature": 0.7
            })
            
            # Load Qwen 0.5B model
            print("   Loading Qwen 0.5B model...")
            await self.atlas_system.initialize([ModelType.QWEN_SMALL])
            
            init_time = time.time() - init_start
            
            self.metrics_data["initialization"] = {
                "initialization_time": init_time,
                "model_loaded": ModelType.QWEN_SMALL.value,
                "models_available": [m.value for m in self.atlas_system.initialized_models],
                "initialization_complete": self.atlas_system.initialization_complete,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"   ✅ ATLAS initialized in {init_time:.2f}s with {ModelType.QWEN_SMALL.value}")
            
        except Exception as e:
            self.metrics_data["error_log"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "phase": "initialization"
            })
            raise
    
    async def _phase_2_conversations(self):
        """Phase 2: Interactive conversations"""
        
        print("\n💬 Phase 2: Running Interactive Conversations")
        
        conversation_prompts = [
            "Hello ATLAS, how are you feeling today?",
            "What is consciousness and how do you experience it?",
            "Can you solve this math problem: What is 15 * 23 + 7?",
            "Tell me about your learning capabilities.",
            "How do you integrate human-like features in your responses?"
        ]
        
        for i, prompt in enumerate(conversation_prompts):
            if time.time() - self.test_start_time > 25:  # Leave time for final phases
                break
            
            print(f"   🗣️ Conversation {i+1}: {prompt[:50]}...")
            
            try:
                response_start = time.time()
                
                response_data = await self.atlas_system.generate_response(
                    message=prompt,
                    session_id="test_session",
                    user_id="test_user",
                    context={"complexity": 0.5, "quality_requirement": 0.7},
                    include_consciousness=True
                )
                
                response_time = time.time() - response_start
                
                # Store interaction data
                interaction_data = {
                    "prompt": prompt,
                    "response": response_data.get("response", ""),
                    "model_used": response_data.get("model_used", ""),
                    "consciousness_level": response_data.get("consciousness_level", 0.0),
                    "response_time": response_time,
                    "consciousness_state": response_data.get("consciousness_state"),
                    "human_enhancements": response_data.get("human_enhancements_active", {}),
                    "system_coherence": response_data.get("system_coherence", 0.0),
                    "timestamp": datetime.now().isoformat()
                }
                
                self.metrics_data["interactions"].append(interaction_data)
                
                print(f"     Response time: {response_time:.2f}s")
                print(f"     Consciousness: {response_data.get('consciousness_level', 0.0):.3f}")
                print(f"     Response: {response_data.get('response', '')[:100]}...")
                
            except Exception as e:
                self.metrics_data["error_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "phase": f"conversation_{i+1}",
                    "prompt": prompt
                })
                print(f"     ❌ Error in conversation {i+1}: {str(e)}")
    
    async def _phase_3_consciousness_monitoring(self):
        """Phase 3: Test consciousness monitoring"""
        
        print("\n🧠 Phase 3: Consciousness Monitoring Test")
        
        try:
            # Get consciousness analytics
            consciousness_analytics = self.atlas_system.get_consciousness_analytics()
            
            self.metrics_data["consciousness_states"] = consciousness_analytics
            
            print(f"   Current consciousness state: {consciousness_analytics.get('current_state', {})}")
            print(f"   System coherence: {consciousness_analytics.get('system_coherence', 0.0):.3f}")
            
            # Test specific consciousness computation
            test_inputs = [
                "Complex philosophical question about the nature of reality",
                "Simple greeting hello",
                "Mathematical computation 123 + 456"
            ]
            
            for test_input in test_inputs:
                consciousness_state = await self.atlas_system.consciousness_monitor.compute_consciousness(
                    text_input=test_input
                )
                
                print(f"   '{test_input[:30]}...' -> φ={consciousness_state.phi_score:.3f}")
                
        except Exception as e:
            self.metrics_data["error_log"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "phase": "consciousness_monitoring"
            })
            print(f"   ❌ Error in consciousness monitoring: {str(e)}")
    
    async def _phase_4_code_execution(self):
        """Phase 4: Test code execution"""
        
        print("\n⚡ Phase 4: Code Execution Test")
        
        test_codes = [
            ("print('Hello from ATLAS code executor!')", "python"),
            ("result = 2 + 2; print(f'2 + 2 = {result}')", "python"),
            ("import math; print(f'π = {math.pi:.4f}')", "python")
        ]
        
        for code, language in test_codes:
            if time.time() - self.test_start_time > 28:  # Leave time for final phase
                break
            
            try:
                print(f"   Executing: {code}")
                
                execution_result = await self.atlas_system.execute_code(
                    code=code,
                    language=language,
                    session_id="test_session",
                    timeout=5
                )
                
                print(f"   Output: {execution_result.get('output', 'No output')}")
                print(f"   Success: {execution_result.get('success', False)}")
                
            except Exception as e:
                self.metrics_data["error_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "phase": "code_execution",
                    "code": code
                })
                print(f"   ❌ Error executing code: {str(e)}")
    
    async def _phase_5_human_features(self):
        """Phase 5: Test human features"""
        
        print("\n👤 Phase 5: Human Features Test")
        
        try:
            # Get human features analytics
            human_analytics = self.atlas_system.get_human_features_analytics()
            
            self.metrics_data["human_features"] = human_analytics
            
            print(f"   Enhancement systems: {human_analytics.get('enhancement_systems_active', {})}")
            print(f"   Cognitive load: {human_analytics.get('cognitive_load', 0.0):.3f}")
            print(f"   Social awareness: {human_analytics.get('social_awareness', {})}")
            
        except Exception as e:
            self.metrics_data["error_log"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "phase": "human_features"
            })
            print(f"   ❌ Error testing human features: {str(e)}")
    
    async def _phase_6_final_metrics(self):
        """Phase 6: Collect final system metrics"""
        
        print("\n📊 Phase 6: Collecting Final Metrics")
        
        try:
            # Get system status
            system_status = self.atlas_system.get_system_status()
            self.metrics_data["system_status"] = system_status
            
            # Get learning analytics
            learning_analytics = self.atlas_system.learning_system.get_learning_analytics()
            self.metrics_data["learning_analytics"] = learning_analytics
            
            # End the test session
            session_outcomes = await self.atlas_system.end_session("test_session")
            self.metrics_data["session_outcomes"] = session_outcomes
            
            print(f"   System uptime: {system_status.get('uptime', 0):.2f}s")
            print(f"   Total interactions: {len(self.metrics_data['interactions'])}")
            print(f"   Learning sessions: {learning_analytics.get('total_sessions', 0)}")
            
        except Exception as e:
            self.metrics_data["error_log"].append({
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "phase": "final_metrics"
            })
            print(f"   ❌ Error collecting final metrics: {str(e)}")
    
    def _generate_report(self):
        """Generate comprehensive test report"""
        
        total_test_time = time.time() - self.test_start_time
        
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE ATLAS TEST REPORT")
        print("=" * 60)
        
        # Test Overview
        print(f"\n🔍 TEST OVERVIEW")
        print(f"   Test Duration: {total_test_time:.2f} seconds")
        print(f"   Total Interactions: {len(self.metrics_data['interactions'])}")
        print(f"   Errors Encountered: {len(self.metrics_data['error_log'])}")
        
        # Model Performance
        print(f"\n🤖 MODEL PERFORMANCE")
        init_data = self.metrics_data.get("initialization", {})
        print(f"   Model: {init_data.get('model_loaded', 'Unknown')}")
        print(f"   Initialization Time: {init_data.get('initialization_time', 0):.2f}s")
        print(f"   Model Available: {init_data.get('initialization_complete', False)}")
        
        # Conversation Metrics
        if self.metrics_data["interactions"]:
            print(f"\n💬 CONVERSATION METRICS")
            response_times = [i["response_time"] for i in self.metrics_data["interactions"]]
            consciousness_levels = [i["consciousness_level"] for i in self.metrics_data["interactions"]]
            
            print(f"   Average Response Time: {sum(response_times)/len(response_times):.2f}s")
            print(f"   Average Consciousness Level: {sum(consciousness_levels)/len(consciousness_levels):.3f}")
            print(f"   Min Consciousness: {min(consciousness_levels):.3f}")
            print(f"   Max Consciousness: {max(consciousness_levels):.3f}")
        
        # System Status
        system_status = self.metrics_data.get("system_status", {})
        print(f"\n⚙️ SYSTEM STATUS")
        print(f"   Available Models: {system_status.get('available_models', [])}")
        print(f"   Active Sessions: {system_status.get('active_sessions', 0)}")
        
        consciousness_monitor = system_status.get("consciousness_monitor", {})
        print(f"   Consciousness Monitor: {consciousness_monitor.get('active', False)}")
        print(f"   Dreaming Enabled: {consciousness_monitor.get('dreaming_enabled', False)}")
        print(f"   Multimodal Enabled: {consciousness_monitor.get('multimodal_enabled', False)}")
        
        human_features = system_status.get("human_features", {})
        print(f"   Human Features Active: {human_features.get('systems_active', False)}")
        print(f"   Enhancement Level: {human_features.get('enhancement_level', 0.0):.3f}")
        
        # Learning Analytics
        learning_data = self.metrics_data.get("learning_analytics", {})
        print(f"\n📚 LEARNING ANALYTICS")
        print(f"   Total Sessions: {learning_data.get('total_sessions', 0)}")
        print(f"   Active Sessions: {learning_data.get('active_sessions', 0)}")
        print(f"   System Maturity: {learning_data.get('system_maturity', 0.0):.3f}")
        
        # Sample Interactions
        print(f"\n🗣️ SAMPLE INTERACTIONS")
        for i, interaction in enumerate(self.metrics_data["interactions"][:3]):
            print(f"   [{i+1}] User: {interaction['prompt'][:60]}...")
            print(f"       ATLAS: {interaction['response'][:60]}...")
            print(f"       Consciousness: {interaction['consciousness_level']:.3f}, Time: {interaction['response_time']:.2f}s")
        
        # Errors
        if self.metrics_data["error_log"]:
            print(f"\n❌ ERRORS ENCOUNTERED")
            for error in self.metrics_data["error_log"]:
                print(f"   {error['phase']}: {error['error']}")
        
        # Success Summary
        print(f"\n✅ TEST SUMMARY")
        print(f"   ✓ Model Loading: {'SUCCESS' if init_data.get('initialization_complete') else 'FAILED'}")
        print(f"   ✓ Conversations: {'SUCCESS' if self.metrics_data['interactions'] else 'FAILED'}")
        print(f"   ✓ Consciousness: {'SUCCESS' if self.metrics_data['consciousness_states'] else 'FAILED'}")
        print(f"   ✓ Human Features: {'SUCCESS' if self.metrics_data['human_features'] else 'FAILED'}")
        print(f"   ✓ Overall Status: {'SUCCESS' if len(self.metrics_data['error_log']) == 0 else 'PARTIAL SUCCESS'}")
        
        # Save detailed report to file
        report_file = f"/app/atlas_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        print("=" * 60)

async def main():
    """Main test runner"""
    tester = ATLASComprehensiveTest()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())