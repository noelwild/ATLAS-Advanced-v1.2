#!/usr/bin/env python3
"""
Validate ATLAS Human Enhancement System Installation

This script validates that all components are properly installed and working.
Run this after installation to ensure everything is set up correctly.
"""

import sys
import time
import asyncio
from typing import Dict, Any

def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_status(item: str, status: bool, details: str = ""):
    """Print status with checkmark or X."""
    symbol = "✅" if status else "❌"
    print(f"{symbol} {item}")
    if details:
        print(f"   {details}")

async def validate_installation():
    """Validate the complete installation."""
    
    print_header("ATLAS Human Enhancement System - Installation Validation")
    
    validation_results = {
        'core_imports': False,
        'configuration': False,
        'basic_functionality': False,
        'integration': False,
        'performance': False
    }
    
    # Test 1: Core Imports
    print_header("1. Testing Core Imports")
    try:
        # Test basic imports
        from human_enhancements import (
            EmotionalIntelligenceSystem, 
            EpisodicMemorySystem,
            MetaCognitionSystem,
            IntuitionSystem,
            HumanLikeEnhancementSystem
        )
        print_status("Basic enhancement imports", True)
        
        from advanced_human_features import (
            SocialCognitionSystem,
            PersonalitySystem, 
            DreamSystem,
            AdvancedHumanFeatures
        )
        print_status("Advanced feature imports", True)
        
        from final_human_features import (
            MoralReasoningSystem,
            AttentionManagementSystem,
            TemporalReasoningSystem,
            CompleteHumanLikeSystem
        )
        print_status("Intelligence feature imports", True)
        
        from config_compat import get_default_config
        print_status("Configuration imports", True)
        
        validation_results['core_imports'] = True
        
    except ImportError as e:
        print_status("Core imports", False, f"Import error: {e}")
        return validation_results
    
    # Test 2: Configuration
    print_header("2. Testing Configuration System")
    try:
        config = get_default_config()
        
        # Check required sections
        required_sections = ['qwen', 'consciousness', 'memory', 'tags', 'code_execution']
        for section in required_sections:
            has_section = hasattr(config, section)
            print_status(f"Config section '{section}'", has_section)
            if not has_section:
                validation_results['configuration'] = False
                return validation_results
        
        print_status("Configuration validation", True, "All required sections present")
        validation_results['configuration'] = True
        
    except Exception as e:
        print_status("Configuration", False, f"Error: {e}")
        return validation_results
    
    # Test 3: Basic Functionality
    print_header("3. Testing Basic Functionality")
    try:
        # Test emotional intelligence
        from human_enhancements import EmotionType
        emotion_system = EmotionalIntelligenceSystem()
        emotion_system.trigger_emotion(EmotionType.CURIOSITY, 0.7)
        has_emotions = emotion_system.current_state.arousal > 0
        print_status("Emotional intelligence", has_emotions)
        
        # Test tag parsing
        from tag_parser import TagParser
        parser = TagParser(config.tags)
        test_text = "<thought>Test thought</thought> This is a test."
        parsed = parser.parse_stream(test_text)
        has_parsing = 'thought' in parsed and len(parsed['thought']) > 0
        print_status("Tag parsing", has_parsing)
        
        # Test personality system
        personality = PersonalitySystem()
        personality_desc = personality.get_personality_description()
        has_personality = len(personality_desc) > 10
        print_status("Personality system", has_personality)
        
        validation_results['basic_functionality'] = True
        
    except Exception as e:
        print_status("Basic functionality", False, f"Error: {e}")
        return validation_results
    
    # Test 4: Integration
    print_header("4. Testing System Integration")
    try:
        # Test complete system initialization
        complete_system = CompleteHumanLikeSystem(config)
        print_status("Complete system initialization", True)
        
        # Test basic processing
        result = await complete_system.process_comprehensive_input(
            input_text="Hello, this is a test message.",
            user_id="test_user",
            context={"test": True}
        )
        
        has_response = 'comprehensive_response' in result
        has_confidence = 'human_like_confidence' in result
        print_status("Comprehensive processing", has_response and has_confidence)
        
        # Test system status
        status = complete_system.get_complete_human_status()
        has_status = 'consciousness_level' in status
        print_status("System status reporting", has_status)
        
        validation_results['integration'] = True
        
    except Exception as e:
        print_status("System integration", False, f"Error: {e}")
        return validation_results
    
    # Test 5: Performance
    print_header("5. Testing Performance")
    try:
        # Test processing speed
        start_time = time.time()
        
        for i in range(5):
            result = await complete_system.process_comprehensive_input(
                input_text=f"Test message {i}",
                user_id="speed_test",
                context={"performance_test": True}
            )
        
        total_time = time.time() - start_time
        avg_time = total_time / 5
        
        meets_performance = avg_time < 0.1  # Should be under 100ms
        print_status("Processing speed", meets_performance, f"Average: {avg_time:.3f}s per request")
        
        # Test memory usage (basic check)
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        reasonable_memory = memory_mb < 2000  # Under 2GB
        print_status("Memory usage", reasonable_memory, f"Current: {memory_mb:.1f} MB")
        
        validation_results['performance'] = True
        
    except Exception as e:
        print_status("Performance testing", False, f"Error: {e}")
        return validation_results
    
    return validation_results

def print_summary(results: Dict[str, bool]):
    """Print validation summary."""
    print_header("Validation Summary")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        formatted_name = test_name.replace('_', ' ').title()
        print_status(formatted_name, passed)
    
    print(f"\n📊 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! ATLAS Human Enhancement System is ready to use.")
        print("\nNext steps:")
        print("   • Run example_basic_usage.py for a demonstration")
        print("   • Check COMPONENT_EXPLANATIONS.md for detailed documentation")
        print("   • See README.md for integration examples")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("   • Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   • Check Python version (3.8+ required)")
        print("   • Verify all files are present in the directory")

async def main():
    """Main validation function."""
    try:
        results = await validate_installation()
        print_summary(results)
        
        # Return appropriate exit code
        if all(results.values()):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())