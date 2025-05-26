#!/usr/bin/env python3
import requests
import json

def test_unified_atlas():
    base_url = "http://localhost:8001/api"
    
    print("🚀 Testing Unified ATLAS System")
    print("=" * 50)
    
    # Test health
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data.get('status')} v{data.get('version', '2.0.0')}")
        else:
            print(f"❌ Health: Failed {response.status_code}")
    except Exception as e:
        print(f"❌ Health: Error {e}")
    
    # Test status and capabilities
    try:
        response = requests.get(f"{base_url}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            capabilities = data.get("capabilities", [])
            print(f"✅ Status: {data.get('status')} with {len(capabilities)} capabilities")
            
            # Check for new capabilities
            new_caps = ["multi_modal_imagination", "consciousness_monitoring", "advanced_learning"]
            found_caps = [cap for cap in new_caps if cap in capabilities]
            print(f"✅ New Features: {len(found_caps)}/3 - {', '.join(found_caps)}")
        else:
            print(f"❌ Status: Failed {response.status_code}")
    except Exception as e:
        print(f"❌ Status: Error {e}")
    
    # Test consciousness
    try:
        response = requests.get(f"{base_url}/consciousness/current", timeout=10)
        if response.status_code == 200:
            data = response.json()
            modalities = len(data.get("modality_states", {}))
            i2c_units = len(data.get("i2c_activations", []))
            cross_modal = data.get("cross_modal_integration", False)
            print(f"✅ Consciousness: {modalities} modalities, {i2c_units} I²C units, Cross-modal: {cross_modal}")
        else:
            print(f"❌ Consciousness: Failed {response.status_code}")
    except Exception as e:
        print(f"❌ Consciousness: Error {e}")
    
    # Test imagination
    try:
        response = requests.post(f"{base_url}/imagination/generate", 
                               json={"prompt": "Test creativity", "modality": "text", "creativity_level": 0.8},
                               timeout=15)
        if response.status_code == 200:
            data = response.json()
            creativity = data.get("creativity_score", 0)
            print(f"✅ Imagination: Generated content with {creativity:.2f} creativity")
        else:
            print(f"❌ Imagination: Failed {response.status_code}")
    except Exception as e:
        print(f"❌ Imagination: Error {e}")
    
    print("=" * 50)
    print("🎯 Unified ATLAS Test Complete!")

if __name__ == "__main__":
    test_unified_atlas()
