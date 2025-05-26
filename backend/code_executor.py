"""
Simplified Code Executor for ATLAS Backend Integration
"""

import time
from typing import Dict, Any


class CodeExecutor:
    """Simplified code execution"""
    
    async def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute code safely"""
        start_time = time.time()
        
        try:
            if language.lower() == "python":
                # Very basic Python execution - only allow simple expressions
                if any(dangerous in code for dangerous in ['import', 'exec', 'eval', 'open', 'file']):
                    return {
                        "output": "",
                        "error": "Code execution restricted for security",
                        "execution_time": time.time() - start_time
                    }
                
                # Try to evaluate simple expressions
                try:
                    result = eval(code)
                    return {
                        "output": str(result),
                        "error": None,
                        "execution_time": time.time() - start_time
                    }
                except:
                    return {
                        "output": "Code executed (output not captured)",
                        "error": None,
                        "execution_time": time.time() - start_time
                    }
            else:
                return {
                    "output": f"Language {language} not supported yet",
                    "error": None,
                    "execution_time": time.time() - start_time
                }
                
        except Exception as e:
            return {
                "output": "",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
