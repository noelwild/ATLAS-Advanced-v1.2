"""
ATLAS-Qwen Code Executor
Executes Python code in a dedicated miniconda environment
"""

import asyncio
import subprocess
import tempfile
import os
import json
import uuid
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import time
import shutil

from config import AtlasQwenConfig


class CodeExecutor:
    """
    Executes Python code in a sandboxed miniconda environment
    """
    
    def __init__(self, config: AtlasQwenConfig):
        self.config = config
        self.env_name = "atlas"
        self.env_path = None
        self.execution_history = []
        
        # Setup environment
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup the atlas miniconda environment"""
        print("🐍 Setting up ATLAS miniconda environment...")
        
        try:
            # Check if conda is available
            result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Conda not found. Installing miniconda...")
                self._install_miniconda()
            
            # Check if atlas environment exists
            result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
            
            if self.env_name not in result.stdout:
                print(f"📦 Creating {self.env_name} environment...")
                self._create_atlas_environment()
            else:
                print(f"✅ {self.env_name} environment already exists")
            
            # Get environment path
            self.env_path = self._get_env_path()
            print(f"🎯 ATLAS environment ready at: {self.env_path}")
            
        except Exception as e:
            print(f"⚠️ Environment setup error: {e}")
            print("Using system Python as fallback")
    
    def _install_miniconda(self):
        """Install miniconda if not present"""
        print("📥 Installing miniconda...")
        
        # Download and install miniconda
        install_script = """
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
        """
        
        try:
            subprocess.run(install_script, shell=True, check=True)
            print("✅ Miniconda installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Miniconda installation failed: {e}")
    
    def _create_atlas_environment(self):
        """Create the atlas conda environment with essential packages"""
        packages = [
            "python=3.11",
            "numpy",
            "pandas", 
            "matplotlib",
            "requests",
            "scipy",
            "scikit-learn",
            "jupyter",
            "pip"
        ]
        
        cmd = ['conda', 'create', '-n', self.env_name, '-y'] + packages
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ {self.env_name} environment created successfully")
                
                # Install additional pip packages
                self._install_pip_packages()
            else:
                print(f"❌ Environment creation failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⏰ Environment creation timed out")
        except Exception as e:
            print(f"❌ Environment creation error: {e}")
    
    def _install_pip_packages(self):
        """Install additional packages via pip in the atlas environment"""
        pip_packages = [
            "torch",
            "transformers", 
            "plotly",
            "seaborn",
            "beautifulsoup4",
            "openai",
            "anthropic"
        ]
        
        for package in pip_packages:
            try:
                cmd = ['conda', 'run', '-n', self.env_name, 'pip', 'install', package]
                subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                print(f"📦 Installed {package}")
            except Exception as e:
                print(f"⚠️ Failed to install {package}: {e}")
    
    def _get_env_path(self) -> Optional[str]:
        """Get the path to the atlas environment"""
        try:
            result = subprocess.run(
                ['conda', 'info', '--envs'], 
                capture_output=True, 
                text=True
            )
            
            for line in result.stdout.split('\n'):
                if self.env_name in line and '*' not in line:
                    return line.split()[-1]
            
        except Exception as e:
            print(f"Environment path detection error: {e}")
        
        return None
    
    async def execute_code(
        self, 
        code: str, 
        execution_id: Optional[str] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute Python code in the atlas environment
        
        Args:
            code: Python code to execute
            execution_id: Optional execution identifier
            timeout: Execution timeout in seconds
            
        Returns:
            Execution result dictionary
        """
        if execution_id is None:
            execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        print(f"🚀 Executing code: {execution_id}")
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap code with capture functionality
            wrapped_code = self._wrap_code_for_execution(code)
            f.write(wrapped_code)
            temp_file = f.name
        
        execution_start = time.time()
        result = {
            'execution_id': execution_id,
            'code': code,
            'success': False,
            'output': '',
            'error': '',
            'execution_time': 0,
            'timestamp': time.time()
        }
        
        try:
            # Execute in atlas environment
            if self.env_path:
                cmd = ['conda', 'run', '-n', self.env_name, 'python', temp_file]
            else:
                cmd = ['python', temp_file]  # Fallback to system python
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                result['success'] = process.returncode == 0 and not stderr.decode('utf-8').strip()
                result['output'] = stdout.decode('utf-8')
                result['error'] = stderr.decode('utf-8')
                result['execution_time'] = time.time() - execution_start
                
                if result['success']:
                    print(f"✅ Code executed successfully: {execution_id}")
                else:
                    print(f"❌ Code execution failed: {execution_id}")
                
            except asyncio.TimeoutError:
                process.kill()
                result['error'] = f"Execution timed out after {timeout} seconds"
                print(f"⏰ Code execution timeout: {execution_id}")
                
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}"
            print(f"💥 Code execution exception: {execution_id} - {e}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Store in history
        self.execution_history.append(result)
        
        # Keep history manageable
        if len(self.execution_history) > 100:
            self.execution_history.pop(0)
        
        return result
    
    def _wrap_code_for_execution(self, code: str) -> str:
        """
        Wrap code to capture outputs and handle imports
        """
        wrapped = f"""
import sys
import io
import contextlib
import traceback
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import requests
from datetime import datetime

# Capture stdout and stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

try:
    # Redirect output
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    # User code execution
{self._indent_code(code)}
    
    # Capture any matplotlib plots
    if plt.get_fignums():
        plt.savefig('atlas_plot.png', dpi=100, bbox_inches='tight')
        print("📊 Plot saved as atlas_plot.png")
        plt.close('all')
    
except Exception as e:
    print(f"Error: {{str(e)}}")
    traceback.print_exc()

finally:
    # Restore stdout/stderr
    sys.stdout = old_stdout  
    sys.stderr = old_stderr
    
    # Print captured output
    captured_out = stdout_capture.getvalue()
    captured_err = stderr_capture.getvalue()
    
    if captured_out:
        print(captured_out, end='')
    if captured_err:
        print(captured_err, end='', file=sys.stderr)
"""
        return wrapped
    
    def _indent_code(self, code: str) -> str:
        """Indent code for wrapping"""
        lines = code.split('\n')
        indented_lines = ['    ' + line for line in lines]
        return '\n'.join(indented_lines)
    
    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:] if self.execution_history else []
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the execution environment"""
        try:
            if self.env_path:
                cmd = ['conda', 'run', '-n', self.env_name, 'python', '-c', 
                      'import sys; print(sys.version); import pkg_resources; print([d.project_name for d in pkg_resources.working_set])']
            else:
                cmd = ['python', '-c', 
                      'import sys; print(sys.version); import pkg_resources; print([d.project_name for d in pkg_resources.working_set])']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                python_version = lines[0] if lines else "Unknown"
                packages = eval(lines[1]) if len(lines) > 1 else []
                
                return {
                    'environment_name': self.env_name,
                    'environment_path': self.env_path,
                    'python_version': python_version,
                    'installed_packages': len(packages),
                    'sample_packages': packages[:10],
                    'executions_count': len(self.execution_history)
                }
        except Exception as e:
            print(f"Environment info error: {e}")
        
        return {
            'environment_name': self.env_name,
            'status': 'error',
            'executions_count': len(self.execution_history)
        }
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        print("🧹 Execution history cleared")


async def test_code_executor():
    """Test the code executor"""
    from config import get_default_config
    
    config = get_default_config()
    executor = CodeExecutor(config)
    
    # Test simple code
    test_code = """
print("Hello from ATLAS!")
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Array mean: {arr.mean()}")

# Test calculation
result = 2 + 2
print(f"2 + 2 = {result}")
"""
    
    result = await executor.execute_code(test_code)
    
    print("🧪 Test Results:")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    print(f"Execution time: {result['execution_time']:.3f}s")
    
    if result['error']:
        print(f"Error: {result['error']}")
    
    # Environment info
    env_info = executor.get_environment_info()
    print(f"\n🔧 Environment: {env_info}")


if __name__ == "__main__":
    asyncio.run(test_code_executor())
