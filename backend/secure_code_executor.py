#!/usr/bin/env python3
"""
Secure Code Execution Environment for ATLAS System
Provides sandboxed code execution with comprehensive security measures
"""

import os
import asyncio
import tempfile
import subprocess
import shutil
import time
import uuid
import logging
import signal
import resource
import sys
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution"""
    output: str
    error: Optional[str]
    execution_time: float
    memory_usage: int  # bytes
    exit_code: int
    timeout: bool
    session_id: str
    language: str
    timestamp: datetime


@dataclass
class SecurityPolicy:
    """Security policy for code execution"""
    max_execution_time: int = 30  # seconds
    max_memory_usage: int = 128 * 1024 * 1024  # 128MB
    max_output_size: int = 1024 * 1024  # 1MB
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_modules: List[str] = None
    blocked_modules: List[str] = None
    network_access: bool = False
    file_system_access: bool = False


class SecureExecutionEnvironment:
    """Secure execution environment for a single code execution"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.temp_dir = None
        self.execution_id = str(uuid.uuid4())
        
    async def __aenter__(self):
        """Async context manager entry"""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"atlas_exec_{self.execution_id}_")
        
        # Set up security constraints
        await self._setup_security()
        
        logger.debug(f"Created secure environment {self.execution_id} at {self.temp_dir}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Clean up temporary directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up environment {self.execution_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up {self.temp_dir}: {e}")
    
    async def _setup_security(self):
        """Set up security constraints for the environment"""
        # Limit file permissions
        os.chmod(self.temp_dir, 0o700)
        
        # Create restricted Python environment
        if not self.policy.file_system_access:
            self._create_restricted_python_env()
    
    def _create_restricted_python_env(self):
        """Create a restricted Python environment"""
        # Create a restricted builtins file
        restricted_builtins = """
# Restricted builtins for secure execution
import builtins

# Safe built-in functions
SAFE_BUILTINS = {
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'callable', 'chr', 'classmethod', 
    'complex', 'dict', 'dir', 'divmod', 'enumerate', 'filter', 'float', 'format',
    'frozenset', 'getattr', 'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'object',
    'oct', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed', 
    'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
    'super', 'tuple', 'type', 'vars', 'zip'
}

# Restricted builtins
restricted_builtins = {}
for name in SAFE_BUILTINS:
    if hasattr(builtins, name):
        restricted_builtins[name] = getattr(builtins, name)

# Override dangerous functions
def restricted_import(name, *args, **kwargs):
    ALLOWED_MODULES = ['math', 'random', 'datetime', 'json', 'string', 're', 'collections', 'itertools']
    BLOCKED_MODULES = ['os', 'sys', 'subprocess', 'socket', 'urllib', 'requests', 'http', 'ftplib', 'smtplib']
    
    if name in BLOCKED_MODULES:
        raise ImportError(f"Import of '{name}' is not allowed")
    
    if name not in ALLOWED_MODULES:
        raise ImportError(f"Import of '{name}' is not in allowed modules")
    
    return builtins.__import__(name, *args, **kwargs)

def restricted_open(*args, **kwargs):
    raise PermissionError("File operations are not allowed")

def restricted_eval(*args, **kwargs):
    raise PermissionError("eval() is not allowed")

def restricted_exec(*args, **kwargs):
    raise PermissionError("exec() is not allowed")

restricted_builtins['__import__'] = restricted_import
restricted_builtins['open'] = restricted_open
restricted_builtins['eval'] = restricted_eval
restricted_builtins['exec'] = restricted_exec

__builtins__ = restricted_builtins
"""
        
        # Write restricted builtins to temp directory
        builtins_path = os.path.join(self.temp_dir, "restricted_builtins.py")
        with open(builtins_path, 'w') as f:
            f.write(restricted_builtins)
    
    async def execute_python(self, code: str) -> ExecutionResult:
        """Execute Python code in secure environment"""
        
        start_time = time.time()
        
        # Create code file
        code_file = os.path.join(self.temp_dir, "user_code.py")
        
        # Wrap code with security imports
        wrapped_code = f"""
import sys
sys.path.insert(0, '{self.temp_dir}')

# Import restricted builtins
try:
    import restricted_builtins
except ImportError:
    pass  # Fallback if restricted builtins not available

# User code
{code}
"""
        
        with open(code_file, 'w') as f:
            f.write(wrapped_code)
        
        try:
            # Set resource limits
            def set_limits():
                # CPU time limit
                resource.setrlimit(resource.RLIMIT_CPU, (self.policy.max_execution_time, self.policy.max_execution_time))
                
                # Memory limit
                resource.setrlimit(resource.RLIMIT_AS, (self.policy.max_memory_usage, self.policy.max_memory_usage))
                
                # File size limit
                resource.setrlimit(resource.RLIMIT_FSIZE, (self.policy.max_file_size, self.policy.max_file_size))
                
                # Process limit
                resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
            
            # Execute code
            process = await asyncio.create_subprocess_exec(
                sys.executable, code_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir,
                preexec_fn=set_limits
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.policy.max_execution_time
                )
                timeout = False
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                stdout, stderr = b"", b"Execution timed out"
                timeout = True
            
            execution_time = time.time() - start_time
            
            # Decode output
            output = stdout.decode('utf-8', errors='replace')[:self.policy.max_output_size]
            error = stderr.decode('utf-8', errors='replace')[:self.policy.max_output_size] if stderr else None
            
            # Get memory usage (approximation)
            memory_usage = len(output.encode('utf-8')) + len((error or "").encode('utf-8'))
            
            return ExecutionResult(
                output=output,
                error=error,
                execution_time=execution_time,
                memory_usage=memory_usage,
                exit_code=process.returncode or 0,
                timeout=timeout,
                session_id="",  # Will be set by caller
                language="python",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                output="",
                error=str(e),
                execution_time=execution_time,
                memory_usage=0,
                exit_code=1,
                timeout=False,
                session_id="",
                language="python",
                timestamp=datetime.now()
            )
    
    async def execute_javascript(self, code: str) -> ExecutionResult:
        """Execute JavaScript code in secure environment"""
        
        start_time = time.time()
        
        # Check if Node.js is available
        try:
            # Test if node is available
            process = await asyncio.create_subprocess_exec(
                'node', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode != 0:
                raise FileNotFoundError("Node.js not available")
                
        except FileNotFoundError:
            return ExecutionResult(
                output="",
                error="JavaScript execution requires Node.js",
                execution_time=time.time() - start_time,
                memory_usage=0,
                exit_code=1,
                timeout=False,
                session_id="",
                language="javascript",
                timestamp=datetime.now()
            )
        
        # Create code file
        code_file = os.path.join(self.temp_dir, "user_code.js")
        
        # Wrap code with security restrictions
        wrapped_code = f"""
// Security restrictions
const process = undefined;
const require = undefined;
const global = undefined;
const Buffer = undefined;

// User code
try {{
    {code}
}} catch (error) {{
    console.error('Error:', error.message);
}}
"""
        
        with open(code_file, 'w') as f:
            f.write(wrapped_code)
        
        try:
            # Execute JavaScript
            process = await asyncio.create_subprocess_exec(
                'node', code_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.policy.max_execution_time
                )
                timeout = False
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                stdout, stderr = b"", b"Execution timed out"
                timeout = True
            
            execution_time = time.time() - start_time
            
            # Decode output
            output = stdout.decode('utf-8', errors='replace')[:self.policy.max_output_size]
            error = stderr.decode('utf-8', errors='replace')[:self.policy.max_output_size] if stderr else None
            
            memory_usage = len(output.encode('utf-8')) + len((error or "").encode('utf-8'))
            
            return ExecutionResult(
                output=output,
                error=error,
                execution_time=execution_time,
                memory_usage=memory_usage,
                exit_code=process.returncode or 0,
                timeout=timeout,
                session_id="",
                language="javascript",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                output="",
                error=str(e),
                execution_time=execution_time,
                memory_usage=0,
                exit_code=1,
                timeout=False,
                session_id="",
                language="javascript",
                timestamp=datetime.now()
            )


class SecureCodeExecutor:
    """
    Main secure code executor with comprehensive security and monitoring
    """
    
    def __init__(self):
        self.default_policy = SecurityPolicy(
            max_execution_time=30,
            max_memory_usage=128 * 1024 * 1024,  # 128MB
            max_output_size=1024 * 1024,  # 1MB
            network_access=False,
            file_system_access=False
        )
        
        self.execution_history = {}
        self.session_limits = {}
        
        # Session limits
        self.max_executions_per_session = 50
        self.max_execution_time_per_session = 300  # 5 minutes total
        
        logger.info("Secure Code Executor initialized")
    
    async def execute_code(self,
                          code: str,
                          language: str = "python",
                          session_id: str = None,
                          timeout: int = None,
                          custom_policy: SecurityPolicy = None) -> Dict[str, Any]:
        """
        Execute code securely with comprehensive monitoring
        
        Args:
            code: Code to execute
            language: Programming language (python, javascript)
            session_id: Session identifier for tracking
            timeout: Custom timeout (will not exceed policy max)
            custom_policy: Custom security policy
            
        Returns:
            Dictionary with execution results
        """
        
        session_id = session_id or str(uuid.uuid4())
        policy = custom_policy or self.default_policy
        
        # Apply timeout from parameter
        if timeout is not None:
            policy.max_execution_time = min(timeout, policy.max_execution_time)
        
        # Check session limits
        if not await self._check_session_limits(session_id):
            return {
                "output": "",
                "error": "Session execution limits exceeded",
                "execution_time": 0.0,
                "session_id": session_id,
                "language": language,
                "security_violation": True
            }
        
        # Validate code
        security_check = await self._security_check_code(code, language)
        if not security_check["safe"]:
            return {
                "output": "",
                "error": f"Security violation: {security_check['reason']}",
                "execution_time": 0.0,
                "session_id": session_id,
                "language": language,
                "security_violation": True
            }
        
        # Execute code in secure environment
        try:
            async with SecureExecutionEnvironment(policy) as env:
                if language.lower() == "python":
                    result = await env.execute_python(code)
                elif language.lower() in ["javascript", "js"]:
                    result = await env.execute_javascript(code)
                else:
                    return {
                        "output": "",
                        "error": f"Unsupported language: {language}",
                        "execution_time": 0.0,
                        "session_id": session_id,
                        "language": language,
                        "security_violation": False
                    }
                
                # Set session_id in result
                result.session_id = session_id
                
                # Update session tracking
                await self._update_session_tracking(session_id, result)
                
                # Store execution history
                self.execution_history[f"{session_id}_{int(time.time())}"] = result
                
                # Convert to dictionary
                return {
                    "output": result.output,
                    "error": result.error,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage,
                    "exit_code": result.exit_code,
                    "timeout": result.timeout,
                    "session_id": result.session_id,
                    "language": result.language,
                    "timestamp": result.timestamp.isoformat(),
                    "security_violation": False
                }
                
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            return {
                "output": "",
                "error": f"Execution environment error: {str(e)}",
                "execution_time": 0.0,
                "session_id": session_id,
                "language": language,
                "security_violation": False
            }
    
    async def _check_session_limits(self, session_id: str) -> bool:
        """Check if session has exceeded execution limits"""
        
        if session_id not in self.session_limits:
            self.session_limits[session_id] = {
                "execution_count": 0,
                "total_time": 0.0,
                "created": datetime.now()
            }
        
        session_info = self.session_limits[session_id]
        
        # Check execution count limit
        if session_info["execution_count"] >= self.max_executions_per_session:
            logger.warning(f"Session {session_id} exceeded execution count limit")
            return False
        
        # Check total time limit
        if session_info["total_time"] >= self.max_execution_time_per_session:
            logger.warning(f"Session {session_id} exceeded total time limit")
            return False
        
        # Check session age (24 hour limit)
        session_age = (datetime.now() - session_info["created"]).total_seconds()
        if session_age > 24 * 3600:
            logger.warning(f"Session {session_id} too old")
            return False
        
        return True
    
    async def _update_session_tracking(self, session_id: str, result: ExecutionResult):
        """Update session tracking information"""
        
        if session_id in self.session_limits:
            session_info = self.session_limits[session_id]
            session_info["execution_count"] += 1
            session_info["total_time"] += result.execution_time
    
    async def _security_check_code(self, code: str, language: str) -> Dict[str, Any]:
        """Perform security check on code before execution"""
        
        # Common dangerous patterns
        dangerous_patterns = [
            "import os", "from os", "__import__", "eval(", "exec(",
            "subprocess", "socket", "urllib", "requests", "http",
            "open(", "file(", "input(", "raw_input(",
            "globals()", "locals()", "vars()", "dir()",
            "getattr", "setattr", "delattr", "hasattr",
            "compile(", "reload(", "exit(", "quit()",
            "while True:", "for i in range(999999)",
            # JavaScript specific
            "require(", "process.", "global.", "Buffer.",
            "setTimeout", "setInterval", "XMLHttpRequest", "fetch("
        ]
        
        code_lower = code.lower()
        
        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                return {
                    "safe": False,
                    "reason": f"Dangerous pattern detected: {pattern}",
                    "severity": "high"
                }
        
        # Check code length
        if len(code) > 10000:  # 10KB limit
            return {
                "safe": False,
                "reason": "Code too long",
                "severity": "medium"
            }
        
        # Check for excessive loops
        if code_lower.count("while") > 5 or code_lower.count("for") > 10:
            return {
                "safe": False,
                "reason": "Too many loops detected",
                "severity": "medium"
            }
        
        return {
            "safe": True,
            "reason": "Code passed security checks",
            "severity": "none"
        }
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        
        if session_id not in self.session_limits:
            return {"error": "Session not found"}
        
        session_info = self.session_limits[session_id].copy()
        session_info["created"] = session_info["created"].isoformat()
        
        # Get execution history for session
        session_executions = []
        for key, result in self.execution_history.items():
            if result.session_id == session_id:
                session_executions.append({
                    "timestamp": result.timestamp.isoformat(),
                    "language": result.language,
                    "execution_time": result.execution_time,
                    "success": result.exit_code == 0 and not result.timeout
                })
        
        session_info["executions"] = session_executions
        session_info["remaining_executions"] = self.max_executions_per_session - session_info["execution_count"]
        session_info["remaining_time"] = self.max_execution_time_per_session - session_info["total_time"]
        
        return session_info
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        
        total_executions = sum(info["execution_count"] for info in self.session_limits.values())
        total_sessions = len(self.session_limits)
        
        # Language usage
        language_stats = {}
        for result in self.execution_history.values():
            lang = result.language
            if lang not in language_stats:
                language_stats[lang] = {"count": 0, "total_time": 0.0, "errors": 0}
            
            language_stats[lang]["count"] += 1
            language_stats[lang]["total_time"] += result.execution_time
            if result.error:
                language_stats[lang]["errors"] += 1
        
        return {
            "total_executions": total_executions,
            "total_sessions": total_sessions,
            "active_sessions": len([s for s in self.session_limits.values() 
                                  if (datetime.now() - s["created"]).total_seconds() < 3600]),
            "language_statistics": language_stats,
            "security_policy": {
                "max_execution_time": self.default_policy.max_execution_time,
                "max_memory_usage": self.default_policy.max_memory_usage,
                "max_output_size": self.default_policy.max_output_size,
                "network_access": self.default_policy.network_access,
                "file_system_access": self.default_policy.file_system_access
            }
        }
    
    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old session data"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up session limits
        sessions_to_remove = []
        for session_id, info in self.session_limits.items():
            if info["created"] < cutoff_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.session_limits[session_id]
        
        # Clean up execution history
        history_to_remove = []
        for key, result in self.execution_history.items():
            if result.timestamp < cutoff_time:
                history_to_remove.append(key)
        
        for key in history_to_remove:
            del self.execution_history[key]
        
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions and {len(history_to_remove)} old executions")


# For backward compatibility
CodeExecutor = SecureCodeExecutor
