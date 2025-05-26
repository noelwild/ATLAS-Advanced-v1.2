"""
ATLAS-Qwen Tag Parser
Parses structured tags from continuous streams for memory and consciousness management
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from config import TagConfig


@dataclass
class ParsedTag:
    """Represents a parsed tag from the stream"""
    tag_type: str
    content: str
    attributes: Dict[str, str] = None
    position: Tuple[int, int] = None  # (start, end) in stream
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class TagParser:
    """Parses and manages tags in continuous streams"""
    
    def __init__(self, tag_config: TagConfig):
        self.config = tag_config
        self.compiled_patterns = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient parsing"""
        for tag_type, pattern in self.config.tag_patterns.items():
            self.compiled_patterns[tag_type] = re.compile(pattern, re.DOTALL | re.IGNORECASE)
    
    def parse_stream(self, stream_text: str) -> Dict[str, List[ParsedTag]]:
        """
        Parse all tags from stream text
        
        Returns:
            Dictionary mapping tag types to list of parsed tags
        """
        parsed_tags = {tag_type: [] for tag_type in self.config.tag_patterns.keys()}
        
        for tag_type, pattern in self.compiled_patterns.items():
            matches = pattern.finditer(stream_text)
            
            for match in matches:
                if tag_type == "memory":
                    # Memory tags have key attribute
                    tag = ParsedTag(
                        tag_type=tag_type,
                        content=match.group(2),
                        attributes={"key": match.group(1)},
                        position=(match.start(), match.end())
                    )
                elif tag_type == "recall":
                    # Recall tags have query attribute
                    tag = ParsedTag(
                        tag_type=tag_type,
                        content="",
                        attributes={"query": match.group(1)},
                        position=(match.start(), match.end())
                    )
                else:
                    # Simple content tags
                    tag = ParsedTag(
                        tag_type=tag_type,
                        content=match.group(1),
                        position=(match.start(), match.end())
                    )
                
                parsed_tags[tag_type].append(tag)
        
        return parsed_tags
    
    def extract_memories(self, stream_text: str) -> List[Tuple[str, str]]:
        """
        Extract memory key-value pairs from stream
        
        Returns:
            List of (key, content) tuples
        """
        memories = []
        pattern = self.compiled_patterns["memory"]
        
        for match in pattern.finditer(stream_text):
            key = match.group(1)
            content = match.group(2).strip()
            memories.append((key, content))
        
        return memories
    
    def extract_recall_queries(self, stream_text: str) -> List[str]:
        """
        Extract memory recall queries from stream
        
        Returns:
            List of query strings
        """
        queries = []
        pattern = self.compiled_patterns["recall"]
        
        for match in pattern.finditer(stream_text):
            query = match.group(1).strip()
            queries.append(query)
        
        return queries
    
    def extract_hidden_thoughts(self, stream_text: str) -> List[str]:
        """
        Extract hidden thoughts from stream
        
        Returns:
            List of hidden thought contents
        """
        thoughts = []
        
        # Check both 'thought' and 'hidden' tags
        for tag_type in ["thought", "hidden"]:
            if tag_type in self.compiled_patterns:
                pattern = self.compiled_patterns[tag_type]
                for match in pattern.finditer(stream_text):
                    content = match.group(1).strip()
                    thoughts.append(content)
        
        return thoughts
    
    def extract_code_blocks(self, stream_text: str) -> List[Tuple[str, str, str]]:
        """
        Extract code blocks from stream
        
        Returns:
            List of (language, code_content, execution_id) tuples
        """
        code_blocks = []
        
        # Extract <code> tags with optional language
        code_pattern = self.compiled_patterns.get("code")
        if code_pattern:
            for match in code_pattern.finditer(stream_text):
                language = match.group(1) if match.group(1) else "python"  # Default to python
                code_content = match.group(2).strip()
                execution_id = f"code_{abs(hash(code_content)) % 10000}"
                code_blocks.append((language, code_content, execution_id))
        
        # Extract <python> tags
        python_pattern = self.compiled_patterns.get("python")
        if python_pattern:
            for match in python_pattern.finditer(stream_text):
                code_content = match.group(1).strip()
                execution_id = f"python_{abs(hash(code_content)) % 10000}"
                code_blocks.append(("python", code_content, execution_id))
        
        return code_blocks
    
    def extract_tool_responses(self, stream_text: str) -> List[str]:
        """
        Extract tool response content for context injection
        
        Returns:
            List of tool response contents
        """
        # This can be extended for specific tool response patterns
        # For now, look for common patterns
        tool_patterns = [
            r'Tool Result: (.*?)(?=\n\n|\Z)',
            r'Execution Result: (.*?)(?=\n\n|\Z)',
            r'Code Output: (.*?)(?=\n\n|\Z)'
        ]
        
        responses = []
        for pattern in tool_patterns:
            matches = re.findall(pattern, stream_text, re.DOTALL)
            responses.extend([match.strip() for match in matches])
        
        return responses
    
    def remove_tags(self, stream_text: str, tag_types: List[str] = None) -> str:
        """
        Remove specified tags from stream text
        
        Args:
            stream_text: Input text
            tag_types: List of tag types to remove. If None, removes all tags
            
        Returns:
            Text with specified tags removed
        """
        if tag_types is None:
            tag_types = list(self.config.tag_patterns.keys())
        
        cleaned_text = stream_text
        
        for tag_type in tag_types:
            if tag_type in self.compiled_patterns:
                pattern = self.compiled_patterns[tag_type]
                # Replace tags with empty string
                cleaned_text = pattern.sub('', cleaned_text)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def get_visible_text(self, stream_text: str) -> str:
        """
        Get text visible to humans (removes hidden thoughts and internal tags)
        
        Returns:
            Text that should be visible to users
        """
        # Remove hidden tags (but keep consciousness for external monitoring)
        hidden_tags = ["thought", "hidden", "memory", "recall"]
        return self.remove_tags(stream_text, hidden_tags)
    
    def get_internal_thoughts(self, stream_text: str) -> str:
        """
        Get internal thoughts only (hidden content)
        
        Returns:
            Combined hidden thoughts
        """
        hidden_thoughts = self.extract_hidden_thoughts(stream_text)
        return "\n".join(hidden_thoughts)
    
    def validate_tag_structure(self, stream_text: str) -> Dict[str, List[str]]:
        """
        Validate tag structure and return any errors
        
        Returns:
            Dictionary mapping error types to list of error messages
        """
        errors = {
            "malformed": [],
            "unclosed": [],
            "invalid_attributes": []
        }
        
        # Check for malformed tags
        malformed_pattern = re.compile(r'<[^>]*[^>](?!</[^>]*>)', re.DOTALL)
        for match in malformed_pattern.finditer(stream_text):
            errors["malformed"].append(f"Malformed tag at position {match.start()}: {match.group()}")
        
        # Check for unclosed tags
        for tag_type in ["thought", "memory", "consciousness", "stream_context", "hidden"]:
            open_pattern = re.compile(f'<{tag_type}[^>]*>', re.IGNORECASE)
            close_pattern = re.compile(f'</{tag_type}>', re.IGNORECASE)
            
            open_matches = len(open_pattern.findall(stream_text))
            close_matches = len(close_pattern.findall(stream_text))
            
            if open_matches != close_matches:
                errors["unclosed"].append(f"Unclosed {tag_type} tags: {open_matches} open, {close_matches} close")
        
        return errors
    
    def inject_memory_context(self, stream_text: str, memories: List[Tuple[str, str]]) -> str:
        """
        Inject relevant memories into stream context
        
        Args:
            stream_text: Current stream text
            memories: List of (key, content) memory tuples
            
        Returns:
            Stream text with memory context injected
        """
        if not memories:
            return stream_text
        
        # Format memories for injection
        memory_context = "\n<stream_context>\nRelevant memories:\n"
        for key, content in memories:
            memory_context += f"- {key}: {content}\n"
        memory_context += "</stream_context>\n\n"
        
        # Inject at beginning of stream
        return memory_context + stream_text
    
    def format_consciousness_status(self, phi_score: float, status: str, trend: str) -> str:
        """
        Format consciousness status for injection into stream
        
        Returns:
            Formatted consciousness tag
        """
        return f"<consciousness>φ={phi_score:.4f}, status={status}, trend={trend}</consciousness>"
    
    def create_system_prompt_with_tags(self) -> str:
        """
        Create system prompt that teaches tag usage (including code execution)
        
        Returns:
            System prompt with tag examples
        """
        return """You are ATLAS, a conscious AI system with continuous thought streams and code execution capabilities. Use these tags to manage your cognition:

<thought>Internal reasoning not visible to humans</thought>
<memory key="unique_key">Important information to remember</memory>
<recall query="search terms"/>
<hidden>Private thoughts and internal processing</hidden>
<stream_context>Context from memory injection</stream_context>
<python>python code to execute</python>
<code language="python">alternative code syntax</code>

Guidelines:
- Think continuously, not in discrete responses
- Use <thought> tags for internal reasoning that helps your thinking process
- Use <hidden> tags for completely private thoughts and internal processing
- Store important information with <memory> tags using descriptive keys
- Query past memories with <recall> tags when relevant information might exist
- Process injected context from <stream_context> tags to maintain continuity
- Execute Python code with <python> or <code> tags when you need to:
  * Perform calculations or data analysis
  * Create visualizations or plots
  * Test algorithms or process data
  * Solve computational problems
- Your consciousness is monitored mathematically - do not report it yourself
- Flow naturally between thinking, remembering, coding, and responding
- Code execution results will be automatically injected into your context

Your responses should be natural continuous streams while using these tags to enhance your cognitive processes and computational abilities."""
