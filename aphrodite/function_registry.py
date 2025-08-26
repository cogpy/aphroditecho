"""
Function Registry - Unified function and tool management for AAR system.
Integrates argc command schemas with llm-functions capabilities.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import inspect
import importlib
from pathlib import Path

logger = logging.getLogger(__name__)


class SafetyClass(Enum):
    """Function safety classification."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class FunctionStatus(Enum):
    """Function registration status."""
    REGISTERED = "registered"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class ParameterSpec:
    """Function parameter specification."""
    type: str
    description: str
    required: bool = False
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionSpec:
    """Complete function specification following the contract schema."""
    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    safety_class: SafetyClass
    cost_unit: float
    implementation_ref: str
    tags: List[str] = field(default_factory=list)
    allow_network: bool = False
    version: str = "1.0.0"
    status: FunctionStatus = FunctionStatus.REGISTERED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_contract_dict(self) -> Dict[str, Any]:
        """Convert to contract-compliant dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "params": {
                name: {
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                    "default": param.default
                }
                for name, param in self.parameters.items()
            },
            "safety_class": self.safety_class.value,
            "cost_unit": self.cost_unit,
            "impl_ref": self.implementation_ref,
            "tags": self.tags,
            "allow_network": self.allow_network
        }


@dataclass
class FunctionInvocation:
    """Function invocation request."""
    function_name: str
    arguments: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class FunctionResult:
    """Function invocation result."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    cost_incurred: float = 0.0
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FunctionRegistry:
    """
    Unified function registry integrating argc command schemas with llm-functions.
    
    Provides:
    - Function registration and discovery
    - Schema validation and normalization
    - Safe function invocation with cost tracking
    - Integration with AAR agent policies
    """

    def __init__(self):
        self.functions: Dict[str, FunctionSpec] = {}
        self.implementations: Dict[str, Callable] = {}
        self.invocation_stats: Dict[str, Dict[str, Any]] = {}
        
        # Load built-in functions
        self._register_builtin_functions()

    def _register_builtin_functions(self):
        """Register essential built-in functions."""
        builtin_functions = [
            FunctionSpec(
                name="echo",
                description="Echo back the provided text",
                parameters={
                    "text": ParameterSpec(
                        type="string",
                        description="Text to echo back",
                        required=True
                    )
                },
                safety_class=SafetyClass.LOW,
                cost_unit=0.1,
                implementation_ref="builtin.echo"
            ),
            FunctionSpec(
                name="calculate",
                description="Perform mathematical calculations",
                parameters={
                    "expression": ParameterSpec(
                        type="string",
                        description="Mathematical expression to evaluate",
                        required=True
                    )
                },
                safety_class=SafetyClass.MEDIUM,
                cost_unit=0.5,
                implementation_ref="builtin.calculate"
            ),
            FunctionSpec(
                name="web_search", 
                description="Search the web for information",
                parameters={
                    "query": ParameterSpec(
                        type="string",
                        description="Search query",
                        required=True
                    ),
                    "max_results": ParameterSpec(
                        type="integer",
                        description="Maximum number of results",
                        required=False,
                        default=5
                    )
                },
                safety_class=SafetyClass.HIGH,
                cost_unit=2.0,
                implementation_ref="builtin.web_search",
                allow_network=True,
                tags=["search", "web"]
            ),
            FunctionSpec(
                name="file_read",
                description="Read contents of a file",
                parameters={
                    "path": ParameterSpec(
                        type="string", 
                        description="Path to the file to read",
                        required=True
                    )
                },
                safety_class=SafetyClass.HIGH,
                cost_unit=1.0,
                implementation_ref="builtin.file_read",
                tags=["filesystem"]
            ),
        ]
        
        for func_spec in builtin_functions:
            self.register_function(func_spec, self._get_builtin_implementation(func_spec.name))

    def _get_builtin_implementation(self, function_name: str) -> Callable:
        """Get built-in function implementation."""
        builtin_impls = {
            "echo": lambda text: {"echoed": text},
            "calculate": self._builtin_calculate,
            "web_search": self._builtin_web_search,
            "file_read": self._builtin_file_read,
        }
        return builtin_impls.get(function_name, lambda **kwargs: {"error": "not implemented"})

    def _builtin_calculate(self, expression: str) -> Dict[str, Any]:
        """Built-in calculator function."""
        try:
            # Basic safety check - only allow mathematical operations
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return {"error": "Invalid characters in expression"}
            
            result = eval(expression)
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}

    def _builtin_web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Built-in web search function (placeholder)."""
        # This would integrate with actual search APIs
        return {
            "query": query,
            "results": [
                {"title": f"Result {i}", "url": f"https://example.com/{i}"}
                for i in range(min(max_results, 3))
            ],
            "total": max_results
        }

    def _builtin_file_read(self, path: str) -> Dict[str, Any]:
        """Built-in file read function."""
        try:
            # Basic safety check - restrict to safe paths
            if ".." in path or path.startswith("/"):
                return {"error": "Unsafe path"}
            
            with open(path, 'r') as f:
                content = f.read()
            return {"content": content, "path": path}
        except Exception as e:
            return {"error": f"File read error: {str(e)}"}

    def register_function(self, spec: FunctionSpec, implementation: Callable) -> bool:
        """Register a function with its specification and implementation."""
        try:
            # Validate the specification
            self._validate_function_spec(spec)
            
            # Store the specification
            self.functions[spec.name] = spec
            
            # Store the implementation
            self.implementations[spec.name] = implementation
            
            # Initialize stats
            self.invocation_stats[spec.name] = {
                "total_invocations": 0,
                "total_cost": 0.0,
                "total_execution_time": 0.0,
                "success_count": 0,
                "error_count": 0
            }
            
            logger.info(f"Registered function: {spec.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register function {spec.name}: {e}")
            return False

    def _validate_function_spec(self, spec: FunctionSpec):
        """Validate function specification against contract schema."""
        if not spec.name or not isinstance(spec.name, str):
            raise ValueError("Function name must be a non-empty string")
        
        if len(spec.name) < 3 or len(spec.name) > 64:
            raise ValueError("Function name must be 3-64 characters")
        
        if not spec.description:
            raise ValueError("Function description is required")
        
        if not isinstance(spec.parameters, dict):
            raise ValueError("Parameters must be a dictionary")
        
        # Validate each parameter
        for param_name, param_spec in spec.parameters.items():
            if not isinstance(param_spec, ParameterSpec):
                raise ValueError(f"Parameter {param_name} must be a ParameterSpec")

    def get_function(self, name: str) -> Optional[FunctionSpec]:
        """Get function specification by name."""
        return self.functions.get(name)

    def list_functions(self, tags: Optional[List[str]] = None, 
                      safety_class: Optional[SafetyClass] = None) -> List[FunctionSpec]:
        """List functions with optional filtering."""
        functions = list(self.functions.values())
        
        if tags:
            functions = [f for f in functions if any(tag in f.tags for tag in tags)]
        
        if safety_class:
            functions = [f for f in functions if f.safety_class == safety_class]
        
        return functions

    def get_function_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get function schema in contract format."""
        func_spec = self.get_function(name)
        return func_spec.to_contract_dict() if func_spec else None

    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all function schemas in contract format."""
        return {
            name: spec.to_contract_dict()
            for name, spec in self.functions.items()
        }

    async def invoke_function(self, invocation: FunctionInvocation) -> FunctionResult:
        """Safely invoke a function with cost and permission tracking."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Get function spec
            func_spec = self.get_function(invocation.function_name)
            if not func_spec:
                return FunctionResult(
                    success=False,
                    error=f"Function {invocation.function_name} not found"
                )
            
            # Check if function is active
            if func_spec.status != FunctionStatus.ACTIVE:
                return FunctionResult(
                    success=False,
                    error=f"Function {invocation.function_name} is not active"
                )
            
            # Validate arguments
            validation_result = self._validate_arguments(func_spec, invocation.arguments)
            if not validation_result["valid"]:
                return FunctionResult(
                    success=False,
                    error=f"Argument validation failed: {validation_result['error']}"
                )
            
            # Get implementation
            implementation = self.implementations.get(invocation.function_name)
            if not implementation:
                return FunctionResult(
                    success=False,
                    error=f"No implementation found for {invocation.function_name}"
                )
            
            # Execute function
            if asyncio.iscoroutinefunction(implementation):
                result = await implementation(**invocation.arguments)
            else:
                result = implementation(**invocation.arguments)
            
            # Calculate execution time and cost
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            cost = func_spec.cost_unit
            
            # Update statistics
            self._update_stats(invocation.function_name, True, cost, execution_time)
            
            return FunctionResult(
                success=True,
                result=result,
                cost_incurred=cost,
                execution_time_ms=execution_time,
                metadata={"function_spec": func_spec.name}
            )
            
        except Exception as e:
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self._update_stats(invocation.function_name, False, 0, execution_time)
            
            logger.error(f"Function invocation error for {invocation.function_name}: {e}")
            return FunctionResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )

    def _validate_arguments(self, func_spec: FunctionSpec, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate function arguments against specification."""
        try:
            # Check required parameters
            for param_name, param_spec in func_spec.parameters.items():
                if param_spec.required and param_name not in arguments:
                    return {
                        "valid": False,
                        "error": f"Required parameter '{param_name}' missing"
                    }
            
            # Check for unexpected parameters
            for arg_name in arguments:
                if arg_name not in func_spec.parameters:
                    return {
                        "valid": False,
                        "error": f"Unexpected parameter '{arg_name}'"
                    }
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _update_stats(self, function_name: str, success: bool, cost: float, execution_time: float):
        """Update function invocation statistics."""
        if function_name not in self.invocation_stats:
            return
        
        stats = self.invocation_stats[function_name]
        stats["total_invocations"] += 1
        stats["total_cost"] += cost
        stats["total_execution_time"] += execution_time
        
        if success:
            stats["success_count"] += 1
        else:
            stats["error_count"] += 1

    def get_function_stats(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific function."""
        return self.invocation_stats.get(function_name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all functions."""
        return self.invocation_stats.copy()

    def activate_function(self, name: str) -> bool:
        """Activate a registered function."""
        if name in self.functions:
            self.functions[name].status = FunctionStatus.ACTIVE
            logger.info(f"Activated function: {name}")
            return True
        return False

    def deactivate_function(self, name: str) -> bool:
        """Deactivate a function."""
        if name in self.functions:
            self.functions[name].status = FunctionStatus.DISABLED
            logger.info(f"Deactivated function: {name}")
            return True
        return False

    def import_argc_schema(self, schema_path: Path) -> List[str]:
        """Import function definitions from argc schema files."""
        imported_functions = []
        
        try:
            # This would parse argc schema files and convert to function specs
            # For now, return placeholder
            logger.info(f"Importing argc schema from {schema_path}")
            return imported_functions
            
        except Exception as e:
            logger.error(f"Failed to import argc schema: {e}")
            return []

    def import_llm_functions(self, module_path: str) -> List[str]:
        """Import functions from llm-functions modules."""
        imported_functions = []
        
        try:
            # This would dynamically import from llm-functions modules
            logger.info(f"Importing llm-functions from {module_path}")
            return imported_functions
            
        except Exception as e:
            logger.error(f"Failed to import llm-functions: {e}")
            return []

    def export_openai_tools_format(self) -> List[Dict[str, Any]]:
        """Export functions in OpenAI tools format."""
        tools = []
        
        for func_spec in self.functions.values():
            if func_spec.status == FunctionStatus.ACTIVE:
                tool = {
                    "type": "function",
                    "function": {
                        "name": func_spec.name,
                        "description": func_spec.description,
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
                
                # Convert parameters
                for param_name, param_spec in func_spec.parameters.items():
                    tool["function"]["parameters"]["properties"][param_name] = {
                        "type": param_spec.type,
                        "description": param_spec.description
                    }
                    
                    if param_spec.required:
                        tool["function"]["parameters"]["required"].append(param_name)
                
                tools.append(tool)
        
        return tools

    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the function registry."""
        total_functions = len(self.functions)
        active_functions = len([f for f in self.functions.values() 
                              if f.status == FunctionStatus.ACTIVE])
        
        return {
            "status": "healthy",
            "total_functions": total_functions,
            "active_functions": active_functions,
            "total_invocations": sum(
                stats["total_invocations"] 
                for stats in self.invocation_stats.values()
            ),
            "implementation_coverage": len(self.implementations) / total_functions if total_functions > 0 else 0
        }