"""
Tool registry - allows agents to call tools dynamically.
Agents can discover and call available tools.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class ToolDef:
    """Definition of an available tool."""
    name: str
    func: Callable
    description: str
    args_schema: Dict[str, Any]
    
    def __call__(self, **kwargs) -> Any:
        """Call the tool with given arguments."""
        return self.func(**kwargs)


class ToolRegistry:
    """
    Registry of tools available to agents.
    Agents can discover, inspect, and call tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolDef] = {}
    
    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        args_schema: Dict[str, Any],
    ) -> None:
        """Register a new tool."""
        self._tools[name] = ToolDef(
            name=name,
            func=func,
            description=description,
            args_schema=args_schema,
        )
    
    def get_tool(self, name: str) -> Optional[ToolDef]:
        """Get a specific tool."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self._tools.keys())
    
    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get tool description and schema."""
        tool = self.get_tool(name)
        if not tool:
            return {}
        return {
            "name": tool.name,
            "description": tool.description,
            "args_schema": tool.args_schema,
        }
    
    def list_tool_info(self) -> List[Dict[str, Any]]:
        """List all tools with descriptions."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema,
            }
            for tool in self._tools.values()
        ]
    
    def call_tool(self, name: str, **kwargs) -> Any:
        """Call a tool and return result."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        return tool(**kwargs)
    
    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"


# Global registry instance
_global_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


def register_tool(
    name: str,
    func: Callable,
    description: str,
    args_schema: Dict[str, Any],
) -> None:
    """Register a tool in the global registry."""
    _global_registry.register(name, func, description, args_schema)
