from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod


@dataclass
class AgentOutput:
    agent: str
    payload: Dict[str, Any]


class BaseAgent(ABC):
    """
    Base class for autonomous agents.
    Agents can:
    - Call tools directly via tool registry
    - Communicate with other agents
    - Make autonomous decisions
    """
    name: str = "base"

    def __init__(self):
        self.communicator = None
        self.tool_registry = None
    
    def set_communicator(self, communicator):
        """Set the communication bus."""
        self.communicator = communicator
    
    def set_tool_registry(self, registry):
        """Set the tool registry."""
        self.tool_registry = registry
    
    def send_message(
        self,
        subject: str,
        content: Any,
        receiver: Optional[str] = None,
        message_type: str = "info",
    ) -> None:
        """Send a message to other agent(s)."""
        if self.communicator:
            self.communicator.send_message(
                sender=self.name,
                subject=subject,
                content=content,
                receiver=receiver,
                message_type=message_type,
            )
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool and log the call."""
        if not self.tool_registry:
            raise RuntimeError(f"Tool registry not set for {self.name}")
        
        result = None
        error = None
        
        try:
            result = self.tool_registry.call_tool(tool_name, **kwargs)
        except Exception as e:
            error = str(e)
            if self.communicator:
                self.communicator.log_tool_call(
                    agent=self.name,
                    tool_name=tool_name,
                    args=kwargs,
                    error=error,
                )
            raise
        
        # Log the successful call
        if self.communicator:
            self.communicator.log_tool_call(
                agent=self.name,
                tool_name=tool_name,
                args=kwargs,
                result=result,
            )
        
        return result
    
    def get_available_tools(self) -> list:
        """List available tools."""
        if not self.tool_registry:
            return []
        return self.tool_registry.list_tool_info()
    
    def remember(self, state: Dict[str, Any]) -> None:
        """Store analysis state for other agents to access."""
        if self.communicator:
            self.communicator.set_agent_state(self.name, state)
    
    def listen(self) -> list:
        """Get messages sent to this agent."""
        if not self.communicator:
            return []
        return self.communicator.get_messages_for(self.name)
    
    def output(self, payload: Dict[str, Any]) -> AgentOutput:
        """Wrap agent output in AgentOutput format."""
        return AgentOutput(agent=self.name, payload=payload)
    
    @abstractmethod
    def run(self, state: "SharedState", llm: "OllamaStructuredAgent") -> AgentOutput:
        """Run the agent. Subclasses must implement."""
        raise NotImplementedError

