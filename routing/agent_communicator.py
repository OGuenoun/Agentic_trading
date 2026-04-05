"""
Inter-agent communication bus with logging.
Allows agents to send messages to each other and provides audit trail.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json


@dataclass
class Message:
    """A message sent between agents."""
    sender: str
    receiver: Optional[str]  # None = broadcast to all
    subject: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_type: str = "info"  # info, question, finding, warning, decision
    
    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "subject": self.subject,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
        }


@dataclass
class ToolCall:
    """Record of a tool call made by an agent."""
    agent: str
    tool_name: str
    args: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            "agent": self.agent,
            "tool_name": self.tool_name,
            "args": self.args,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class AgentCommunicator:
    """
    Communication bus for agents with full logging.
    Provides inter-agent messaging and tool call tracking.
    """
    
    def __init__(self):
        self.messages: List[Message] = []
        self.tool_calls: List[ToolCall] = []
        self.agent_states: Dict[str, Any] = {}
    
    def send_message(
        self,
        sender: str,
        subject: str,
        content: Any,
        receiver: Optional[str] = None,
        message_type: str = "info",
    ) -> None:
        """Send a message from one agent to another or broadcast."""
        msg = Message(
            sender=sender,
            receiver=receiver,
            subject=subject,
            content=content,
            message_type=message_type,
        )
        self.messages.append(msg)
        self._log_message(msg)
    
    def log_tool_call(
        self,
        agent: str,
        tool_name: str,
        args: Dict[str, Any],
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """Log a tool call made by an agent."""
        call = ToolCall(
            agent=agent,
            tool_name=tool_name,
            args=args,
            result=result,
            error=error,
        )
        self.tool_calls.append(call)
        self._log_tool_call(call)
    
    def get_messages_for(self, agent: str) -> List[Message]:
        """Get all messages directed to an agent or broadcasted."""
        return [
            m for m in self.messages
            if m.receiver is None or m.receiver == agent
        ]
    
    def get_messages_from(self, agent: str) -> List[Message]:
        """Get all messages sent by an agent."""
        return [m for m in self.messages if m.sender == agent]
    
    def get_agent_messages(self, agent: str) -> List[Message]:
        """Get all messages related to an agent (sent or received)."""
        return self.get_messages_for(agent) + self.get_messages_from(agent)
    
    def get_tool_calls_by_agent(self, agent: str) -> List[ToolCall]:
        """Get all tool calls made by an agent."""
        return [tc for tc in self.tool_calls if tc.agent == agent]
    
    def set_agent_state(self, agent: str, state: Dict[str, Any]) -> None:
        """Store agent's analysis state."""
        self.agent_states[agent] = {
            "timestamp": datetime.utcnow().isoformat(),
            "state": state,
        }
    
    def get_agent_state(self, agent: str) -> Optional[Dict[str, Any]]:
        """Retrieve agent's analysis state."""
        return self.agent_states.get(agent, {}).get("state")
    
    def dump(self) -> Dict[str, Any]:
        """Export full communication log."""
        return {
            "messages": [m.to_dict() for m in self.messages],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "agent_states": {
                agent: state["state"]
                for agent, state in self.agent_states.items()
            },
            "total_messages": len(self.messages),
            "total_tool_calls": len(self.tool_calls),
        }
    
    def get_execution_summary(self) -> str:
        """Get human-readable execution summary."""
        lines = []
        lines.append("\n" + "="*60)
        lines.append("AGENT EXECUTION LOG")
        lines.append("="*60)
        
        # Timeline
        lines.append("\n📋 TIMELINE:")
        lines.append("-" * 60)
        
        all_events = []
        for msg in self.messages:
            all_events.append({
                "time": msg.timestamp,
                "type": "message",
                "text": f"[{msg.sender}] → {msg.receiver or 'All'}: {msg.subject}",
            })
        for tc in self.tool_calls:
            status = "[SUCCESS]" if tc.error is None else "[FAILED]"
            all_events.append({
                "time": tc.timestamp,
                "type": "tool_call",
                "text": f"{status} [{tc.agent}] called {tc.tool_name}()",
            })
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x["time"])
        for event in all_events:
            timestamp = event["time"].strftime("%H:%M:%S")
            lines.append(f"{timestamp} | {event['text']}")
        
        # Tool Summary
        lines.append("\nTOOL CALLS BY AGENT:")
        lines.append("-" * 60)
        for agent in sorted(set(tc.agent for tc in self.tool_calls)):
            calls = self.get_tool_calls_by_agent(agent)
            successful = len([tc for tc in calls if tc.error is None])
            failed = len([tc for tc in calls if tc.error is not None])
            tools_called = set(tc.tool_name for tc in calls)
            lines.append(f"{agent}: {len(calls)} calls ({successful} ok, {failed} failed)")
            lines.append(f"  Tools: {', '.join(tools_called)}")
        
        # Messages Summary
        lines.append("\nINTER-AGENT COMMUNICATION:")
        lines.append("-" * 60)
        for agent in sorted(set(m.sender for m in self.messages)):
            sent = len(self.get_messages_from(agent))
            received = len(self.get_messages_for(agent))
            lines.append(f"{agent}: sent {sent}, received {received}")
        
        lines.append("\n" + "="*60)
        return "\n".join(lines)
    
    def _log_message(self, msg: Message) -> None:
        """Internal: format message for logging."""
        msg_type_label = {
            "info": "[INFO]",
            "question": "[QUESTION]",
            "finding": "[FINDING]",
            "warning": "[WARNING]",
            "decision": "[DECISION]",
        }.get(msg.message_type, "[MESSAGE]")
        
        receiver = msg.receiver or "All"
        print(f"{msg_type_label} [{msg.sender}] → [{receiver}] {msg.subject}")
        if isinstance(msg.content, (dict, list)):
            print(f"   {json.dumps(msg.content, indent=2)[:100]}...")
        else:
            print(f"   {str(msg.content)[:100]}")
    
    def _log_tool_call(self, call: ToolCall) -> None:
        """Internal: format tool call for logging."""
        status_icon = "[OK]" if call.error is None else "[ERROR]"
        print(f"{status_icon} [{call.agent}] → {call.tool_name}({list(call.args.keys())})")
        if call.error:
            print(f"   ERROR: {call.error}")
