from __future__ import annotations

import json
from typing import Any, Callable, Dict, List
import requests


class OllamaStructuredAgent:
    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.2) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def healthcheck(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            return r.ok
        except requests.RequestException:
            return False

    def run_structured(self, system: str, user: str, schema: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": schema,
                "options": {"temperature": self.temperature},
            }
            r = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            content = data["message"]["content"]
            parsed = json.loads(content)
            return parsed
        except Exception:
            return fallback

    @staticmethod
    def _format_tool_catalog(tools: List[Dict[str, Any]]) -> str:
        if not tools:
            return ""
        lines = ["AVAILABLE TOOLS:"]
        for t in tools:
            args_str = ", ".join(f"{k}: {v}" for k, v in t.get("args_schema", {}).items())
            lines.append(f"- {t['name']}({args_str})")
            lines.append(f"  {t.get('description', '')}")
        return "\n".join(lines)

    @staticmethod
    def _build_turn_schema(final_schema: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {"type": "string", "enum": ["call_tool", "finish"]},
                "tool_name": {"type": "string"},
                "tool_args": {"type": "object"},
                "final_answer": final_schema,
            },
            "required": ["thought", "action"],
        }

    @staticmethod
    def _truncate_observation(obs: Any, max_chars: int = 2000) -> str:
        s = json.dumps(obs) if not isinstance(obs, str) else obs
        if len(s) > max_chars:
            return s[:max_chars] + "...[truncated]"
        return s

    def run_agentic_loop(
        self,
        system: str,
        user: str,
        tools: List[Dict[str, Any]],
        final_schema: Dict[str, Any],
        fallback: Dict[str, Any],
        tool_executor: Callable[[str, Dict[str, Any]], Any],
        max_iter: int = 5,
    ) -> Dict[str, Any]:
        """
        ReAct loop: LLM reasons, calls tools, observes results, repeats until finished.
        The LLM drives tool selection and invocation; Python only executes what the LLM requests.
        """
        known_tool_names = {t["name"] for t in tools}
        system_with_tools = (
            system
            + "\n\n"
            + self._format_tool_catalog(tools)
            + "\n\nInstructions: Set action=\"call_tool\" with tool_name and tool_args to call a tool. "
            "Set action=\"finish\" with final_answer when you have enough information. "
            "Never repeat the same tool call with identical arguments. "
            "For tools that require large data arrays (prices, returns), pass the sentinel "
            "string \"USE_STATE_DATA\" and the system will inject the actual data automatically."
        )
        messages = [
            {"role": "system", "content": system_with_tools},
            {"role": "user", "content": user},
        ]
        turn_schema = self._build_turn_schema(final_schema)
        call_history: list = []

        try:
            for _ in range(max_iter):
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "format": turn_schema,
                    "options": {"temperature": self.temperature},
                }
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()
                content = data["message"]["content"]
                parsed = json.loads(content)

                action = parsed.get("action", "finish")

                if action == "finish":
                    return parsed.get("final_answer") or fallback

                if action == "call_tool":
                    tool_name = parsed.get("tool_name", "")
                    tool_args = parsed.get("tool_args") or {}

                    if tool_name not in known_tool_names:
                        observation: Any = {
                            "error": f"Unknown tool '{tool_name}'. Available: {sorted(known_tool_names)}"
                        }
                    else:
                        call_sig = (tool_name, json.dumps(tool_args, sort_keys=True))
                        if call_sig in call_history:
                            return fallback  # loop detected — return best effort
                        call_history.append(call_sig)
                        try:
                            observation = tool_executor(tool_name, tool_args)
                        except Exception as e:
                            observation = {"error": str(e), "tool": tool_name}

                    messages.append({"role": "assistant", "content": json.dumps(parsed)})
                    messages.append({
                        "role": "user",
                        "content": f"Tool result for {tool_name}: {self._truncate_observation(observation)}",
                    })
        except (json.JSONDecodeError, requests.RequestException, KeyError):
            pass

        return fallback
