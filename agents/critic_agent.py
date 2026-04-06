from __future__ import annotations

from agents.base import BaseAgent, AgentOutput


class CriticAgent(BaseAgent):
    name = "critic"

    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "challenge"]},
            "concerns": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["verdict", "concerns", "summary"],
    }

    def run(self, state, llm):
        """Critic agent listens to others and challenges weak theses using LLM reasoning."""
        
        # Announce intent
        self.send_message(
            subject="Starting to challenge weak theses",
            content={},
            message_type="info"
        )
        
        # Listen to what other agents said
        incoming_messages = self.listen()
        findings = [m.content for m in incoming_messages if m.message_type == "finding"]
        
        # Get all agent views from state
        news_view = state.get("news_view", {})
        quant_view = state.get("quant_view", {})
        technical_view = state.get("technical_view", {})
        bull_view = state.get("output::BullAgent", {})
        bear_view = state.get("output::BearAgent", {})

        # Ask LLM to identify concerns and weaknesses
        llm_prompt = f"""
You are a critical analyst reviewing this multi-agent debate:

News View:
{news_view}

Quant View:
{quant_view}

Technical View:
{technical_view}

Bull Argument (signal={bull_view.get("signal", "N/A")}, confidence={bull_view.get("confidence", 0):.2f}):
{bull_view.get("rationale", "No rationale")}

Bear Argument (signal={bear_view.get("signal", "N/A")}, confidence={bear_view.get("confidence", 0):.2f}):
{bear_view.get("rationale", "No rationale")}

Identify any logical inconsistencies, weak edges, excessive risks, or conflicting signals.
Challenge the bull/bear rationales if they contradict the quant or news data.
Should we challenge this thesis or let it pass?
Return a verdict (pass or challenge) and list specific concerns.
"""
        
        llm_result = llm.run_structured(
            system="You are a rigorous critic who challenges weak analytical theses. Identify risks, conflicts, and weak edges.",
            user=llm_prompt,
            schema=self.schema,
            fallback={
                "verdict": "pass",
                "concerns": [],
                "summary": "Fallback critical analysis used because Ollama was unavailable."
            }
        )
        
        payload = {
            "verdict": llm_result.get("verdict", "challenge"),
            "concerns": llm_result.get("concerns", []),
            "summary": llm_result.get("summary", f"Critic verdict: {llm_result.get('verdict', 'challenge')}"),
        }
        
        # Broadcast verdict to others
        self.send_message(
            subject="Critic verdict",
            content=payload,
            message_type="decision"
        )
        
        # Remember analysis
        self.remember(payload)
        
        state.set("critic_view", payload)
        state.publish(self.name, payload["summary"])
        return AgentOutput(agent=self.name, payload=payload)
