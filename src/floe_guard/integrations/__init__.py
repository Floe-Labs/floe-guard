"""Optional framework adapters for floe-guard.

Each adapter lives behind an optional extra so the core stays dependency-free:

    pip install floe-guard[litellm]
    pip install floe-guard[crewai]
    pip install floe-guard[langchain]
    pip install floe-guard[langgraph]

The voice adapters are split: Pipecat and LiveKit pull their framework extras
(``[pipecat]`` / ``[livekit]``); the Vapi and Retell adapters are
framework-free — they speak Vapi's and Retell's plain wire formats (OpenAI-format
JSON/SSE and WebSocket dicts) with no SDK dependency, so they need no extra.
"""
