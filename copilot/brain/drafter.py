"""Message drafter — draft-and-approve only.

Builds an opener or a reply in the owner's voice, using a style corpus of his own
past messages as few-shot context. Drafts land in a queue for one-tap approval;
nothing is sent autonomously. Fully-autonomous conversation is explicitly out of
scope of this module.

The LLM is behind an interface with two implementations:
  StubLLM         deterministic templated draft, no network — for dev/tests
  AnthropicLLM    real drafts via the Anthropic SDK (defaults to claude-opus-5)
"""

from __future__ import annotations

from typing import Protocol

_SYSTEM = (
    "You draft short, casual dating-app messages in the owner's own voice. "
    "You are given examples of how he writes; match that voice — his tone, "
    "length, and punctuation. Keep it to one or two sentences. Reference "
    "something concrete from her profile or the thread. No emoji unless his "
    "examples use them. Output only the message text, nothing else."
)


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class StubLLM:
    """Deterministic, network-free drafter for dev and tests."""

    def complete(self, system: str, user: str) -> str:
        # Pull the first concrete hook we can find in the prompt for a plausible
        # opener; otherwise a safe generic line.
        hook = ""
        for line in user.splitlines():
            line = line.strip()
            if line.lower().startswith("bio:") and len(line) > 4:
                hook = line[4:].strip()
                break
        if hook:
            return f"okay I have to ask about {hook.rstrip('.').lower()} — tell me more?"
        return "hey! your profile made me smile. how's your week going?"


class AnthropicLLM:
    """Real drafts via the Anthropic SDK. Imported lazily."""

    def __init__(self, model: str = "claude-opus-5") -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "AnthropicLLM needs the `anthropic` SDK installed and "
                "ANTHROPIC_API_KEY set. Use COPILOT_DRAFTER=stub otherwise."
            ) from exc
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, system: str, user: str) -> str:  # pragma: no cover - needs network
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Guard the refusal stop reason before reading content.
        if response.stop_reason == "refusal":
            return ""
        for block in response.content:
            if block.type == "text":
                return block.text.strip()
        return ""


class MessageDrafter:
    def __init__(self, llm: LLMClient, style_corpus: list[str] | None = None,
                 max_examples: int = 12) -> None:
        self.llm = llm
        self.style_corpus = style_corpus or []
        self.max_examples = max_examples

    def _voice_block(self) -> str:
        if not self.style_corpus:
            return "(no example messages available yet)"
        examples = self.style_corpus[-self.max_examples :]
        return "\n".join(f"- {ex}" for ex in examples)

    def draft_opener(self, profile) -> str:
        user = (
            "Here is how the owner writes (examples of his sent messages):\n"
            f"{self._voice_block()}\n\n"
            "Draft an opener for this new match.\n"
            f"Age: {profile.age}\n"
            f"Bio: {profile.bio_text}\n"
        )
        return self.llm.complete(_SYSTEM, user)

    def draft_reply(self, conversation) -> str:
        thread = "\n".join(
            f"{m['from']}: {m['text']}" for m in conversation.messages
        )
        user = (
            "Here is how the owner writes (examples of his sent messages):\n"
            f"{self._voice_block()}\n\n"
            "Here is the conversation so far:\n"
            f"{thread}\n\n"
            "Draft his next reply in his voice."
        )
        return self.llm.complete(_SYSTEM, user)


def get_llm(config) -> LLMClient:
    if config.drafter == "anthropic":
        try:
            return AnthropicLLM(config.draft_model)
        except RuntimeError:
            return StubLLM()
    return StubLLM()
