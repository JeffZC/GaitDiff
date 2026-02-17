import os
from typing import List, Dict, Any

# Lazy import inside function so module can be imported without Azure SDK or env vars present.

def _ensure_env():
    endpoint = "https://zyang-mknld564-eastus2.cognitiveservices.azure.com/"
    deploy = "gpt-5-mini-2"
    return endpoint, deploy


def call_azure_chat(messages: List[Dict[str, Any]], max_tokens: int = 512) -> str:
    """Call Azure OpenAI responses API via API key auth."""
    endpoint, deploy = _ensure_env()

    try:
        from openai import AzureOpenAI
    except Exception as e:
        raise RuntimeError("openai package is required. Install with: pip install openai") from e

    subscription_key = os.environ.get("AZURE_OPENAI_KEY")
    if not subscription_key:
        raise RuntimeError("AZURE_OPENAI_KEY is not set; cannot call Azure OpenAI.")

    client = AzureOpenAI(
        api_version="2025-03-01-preview",
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    resp = client.responses.create(
        model=deploy,
        input=[
            {
                "role": m.get("role", "user"),
                "content": m.get("content", ""),
            }
            for m in messages
        ],
        max_output_tokens=max(max_tokens, 1024),
        reasoning={"effort": "high"},
        text={"format": {"type": "text"}},
    )

    assistant_text = getattr(resp, "output_text", None)
    if not assistant_text:
        try:
            raw = resp.model_dump()
        except Exception:
            raw = {"response": resp}
        assistant_text = str(raw)
    return assistant_text
