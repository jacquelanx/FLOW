"""Smoke-test the Azure OpenAI (or APIM-gateway) provider end to end.

Reads AZURE_OPENAI_* from your .env / shell, then:
  1. sends a trivial free-text chat completion (transport + auth check), and
  2. forces a tool call (what the FLOW agent actually relies on every step).

Usage:
    .venv/bin/python scripts/test_azure.py <deployment-name>
    # e.g. .venv/bin/python scripts/test_azure.py gpt-35-turbo

Exit code 0 means the endpoint authenticated and returned a usable tool call.
No dataset, no Docker, no real analysis — just a connectivity/capability probe.
"""

from __future__ import annotations

import sys

from flow.envfile import load_dotenv
from flow.aviary.message import Message
from flow.aviary.tool import Tool
from flow.providers.registry import build_provider


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    deployment = sys.argv[1]

    load_dotenv()  # pulls AZURE_OPENAI_* from .env if present

    try:
        provider = build_provider("azure", deployment)
    except Exception as e:
        print(f"FAILED to build provider: {e}")
        print("Check AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT in your .env.")
        return 1

    print(f"Endpoint : {provider.endpoint}")
    print(f"URL      : {provider._chat_url()}")
    print(f"Auth hdr : {provider.auth_header}")
    print(f"Model    : {provider.model}")
    print("-" * 60)

    # 1) Plain completion — proves the host/auth/api-version are all correct.
    print("[1/2] free-text completion ...")
    try:
        text = provider.complete(
            [Message(role="user", content="Reply with exactly the word: pong")]
        )
        print(f"      OK -> {text!r}")
    except Exception as e:
        print(f"      FAILED: {e}")
        return 1

    # 2) Forced tool call — the FLOW agent forces one tool call per step, so the
    #    deployment must support function calling for a real run to work.
    print("[2/2] forced tool call (function calling) ...")
    tool = Tool(
        name="report_number",
        description="Report a single integer back to the caller.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
        fn=lambda value: value,  # unused here; Tool requires a callable
    )
    try:
        call = provider.generate(
            [Message(role="user", content="Call report_number with value 42.")],
            [tool],
        )
        print(f"      OK -> tool={call.name!r} args={call.arguments!r}")
    except Exception as e:
        print(f"      FAILED: {e}")
        print("      If completion (step 1) worked but this did not, the deployment")
        print("      likely does not support function calling (e.g. a legacy GPT-3")
        print("      completion model). FLOW needs a chat model with tool calling.")
        return 1

    print("-" * 60)
    print("SUCCESS: this deployment works with FLOW.")
    print(f"Run it with:  flow run --provider azure --model {deployment} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
