"""Discover which Azure OpenAI deployment names are valid on your endpoint.

The APIM dev portal doesn't list deployments, so this probes the chat-completions URL
with a set of common deployment names and reports what the gateway says for each:

    200            -> VALID deployment (usable as your --model)
    404 NotFound   -> not a deployment name here
    401 / 403      -> auth problem (fix your key/header before trusting other results)
    other          -> shown verbatim so you can eyeball it

Reads AZURE_OPENAI_* from .env / shell. Add extra candidate names as CLI args, e.g. if
your boss says the name is "dig-gpt5":

    .venv/bin/python scripts/discover_azure_deployments.py dig-gpt5 gpt-5-chat

This only sends a 1-token request per name; it writes nothing and runs no analysis.
"""

from __future__ import annotations

import os
import sys

import httpx

from flow.envfile import load_dotenv

# Common Azure deployment names circa 2025-2026. Deployments are often named after the
# base model, so these are worth trying — but an org can name a deployment anything, so a
# miss here does NOT mean the model is unavailable (ask the admin for the exact name).
DEFAULT_CANDIDATES = [
    "gpt-5", "gpt-5-chat", "gpt-5-mini", "gpt-5-nano", "gpt-5.1",
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
    "gpt-4o", "gpt-4o-mini", "gpt-4o-2", "gpt-4",
    "gpt-4-turbo", "gpt-4-32k",
    "gpt-35-turbo", "gpt-35-turbo-16k", "gpt-3.5-turbo",
    "o1", "o1-mini", "o3", "o3-mini", "o4-mini",
]


def main() -> int:
    load_dotenv()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
    auth_header = os.environ.get("AZURE_OPENAI_AUTH_HEADER", "api-key")

    if not endpoint or not key:
        print("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env first.")
        return 2

    candidates = sys.argv[1:] + DEFAULT_CANDIDATES
    # de-dupe, preserve order (user-supplied names probed first)
    seen: set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    headers = {auth_header: key, "Content-Type": "application/json"}
    body = {"messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}

    print(f"Endpoint   : {endpoint}")
    print(f"api-version: {api_version}   auth header: {auth_header}")
    print(f"Probing {len(candidates)} candidate deployment names ...\n")

    valid: list[str] = []
    auth_bad = False
    for name in candidates:
        url = (
            f"{endpoint}/openai/deployments/{name}"
            f"/chat/completions?api-version={api_version}"
        )
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=30.0)
        except httpx.HTTPError as e:
            print(f"  {name:<18} ERROR  {e}")
            continue

        code = r.status_code
        if code == 200:
            print(f"  {name:<18} 200    VALID  <-- usable as --model {name}")
            valid.append(name)
        elif code == 404:
            print(f"  {name:<18} 404    not a deployment here")
        elif code in (401, 403):
            auth_bad = True
            print(f"  {name:<18} {code}    AUTH problem: {r.text[:120]}")
        else:
            print(f"  {name:<18} {code}    {r.text[:120]}")

    print()
    if auth_bad:
        print("Saw 401/403 — the key or auth header is wrong. Fix that first; the 404s")
        print("above are not trustworthy until auth succeeds.")
        return 1
    if valid:
        print("Valid deployments found:", ", ".join(valid))
        print(f"Test one end to end:  .venv/bin/python scripts/test_azure.py {valid[0]}")
        return 0
    print("No common names matched. The deployment is probably custom-named —")
    print("ask your boss/admin for the exact string (Azure portal -> resource -> Deployments).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
