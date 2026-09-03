"""Verify the configured LLM provider, and list the models it will actually accept.

Model names change. Rather than trusting a default baked into the code, this asks the
provider what it serves and flags whether the configured model is among them.

    python scripts/check_llm.py            # check what is configured
    python scripts/check_llm.py --models   # also list every model on offer
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    for env_file in (".env", "config/.env"):
        if os.path.exists(env_file):
            load_dotenv(env_file)
except ImportError:
    pass

from src.ai.blind_sentiment import PROVIDERS, resolve_llm_provider


def fetch_models(base_url, api_key):
    req = urllib.request.Request(f"{base_url}/models",
                                 headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="list every available model")
    args = ap.parse_args()

    print("keys present:")
    for name, cfg in PROVIDERS.items():
        print(f"  {cfg['key_env']:<20} {'yes' if os.getenv(cfg['key_env']) else 'no'}")

    provider, base_url, api_key, model = resolve_llm_provider()
    if not provider:
        print("\nNo provider configured. Set GROQ_API_KEY or OPENROUTER_API_KEY.")
        return 1

    print(f"\nactive provider : {provider}")
    print(f"base url        : {base_url}")
    print(f"model           : {model}")
    for override in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL"):
        if os.getenv(override):
            print(f"  (overridden by {override}={os.getenv(override)})")

    try:
        models = fetch_models(base_url, api_key)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        print(f"\nHTTP {e.code} listing models: {body}")
        if e.code in (401, 403):
            print("-> the key is not valid for this provider.")
        return 1
    except Exception as e:
        print(f"\nCould not reach {base_url}: {type(e).__name__}: {e}")
        return 1

    print(f"\n{len(models)} model(s) available.")
    if model in models:
        print(f"OK: '{model}' is served by {provider}.")
    else:
        print(f"WARNING: '{model}' is NOT in the list. Set LLM_MODEL to one of these.")
        args.models = True

    if args.models:
        print()
        for m in sorted(models):
            print(f"  {m}{'   <- configured' if m == model else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
