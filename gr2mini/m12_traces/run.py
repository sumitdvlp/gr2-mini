"""Try it:  python -m gr2mini.m12_traces.run --config configs/tiny.yaml --mode rejection

Needs a running teacher server (README table). Writes <artifacts_dir>/traces_<mode>.jsonl.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reasoning traces with a teacher LLM.")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--mode", choices=["targeted", "rejection"], required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    # TODO(m12):
    #   1. cfg; items; index; examples = load_jsonl(candidates_train.jsonl)[: traces.num_examples]
    #   2. api_key = os.environ[cfg["traces"]["api_key_env"]] if that's set, else None
    #      client = OpenAICompatClient(base_url, model, api_key, timeout)
    #   3. Smoke-test the server first: client.chat([{"role": "user", "content": "Say OK."}], max_tokens=10)
    #      and print a helpful message if the connection fails (is Ollama running?)
    #   4. stats = generate_traces(...); print stats
    #   5. Print one accepted trace's explanation, and the counts of each rejection reason in the file.
    raise NotImplementedError("TODO(m12): implement run.main")


if __name__ == "__main__":
    main()
