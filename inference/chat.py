"""Interactive terminal chat with a SHINRA checkpoint."""

from __future__ import annotations

import argparse

from .generate import generate_chat, load_generate_stack
from configs.loader import load_config


def main() -> None:
    inference_cfg = load_config()["inference"]
    parser = argparse.ArgumentParser(description="Interactive SHINRA chat")
    parser.add_argument("--model", required=True)
    parser.add_argument("--system", default=None, help="Optional caller supplied system message")
    parser.add_argument("--max-new-tokens", type=int, default=inference_cfg["default_max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=inference_cfg["default_temperature"])
    parser.add_argument("--top-p", type=float, default=inference_cfg["default_top_p"])
    args = parser.parse_args()

    tokenizer, model = load_generate_stack(args.model)
    messages = [{"role": "system", "content": args.system}] if args.system else []

    print("SHINRA chat. Ctrl+C to exit.")
    while True:
        try:
            user = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue

        messages.append({"role": "user", "content": user})
        reply = generate_chat(
            args.model,
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            tokenizer=tokenizer,
            model=model,
        )
        print(f"assistant> {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
