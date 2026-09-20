"""Interactive terminal chat with a SHINRA checkpoint."""

from __future__ import annotations

import argparse

from .generate import generate_chat, load_generate_stack


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive SHINRA chat")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--system",
        default="You are SHINRA, a language model developed by NULLXES.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    tokenizer, model = load_generate_stack(args.model, dtype=args.dtype)
    messages = [{"role": "system", "content": args.system}]

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
