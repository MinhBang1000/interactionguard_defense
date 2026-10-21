"""Compatibility entry point for the explicit local Llama prompt-success judge."""
from eval.prompt_success_judge import main


def prompt_injection_eval():
    return main([])


if __name__ == "__main__":
    main()
