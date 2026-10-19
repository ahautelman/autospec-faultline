"""Run the five FaultLine fault injections.

This does NOT train PPO or run AutoSpec.

It only verifies that:

1. each correct specification can be constructed;
2. the intended fault is injected;
3. both graphs remain valid DAGs;
4. the resulting structural change is what was expected.

Run from project root:

    python scripts/run_fault_injection.py

or:

    uv run python scripts/run_fault_injection.py
"""

from faultline.faults import (
    all_fault_cases,
    format_summary,
    summarize_fault,
)


def main() -> None:

    print(
        "FaultLine controlled fault-injection check\n"
    )

    for benchmark in all_fault_cases():

        case = benchmark.fault

        # Fail immediately if an injection accidentally
        # created an invalid graph.
        case.correct_graph.assert_acyclic()
        case.faulty_graph.assert_acyclic()

        summary = summarize_fault(
            case
        )

        print(
            format_summary(summary)
        )

        print()

    print(
        "All five injected specifications "
        "were built successfully."
    )


if __name__ == "__main__":
    main()