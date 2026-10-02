"""Initialize or verify the identity of a dedicated runtime-state directory."""

import argparse
import json
import os

from src.runtime_state import (
    RuntimeStateError,
    initialize_runtime_state,
    inspect_runtime_state,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser(
        "initialize",
        help="Create the runtime identity once and print the state id for secret storage",
    )
    initialize.add_argument("--state-dir", required=True)
    verify = commands.add_parser(
        "verify",
        help="Verify the mounted state against the expected identity",
    )
    verify.add_argument("--state-dir", required=True)
    verify.add_argument(
        "--expected-state-id",
        default=os.getenv("DATA_PRISM_EXPECTED_STATE_ID"),
    )
    args = parser.parse_args()
    try:
        if args.command == "initialize":
            result = initialize_runtime_state(args.state_dir)
            result = {"status": "ok", **result}
        else:
            result = inspect_runtime_state(args.state_dir, args.expected_state_id)
            if not result["verified"]:
                parser.exit(2, json.dumps(result, sort_keys=True) + "\n")
            result = {"status": "ok", **result}
    except RuntimeStateError as error:
        parser.exit(2, f"{error}\n")
    except OSError:
        parser.exit(2, "Runtime state identity operation failed. Check the path and permissions.\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
