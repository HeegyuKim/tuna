import os
import jsonlines


def estimate_skip_length(output_path: str):
    if os.path.exists(output_path):
        with jsonlines.open(output_path, "r") as f:
            skip_length = len(list(f))
    else:
        skip_length = 0

    return skip_length