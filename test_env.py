import os
import argparse

val = os.getenv("INTERVALS_COUNT", "7")
print(f"Env INTERVALS_COUNT: {val}")

parser = argparse.ArgumentParser()
parser.add_argument("--intervals_count", type=int, default=int(os.getenv("INTERVALS_COUNT", "7")))
args = parser.parse_args()
print(f"Args intervals_count: {args.intervals_count}")
