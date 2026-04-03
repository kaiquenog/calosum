import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--changed-files", required=True)
    parser.add_argument("--minimum", type=float, default=80.0)
    args = parser.parse_args()

    try:
        with open(args.coverage_json) as f:
            cov_data = json.load(f)
        with open(args.changed_files) as f:
            changed_files = {line.strip() for line in f if line.strip()}
    except Exception as e:
        print(f"Error reading input files: {e}")
        sys.exit(1)

    files_cov = cov_data.get("files", {})
    failed = False

    for changed_file in changed_files:
        if changed_file in files_cov:
            stats = files_cov[changed_file].get("summary", {})
            percent = stats.get("percent_covered", 0.0)
            if percent < args.minimum:
                print(f"FAIL: {changed_file} has {percent:.2f}% coverage (< {args.minimum}%)")
                failed = True
            else:
                print(f"PASS: {changed_file} has {percent:.2f}% coverage")

    if failed:
        sys.exit(1)
    else:
        print("Coverage gate passed.")

if __name__ == "__main__":
    main()
