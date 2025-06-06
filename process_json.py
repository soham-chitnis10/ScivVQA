import pandas as pd
import argparse

def main(args):
    df = pd.read_json(args.json_file)
    df.drop(df[df['qa_pair_type']=="unanswerable"].index, inplace=True)
    df.to_json(args.output_path, orient="records")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True, help="Path to json file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output json file")

    args = parser.parse_args()
    main(args)