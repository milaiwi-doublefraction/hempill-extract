import csv, json
from pathlib import Path
import argparse
from typing import List, Dict


def extract_metadata_files(input_path: Path, verbose: bool) -> List[Path]:
    data_dir_files = {}

    for data_dir in input_path.iterdir():
        if verbose:
            print(f"Data_dir is {data_dir}")

        if not data_dir.is_dir():
            continue

        ref_files = list(data_dir.glob("*Reference.txt"))
        data_files = list(data_dir.glob("Index*.txt"))

        if len(ref_files) != 1 or len(data_files) != 1:
            if verbose:
                print(f"Skipping {data_dir} because it does not contain exactly one reference and one data file")
            continue

        data_dir_files[data_dir] = {
            "ref_file": ref_files[0],
            "data_file": data_files[0]
        }
    
    return data_dir_files


def process_operation_data(, verbose: bool) -> Dict:
    
    for data_dir, files in data_dir_files.items():
        ref_file = files["ref_file"]
        data_file = files["data_file"]

        opr_path = data_dir / "OPR.txt" # "../hempill/1951-1970/OPR"

        if not opr_path.exists():
            if verbose:
                print(f"Skipping {data_dir} because it does not contain an OPR file")
            continue

        print(f"OPR path is {opr_path}")

    return {}



def main(input_path, output_path, verbose):
    contained_metadata_files = extract_metadata_files(input_path, verbose)
    process_operation_data(contained_metadata_files, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("-v", "--verbose", required=False, default=False, action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    verbose = args.verbose if args.verbose else False

    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist")

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Created output path {output_path}")
            
    main(input_path, output_path, verbose)
