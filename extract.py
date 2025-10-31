import csv
import json
from pathlib import Path
import argparse
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
import os


load_dotenv(dotenv_path=".env.dev")


REFERENCE_SCHEMAS = {
"IndexDocs": [
        "DocNumber", "RecordDateTime", "DocumentDate", "Instrument", "Pages", "Volume",
        "Page", "Book", "Consideration", "MortgageAmount", "ExpirationDate", "FileNumber",
        "UCCDesc", "ReturnTo", "ReturnAdd1", "ReturnCity", "ReturnState", "Field18Placeholder", "ReturnZip"
    ],
    "IndexLegals": [
        "DocNumber", "Block", "Lot", "Twp", "Rge", "Sec", "Legal", "Notes", "City",
        "Subdivision", "SurveyName", "AbstractNumber", "Section", "Tract", "Acres"
    ],
    "IndexParcels": ["DocNumber", "ParcelNumber"],
    "IndexParties": [
        "DocNumber", "PartyName", "PartyType", "PartyAddress1", "PartyAddress2",
        "PartyCity", "PartyState", "PartyZip"
    ],
    "IndexRelated": ["DocNumber", "RelatedDocNumber", "RelatedDocID", "Volume", "Page", "Book"],
}


# ---------- Core Data Classes ---------- #

class MetadataFiles:
    """Holds paths for a single date range folder’s metadata files."""
    def __init__(self, data_dir: Path, ref_file: Path, data_file: Path):
        self.data_dir = data_dir
        self.ref_file = ref_file
        self.data_file = data_file

    def __repr__(self):
        return f"<MetadataFiles dir={self.data_dir.name} ref={self.ref_file.name} data={self.data_file.name}>"


class MetadataExtractor:
    """Responsible for discovering metadata files in a given directory tree."""

    def __init__(self, input_path: Path, verbose: bool = False):
        self.input_path = input_path
        self.verbose = verbose

    def extract_metadata_files(self) -> Dict[Path, list[MetadataFiles]]:
        """Scans each date-range folder and returns mapping of folder -> list of MetadataFiles pairs."""
        data_dir_files: Dict[Path, list[MetadataFiles]] = {}

        for data_dir in self.input_path.iterdir():
            if not data_dir.is_dir():
                continue

            if self.verbose:
                print(f"[+] Checking directory: {data_dir}")

            ref_files = list(data_dir.glob("*Reference.txt"))
            data_files = [f for f in data_dir.glob("Index*.txt") if "Reference" not in f.name]

            if not ref_files or not data_files:
                if self.verbose:
                    print(f"[-] Skipping {data_dir} (no valid metadata files)")
                continue

            pairs: list[MetadataFiles] = []
            for ref_file in ref_files:
                # Match Reference.txt with its Index file based on prefix
                base_prefix = ref_file.stem.replace("Reference", "")
                match = next(
                    (
                        df
                        for df in data_files
                        if df.stem.startswith(base_prefix)
                        or base_prefix.startswith(df.stem)
                        or base_prefix.replace("Parcels", "ParcelNumbers") in df.stem
                        or base_prefix.replace("Related", "RelatedDocs") in df.stem
                    ),
                    None
                )
                if match:
                    pairs.append(MetadataFiles(data_dir, ref_file, match))
                    if self.verbose:
                        print(f"[✓] Matched {match.name} ↔ {ref_file.name}")
                else:
                    if self.verbose:
                        print(f"Data files: {data_files} did not match {base_prefix}")
                        print(f"[!] No Index file found for {ref_file.name}")

            if pairs:
                data_dir_files[data_dir] = pairs

        return data_dir_files


class MetadataProcessor:
    """Processes metadata directories and locates associated OPR image folders."""

    def __init__(self, input_path: Path, output_path: Path, verbose: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.verbose = verbose

    def process(self, metadata_map: Dict[Path, list[MetadataFiles]]) -> Dict:
        combined_data = {}

        for data_dir, pairs in metadata_map.items():
            opr_path = data_dir / "OPR"
            if not opr_path.exists():
                if self.verbose:
                    print(f"[-] Skipping {data_dir.name} (no OPR directory found)")
                continue

            if self.verbose:
                print(f"[✓] Found OPR path: {opr_path}")

            # --- NEW AGGREGATION STRUCTURE ---
            # 1. Stores the primary document metadata (IndexDocs, IndexParcels, etc.)
            main_records: Dict[str, Dict] = {}
            # 2. Stores one-to-many records (IndexLegals, IndexParties, etc.)
            legal_map: Dict[str, List[Dict]] = {}
            party_map: Dict[str, List[Dict]] = {}
            # ---------------------------------

            for files in pairs:
                if self.verbose:
                    print(f"[→] Processing {files.data_file.name} for {data_dir.name}")

                field_names = self._parse_reference_file(files.ref_file)
                parsed_records = self._parse_data_file(files.data_file, field_names)
                base_name = files.data_file.stem

                # Aggregate the records based on the file type
                if base_name == "IndexDocs":
                    main_records.update(parsed_records)
                elif base_name == "IndexLegals":
                    for doc_num, record in parsed_records.items():
                        legal_map.setdefault(doc_num, []).append(record)
                elif base_name == "IndexParties":
                    for doc_num, record in parsed_records.items():
                        party_map.setdefault(doc_num, []).append(record)
                else:
                    # For other files (Parcels, Related), merge into the main record
                    for doc_num, record in parsed_records.items():
                        main_records.setdefault(doc_num, {}).update(record)
            
            if self.verbose:
                print(f"[i] Aggregated {len(main_records)} documents for {data_dir.name}")

            # NEW: Call the new mapping function to create the combined Mongo records
            combined_map = self._create_combined_mongo_records(
                opr_path, main_records, legal_map, party_map
            )
            
            # Write a single JSON output for the entire folder
            key = f"{data_dir.name}_Combined"
            combined_data[key] = combined_map
            self._write_json_output(key, combined_map)

        return combined_data

    def _create_combined_mongo_records(self, opr_dir: Path, main_records: Dict[str, Dict], legal_map: Dict[str, List[Dict]], party_map: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        mapping = {}
        # Normalize keys for image matching
        normalized_records = {
            k.strip().strip('"').replace("_", "-").lower(): v for k, v in main_records.items()
        }

        for tif_path in opr_dir.rglob("*.[Tt][Ii][Ff]"):
            doc_number_key = tif_path.stem.strip().replace("_", "-").lower()

            if doc_number_key not in normalized_records:
                if self.verbose:
                    print(f"[x] No primary metadata for image: {tif_path.name}")
                continue

            rec = normalized_records[doc_number_key]
            
            # Lookup non-primary records using the original, un-normalized DocNumber
            # We assume the DocNumber in the aggregation maps is consistently formatted from the TXT files
            original_doc_number = rec.get("DocNumber", doc_number_key.upper()) # Use the DocNumber field if available

            mongo_record = {
                "doc_no": rec.get("DocNumber", original_doc_number),
                "doc_id": rec.get("DocNumber", original_doc_number),
                "doc_type": rec.get("Instrument") or None,
                "doc_date": rec.get("DocumentDate") or None,
                "recording_date": self._convert_date(rec.get("RecordDateTime")),
                "vol": rec.get("Volume") or None,
                "page": rec.get("Page") or None,
                "book": rec.get("Book") or None,
                "num_pages": int(rec.get("Pages") or 0),
                "b2_bucket": os.getenv("HEMPILL_B2_BUCKET"),
                "b2_key": f"tx/hempill/{tif_path.name}",
                "file_name": tif_path.name,
                "downloaded": True,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "last_scraped": datetime.utcnow().isoformat() + "Z",
                "last_downloaded": datetime.utcnow().isoformat() + "Z",
                "grantees": [],
                "grantors": [],
                "legal_descriptions": [],
                "year_folder": opr_dir.parent.name,
                "image_path": str(tif_path.relative_to(self.input_path))
            }

            # --- ENRICHMENT STEP: Add data from one-to-many files ---
            
            # 1. Add Parties (Grantors/Grantees)
            for party_record in party_map.get(original_doc_number, []):
                party_type = party_record.get("PartyType", "").lower()
                party_name = party_record.get("PartyName")
                
                party_data = {
                    "name": party_name,
                    "address1": party_record.get("PartyAddress1"),
                    "city": party_record.get("PartyCity"),
                    "state": party_record.get("PartyState"),
                    # ... add more party fields as needed
                }
                
                if "grantor" in party_type:
                    mongo_record["grantors"].append(party_data)
                elif "grantee" in party_type:
                    mongo_record["grantees"].append(party_data)
                    
            # 2. Add Legal Descriptions
            for legal_record in legal_map.get(original_doc_number, []):
                # Map legal fields to your desired legal_description schema
                mongo_record["legal_descriptions"].append({
                    "block": legal_record.get("Block"),
                    "lot": legal_record.get("Lot"),
                    "subdivision": legal_record.get("Subdivision"),
                    "survey_name": legal_record.get("SurveyName"),
                    "acres": legal_record.get("Acres"),
                    # ... add more legal fields as needed
                })

            mapping[doc_number_key] = mongo_record

        if self.verbose:
            print(f"[i] Mapped {len(mapping)} combined records to Mongo schema")
        return mapping

    def _parse_reference_file(self, ref_path: Path) -> list[str]:
        """Return predefined schema if known; fallback to dynamic parse."""
        base_name = ref_path.stem.replace("Reference", "")
        if base_name in REFERENCE_SCHEMAS:
            fields = REFERENCE_SCHEMAS[base_name]
            if self.verbose:
                print(f"[i] Using predefined schema for {base_name} ({len(fields)} fields)")
            return fields

        # fallback dynamic
        fields = []
        with open(ref_path, encoding="utf-8") as f:
            for line in f:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2:
                    fields.append(parts[1])
        if self.verbose:
            print(f"[i] Parsed {len(fields)} fields dynamically from {ref_path.name}")
        
        print(f"[+] Parsed {len(fields)} fields dynamically from {ref_path.name}")
        return fields

    def _parse_data_file(self, data_path: Path, field_names: list[str]) -> dict[str, dict]:
        """Reads IndexDocs.txt or similar data file and returns dict of DocNumber -> metadata."""
        doc_map = {}
        expected_fields = len(field_names)

        with open(data_path, encoding="utf-8") as f:
            reader = csv.reader(f, skipinitialspace=True)
            for line_num, row in enumerate(reader, start=1):
                if not row or not row[0].strip():
                    continue

                # Ensure we have correct number of fields
                if len(row) != expected_fields:
                    if self.verbose:
                        print(f"[!] Warning: line {line_num} in {data_path.name} has {len(row)} fields (expected {expected_fields})")
                        # Print small preview to debug
                        preview = ', '.join(row[:5])
                        print(f"    → Row preview: {preview}")
                    # Skip malformed line
                    continue

                doc_number = row[0].strip().strip('"').strip("'")
                record = {}
                for i, field_name in enumerate(field_names):
                    value = row[i].strip().strip('"')
                    record[field_name] = value

                doc_map[doc_number] = record

        if self.verbose:
            print(f"[i] Parsed {len(doc_map)} valid records from {data_path.name}")
        return doc_map

    def _map_images_to_metadata(self, opr_dir: Path, records: Dict[str, Dict]) -> Dict[str, Dict]:
        mapping = {}
        normalized_records = {
            k.strip().strip('"').replace("_", "-").lower(): v for k, v in records.items()
        }

        for tif_path in opr_dir.rglob("*.[Tt][Ii][Ff]"):
            doc_number = tif_path.stem.strip().replace("_", "-").lower()
            if doc_number not in normalized_records:
                if self.verbose:
                    print(f"[x] No metadata for image: {tif_path.name}")
                continue

            rec = normalized_records[doc_number]


            mongo_record = {
                "doc_no": rec.get("DocNumber", doc_number),
                "doc_id": rec.get("DocNumber", doc_number),
                "doc_type": rec.get("Instrument") or None,
                "doc_date": rec.get("DocumentDate") or None,
                "recording_date": self._convert_date(rec.get("RecordDateTime")),
                "vol": rec.get("Volume") or None,
                "page": rec.get("Page") or None,
                "book": rec.get("Book") or None,
                "num_pages": int(rec.get("Pages") or 0),
                "b2_bucket": os.getenv("HEMPILL_B2_BUCKET"),
                "b2_key": f"tx/hempill/{tif_path.name}",
                "file_name": tif_path.name,
                "downloaded": True,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "last_scraped": datetime.utcnow().isoformat() + "Z",
                "last_downloaded": datetime.utcnow().isoformat() + "Z",
                "grantees": [],
                "grantors": [],
                "legal_descriptions": [],
                "year_folder": opr_dir.parent.name,
                "image_path": str(tif_path.relative_to(self.input_path))
            }

            mapping[doc_number] = mongo_record

        if self.verbose:
            print(f"[i] Mapped {len(mapping)} / {len(records)} to Mongo schema")
        return mapping

    def _convert_date(self, date_str: str):
        """Convert '1/1/1800' to ISO date string."""
        if not date_str or date_str.strip() in {"", "0"}:
            return None
        try:
            month, day, year = [int(x) for x in date_str.split("/")]
            return datetime(year, month, day).isoformat() + "Z"
        except Exception:
            return None

    def _write_json_output(self, folder_name: str, image_map: Dict[str, Dict]):
        """Writes JSON output for a given date range folder."""
        folder_output = self.output_path / f"{folder_name}.json"
        with open(folder_output, "w", encoding="utf-8") as out:
            json.dump(image_map, out, indent=2)
        if self.verbose:
            print(f"[+] Wrote {len(image_map)} mapped records to {folder_output}")


# ---------- CLI Entry Point ---------- #

class CLI:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Map TexasFile-style metadata to OPR TIFF images.")
        parser.add_argument("--input_path", type=str, required=True, help="Path to root data directory")
        parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON directory")
        parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
        self.args = parser.parse_args()

    def run(self):
        input_path = Path(self.args.input_path)
        output_path = Path(self.args.output_path)
        verbose = self.args.verbose

        if not input_path.exists():
            raise FileNotFoundError(f"Input path {input_path} does not exist")

        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"[+] Created output directory {output_path}")

        extractor = MetadataExtractor(input_path, verbose)
        contained_metadata_files = extractor.extract_metadata_files()

        processor = MetadataProcessor(input_path, output_path, verbose)
        processor.process(contained_metadata_files)


if __name__ == "__main__":
    CLI().run()
