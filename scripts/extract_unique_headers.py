from scripts.headers_utils import extract_unique_headers_from_file, save_headers


def main():
    input_path = "data/train.tables.jsonl"
    output_path = "unique_headers.txt"

    headers = extract_unique_headers_from_file(input_path)
    save_headers(headers, output_path)

    print(f"Saved {len(headers)} unique headers to {output_path}")


if __name__ == "__main__":
    main()
