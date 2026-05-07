import pandas as pd


def remove_empty_questions(input_path: str, output_path: str) -> int:
    """Remove rows with empty question from a CSV and save to output."""
    df = pd.read_csv(input_path)
    n_before = len(df)
    df = df[df["question"].notna() & df["question"].str.strip().ne("")]
    n_removed = n_before - len(df)
    df.to_csv(output_path, index=False)
    return n_removed


if __name__ == "__main__":
    base = "/home/yosef/ws/thesis/data/coordinate-other"

    n1 = remove_empty_questions(f"{base}/test.csv", f"{base}/test.csv")
    print(f"test.csv: removed {n1} rows with empty question")

    n2 = remove_empty_questions(f"{base}/test_infer.csv", f"{base}/test_infer.csv")
    print(f"test_infer.csv: removed {n2} rows with empty question")
