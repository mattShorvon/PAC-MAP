from pathlib import Path
from spn.io.file import from_file

ROOT = Path("20-datasets")
SUFFIXES = (
    "_0.1q_0.9e.map",
    "_0.25q_0.75e.map",
    "_0.5q_0.5e.map",
)

def load_spn_and_num_features(dataset_dir: Path):
    """Load SPN for dataset and return (spn, num_features)."""
    dataset = dataset_dir.name
    candidates = [
        dataset_dir / f"{dataset}.spn",
        dataset_dir / f"{dataset}_em.spn",
    ]
    for spn_path in candidates:
        if spn_path.exists():
            spn = from_file(spn_path)
            num_features = len(spn.scope())
            return spn, num_features
    print(f"  WARNING: no SPN found for {dataset_dir.name}, skipping")
    return None, None

def analyze_map_file(map_path: Path, num_features: int):
    with map_path.open() as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) % 2 != 0:
        print(f"    WARNING: {map_path.name} has odd number of lines ({len(lines)}), skipping")
        return

    num_pairs = len(lines) // 2
    full_cover_indices = []   # indices where |Q| + |E| == num_features
    marg_indices = []         # indices where some vars are marginalized (|Q|+|E| < num_features)

    for i in range(num_pairs):
        q_line = lines[2 * i]
        e_line = lines[2 * i + 1]

        # .map format from make_queries_and_evids.py:
        #   q_line: "q_id q_id ..."
        #   e_line: "var val var val ..."
        q_ids = q_line.split()
        e_tokens = e_line.split()
        e_ids = e_tokens[::2]  # variable indices at even positions

        total_covered = len(q_ids) + len(e_ids)

        if total_covered == num_features:
            full_cover_indices.append(i)
        else:
            marg_indices.append((i, total_covered))

    print(f"  {map_path.name}:")
    print(f"    total pairs        : {num_pairs}")
    print(f"    no-marg pairs      : {len(full_cover_indices)}")
    print(f"    with-marg pairs    : {len(marg_indices)}")
    if marg_indices:
        # Show first few marginal pairs
        preview = ", ".join(
            f"{idx} (covers {covered})" for idx, covered in marg_indices[:10]
        )
        print(f"    example marg pairs : {preview}")

def main():
    for dataset_dir in ROOT.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        print(f"\nDataset: {dataset}")

        spn, num_features = load_spn_and_num_features(dataset_dir)
        if spn is None:
            continue
        print(f"  num_features (from SPN.scope) = {num_features}")

        for map_path in dataset_dir.glob("*.map"):
            if not map_path.name.endswith(SUFFIXES):
                continue
            analyze_map_file(map_path, num_features)

if __name__ == "__main__":
    main()