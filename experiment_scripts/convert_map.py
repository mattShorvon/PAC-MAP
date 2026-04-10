from pathlib import Path
from spn.io.file import from_file

ROOT = Path("20-datasets")
SUFFIXES = (
    "_0.1q_0.9e.map",
    "_0.25q_0.75e.map",
    "_0.5q_0.5e.map",
)

def load_spn_and_num_features(dataset_dir: Path):
    dataset = dataset_dir.name
    candidates = [
        dataset_dir / f"{dataset}.spn",
        dataset_dir / f"{dataset}_em.spn",
    ]
    for spn_path in candidates:
        if spn_path.exists():
            spn = from_file(spn_path)
            return spn, len(spn.scope())
    print(f"  WARNING: no SPN found for {dataset_dir.name}, skipping")
    return None, None

def convert_map(map_path: Path, num_features: int):
    with map_path.open() as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) % 2 != 0:
        print(f"  WARNING: {map_path.name} has odd number of lines ({len(lines)}), skipping")
        return

    num_pairs = len(lines) // 2
    keep_q_lines = []
    keep_e_lines = []

    for i in range(num_pairs):
        q_line = lines[2 * i]
        e_line = lines[2 * i + 1]

        # .map format:
        #   q_line: "q_id q_id ..."
        #   e_line: "var val var val ..."
        q_ids = [int(x) for x in q_line.split()]
        e_tokens = e_line.split()
        e_ids = [int(e_tokens[j]) for j in range(0, len(e_tokens), 2)]

        total_covered = len(q_ids) + len(e_ids)
        if total_covered != num_features:
            continue  # this pair has marginalised variables, skip

        # Build .query line: "k q1 q2 ... qk"
        q_out = f"{len(q_ids)} " + " ".join(str(v) for v in q_ids)

        # Build .evid line: "k v1 val1 v2 val2 ..."
        k = len(e_ids)
        e_out = f"{k} " + " ".join(e_tokens)

        keep_q_lines.append(q_out)
        keep_e_lines.append(e_out)

    print(f"  {map_path.name}:")
    print(f"    total pairs     : {num_pairs}")
    print(f"    no-marg pairs   : {len(keep_q_lines)}")

    if not keep_q_lines:
        return

    dataset_dir = map_path.parent
    dataset = dataset_dir.name
    base = map_path.name.replace(".map", "")  # e.g. nltcs_0.1q_0.9e
    out_q = dataset_dir / f"{base}_nomarg.query"
    out_e = dataset_dir / f"{base}_nomarg.evid"

    out_q.write_text("\n".join(keep_q_lines))
    out_e.write_text("\n".join(keep_e_lines))
    print(f"    wrote: {out_q.name}, {out_e.name}")

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
            convert_map(map_path, num_features)

if __name__ == "__main__":
    main()