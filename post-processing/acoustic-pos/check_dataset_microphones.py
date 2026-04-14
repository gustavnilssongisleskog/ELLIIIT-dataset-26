import acoustic_local_functions as alf
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.json"
EXPERIMENT_ID = "EXP008"
DATASET_PATH = None


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        expected_mics = sorted(str(x) for x in json.load(f)["all_mics"])

    ds, dataset_path = alf.open_acoustic_dataset(EXPERIMENT_ID, DATASET_PATH)
    try:
        available_mics = sorted(str(x) for x in ds["microphone_label"].values)
    finally:
        ds.close()

    missing_mics = sorted(set(expected_mics) - set(available_mics))
    extra_mics = sorted(set(available_mics) - set(expected_mics))

    print(f"Dataset: {dataset_path}")
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Expected microphones: {len(expected_mics)}")
    print(f"Available microphones: {len(available_mics)}")
    print(f"Missing microphones: {len(missing_mics)}")

    print("\nMissing microphone IDs:")
    print(", ".join(missing_mics) if missing_mics else "None")

    print("\nAvailable microphone IDs:")
    print(", ".join(available_mics))

    if extra_mics:
        print("\nExtra IDs in dataset (not in config):")
        print(", ".join(extra_mics))


if __name__ == "__main__":
    main()
