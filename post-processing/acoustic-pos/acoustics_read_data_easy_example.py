import acoustic_local_functions as alf
import matplotlib.pyplot as plt

EXPERIMENT_ID = "EXP008"
CYCLE_ID = 20  # Set the exact acoustic cycle you want to inspect.
MICROPHONE_LABEL = "D06"  # Set the exact microphone label for the acoustic waveform.
DATASET_PATH = None  # Set this to a specific .nc file when you do not want the newest match.
CSI_DATASET_PATH = None  # Optional: provide a specific CSI dataset path if needed.

if __name__ == "__main__":

    ds, dataset_path = alf.open_acoustic_dataset(EXPERIMENT_ID, DATASET_PATH)
    print(f"Loaded dataset: {dataset_path}")
    print(f"Selected experiment: {EXPERIMENT_ID}")

    # Get all microphone labels
    microphone_labels = ds["microphone_label"].values.astype(str).tolist()
    print(f"All microphone labels: {microphone_labels}")
    print(f"Number of microphones: {len(microphone_labels)}")

    shape = alf.get_acoustic_dataset_shape(EXPERIMENT_ID, DATASET_PATH)
    print(f"Dataset shape: {shape}")

    cycle_ids = ds["cycle_id"].values.astype(int)
    if cycle_ids.size:
        print(f"Cycle ID range: {int(cycle_ids.min())} .. {int(cycle_ids.max())}")
    ds.close()

    position = alf.get_rover_position(EXPERIMENT_ID, CYCLE_ID, CSI_DATASET_PATH)
    if position["position_available"]:
        print(
            f"Rover position for cycle {CYCLE_ID}: "
            f"x={position['rover_x']:.2f}, y={position['rover_y']:.2f}, z={position['rover_z']:.2f}"
        )
    else:
        print(f"No rover position available for cycle {CYCLE_ID}")

    microphone_positions = alf.load_microphone_positions()
    mic_pos = microphone_positions.get(MICROPHONE_LABEL.upper())
    if mic_pos is not None:
        print(
            f"Microphone {MICROPHONE_LABEL} position: "
            f"x={mic_pos[0]:.2f}, y={mic_pos[1]:.2f}, z={mic_pos[2]:.2f}"
        )
    else:
        print(f"Position not found for microphone {MICROPHONE_LABEL}")

    waveform, waveform_values, sample_index = alf.get_acoustic_waveform(
        EXPERIMENT_ID,
        CYCLE_ID,
        MICROPHONE_LABEL,
        DATASET_PATH,
    )
    print(f"Waveform dims: {dict(waveform.sizes)}")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sample_index, waveform_values, color="navy", linewidth=1.2)
    ax.set_title(
        f"Acoustic waveform for {EXPERIMENT_ID} / cycle {CYCLE_ID} / mic {MICROPHONE_LABEL}"
    )
    ax.set_xlabel("sample_index")
    ax.set_ylabel("values")
    ax.grid(True, alpha=0.25)
    plt.show()
