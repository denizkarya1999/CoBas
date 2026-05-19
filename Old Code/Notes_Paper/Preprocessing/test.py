from glob import glob
import collections

label_map = {
    "control": 0,
    "0p": 1,
    "50p": 2,
    "100p": 3
}

def get_sample_labels(file_paths: list[str]) -> list[int]:
    labels = []

    for path in file_paths:
        if "_100p_" in path:
            labels.append(label_map["100p"])
        elif "_50p_" in path:
            labels.append(label_map["50p"])
        elif "_0p_" in path:
            labels.append(label_map["0p"])
        elif "_control_" in path:
            labels.append(label_map["control"])
        else:
            raise ValueError(f"[ERROR]: Could not infer label from path: {path}")

    return labels


file_paths = sorted(glob("/Users/pedropaiva/Documents/Dev/Research/CoBasE-Energy/cobas/Acoustic_Dataset_Collection/BeaconDataset/official00/test/test_ffts/*"))

# for file in file_paths:
#     print(file)

labels = get_sample_labels(file_paths)

print(collections.Counter(labels))
print()
print(sorted(labels))