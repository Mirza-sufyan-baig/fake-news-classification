import os


def get_next_model_version(model_dir="models", prefix="baseline"):

    os.makedirs(model_dir, exist_ok=True)

    existing = [f for f in os.listdir(model_dir) if f.startswith(prefix)]

    if not existing:
        return f"{prefix}_v1"

    versions = []

    for name in existing:
        try:
            v = int(name.split("_v")[-1].split(".")[0])
            versions.append(v)
        except:
            pass

    next_version = max(versions) + 1

    return f"{prefix}_v{next_version}"