import os


def get_next_model_version(model_dir="models", prefix="baseline"):

    os.makedirs(model_dir, exist_ok=True)

    existing = [f for f in os.listdir(model_dir) if f.startswith(prefix)]

    if not existing:
        return f"{prefix}_v1"

# Inside get_next_model_version()
    # We only want files that are JUST the version number, not the vectorizers
    versions = [
        int(f.split('.')[0]) 
        for f in os.listdir("models") 
        if f.endswith('.pkl') and f[0].isdigit() and "vectorizer" not in f
    ]

    for name in existing:
        try:
            v = int(name.split("_v")[-1].split(".")[0])
            versions.append(v)
        except:
            pass

    next_version = max(versions) + 1

    return f"{prefix}_v{next_version}"