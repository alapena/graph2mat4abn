from pathlib import Path


def train_crystalls_with_some_amorph(paths):
    # In this case, we want to train with only crystals, including also some amorphous. Thus, we need to 1. exclude B-B and N-N bonds and 2. exclude 64 atms.
    # First one is difficult. Second is easy.

    # Exclude 64 atm:
    # // x_atoms_paths = [Path('/'.join(path.parts[:-1])) for path in paths]
    train_paths = [path for path in paths if int(Path(path.parts[1]).stem.split('_')[2]) != 64] # Define here the amorphous structures
    val_paths = [path for path in paths if path not in train_paths]

    
    # Now include some of them
    count_64 = 0
    count_8 = 0
    for path in paths:
        if int(Path(path.parts[1]).stem.split('_')[2]) == 64 and count_64 < 3:
            train_paths.append(path)
            val_paths.remove(path)
            count_64 += 1

        # Also move some crystalls to validation:
        elif int(Path(path.parts[1]).stem.split('_')[2]) == 8 and count_8 < 3:
            train_paths.remove(path)
            val_paths.append(path)
            count_8 += 1

        if count_64 >= 3 and count_8 >= 3:
            break