def filter_files(path, ext):
    import os

    files = os.listdir(path)

    result = []

    for file in files:
        if file.endswith(ext):
            result.append(file)

    return result
