import glob as g

def glob(path):
    return [fname.replace('\\', '/') for fname in g.glob(path)]