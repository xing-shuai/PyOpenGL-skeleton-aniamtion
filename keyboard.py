keys = {
    "escape": False,
    "w": False,
    "s": False,
    "a": False,
    "d": False
}

def keys_down(key, x, y):
    if ord(key) == 27:
        keys["escape"] = True

    if ord(key) == 119:
        keys["w"] = True

    if ord(key) == 115:
        keys["s"] = True

    if ord(key) == 97:
        keys["a"] = True

    if ord(key) == 100:
        keys["d"] = True


def keys_up(key, x, y):
    if ord(key) == 27:
        keys["escape"] = False

    if ord(key) == 119:
        keys["w"] = False

    if ord(key) == 115:
        keys["s"] = False

    if ord(key) == 97:
        keys["a"] = False

    if ord(key) == 100:
        keys["d"] = False

