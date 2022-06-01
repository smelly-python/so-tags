from functools import cmp_to_key


def compare_lists_by_element(index):
    def compare(xs, ys):
        x = xs[index]
        y = ys[index]
        if x < y:
            return -1
        elif y < x:
            return 1
        else:
            return 0

    return cmp_to_key(compare)


def increase_dict_count(d, key):
    if key in d:
        d[key] += 1
    else:
        d[key] = 1
