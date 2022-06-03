from tabulate import tabulate


def tabulate_dict(data, comparator, reverse, key_header="key", value_header="value"):
    extracted = [[key, value] for key, value in data.items()]

    print(tabulate(
        sorted(extracted, key=comparator, reverse=reverse),
        headers=[key_header, value_header]
    ))
