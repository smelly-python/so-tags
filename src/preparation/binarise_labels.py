from sklearn.preprocessing import MultiLabelBinarizer


def get_tags_count(y):
    # Dictionary of all tags from train corpus with their counts.
    tags_counts = {}

    for tags in y:
        for tag in tags:
            if tag in tags_counts:
                tags_counts[tag] += 1
            else:
                tags_counts[tag] = 1

    return tags_counts


def binarise(y_train, y_val):
    mlb = MultiLabelBinarizer(classes=sorted(get_tags_count(y_train).keys()))
    return mlb.fit_transform(y_train), mlb.fit_transform(y_val)
