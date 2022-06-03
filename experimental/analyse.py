from ast import literal_eval
import pandas as pd
from util import increase_dict_count


def read_data(filename):
    """
    Read the data.
    """
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def get_data(filename):
    data = read_data(filename)
    titles, tags = data['title'].values, data['tags'].values
    return {title: tag for (title, tag) in zip(titles, tags)}


def get_tag_counts(tags_list):
    result = {}
    for tags in tags_list:
        for tag in tags:
            increase_dict_count(result, tag)
    return result


def get_tags_amounts(data):
    result = {}
    for key in data:
        increase_dict_count(result, len(data[key]))
    return result
