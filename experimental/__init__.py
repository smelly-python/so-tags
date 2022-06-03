from experimental.analyse import get_data, get_tag_counts, get_tags_amounts
from experimental.plot import plot_dict_as_pie, plot_dict_as_bar
from experimental.tabulate_data import tabulate_dict

from util import compare_lists_by_element

if __name__ == '__main__':
    data = get_data("../data/train.tsv")

    # tabulate the tags from most to least common
    tabulate_dict(
        get_tag_counts(data.values()),
        compare_lists_by_element(1),
        True,
        "tag",
        "count"
    )

    # plot in a bar the often-ness of tag amounts
    plot_dict_as_bar(
        get_tags_amounts(data),
        "#tags",
        "#titles with this #tags"
    )
