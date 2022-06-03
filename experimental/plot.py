import matplotlib.pyplot as plt


def plot_dict_as_pie(data):
    """
    Plots a pie chart from a dictionary.
    :param data: the data in a dictionary of labels to integers
    """
    plt.pie(data.values(), labels=data.keys())
    plt.show()


def plot_dict_as_bar(data, x_label, y_label):
    """
    Plots a bar graph from a dictionary.
    :param data: the data in a dictionary of labels to integers
    :param x_label: the label on the x-axis
    :param y_label: the label on the y-axis
    """
    plt.bar(list(data.keys()), list(data.values()))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
