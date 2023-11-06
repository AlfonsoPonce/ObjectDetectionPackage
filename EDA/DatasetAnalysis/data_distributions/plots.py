'''
This module serves plots distributions in number of classes and relative object sizes.

Date: 25/10/2023
Author: Alfonso Ponce Navarro
'''
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histogram(ocurrence_list: list, file_name: str) -> None:
    '''
    Plots histogram of classes given a list of class appearances.
    :param file_name (str): File name
    :param ocurrence_list (list): Class appearances in the dataset (e.g.: ocurrence = [C1, C1, C3, C2, ...]
    :return: None
    '''

    try:
        sns.histplot(data=ocurrence_list).figure.savefig(file_name)
        plt.close()
    except Exception as error:
        raise error

def plot_pie(distribution_dict: dict) -> None:
    '''
    Plots pie chart of a given distribution dictionary
    :param distribution_dict (dict): distribution of classes
    :return: None
    '''
    data = list(distribution_dict.values())
    labels = list(distribution_dict.keys())
    try:
        plt.title("General Size Distribution")
        plt.pie(data, labels=labels,  autopct='%.0f%%')
        plt.savefig("./Piechart")
    except Exception as error:
        raise error

