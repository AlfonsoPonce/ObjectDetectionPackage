from scipy.stats import ttest_ind
import logging

def pairwise_ttest(result_list_first_model: list, result_list_second_model: list,significance_level: float) -> bool:
    '''
    Executes the pairwise ttest for model comparison at a given significance level.

    :param result_list_first_model: A list of model results along a comparison method.
    :param result_list_second_model: A list of model results along a comparison method.
    :param significance_level: Significance level to execute the hypothesis test.
    :return: True if Null hypothesis is rejected, False if Null Hypothesis is not rejected.
    '''
    statistic, p_value = ttest_ind(result_list_first_model, result_list_second_model)

    logging.info(f"Statistic value: {statistic}")
    logging.info(f"P-Value: {p_value}")

    if p_value < significance_level:
        logging.info("Null hypothesis rejected. There is a significant difference between the models.")
        result = True
    else:
        logging.info("Null hypothesis can not be rejected. There is not a significant difference between the models.")
        result = False

    return result