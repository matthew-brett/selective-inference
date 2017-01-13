import numpy as np
import matplotlib.pyplot as plt
from selection.approx_ci.tests.test_glm import test_glm


def compute_coverage():
    n, p, s = 500, 50, 5
    snr_grid = np.linspace(0, 10, num=4)
    snr_grid_length = snr_grid.shape[0]
    niter = 200
    lasso_selective_xaxis = []
    lasso_naive_xaxis = []
    logistic_selective_xaxis = []
    logistic_naive_xaxis = []

    for i in range(snr_grid_length):
        lasso_selective = []
        lasso_naive = []
        logistic_selective = []
        logistic_naive = []
        for _ in range(niter):
            result_lasso = test_glm(n=n, p=p, s=s, snr = snr_grid[i], loss='gaussian')
            #result_logistic = test_glm(n=n, p=p, s=s, snr = snr_grid[i], loss='logistic')
            if result_lasso is not None:
                covered, ci_length, pivots, naive_covered, naive_pvals = result_lasso[1]
                lasso_selective.append(np.true_divide(covered.sum(), covered.shape[0]))
                lasso_naive.append(np.true_divide(naive_covered.sum(), naive_covered.shape[0]))

            #if result_logistic is not None:
            #    covered, ci_length, pivots, naive_covered, naive_pvals = result_logistic[1]
            #    logistic_selective.append(np.true_divide(covered.sum(), covered.shape[0]))
            #    logistic_naive.append(np.true_divide(naive_covered.sum(), naive_covered.shape[0]))

        lasso_selective_xaxis.append(np.mean(lasso_selective))
        lasso_naive_xaxis.append(np.mean(lasso_naive))
        #logistic_selective_xaxis.append(np.mean(logistic_selective))
        #logistic_naive_xaxis.append(np.mean(logistic_selective))

    return snr_grid, lasso_selective_xaxis, lasso_naive_xaxis, logistic_selective_xaxis, logistic_naive_xaxis



if __name__=='__main__':
    np.random.seed(500)

    snr_grid, lasso_selective_xaxis, lasso_naive_xaxis, logistic_selective_xaxis, logistic_naive_xaxis = compute_coverage()

    plt.plot(snr_grid, lasso_selective_xaxis, label = 'Lasso selective')
    plt.plot(snr_grid, lasso_naive_xaxis, label = "Lasso naive")
    #plt.plot(snr_grid, logistic_selective_xaxis, label = 'Logistic selective')
    #plt.plot(snr_grid, logistic_naive_xaxis, label = "Logistic naive")
    plt.legend(loc='lower right')
    plt.xlabel("coverage")
    plt.ylabel("snr")
    print(lasso_selective_xaxis)
    print(lasso_naive_xaxis)
    #print(logistic_selective_xaxis)
    #print(logistic_naive_xaxis)
    plt.savefig('lasso_coverage.pdf')




