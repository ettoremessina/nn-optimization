import matplotlib.pyplot as plt

def scatter_attempt_infos(attempt_infos, x_test, y_test):
    num_of_cols = 6
    num_of_ais = len(attempt_infos)
    num_of_bands = num_of_ais // num_of_cols
    if num_of_ais % num_of_cols != 0:
        num_of_bands += 1
    fig, axs = plt.subplots(num_of_bands, num_of_cols)
    for idx in range(0, num_of_ais):
        ai = attempt_infos[idx]
        i = idx // num_of_cols
        j = idx % num_of_cols
        axs[i, j].set_title(ai.name)
        axs[i, j].scatter(x_test, y_test, color='green', s=2, marker='.')
        axs[i, j].scatter(x_test, ai.y_pred, color='red', s=1, marker='.')
    plt.show()
