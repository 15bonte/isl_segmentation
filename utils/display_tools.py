import sys
from matplotlib import pyplot as plt
import numpy as np
import psutil


TITLE_FONT_SIZE = 25
LABELS_AXES_FONT_SIZE = 20
PLOT_FONT_SIZE = 20

# Set general font size
plt.rcParams["font.size"] = str(PLOT_FONT_SIZE)


def display_progress(
    message, current, total, precision=1, additional_message="", cpu_memory=False
):
    percentage = round(current / total * 100, precision)
    padded_percentage = str(percentage).ljust(precision + 3, "0")
    display_message = f"\r{message}: {padded_percentage}%"
    # Display additional message
    if additional_message:
        display_message += " | " + additional_message
    # Display CPU memory usage
    if cpu_memory:
        cpu_available = round(
            psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
        )
        cpu_message = f"CPU available: {cpu_available}%"
        display_message += " | " + cpu_message
    sys.stdout.write(display_message)
    sys.stdout.flush()


def display_accuracy(coefficients, directory, loss_type, col_names):
    # Create plot
    plt.figure(figsize=(10, 7))

    coefficients_np = np.array(coefficients)

    mean = np.mean(coefficients_np[0])
    std = np.std(coefficients_np[0])

    accuracy_message = f"Average {loss_type.name}: {round(mean, 2)} +/- {round(std, 2)}"
    print("\n" + accuracy_message)

    plt.boxplot(coefficients)
    if len(coefficients) == 1:
        plt.title(accuracy_message, fontsize=TITLE_FONT_SIZE, fontweight="bold")
    else:
        print(coefficients[0])
        print(coefficients[1])
    plt.xticks(
        range(1, len(col_names) + 1), col_names, fontsize=LABELS_AXES_FONT_SIZE, fontweight="bold"
    )
    plt.ylabel(loss_type.name, fontsize=LABELS_AXES_FONT_SIZE, fontweight="bold")

    # Save plot
    if directory is not None:
        plt.savefig(f"{directory}/box_plot.png")
