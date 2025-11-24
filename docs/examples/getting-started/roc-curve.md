# Receiver Operating Characteristic (ROC) Curve Plots

Receiver Operating Characteristic (ROC) curves are a fundamental tool for evaluating the performance of binary classification models. They illustrate the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The curve plots the Probability of Detection (POD, also known as True Positive Rate or Sensitivity) against the Probability of False Detection (POFD, also known as False Positive Rate or 1 - Specificity) at various threshold settings. The Area Under the Curve (AUC) provides a single scalar measure of overall performance.

## Prerequisites

-   Basic Python knowledge
-   Understanding of numpy and pandas
-   MONET Plots installed (`pip install monet_plots`)
-   Matplotlib installed (`pip install matplotlib`)

## Plotting Workflow

To create an ROC curve with `monet_plots`:

1.  **Prepare Data**: Your data should be a `pandas.DataFrame` containing columns for Probability of False Detection (POFD) and Probability of Detection (POD) for various classification thresholds.
2.  **Initialize `ROCCurvePlot`**: Create an instance of the `ROCCurvePlot` class.
3.  **Call `plot` method**: Pass your data, specifying the `x_col` for POFD and `y_col` for POD.
4.  **Customize (Optional)**: Add titles, labels, or other visual enhancements.
5.  **Display/Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1: Basic ROC Curve for a Single Model

Let's create a basic ROC curve using synthetic POD and POFD values for a single binary classification model.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.roc_curve import ROCCurvePlot

# 1. Prepare sample data
np.random.seed(42) # for reproducibility

# Simulate POD and POFD values for different thresholds
# A good model will have high POD for low POFD
thresholds = np.linspace(0, 1, 50)
pod = 0.5 * (1 + np.tanh(5 * (thresholds - 0.5))) + np.random.normal(0, 0.05, 50)
pofd = thresholds + np.random.normal(0, 0.03, 50)

# Ensure values are within [0, 1] and sorted for plotting
pod = np.clip(pod, 0, 1)
pofd = np.clip(pofd, 0, 1)

# Sort by POFD to ensure correct curve plotting
df = pd.DataFrame({'pofd': pofd, 'pod': pod}).sort_values(by='pofd').reset_index(drop=True)

# 2. Initialize and create the plot
plot = ROCCurvePlot(figsize=(8, 8))
plot.plot(
    df,
    x_col='pofd',
    y_col='pod',
    color='blue',
    linewidth=2
)

# 3. Add titles and labels
plot.ax.set_title("Receiver Operating Characteristic (ROC) Curve")
# Legend includes AUC automatically if show_auc=True (default)

plt.tight_layout()
plt.show()
```

### Expected Output

A square plot will be displayed with "Probability of False Detection (POFD)" on the x-axis and "Probability of Detection (POD)" on the y-axis, both ranging from 0 to 1. A dashed black line will represent the "No Skill" diagonal. A blue curve will represent the ROC curve of the model, starting from (0,0) and ending at (1,1). The legend will include the calculated Area Under the Curve (AUC) for the model. The area under the curve will also be lightly shaded.

### Key Concepts

-   **`ROCCurvePlot`**: The class used to generate ROC curves.
-   **`x_col='pofd'`**: Specifies the column containing the Probability of False Detection values.
-   **`y_col='pod'`**: Specifies the column containing the Probability of Detection values.
-   **No Skill Line**: The diagonal line from (0,0) to (1,1), representing a classifier that performs no better than random chance.
-   **Area Under the Curve (AUC)**: A scalar value (0 to 1) that summarizes the overall performance of the classifier across all possible thresholds. An AUC of 0.5 indicates no skill, while an AUC of 1.0 indicates a perfect classifier.

## Example 2: Comparing Multiple Models with ROC Curves

The `label_col` parameter allows you to plot and compare the ROC curves of multiple binary classification models on the same diagram, making it easy to visually assess their relative performance.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monet_plots.plots.roc_curve import ROCCurvePlot

# 1. Prepare sample data for two different models
np.random.seed(42) # for reproducibility

# Model 1: Better performance (higher POD for lower POFD)
thresholds_1 = np.linspace(0, 1, 50)
pod_1 = 0.6 * (1 + np.tanh(6 * (thresholds_1 - 0.4))) + np.random.normal(0, 0.03, 50)
pofd_1 = thresholds_1 * 0.8 + np.random.normal(0, 0.02, 50)
pod_1 = np.clip(pod_1, 0, 1)
pofd_1 = np.clip(pofd_1, 0, 1)
df_model1 = pd.DataFrame({'pofd': pofd_1, 'pod': pod_1, 'model': 'Model 1'}).sort_values(by='pofd')

# Model 2: Moderate performance
thresholds_2 = np.linspace(0, 1, 50)
pod_2 = 0.5 * (1 + np.tanh(4 * (thresholds_2 - 0.6))) + np.random.normal(0, 0.05, 50)
pofd_2 = thresholds_2 * 1.2 + np.random.normal(0, 0.04, 50)
pod_2 = np.clip(pod_2, 0, 1)
pofd_2 = np.clip(pofd_2, 0, 1)
df_model2 = pd.DataFrame({'pofd': pofd_2, 'pod': pod_2, 'model': 'Model 2'}).sort_values(by='pofd')

# Combine dataframes
df_combined = pd.concat([df_model1, df_model2]).reset_index(drop=True)

# 2. Initialize and create the plot, using 'model' as label_col
plot = ROCCurvePlot(figsize=(8, 8))
plot.plot(
    df_combined,
    x_col='pofd',
    y_col='pod',
    label_col='model',
    linewidth=2
)

plot.ax.set_title("ROC Curve: Comparing Two Classification Models")
# Legend with AUC is automatically added when label_col is used

plt.tight_layout()
plt.show()
```

### Expected Output

A single ROC curve plot will display two distinct curves, one for 'Model 1' and one for 'Model 2', distinguished by color and an automatic legend. Each legend entry will also include the calculated AUC for that model. This allows for a direct visual comparison of which model achieves a higher POD for a given POFD, and which has a higher overall AUC.

### Key Concepts

-   **`label_col`**: Allows plotting multiple ROC curves on the same axes, grouped by a specified categorical column (e.g., different models).
-   **Comparative Analysis**: Essential for selecting the best performing model for a given application, especially when considering the trade-off between true positives and false positives.