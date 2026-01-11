"""
Taylor Diagram
==============

This example demonstrates how to create a Taylor Diagram using the
:class:`monet_plots.plots.taylor_diagram.TaylorDiagram` class.

Taylor Diagrams are a powerful tool for comparing a "model" dataset against a
"reference" dataset. They provide a concise summary of how well the model
matches the reference in terms of correlation, standard deviation, and
centered root-mean-square error.

For this example, we will treat our synthetic dataset as the "reference" and
create a slightly noisy version of it to act as our "model".
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monet_plots.plots.taylor_diagram import TaylorDiagramPlot
from data import create_dataset

# 1. Create the base synthetic dataset to act as our "reference"
ds_ref = create_dataset()
ref_temp = ds_ref["temperature"].isel(time=0, level=0)

# 2. Create a "model" dataset by adding some random noise
# This simulates a common use case where a model output is compared
# to observations or a reanalysis product.
noise = np.random.normal(0, 1.5, size=ref_temp.shape)
ds_model = ds_ref.copy()
ds_model["temperature"] = ds_model["temperature"] + noise

model_temp = ds_model["temperature"].isel(time=0, level=0)

# 3. Prepare the data for the TaylorDiagram class
# The class expects 1D arrays of paired reference and model points. We can
# get these directly from the xarray DataArrays.
ref_flat = ref_temp.values.flatten()
model_flat = model_temp.values.flatten()

# 4. Create the Taylor Diagram
fig = plt.figure(figsize=(10, 10))
df = pd.DataFrame({"reference": ref_flat, "model": model_flat})
plot = TaylorDiagramPlot(df, col1="reference", col2="model", label1="Reference", fig=fig)
plot.plot(marker="o", ms=10, ls="")

# Set a title
fig.suptitle("Taylor Diagram for Surface Temperature", size="x-large")

# Let mkdocs-gallery display the plot
plt.show()
