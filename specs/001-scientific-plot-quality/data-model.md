# Figure data and invariants

- **Taylor sample** holds the model label, standard deviation, correlation, and plotted artist. A finite standard deviation and defined correlation produce a point. Record invalid samples with the model label and reason. Coincident samples retain their data coordinates and separate identifiers.
- **Taylor domain** includes a reference standard deviation, initial radial scale, current radial maximum, and angular coverage. It grows monotonically as valid samples are added. Contours use the same spread convention and reference as their points.
- **Figure presentation** combines selected preset, explicit overrides, final size, and export parameters for one figure. The choice is scoped to that figure.
- **Reference fixture** records data identity, units, time basis or event definition, expected values, and settings. Before and after exports share fixture values.

- **Comparison group** identifies a variable or regime, paired reference and model series, and common physical units when known.
- **Companion row** contains model and reference identifiers, paired sample count, raw signed bias, centered RMS difference, total RMSE, reference spread, dimensionless error equivalents when defined, and a reason for any unavailable coordinates.
