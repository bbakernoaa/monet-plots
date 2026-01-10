# Basic Styling Guide

Learn how to quickly customize the appearance of your plots using MONET Plots' built-in styling system and basic customization options.

## Overview

This guide covers fundamental styling techniques to make your plots look professional and consistent. We'll explore the default styles, quick customizations, and basic plot modifications.

### Learning Objectives

- Apply built-in style presets
- Modify basic plot elements
- Change colors and fonts
- Adjust plot dimensions and layout
- Create simple custom styles

## Built-in Style Presets

MONET Plots comes with professionally designed style presets that follow publication standards.

### Wiley Style

The default Wiley-compliant style provides a clean, professional appearance suitable for scientific publications.

```python
import matplotlib.pyplot as plt
from monet_plots import style, TimeSeriesPlot

# Apply Wiley style (default)
style.set_style("wiley")

# Create a plot
plot = TimeSeriesPlot(figsize=(12, 6))
plot.plot(df, x='time', y='value', title="Wiley Style Plot")
plot.save("wiley_style.png")
```

**Wiley Style Characteristics:**
- **Font**: Times New Roman serif font
- **Size**: 10pt for body text, 12pt for titles
- **Grid**: Light gray dotted lines
- **Lines**: 1.5pt width
- **Colors**: Professional color palette
- **Save Format**: TIFF by default, 300 DPI

### APS Style

For American Physical Society publications, use the `paper` style context:

```python
from monet_plots import style
style.set_style("paper")
```

### Nature Style

For Nature journal publications, use the `paper` style context (or customize further if needed):

```python
from monet_plots import style
style.set_style("paper")
# Further customizations can be applied after setting the base style
# plt.rcParams.update({'font.size': 7, 'figure.figsize': (3.5, 2)})
```

## Quick Customizations

### Changing Colors

```python
import matplotlib.pyplot as plt
from monet_plots import TimeSeriesPlot

# Use a custom color palette
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Apply custom style
plt.style.use({
    'axes.prop_cycle': plt.cycler('color', custom_colors),
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 11
})

plot = TimeSeriesPlot()
plot.plot(df, x='time', y='value', color=custom_colors[0])
plot.save("custom_colors.png")
```

### Font Customization

```python
# Custom font configuration
font_style = {
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
}

plt.style.use(font_style)
```

### Size and Layout

```python
# Figure size and layout customization
layout_style = {
    'figure.figsize': (14, 10),  # Large figure
    'figure.dpi': 150,           # Lower DPI for web
    'figure.autolayout': True,   # Automatic layout
    'axes.titlesize': 16,        # Larger titles
    'axes.labelsize': 14,        # Larger labels
    'savefig.dpi': 300,          # High quality saves
    'savefig.bbox': 'tight'      # Tight bounding box
}

plt.style.use(layout_style)
```

## Basic Plot Customization

### Title and Labels

```python
from monet_plots import SpatialPlot

plot = SpatialPlot(figsize=(12, 8))

# Plot data
plot.plot(data, title="Custom Title with Subtitle")

# Add custom labels
plot.xlabel("Longitude (degrees)")
plot.ylabel("Latitude (degrees)")

# Add subtitle
plot.ax.text(0.5, 0.95, "Regional Analysis",
            transform=plot.ax.transAxes,
            ha='center', fontsize=11, style='italic')

plot.save("custom_labels.png")
```

### Grid and Ticks

```python
from monet_plots import TimeSeriesPlot

plot = TimeSeriesPlot(figsize=(12, 6))

# Plot data
plot.plot(df, x='time', y='value')

# Customize grid
plot.ax.grid(True, linestyle='--', alpha=0.7, color='gray')

# Customize ticks
plot.ax.tick_params(axis='both', which='major', labelsize=10, length=6, width=1.5)
plot.ax.tick_params(axis='both', which='minor', length=3, width=1)

# Add minor grid
plot.ax.grid(True, axis='both', which='minor', linestyle=':', alpha=0.3)

plot.save("custom_grid.png")
```

### Legend Customization

```python
from monet_plots import TimeSeriesPlot

plot = TimeSeriesPlot(figsize=(12, 6))

# Plot with custom legend
plot.plot(
    df,
    x='time',
    y='value',
    title="Custom Legend Example",
    label="Main Dataset"
)

# Customize legend
legend = plot.ax.legend(
    loc='upper right',
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.9,
    fontsize=10,
    markerscale=1.2,
    edgecolor='black',
    facecolor='white'
)

# Add legend title
legend.set_title("Data Sources", prop={'size': 11, 'weight': 'bold'})

plot.save("custom_legend.png")
```

## Color Schemes

### Sequential Color Schemes

```python
from monet_plots import SpatialPlot

# Use sequential colormap for continuous data
plot = SpatialPlot(figsize=(12, 8))

# Sequential data
sequential_data = np.random.random((20, 30)) * 100

plot.plot(
    sequential_data,
    cmap='viridis',  # Sequential colormap
    title="Sequential Color Scheme"
)

plot.save("sequential_colors.png")
```

### Diverging Color Schemes

```python
from monet_plots import SpatialPlot

# Use diverging colormap for data with center point
plot = SpatialPlot(figsize=(12, 8))

# Diverging data (positive and negative values)
diverging_data = np.random.normal(0, 50, (20, 30))

plot.plot(
    diverging_data,
    cmap='RdBu_r',  # Red-Blue diverging
    title="Diverging Color Scheme"
)

plot.save("diverging_colors.png")
```

### Qualitative Color Schemes

```python
from monet_plots import ScatterPlot

# Use qualitative colormap for categorical data
plot = ScatterPlot(figsize=(12, 8))

# Create categorical data
categories = ['A', 'B', 'C', 'D', 'E']
colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

for i, (category, color) in enumerate(zip(categories, colors)):
    subset = df[df['category'] == category]
    plot.ax.scatter(subset['x'], subset['y'],
                   color=color, label=category, s=50, alpha=0.7)

plot.ax.legend(title="Categories")
plot.title("Qualitative Color Scheme")
plot.save("qualitative_colors.png")
```

## Colorblind-Friendly Practices

Ensuring your plots are accessible to individuals with color vision deficiencies is crucial for effective communication. MONET Plots encourages the use of colorblind-friendly palettes and practices.

### Recommended Colormaps

Matplotlib offers several perceptually uniform colormaps that are suitable for colorblind individuals.

**For Sequential Data (ordered data, e.g., temperature, elevation):**

*   `viridis`
*   `plasma`
*   `inferno`
*   `magma`
*   `cividis`

**Example (Sequential):**

```python
import matplotlib.pyplot as plt
from monet_plots import SpatialPlot
import numpy as np

plot = SpatialPlot(figsize=(12, 8))
data = np.random.random((20, 30)) * 100

plot.plot(
    data,
    cmap='viridis',  # Colorblind-friendly sequential colormap
    title="Sequential Data with Viridis Colormap"
)
plot.save("colorblind_sequential.png")
```

**For Diverging Data (data with a critical central value, e.g., anomalies, differences):**

*   `BrBG` (Brown-BlueGreen)
*   `PiYG` (Pink-YellowGreen)
*   `PRGn` (Purple-Green)
*   `PuOr` (Purple-Orange)
*   `RdGy` (Red-Gray)
*   `RdBu` (Red-Blue)

*(Note: Adding `_r` to the end of a colormap name reverses it, e.g., `RdBu_r`)*

**Example (Diverging):**

```python
import matplotlib.pyplot as plt
from monet_plots import SpatialPlot
import numpy as np

plot = SpatialPlot(figsize=(12, 8))
diverging_data = np.random.normal(0, 50, (20, 30))

plot.plot(
    diverging_data,
    cmap='RdBu_r',  # Colorblind-friendly diverging colormap
    title="Diverging Data with RdBu_r Colormap"
)
plot.save("colorblind_diverging.png")
```

**For Qualitative/Categorical Data (distinct categories, no inherent order):**

When plotting categorical data, avoid using too many distinct colors that might be hard to differentiate. Consider using:

*   **Colorblind-friendly palettes:** Matplotlib's `tab10`, `tab20`, `Paired` are often good starting points.
*   **Varying line styles or markers:** In addition to color, use different line styles (e.g., solid, dashed, dotted) or marker shapes (e.g., circles, squares, triangles) to distinguish categories.
*   **Direct labeling:** Label categories directly on the plot rather than relying solely on a legend.

**Example (Qualitative):**

```python
import matplotlib.pyplot as plt
from monet_plots import ScatterPlot
import pandas as pd
import numpy as np

plot = ScatterPlot(figsize=(12, 8))

# Create categorical data
categories = ['Group A', 'Group B', 'Group C', 'Group D']
data = {
    'x': np.random.rand(100) * 10,
    'y': np.random.rand(100) * 10,
    'category': np.random.choice(categories, 100)
}
df = pd.DataFrame(data)

# Use a colorblind-friendly qualitative palette (tab10)
colors = plt.cm.get_cmap('tab10', len(categories))

for i, category in enumerate(categories):
    subset = df[df['category'] == category]
    plot.ax.scatter(subset['x'], subset['y'],
                   color=colors(i), label=category, s=50, alpha=0.7)

plot.ax.legend(title="Categories")
plot.title("Qualitative Data with Colorblind-Friendly Palette")
plot.save("colorblind_qualitative.png")
```

### General Best Practices

*   **Avoid red-green combinations:** These are particularly problematic for the most common forms of colorblindness (deuteranomaly and protanomaly).
*   **Use sufficient contrast:** Ensure there is enough contrast between colors and between colors and the background.
*   **Provide redundant coding:** Whenever possible, use other visual cues in addition to color, such as different line styles, marker shapes, patterns, or direct labels.
*   **Test your plots:** Use online simulators or tools (e.g., Color Oracle) to check how your plots appear to individuals with different types of colorblindness.

## Quick Style Templates

MONET Plots provides convenient `set_style` contexts for common use cases:

### Presentation Style

Apply a style optimized for presentations:

```python
from monet_plots import style
style.set_style("presentation")
```

### Paper Publication Style

Apply a style suitable for academic papers:

```python
from monet_plots import style
style.set_style("paper")
```

### Web Publication Style

Apply a style designed for web content:

```python
from monet_plots import style
style.set_style("web")
```

## Common Styling Patterns

### Pattern 1: Quick Style Application

```python
# Apply style before creating plot
plt.style.use('seaborn-v0_8-whitegrid')

plot = TimeSeriesPlot()
plot.plot(df, x='time', y='value')
plot.save("quick_style.png")
```

### Pattern 2: Style Context Manager

```python
# Apply style temporarily
with plt.style.context('seaborn-v0_8-darkgrid'):
    plot = TimeSeriesPlot()
    plot.plot(df, x='time', y='value')
    plot.save("temp_style.png")
```

### Pattern 3: Style Combination

```python
# Combine multiple styles
base_style = plt.style.library['seaborn-v0_8-whitegrid']
custom_colors = {'axes.prop_cycle': plt.cycler('color', ['#e74c3c', '#3498db'])}

combined_style = {**base_style, **custom_colors}
plt.style.use(combined_style)

plot = TimeSeriesPlot()
plot.plot(df, x='time', y='value')
plot.save("combined_style.png")
```

## Practice Exercises

### Exercise 1: Wiley Style Modification
Create a modified Wiley style with larger fonts and a different color scheme.

### Exercise 2: Custom Presentation Style
Design a style optimized for academic presentations with readable fonts and high contrast.

### Exercise 3: Journal-Specific Style
Create a style for a specific journal of your choice based on their submission guidelines.

### Exercise 4: Color Scheme Experimentation
Test different colormaps (viridis, plasma, inferno, magma) on your data and compare.

### Exercise 5: Layout Optimization
Create a style that optimizes plot layout for multi-panel figures.

## Troubleshooting

### Issue 1: Style Not Applying

```python
# Reset matplotlib defaults before applying new style
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('your_style')
```

### Issue 2: Font Not Available

```python
# Check available fonts
import matplotlib.font_manager as fm
print([f.name for f in fm.fontManager.ttflist])

# Use fallback fonts
font_style = {
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'serif']  # Fallback chain
}
```

### Issue 3: Poor Text Readability

```python
# Improve text readability
readable_style = {
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'axes.titlesize': 16,
    'axes.labelweight': 'bold'
}
```

## Next Steps

After mastering basic styling, explore:

1. **[Custom Styles](../custom-styles)** - Create your own comprehensive style themes
2. **[Advanced Customization](../advanced-customization)** - Fine-tune individual plot elements
3. **[Color Management](../colors)** - Advanced color palette management
4. **[Theming Guide](../theming)** - Create consistent branded styles

## Quick Reference

| Style Element | Key | Common Values |
|---------------|-----|---------------|
| Preset Style | `style.set_style(context)` | `"wiley"`, `"presentation"`, `"paper"`, `"web"`, `"default"` |
| Font Family | `font.family` | `serif`, `sans-serif`, `monospace` |
| Font Size | `font.size` | 8-16pt for most uses |
| Figure Size | `figure.figsize` | (8, 6), (12, 8), (10, 6) |
| Save Format | `savefig.format` | `png`, `tiff`, `pdf`, `eps` |
| DPI | `savefig.dpi` | 300 for print, 150 for web |
| Grid Style | `grid.linestyle` | `:`, `--`, `-`, `None` |

---

**Navigation**:

- [Configuration Index](../index) - All configuration guides
- [Custom Styles](../custom-styles) - Advanced style creation
- [Advanced Customization](../advanced-customization) - Full plot control
- [Color Management](../colors) - Color schemes and palettes
