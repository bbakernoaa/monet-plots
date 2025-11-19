import monet_plots.plots as plots
import matplotlib.pyplot as plt
import seaborn as sns

def test_default_sns_context():
    @plots._default_sns_context
    def plotting_function():
        pass

    with sns.plotting_context("poster"), sns.color_palette(plots.colors):
        plotting_function()
