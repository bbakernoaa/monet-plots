# Forecast plot review fixes

Branch baseline is 478262e. Keep the existing display and export APIs.

1. Taylor geometry and shared figures. Test accepted floating-point boundary values, labeled contour expansion, and interleaved wrappers. Use the validated angle, regenerate labels with the contours, and read shared error state from the diagram. Explain the common-pair requirement when it leaves too few raw observations. Preserve the centered RMS identity.
2. Scorecards and station bias. Test nullable numeric data, custom norms, and missing stations. Use float arrays with NaN for rendering and align coordinates with the source grid. Check significant markers with numbers hidden and grouped headers with a key.
3. ROC and reliability. Test AUC for permuted vertical segments and group raw forecast samples before binning. Keep each group's gaps, show meaningful histogram counts, and validate climatology on the plotted sample. Use the same valid-rank count in rank histogram facets and expected-frequency lines.
4. Export. Test narrow fixed-size PNG and PDF exports with long titles. Make room for extra title lines during export and restore the figure afterward.
5. Run focused regressions, the full headless suite, and repository hooks. Review the diff, commit, and update the branch on the fork.
