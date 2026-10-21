# Spatial rendering benchmark

Python 3.12.14; Linux-6.18.55-x86_64-with-glibc2.41; 2 isolated repetitions per case.

Times are medians. Render = scale selection/artist construction + an actual canvas draw; PNG is a second draw/export.
RSS includes native allocations. Peak increase is relative to warmed imports before creating inputs; sampled peaks may miss allocations shorter than 10 ms.

| Case | Grid | Panels | Render (s) | PNG (s) | Peak RSS (MiB) | Peak increase (MiB) | Dask tasks: construct / draw / PNG |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| mesh_numpy | 1024² | 1 | 0.425 | 0.222 | 285.8 | 113.5 | 0 / 0 / 0 |
| mesh_dask | 1024² | 1 | 0.454 | 0.217 | 277.6 | 106.1 | 33 / 0 / 0 |
| facets_auto | 1024² | 4 | 1.705 | 1.036 | 423.4 | 251.4 | 632 / 0 / 0 |
| facets_explicit | 1024² | 4 | 1.764 | 1.090 | 427.4 | 255.0 | 260 / 0 / 0 |
| images_explicit | 1024² | 4 | 0.366 | 0.368 | 268.2 | 96.0 | 260 / 0 / 0 |
| comparison_auto | 1024² | 3 | 1.365 | 0.841 | 386.4 | 214.8 | 353 / 0 / 0 |
| comparison_explicit | 1024² | 3 | 1.355 | 0.848 | 386.5 | 214.6 | 131 / 0 / 0 |
| comparison_rows | 1024² | 12 | 5.118 | 3.307 | 823.7 | 651.4 | 844 / 0 / 0 |
| comparison_panel_auto | 1024² | 3 | 1.369 | 0.838 | 370.9 | 199.3 | 321 / 0 / 0 |
| comparison_panel_explicit | 1024² | 3 | 1.326 | 0.827 | 373.1 | 201.2 | 99 / 0 / 0 |
| mesh_2048 | 2048² | 1 | 1.676 | 0.865 | 567.9 | 396.0 | 97 / 0 / 0 |

Synthetic fields, resident/lazy input generation, geographic PlateCarree, Agg backend, no map features or file/network I/O. Input generation is reported separately in JSON. These are machine-specific measurements, not performance guarantees.

Automatic scales perform global reductions; explicit limits avoid them. All displayed cells still enter Matplotlib. Rasterizing vector output does not reduce the number of in-memory cells. Dask does not make a Matplotlib figure out-of-core.
