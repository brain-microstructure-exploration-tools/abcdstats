# abcdstats Python module

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/brain-microstructure-exploration-tools/abcdstats/workflows/CI/badge.svg
[actions-link]:             https://github.com/brain-microstructure-exploration-tools/abcdstats/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/abcdstats
[conda-link]:               https://github.com/conda-forge/abcdstats-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/brain-microstructure-exploration-tools/abcdstats/discussions
[pypi-link]:                https://pypi.org/project/abcdstats/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/abcdstats
[pypi-version]:             https://img.shields.io/pypi/v/abcdstats
[rtd-badge]:                https://readthedocs.org/projects/abcdstats/badge/?version=latest
[rtd-link]:                 https://abcdstats.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

## Overview

The abcdstats Python module implements a hypothesis generation workflow for the
[Adolescent Brain Cognitive Development Study (ABCD Study®)](https://abcdstudy.org/)
data set.

The module uses subject data and metadata about their images to predict voxel
values in supplied images. Voxels that can be predicted with statistical
significance are hypotheses about which voxels are associated with those data.
The workflow is directed by a supplied
[YAML](https://en.wikipedia.org/wiki/YAML) configuration file and, optionally,
additional information about target images is supplied via a
[CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file.

### Input & Output

The input data are divided into three categories:

- **tested variables:** these are variables that we are seeking significant
  voxels for. These are typically
  [KSADS](https://en.wikipedia.org/wiki/Kiddie_Schedule_for_Affective_Disorders_and_Schizophrenia)
  data from `mental-health/mh_y_ksads_ss`.
- **target data (voxels):** the intensity associated with each voxel of each
  registered image. These can be `fa`, `md`, or other data associated with each
  voxel of an image.
- **confounding variables:** these are variables that may affect voxel values
  and hide the significant signal that we are hoping to detect for the tested
  variables. These can be variables like age
  (`abcd-general/abcd_y_lt/interview_age`), hospital
  (`abcd-general/abcd_y_lt/site_id_l`), gender
  (`gender-identity-sexual-health/gish_p_gi/demo_gender_id_v2`), or family
  (`abcd-general/abcd_y_lt/rel_family_id`).
  - These variables can be used for both intercept and slope statistical random
    effects in a
    [multilevel model](https://en.wikipedia.org/wiki/Multilevel_model).

The output includes:

- **a computed statistical significance for each voxel for each tested
  variable:** This is output as one image per tested variable. Each of these
  images can be visualized in software such as
  [3D Slicer](https://download.slicer.org/), alongside a registered segmentation
  image that labels brain regions.

### Methods

Mathematically speaking, for each tested variable separately, we process each
voxel separately. We use a
[multlevel model](https://en.wikipedia.org/wiki/Multilevel_model) to predict the
voxel's intensity as a function of the tested variable and the confounding
variables. This is a regression model with a non-zero constant term permitted.
When a tested variable or a confounding variable is an ordered variable, it is
used directly. If it is an unordered variable, (such as a site id or other
category indicator), it is represented in
[one-hot](https://en.wikipedia.org/wiki/One-hot) notation; with a separate
[indicator term](https://en.wikipedia.org/wiki/Indicator_function) for all
categories but one.

When a confounding variable is labeled as `intercept` these values are used
directly as just described. When one confounding variable is labeled as `time`
each confounding variable that is labeled as `slope` adds a term to the
regression that is the product of the time variable and the `slope` variable. (A
given confounding variable can be labeled as more than one of `time`,
`intercept`, and `slope`, though only one confounding variable can be labeled as
`time`.) Together these intercept and slope variables are often called
[random effects](https://en.wikipedia.org/wiki/Random_effects_model).

In equations: $$v_i = \sum_c D_{ic} \beta_c + \epsilon_i$$ where

- $v_i$ is the intensity of the voxel in image $i$
- $c$ indexes the independent variables of the regression, which are the form of
  any of:
  - an ordered confounding variable
  - a category of an unordered confounding variable
  - one of the above two possibilities multiplied by a time factor (so as to
    make a random effects slope)
  - the tested variable
  - a constant term
- $\beta_c$ is the regression coefficient for term $c$, which we are solving for
- $D_{ic}$ is the input-data value of the $c$ term for image $i$.
- $\epsilon_i$ is the error term for image $i$. The regression minimizes the sum
  of squares of these values.

The regression is solved and the return value for a voxel for a tested variable
is $$-\log_{10}(\operatorname{pvalue}(\operatorname{tstatistic}(\beta_T)))\,,$$
where

- $\beta_T$ is the coefficient for the tested variable,
- the t-statistic is quantifying the belief that $\beta_T \ne 0$, and
- the p-value is estimated via random permutations of the null hypothesis
  ($\beta_T =
  0$) residuals $\{\epsilon_i\}$ in the manner of Freedman-Lane
  (see
  [Wingler NeuroImage 2014](https://doi.org/10.1016/j.neuroimage.2014.01.060),
  p. 385 for a good description).

A value of $2.0$ indicates a p-value of 1%, a value of $1.3$ indicates a p-value
of 5%, a value of $1.0$ indicates a p-value of 10%, and so on.

## Installation

For general users, install with pip:

```bash
pip install abcdstats
```

Developers can add additional options:

```bash
pip install abcdstats[test,dev,docs]
```

## Examples

Example uses of the code can be found in the [examples](examples/) directory.
