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
values in the supplied images. Voxels that can be predicted with statistical
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
[multilevel model](https://en.wikipedia.org/wiki/Multilevel_model) to predict
the voxel's intensity as a function of the tested variable and the confounding
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
is $$-\log_{10}(\mathrm{pvalue}(\mathrm{tstatistic}(\beta_T)))\,,$$ where

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

## The YAML configuration file

Each run is configured with a YAML file and, optionally, a CSV file. The CSV
file is described below with the `table_of_filenames_and_metadata` field below.
The YAML configuration file is organized into four sections as follows:

- **version**: Set to `1.0`.
- **tested_variables**: This section specifies which variables are to be tested.
  Typically they are KSADS variables. The specific fields are:
  - **source_directory**: All relative paths supplied in the tested_variables
    section will be considered as relative to this directory.
  - **variable_default**: Allows one to specify defaults for all variables in
    the tested_variables section so that those defaults need not be repeated
    multiple times.
  - **variable**: Each entry in this section is the name of a tested variable
    (such as "ksads_1_187_t"). Associated with it is information on how to treat
    the variable.
    - **filename**: ABCD CSV file that contains the data for the tested variable
    - **convert**: A dictionary that explains how to convert the tested
      variable's data. Each (key, value) pair of the dictionary specifies what
      the datum (the key) is to be converted to (the value).
    - **handle_missing**: Can be set to any of
      - **invalidate**: throw away target image if it has this field missing
      - **together**: all target images marked as "missing" are put in one
        category that is dedicated to all missing data
      - **by_value**: for each `is_missing` value, all target images with that
        value are put in a category
      - **separately**: each row with a missing value is its own category; e.g.,
        a patient with no siblings in the study
    - **type**: whether the variable is "ordered" (such as a number representing
      an intensity) or "unordered" (as in a number representing a category)
    - **is_missing**: a list of values that should be considered as missing such
      as `.nan` or `""`
    - **description**: an optional, user-supplied string
    - **internal_name**: optionally, the name by which this tested variable will
      be known within `filename`.
- **target_variables**: This section specifies the target images that are to be
  analyzed. The specific fields are:
  - **source_directory**: All relative paths supplied in the target_variables
    section will be considered as relative to this directory.
  - **desired_modality**: Which images in filenames_and_metadata should be
    processed, such as `fa` or `md`.
  - **table_of_filenames_and_metadata**: the file location for the CSV file that
    lists target images to be analyzed. The columns of the CSV file are
    `filename`, `src_subject_id`, `eventname`, `modality`, and `description`, as
    described next.
  - **individual_filenames_and_metadata**: a way to specify target images in the
    YAML file (rather than in the CSV file).
    - **filename**: the location of the .nii.gz file containing the voxel
      intensities
    - **src_subject_id**: the ABCD subject ID associated with this file (with
      prefix `NDAR_`).
    - **eventname**: the ABCD eventname associated with this file (such as
      `2_year_follow_up_y_arm_1`).
    - **modality**: Such as `fa` or `md`
    - **description**: an optional, user-supplied string
  - **mask**: describes mask (e.g., white matter mask) that specifies which
    voxels are meaningful
    - **filename**: the location of the .nii.gz file containing the mask
    - **threshold**: the threshold intensity for determining which voxels to
      include. Defaults to `0.5`.
  - **segmentation**:
    - **filename**: the location of the .nii.gz file containing segmentation
      information
    - **background_index**: The numerical value of the segmentation segment that
      represents background. Defaults to `0`.
  - **template**:
    - **filename**: the location of the .nii.gz file containing a template image
      on which output information is overlaid.
- **confounding_variables**: These are variables that may affect voxel values
  and hide the significant signal that we are hoping to detect for the tested
  variables. The specific fields are:
  - **source_directory**: All relative paths supplied in the
    confounding_variables section will be considered as relative to this
    directory.
  - **variable_default**: Allows one to specify defaults for all variables in
    the confounding_variables section so that those defaults need not be
    repeated multiple times.
  - **variable**: Each entry in this section is the name of a confounding
    variable (such as "interview_age", "site_id_l", or "demo_gender_id_v2").
    Associated with it is information on how to treat the variable.
    - Any field from `tested_variable.variable` above.
    - **minimum_perplexity**: A numerical value at least equal to 1.0. A
      confounding variable with perplexity (i.e., exponentiated entropy) below
      this threshold will be deemed too constant and will be discarded. (If a
      variable is equally likely to be any of $N$ possibilities, its perplexity
      is $N$.)
    - **longitudinal**: This describes whether the confounding variable should
      be used as an intercept and/or slope random effect. Can be any one or more
      of `time`, `intercept`, or `slope`; see the [Methods](#Methods) section
      above. Note that `time` will not be a confounding variable itself unless
      it is also `intercept` or, if it is also `slope` then its square will be
      included.
- **output**: This specifies how to post-process and save the results of the
  analysis
  - **destination_directory**: Where to write the output
  - **local_maxima**: How do define a local maximum of statistical significance.
    - **minimum_negative_log10_p**: Show all peaks with $-\log_{10} p$ at least
      as large as the specified value.
    - **cluster_radius**: If peak falls into a background segment, search of up
      to this radius for a non-background label for this peak.
