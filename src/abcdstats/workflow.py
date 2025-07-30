from __future__ import annotations

import collections
import copy
import itertools
import math
import os
import pathlib
from typing import Any, TypeAlias, Union, cast

import matplotlib as mpl  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nibabel as nib  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.glm  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.maskers  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.masking  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.mass_univariate  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nrrd  # type: ignore[import-not-found,import-untyped,unused-ignore]
import numpy as np  # type: ignore[import-not-found,import-untyped,unused-ignore]
import numpy.typing as npt  # type: ignore[import-not-found,import-untyped,unused-ignore]
import pandas as pd  # type: ignore[import-not-found,import-untyped,unused-ignore]
import scipy.signal  # type: ignore[import-not-found,import-untyped,unused-ignore]
import yaml  # type: ignore[import-not-found,import-untyped,unused-ignore]

BasicValue: TypeAlias = bool | int | float | str | None
ConfigurationValue: TypeAlias = BasicValue | list[Any]
ConfigurationType: TypeAlias = dict[str, Union[ConfigurationValue, "ConfigurationType"]]


class Basic:
    """abcdstats.workflow.Basic is is for hypothesis generation from ABCD MR data.  It
    tests variables (usually KSADS diagnoses), against target variables (such as data
    from fractional anisotropy MR scans), in the presence of confounding variables (such
    as interview_age, site_id_l).

    The workflow is configured via a YAML file.  Changes not achievable via
    configuration can be made by subclassing this class and overriding relevant methods.

    """

    def __init__(self, *, yaml_file: str | pathlib.Path | None = None) -> None:
        """Create an instance of the abcdstats.workflow.Basic class for running a
        workflow.

        Args:
            yaml_file (str | pathlib.Path | None): The location of the YAML file
                configure the workflow.  If omitted, one can later invoke the
                configure(yaml_file) method.

        Returns:
            abcdstats.workflow.Basic: The class instance for running the workflow.

        """

        # globally useful values
        self.join_keys: list[str] = ["src_subject_id", "eventname"]

        # Set some default configuration values.  The user can override these with
        # self.configure(yaml_file).  The user can go back to these with
        # self.configure(None).
        variable_default: ConfigurationType = {
            "convert": {},
            "handle_missing": "invalidate",
            "is_missing": ["", np.nan],
            "minimum_perplexity": 1.0,
        }
        self.config_default: ConfigurationType = {
            "tested_variables": {"variable_default": variable_default},
            "target_variables": {
                "mask": {"threshold": 0.5},
                "segmentation": {"background_index": 0},
            },
            "confounding_variables": {
                "variable_default": {**variable_default, "longitudinal": ["intercept"]}
            },
            "output": {
                "local_maxima": {
                    "minimum_negative_log10_p": 0.1,
                    "cluster_radius": 3,
                    "label_threshold": 0.1,
                },
                "permuted_ols": {
                    "model_intercept": True,
                    "n_perm": 100,  # TODO: Restore me to 10000 when testing is done
                    "two_sided_test": True,
                    "random_state": None,
                    "n_jobs": -1,  # All available
                    "verbose": 1,
                    "tfce": False,
                    "threshold": None,
                    "output_type": "dict",
                },
                "images": {"gamma": 1.0},
            },
        }

        # First, initialize with defaults
        self.config: ConfigurationType
        self.configure(yaml_file=None)
        # Second, use user-supplied values from file, if any
        if yaml_file is not None:
            self.configure(yaml_file=yaml_file)

    def configure(self, *, yaml_file: str | pathlib.Path | None) -> None:
        """Designate the workflow configuration via a YAML file.

        Args:
            yaml_file (str | pathlib.Path | None): Use the provided YAML file to
                incrementally change the configuration of workflow.  If supplied as
                `None` then the configuration will be reset to package defaults.

        Returns:
            None

        """

        if yaml_file is not None:
            with pathlib.Path(yaml_file).open("r", encoding="utf-8") as file:
                self.copy_keys_into(src=yaml.safe_load(file), dest=self.config)
        else:
            # The user requests the system defaults
            self.config = copy.deepcopy(self.config_default)

    def copy_keys_into(
        self, *, src: ConfigurationType, dest: ConfigurationType
    ) -> None:
        """Updates configuration information from a source to a destination.  Values in
        the destination will be retained unless they are overwritten from the source.

        Args:
            src (ConfigurationType): The configuration from which information is
                transferred into the other configuration.
            dest (ConfigurationType): The configuration to which information is
                transferred from the other configuration.

        Returns:
            None

        """

        for key, value in src.items():
            if isinstance(value, dict) and key in dest and isinstance(dest[key], dict):
                self.copy_keys_into(src=value, dest=cast(ConfigurationType, dest[key]))
            else:
                dest[key] = copy.deepcopy(value)

    def run(self) -> None:
        """Run the workflow.  As a preliminary step, lint the YAML file and, if
        supplied, the CSV file.

        Args:
            None

        Returns:
            None

        """

        # Validate the supplied YAML file
        warnings: list[str] = self.lint()
        if warnings:
            print("\n".join(warnings))  # noqa: T201

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        tested_variables: dict[str, dict[str, Any]]
        tested_frame: pd.core.frame.DataFrame
        # Assembly can include conversion to one-hot.
        tested_variables, tested_frame = self.get_tested_data()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        target_frame: pd.core.frame.DataFrame
        target_frame = self.get_target_data()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        confounding_variables: dict[str, dict[str, Any]]
        confounding_frame: pd.core.frame.DataFrame
        # Assembly can include conversion to one-hot, as well as use as intercept or
        # slope random effects.
        confounding_variables, confounding_frame = self.get_confounding_data()

        # Merge frames by join_keys and construct inputs useful for nilearn
        tested_array: npt.NDArray[np.float64]
        target_images: list[nib.filebasedimages.FileBasedImage]
        target_affine: npt.NDArray[np.float64]
        confounding_array: npt.NDArray[np.float64]
        (
            tested_frame,
            target_frame,
            confounding_frame,
            tested_array,
            target_images,
            target_affine,
            confounding_array,
        ) = self.make_arrays(
            tested_variables,
            tested_frame,
            target_frame,
            confounding_variables,
            confounding_frame,
        )

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        mask_masker: nilearn.maskers.NiftiMasker | None
        mask_affine: npt.NDArray[np.float64] | None
        mask_masker, mask_affine = self.get_source_mask()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        template_voxels: nib.filebasedimages.FileBasedImage | None
        template_affine: npt.NDArray[np.float64] | None
        template_voxels, template_affine = self.get_template()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        segmentation_voxels: npt.NDArray[np.float64 | int] | None
        segmentation_affine: npt.NDArray[np.float64] | None
        segmentation_header: collections.OrderedDict[str, Any] | None
        segmentation_map: dict[int, str] | None
        background_index: int | None
        # Can be a labelmap (all-or-none segmentation) or a cloud that gives
        # probabilities for each segment.
        (
            segmentation_voxels,
            segmentation_affine,
            segmentation_header,
            segmentation_map,
            background_index,
        ) = self.get_brain_segmentation()

        self.allclose_affines(
            [target_affine, mask_affine, template_affine, segmentation_affine],
            "Supplied affine transformations for"
            " target, mask, template, and segmentation must match.",
        )

        # Process inputs to compute statistically significant voxels.
        permuted_ols: dict[str, npt.NDArray[np.float64]]
        glm_ols: npt.NDArray[np.float64]
        permuted_ols, glm_ols = self.compute_significant_voxels(
            tested_vars=tested_array,
            target_images=target_images,
            confounding_vars=confounding_array,
            masker=mask_masker,
        )
        logp_max_t: npt.NDArray[np.float64] = permuted_ols["logp_max_t"]

        # Process output to compute local maxima.
        # For each tested variable, for each local maximum, output coordinates and a
        # description
        local_maxima_description: dict[str, list[tuple[list[int], str]]]
        local_maxima_description = self.compute_local_maxima(
            tested_names=list(tested_frame.columns),
            logp_max_t=logp_max_t,
            segmentation_voxels=segmentation_voxels,
            segmentation_map=segmentation_map,
            background_index=background_index,
        )

        self.save_output(
            template_voxels=template_voxels,
            affine=template_affine,
            segmentation_voxels=segmentation_voxels,
            segmentation_header=segmentation_header,
            segmentation_map=segmentation_map,
            permuted_ols=permuted_ols,
            glm_ols=glm_ols,
            local_maxima_description=local_maxima_description,
        )

    def get_target_data(self) -> pd.core.frame.DataFrame:
        """Read the meta-information about the target data (voxel intensities).  This is
        used for filtering and, later, for retrieving the needed data.

        Args:
            None

        Returns:
            pd.core.frame.DataFrame: A table of meta-information of images.  Each image
                that might be analyzed is a row in this table.

        """

        table: pathlib.Path | None
        individuals: list[dict[str, Any]] | None
        table, individuals, target_directory = (
            self.get_target_data_table_and_individuals()
        )

        target_frame: pd.core.frame.DataFrame
        target_frame = self.get_target_data_frame(table, individuals, target_directory)

        return target_frame

    def get_target_data_table_and_individuals(
        self,
    ) -> tuple[pathlib.Path | None, list[dict[str, Any]] | None, pathlib.Path | None]:
        """Extract information supplied in the YAML configuration file for each image
        and locate the optional CSV file that lists additional images.

        Args:
            None

        Returns:
            pathlib.Path | None: The location of the CSV file if any.
            list[dict[str, Any]] | None: A list of images described in the YAML file, if
                any.  The dict associated with an image is meta-information about the
                image supplied in the YAML file.
            pathlib.Path | None: source_directory to use for filenames found in the
                table (once we read it).

        """

        mesg: str
        d_raw: ConfigurationType | ConfigurationValue | None
        d_raw = self.config_get(["target_variables", "source_directory"])
        directory: pathlib.Path | None
        directory = pathlib.Path(cast(str, d_raw)) if d_raw is not None else None

        t_raw: ConfigurationType | ConfigurationValue | None
        t_raw = self.config_get(["target_variables", "table_of_filenames_and_metadata"])
        table: pathlib.Path | None
        table = pathlib.Path(cast(str, t_raw)) if t_raw is not None else None

        # We will be modifying and returning `individuals` so make a deepcopy
        i_raw: ConfigurationType | ConfigurationValue | None
        i_raw = self.config_get(
            ["target_variables", "individual_filenames_and_metadata"]
        )
        individuals: list[dict[str, Any]] | None
        individuals = (
            cast(list[dict[str, Any]], copy.deepcopy(i_raw))
            if i_raw is not None
            else None
        )

        if table is None and individuals is None:
            mesg = (
                "Must supply at least one of `table_of_filenames_and_metadata`"
                " or `individual_filenames_and_metadata`."
            )
            raise KeyError(mesg)

        # Prepend directory to table location if appropriate
        if table is not None and directory is not None:
            table = directory / table
        # We will prepend directory to entries within table and individuals later

        return table, individuals, directory

    def get_target_data_frame(
        self,
        table: pathlib.Path | None,
        individuals: list[dict[str, Any]] | None,
        target_directory: pathlib.Path | None,
    ) -> pd.core.frame.DataFrame:
        """Combine images from YAML and CSV sources, filter on modality (e.g., "fa" or
        "md") and check that there are no duplicates.

        Args:
            table (pathlib.Path | None): The location of the CSV file if any.
            individuals (list[dict[str, Any]] | None): A list of images described in the
                YAML file, if any.  The dict associated with an image is
                meta-information about the image supplied in the YAML file.
            target_directory: A directory to start with, for relative paths within table
                and individuals.

        Returns:
            pd.core.frame.DataFrame: Combined table of meta-information about all
                images, whether initially supplied in the YAML file or CSV file.

        """

        target_frame: pd.core.frame.DataFrame = pd.concat(
            [
                pd.read_csv(table) if table is not None else None,
                pd.DataFrame(individuals) if individuals is not None else None,
            ],
            ignore_index=True,
        )

        if len(target_frame) == 0:
            mesg = "No target images were supplied, via table or individuals."
            raise ValueError(mesg)

        if target_directory is not None:
            target_frame["filename"] = target_frame["filename"].apply(
                lambda filename: str(target_directory / pathlib.Path(filename))
            )

        # Keep only those images that are the `desired_modality`.
        m_raw: ConfigurationType | ConfigurationValue | None
        m_raw = self.config_get(["target_variables", "desired_modality"])
        if m_raw is None:
            mesg = (
                "You must supply a target_variables.desired_modality in the YAML file,"
                ' often "fa" or "md".'
            )
            raise ValueError(mesg)
        modality: str = cast(str, m_raw)
        target_frame = target_frame[target_frame["modality"] == modality]

        if len(target_frame) == 0:
            mesg = f"No target images of modality {modality!r} were supplied."
            raise ValueError(mesg)

        # Complain if there are duplicate filenames
        dups: dict[str, int]
        dups = dict(target_frame["filename"].value_counts(dropna=True))
        dups = {file: count for file, count in dups.items() if count > 1}
        if dups:
            mesg = (
                "Each image can be a target only once:\n  "
                + ",\n  ".join(
                    [f"{file} appears {count} times." for file, count in dups.items()]
                )
                + "."
            )
            raise ValueError(mesg)

        return target_frame

    def get_target_data_voxels_and_affine(
        self, target_frame: pd.core.frame.DataFrame
    ) -> tuple[list[nib.filebasedimages.FileBasedImage], npt.NDArray[np.float64]]:
        """Fetch lazy-lodable voxel data and associated the affine transform.

        Args:
            pd.core.frame.DataFrame: Combined table of meta-information about all
                images, whether initially supplied in the YAML file or CSV file.

        Returns:
            list[nib.filebasedimages.FileBasedImage]: List of nibabel loaded images.
                These are lazy-loading images that have not loaded the large voxel data
                sets yet, but can do so efficiently when needed.
            npt.NDArray[np.float64]: The affine transformation image used by all the
                nibabel images.

        """

        mesg: str
        # Create nib.filebasedimages.FileBasedImage for each input file.  Note that the
        # order of the elements in `target_images` must match the order of the rows in
        # `target_frame`.
        target_images: list[nib.filebasedimages.FileBasedImage]
        target_images = [nib.load(path) for path in target_frame["filename"].tolist()]

        # Check that shapes match
        all_shapes: list[tuple[int, ...]] = [img.shape for img in target_images]
        if len(all_shapes) > 1 and not all(all_shapes[0] == s for s in all_shapes[1:]):
            mesg = "The target images are not all the same shape."
            raise ValueError(mesg)

        # Find the common affine transformation
        all_affines: list[npt.NDArray[np.float64]] = [
            img.affine for img in target_images
        ]
        self.allclose_affines(
            all_affines,
            "The affine transformation matrices for the images do not all match;"
            " are the images registered?",
        )
        affine: npt.NDArray[np.float64]
        affine = all_affines[0] if len(all_affines) > 0 else None
        return target_images, affine

    def get_tested_data(
        self,
    ) -> tuple[dict[str, dict[str, Any]], pd.core.frame.DataFrame]:
        """Fetch tested data (KSADS diagnoses).  Note that the variables returned may
        not exactly match the requested variables.  For example, an unordered variable
        (such as one that uses integers to represent different categories) will be
        one-hot converted into multiple variables.

        Args:
            None

        Returns:
            dict[str, dict[str, Any]]: A dictionary of meta-information for each
                (possibly converted) tested variable.
            pd.core.frame.DataFrame: A data table with row for each image that provides
                the image's values for each (possibly converted) tested variable.

        """

        mesg: str

        d_raw: ConfigurationType | ConfigurationValue | None
        d_raw = self.config_get(["tested_variables", "source_directory"])
        directory: pathlib.Path | None
        directory = pathlib.Path(cast(str, d_raw)) if d_raw is not None else None

        z_raw: ConfigurationType | ConfigurationValue | None
        z_raw = self.config_get(["tested_variables", "variable_default"])
        variable_default: dict[str, Any]
        variable_default = cast(dict[str, Any], z_raw if z_raw is not None else {})

        v_raw: ConfigurationType | ConfigurationValue | None
        v_raw = self.config_get(["tested_variables", "variable"])
        if v_raw is None or not cast(dict[str, Any], v_raw):
            mesg = "Must supply at least one `variable` in `tested_variables`."
            raise KeyError(mesg)
        variables: dict[str, dict[str, Any]]
        variables = copy.deepcopy(cast(dict[str, dict[str, Any]], v_raw))

        # Note: fetch_variables will use one-hot for an unordered tested variable, which
        # transforms a single test into multiple tests.
        df_by_var: dict[str, pd.core.frame.DataFrame]
        variables, df_by_var = self.fetch_variables(
            directory, variable_default, variables
        )
        df_var: pd.core.frame.DataFrame
        df_var = self.merge_df_list(list(df_by_var.values()))

        return variables, df_var

    def get_confounding_data(
        self,
    ) -> tuple[dict[str, dict[str, Any]], pd.core.frame.DataFrame]:
        """Fetch the data for the confounding variables (such as interview_age,
        site_id_l).  Note that the variables returned may not exactly match the
        requested variables.  For example, an unordered variable (such as one that uses
        integers to represent different categories) will be one-hot converted into
        multiple variables.  For example, a variable that is to be used as a slope
        random effect will be converted to be the product of its original value and an
        appropriate time value.

        Args:
            None

        Returns:
            dict[str, dict[str, Any]]: A dictionary of meta-information for each
                (possibly converted) confounding variable.
            pd.core.frame.DataFrame: A data table with row for each image that provides
                the image's values for each (possibly converted) confounding variable.

        """

        mesg: str

        d_raw: ConfigurationType | ConfigurationValue | None
        d_raw = self.config_get(["confounding_variables", "source_directory"])
        directory: pathlib.Path | None
        directory = pathlib.Path(cast(str, d_raw)) if d_raw is not None else None

        z_raw: ConfigurationType | ConfigurationValue | None
        z_raw = self.config_get(["confounding_variables", "variable_default"])
        variable_default: dict[str, Any]
        variable_default = cast(dict[str, Any], z_raw if z_raw is not None else {})

        v_raw: ConfigurationType | ConfigurationValue | None
        v_raw = self.config_get(["confounding_variables", "variable"])
        if v_raw is None or not cast(dict[str, Any], v_raw):
            mesg = "Must supply at least one `variable` in `confounding_variables`."
            raise KeyError(mesg)
        variables: dict[str, dict[str, Any]]
        variables = copy.deepcopy(cast(dict[str, dict[str, Any]], v_raw))

        df_by_var: dict[str, pd.core.frame.DataFrame]
        variables, df_by_var = self.fetch_variables(
            directory, variable_default, variables
        )
        df_by_var = self.handle_longitudinal(variables, df_by_var)

        df_var: pd.core.frame.DataFrame
        df_var = self.merge_df_list(list(df_by_var.values()))
        return variables, df_var

    def fetch_variables(
        self,
        directory: pathlib.Path | None,
        variable_default: dict[str, Any],
        variables: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, pd.core.frame.DataFrame]]:
        """Fetch data (tested, target, or confounding) and convert it (e.g. via one-hot)
        as needed or requested.

        Args:
            directory (pathlib.Path | None): source_directory that is applicable to this
                data type (tested, target, or confounding).
            variable_default (dict[str, Any]): default meta-information values to be
                used for each variable when not specifically provided for a variable.
            variables (dict[str, dict[str, Any]]): meta-information values to be
                used for each variable

        Returns:
            dict[str, dict[str, Any]]: Updated `variables` information that reflects any
                conversions that happened.
            dict[str, pd.core.frame.DataFrame]: For each of the updated variables, a
                data table with a row for each image that provides the image's values
                for each variable.

        """

        # Use defaults unless a variable-specific value is supplied
        variables = {
            variable: variable_default | var_config
            for variable, var_config in variables.items()
        }

        # Check that each var_config has required fields
        mesg = "\n".join(
            [
                f"`{field}` is missing for tested variable `{variable}`."
                for field in [
                    "filename",
                    "convert",
                    "handle_missing",
                    "is_missing",
                    "type",
                    "minimum_perplexity",
                ]
                for variable, var_config in variables.items()
                if field not in var_config
            ]
        )
        if mesg:
            raise KeyError(mesg)

        # Compute full paths
        if directory is not None:
            variables = {
                variable: var_config
                | {"filename": str(directory / pathlib.Path(var_config["filename"]))}
                for variable, var_config in variables.items()
            }

        # The `variable` is the internal_name unless the `var_config` already has it
        variables = {
            variable: {"internal_name": variable} | var_config
            for variable, var_config in variables.items()
        }

        # Load each file (once) into a pandas DataFrame and select the variables and
        # join keys.
        df_by_file: dict[str, pd.core.frame.DataFrame]
        df_by_file = {
            fn: pd.read_csv(fn)[
                [
                    *self.join_keys,
                    *{
                        var_config["internal_name"]
                        for var_config in variables.values()
                        if var_config["filename"] == fn
                    },
                ]
            ]
            for fn in {var_config["filename"] for var_config in variables.values()}
        }

        # Switch to user-specified names.
        df_by_file = {
            fn: df_var.rename(
                columns={
                    var_config["internal_name"]: variable
                    for variable, var_config in variables.items()
                    if var_config["filename"] == fn
                    and var_config["internal_name"] != variable
                }
            )
            for fn, df_var in df_by_file.items()
        }

        # Segregate dataframes by variable.
        by_var: dict[str, tuple[dict[str, dict[str, Any]], pd.core.frame.DataFrame]]
        by_var = {
            variable: self.handle_variable_config(variable, var_config, df_by_file)
            for variable, var_config in variables.items()
        }
        # Add newly created variables
        variables = {
            **variables,
            **{
                variable: var_config
                for value in by_var.values()
                for variable, var_config in value[0].items()
            },
        }
        df_by_var: dict[str, pd.core.frame.DataFrame]
        df_by_var = {variable: value[1] for variable, value in by_var.items()}

        return variables, df_by_var

    def handle_variable_config(
        self,
        variable: str,
        var_config: dict[str, Any],
        df_by_file: dict[str, pd.core.frame.DataFrame],
    ) -> tuple[dict[str, dict[str, Any]], pd.core.frame.DataFrame]:
        """Use meta-information about a variable to process and/or convert it.  This
        includes "convert" of some values to others, handling missing data, handling
        unordered categories, etc.

        Args:
            variable (str): The name of the variable being handled
            var_config (dict[str, Any]): The meta-information for the variable.
            df_by_file (dict[str, pd.core.frame.DataFrame]): The data for each image,
                organized by the file name (rather than variable name) from which it
                comes.

        Returns:
            dict[str, dict[str, Any]]: The meta-information for any newly created
                variables.
            pd.core.frame.DataFrame: Converted data for each (possibly just created)
                variable.

        """

        convert: dict[str, Any]
        convert = var_config["convert"]
        handle_missing: str
        handle_missing = var_config["handle_missing"]
        is_missing: list[Any]
        is_missing = var_config["is_missing"]
        var_type: str
        var_type = var_config["type"]
        df_var: pd.core.frame.DataFrame
        df_var = df_by_file[var_config["filename"]][[*self.join_keys, variable]].copy()

        # Apply convert, handle_missing and is_missing
        for src, dst in convert.items():
            df_var[variable] = self.pandas_replace(df_var[variable], src, dst)

        # Meaning of `handle_missing`:
        # * "invalidate" (throw away scan if it has this field missing)
        # * "together" (all scans marked as "missing" are put in one category that is
        #   dedicated to all missing data)
        # * "by_value" (for each IsMissing value, all scans with that value are put in a
        #   category)
        # * "separately" (each row with a missing value is its own category; e.g., a
        #   patient with no siblings in the study)

        if handle_missing != "by_value" and len(is_missing) > 1:
            # Prior to handling the separate cases, make all missing values equal to
            # is_missing[0]
            for val in is_missing[1:]:
                df_var[variable] = self.pandas_replace(
                    df_var[variable], val, is_missing[0]
                )
        # Process each handle_missing case
        if handle_missing == "invalidate" and len(is_missing) > 0:
            # Remove rows for missing values
            df_var = df_var[df_var[variable] != is_missing[0]]
        if handle_missing == "together" and len(is_missing) > 0:
            if var_type == "unordered":
                # Nothing to do
                pass
            if var_type == "ordered":
                # We will interpret missing as 0, but add a one-hot column so that it
                # can effectively be any constant.  In particular, it will not be
                # confounded with actual values of 0.
                new_missing_name: str = variable + "_missing"
                if new_missing_name in df_var.columns:
                    mesg = f"Failed to get unique column name for {new_missing_name!r}."
                    raise ValueError(mesg)
                df_var[new_missing_name] = (df_var[variable] == is_missing[0]).astype(
                    int
                )
                df_var.loc[df_var[variable] == is_missing[0], variable] = 0
        if handle_missing == "by_value":
            if var_type == "unordered":
                # There is nothing more to do.
                pass
            if var_type == "ordered":
                mesg = (
                    "We do not currently handle the case that"
                    ' handle_missing == "by_value" and var_type == "ordered".'
                )
                raise ValueError(mesg)
        if handle_missing == "separately":
            if var_type == "ordered":
                mesg = (
                    "We do not currently handle the case that"
                    ' handle_missing == "separately" and var_type == "ordered".'
                )
                raise ValueError(mesg)
            # We want to use unused distinct values for each type of "missing".  We add
            # unique values so the pd.get_dummies call creates a one-hot column for each
            # missing value.
            unused_numeric_value: int = 1 + max(
                [int(x) for x in df_var[variable] if isinstance(x, int | float)] + [0]
            )
            number_needed_values: int = (df_var[variable] == is_missing[0]).sum()
            if var_type == "unordered":
                df_var.loc[df_var[variable] == is_missing[0], variable] = range(
                    unused_numeric_value, unused_numeric_value + number_needed_values
                )

        # Convert categorical data to a multicolumn one-hot representation.  Note that
        # setting drop_first=True will drop one of the categories; and in such a case,
        # we should drop the single, "missing" category if there is such a thing.

        df_var_columns: set[str]
        df_var_columns = set(df_var.columns)
        if var_type == "unordered":
            # Note: if an unordered variable has high perplexity but its individual
            # categories (in one-hot representation) do not, the categories with low
            # perplexity will ultimately be removed.
            df_var = (
                pd.get_dummies(
                    df_var, dummy_na=True, columns=[variable], drop_first=False
                )
                if df_var[variable].isna().any()
                else pd.get_dummies(
                    df_var, dummy_na=False, columns=[variable], drop_first=False
                )
            )

        # Each newly created column, if any, is a new variable, so prepare an entry for
        # `variables` by copying from its origin variable.
        variables: dict[str, dict[str, Any]]
        variables = {
            new_variable: {**var_config, "internal_name": new_variable}
            for new_variable in set(df_var.columns) - df_var_columns
        }

        return variables, df_var

    def enforce_perplexity(
        self, df_var: pd.core.frame.DataFrame, variables: dict[str, dict[str, Any]]
    ) -> pd.core.frame.DataFrame:
        """Eliminate each variable (column) that is too close to constant if the user
        has set a minimum_perplexity for that variable (or via variable_default).

        Args:
            df_var (pd.core.frame.DataFrame): Data from which variables (columns) may be
                culled.
            variables (dict[str, dict[str, Any]]): Meta-information about each variable
                including its `minimum_perplexity`.

        Returns:
            pd.core.frame.DataFrame: The dataframe with its low complexity variables
            removed.

        """

        return df_var.drop(
            columns=[
                col
                for col in df_var.columns
                if col in variables and "minimum_perplexity" in variables[col]
                for min_perp in [variables[col]["minimum_perplexity"]]
                if min_perp > 1.0
                for counts in (df_var[col].value_counts(dropna=False),)
                for total in (sum(counts),)
                for perp in (
                    math.prod(
                        (total / count) ** (count / total)
                        for count in counts
                        if count > 0
                    ),
                )
                if perp < min_perp and not math.isclose(perp, min_perp)
            ]
        )

    def handle_longitudinal(
        self,
        variables: dict[str, dict[str, Any]],
        df_by_var: dict[str, pd.core.frame.DataFrame],
    ) -> dict[str, pd.core.frame.DataFrame]:
        """Handle random effects for confounding variables

        Args:
            variables (dict[str, dict[str, Any]]): meta-information values to be used
                for each variable, including "time", "intercept", and "slope" values for
                the "longitudinal" key.
            df_by_var (dict[str, pd.core.frame.DataFrame]): a data frame with image data
                for each variable.

        Returns:
            dict[str, pd.core.frame.DataFrame]: an updated df_by_var that includes the
                requested random effects.

        """

        kinds: list[str] = ["time", "intercept", "slope"]
        mesg: str

        has: dict[str, list[str]]
        has = {
            kind: [
                variable
                for variable, var_config in variables.items()
                if kind in var_config["longitudinal"]
            ]
            for kind in kinds
        }

        mesgs: list[str]
        mesgs = []
        if len(has["time"]) > 1:
            mesgs.append(
                f'{len(has["time"])} confounding_variables variables {has["time"]} were'
                ' specified as "time" but having more than 1 is not permitted.'
            )
        if len(has["time"]) == 1 and len(has["slope"]) == 0:
            mesgs.append(
                f'When one confounding_variables variable {has["time"]} is specified as'
                ' "time" then at least one must be specified as "slope".'
            )
        if len(has["time"]) == 0 and len(has["slope"]) > 0:
            mesgs.append(
                f'{len(has["slope"])} confounding_variables variables {has["slope"]}'
                ' were supplied as "slope" but none are permitted because no'
                ' confounding variables were supplied as "time".'
            )
        if mesgs:
            mesg = "\n".join(mesgs)
            raise ValueError(mesg)

        if len(has["time"]) > 0:
            new_time_name: str = "E3McmbxXubhzaeZT"
            if any(new_time_name in df_var.columns for df_var in df_by_var.values()):
                mesg = "Failed to get unique column name for df_time."
                raise ValueError(mesg)
            df_time: pd.core.frame.DataFrame
            df_time = (
                df_by_var[has["time"][0]]
                .copy(deep=True)
                .rename(columns={has["time"][0]: new_time_name})
            )

            df_intercept_by_var: dict[str, pd.core.frame.DataFrame]
            # df_by_var is indexed by unconverted variables but includes columns for the
            # converted variables (e.g. added one-hot or slope columns).
            df_intercept_by_var = {
                k: v for k, v in df_by_var.items() if k in has["intercept"]
            }

            slope: str = "_slope"
            bad_names: set[str]
            bad_names = {
                v + slope
                for v in has["slope"]
                for df in df_by_var.values()
                if v + slope in df.columns
            }
            if bad_names:
                mesg = (
                    "Failed to get unique column name for slope "
                    f'variable{"s" if len(bad_names) > 1 else ""} {bad_names!r}.'
                )
                raise ValueError(mesg)
            df_slope_by_var: dict[str, pd.core.frame.DataFrame]
            df_slope_by_var = {
                v: df_by_var[v].merge(
                    df_time, on=self.join_keys, how="inner", validate="one_to_one"
                )
                for v in has["slope"]
            }
            df_slope_by_var = {
                v + slope: df.assign(**{v + slope: df[new_time_name] * df[v]}).drop(
                    columns=[v, new_time_name]
                )
                for v, df in df_slope_by_var.items()
            }
            # Note that we do not keep the confounding variable has["time"][0] unless it
            # is also listed in has["intercept"].
            df_by_var = {**df_intercept_by_var, **df_slope_by_var}

        return df_by_var

    def make_arrays(
        self,
        tested_variables: dict[str, dict[str, Any]],
        tested_frame: pd.core.frame.DataFrame,
        target_frame: pd.core.frame.DataFrame,
        confounding_variables: dict[str, dict[str, Any]],
        confounding_frame: pd.core.frame.DataFrame,
    ) -> tuple[
        pd.core.frame.DataFrame,
        pd.core.frame.DataFrame,
        pd.core.frame.DataFrame,
        npt.NDArray[np.float64],
        list[nib.filebasedimages.FileBasedImage],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Combine tested, target, and confounding variables information and create
        objects directly usable with nilearn.permuted_ols.  Reject an image if it is not
        sufficiently present across the three variable data sets.

        Args:
            tested_variables (dict[str, dict[str, Any]]): meta-information about the
                tested variables.
            tested_frame (pd.core.frame.DataFrame): values for the tested variables
            target_frame (pd.core.frame.DataFrame): (lazy) values for the target
                variables.
            confounding_variables (dict[str, dict[str, Any]]): meta-information about
                the confounding variables.
            confounding_frame (pd.core.frame.DataFrame): values for the confounding
                variables.

        Returns:
            pd.core.frame.DataFrame: updated tested_frame
            pd.core.frame.DataFrame: updated target_frame
            pd.core.frame.DataFrame: updated confounding_frame
            npt.NDArray[np.float64]: version of tested_frame for nilearn.permuted_ols.
            list[nib.filebasedimages.FileBasedImage]: Lazy-loading voxel data
            npt.NDArray[np.float64]: affine transformation associated with target images
            npt.NDArray[np.float64]: version of confounding_frame for
                nilearn.permuted_ols.

        """

        # Note that the join with target_frame is "one_to_many" because there could be
        # multiple images associated with a particular (src_subject_id, eventname) pair.
        # For example, they could be images with different modalities (such as FA
        # vs. MD).
        all_frame: pd.core.frame.DataFrame
        all_frame = tested_frame.merge(
            confounding_frame, on=self.join_keys, how="inner", validate="one_to_one"
        )
        all_frame = all_frame.merge(
            target_frame, on=self.join_keys, how="inner", validate="one_to_many"
        )
        # TODO: Do we want `all_frame = all_frame.dropna()` here?  Although convert may
        # already have converted all NaNs that we wish to retain, it is also the case
        # that it might not have.
        all_frame = all_frame.dropna()

        # Now that we have determined which rows are actually going to be processed,
        # let's remove columns that do not meet the perplexity requirement.
        variables: dict[str, dict[str, Any]]
        variables = {**tested_variables, **confounding_variables}
        all_frame = self.enforce_perplexity(all_frame, variables)

        # Recreate the input frames, respecting any selection, any replication, and any
        # reordering of rows to produce all_frame, and respecting any dropping of
        # columns.
        tested_frame = all_frame[all_frame.columns.intersection(tested_frame.columns)]
        tested_keys: list[str]
        tested_keys = list(set(tested_frame.columns) - set(self.join_keys))
        tested_array: npt.NDArray[np.float64]
        tested_array = tested_frame[tested_keys].to_numpy(dtype=np.float64)

        target_frame = all_frame[all_frame.columns.intersection(target_frame.columns)]
        target_keys: list[str]
        target_keys = list(set(target_frame.columns) - set(self.join_keys))
        target_images: list[nib.filebasedimages.FileBasedImage]
        target_affine: npt.NDArray[np.float64]
        target_images, target_affine = self.get_target_data_voxels_and_affine(
            target_frame[target_keys]
        )

        confounding_frame = all_frame[
            all_frame.columns.intersection(confounding_frame.columns)
        ]
        confounding_keys: list[str]
        confounding_keys = list(set(confounding_frame.columns) - set(self.join_keys))
        confounding_array: npt.NDArray[np.float64]
        confounding_array = confounding_frame[confounding_keys].to_numpy(
            dtype=np.float64
        )

        return (
            tested_frame,
            target_frame,
            confounding_frame,
            tested_array,
            target_images,
            target_affine,
            confounding_array,
        )

    def get_source_mask(
        self,
    ) -> tuple[nilearn.maskers.NiftiMasker | None, npt.NDArray[np.float64] | None]:
        """Read in mask for choosing which voxels should be target variables.  Returns
        `(None, None)` if no mask is requested.

        Args:
           None

        Returns:
            nilearn.maskers.NiftiMasker | None: Mask object used by nilearn
            npt.NDArray[np.float64] | None: Affine transformation for mask image

        """

        d_raw: ConfigurationType | ConfigurationValue | None
        d_raw = self.config_get(["target_variables", "source_directory"])
        directory: pathlib.Path | None
        directory = pathlib.Path(cast(str, d_raw)) if d_raw is not None else None

        f_raw: ConfigurationType | ConfigurationValue | None
        f_raw = self.config_get(["target_variables", "mask", "filename"])
        filename: pathlib.Path | None
        filename = pathlib.Path(cast(str, f_raw)) if f_raw is not None else None
        if directory is not None and filename is not None:
            filename = directory / filename

        t_raw: ConfigurationType | ConfigurationValue | None
        t_raw = self.config_get(["target_variables", "mask", "threshold"])
        threshold: float | None
        threshold = cast(float, t_raw) if t_raw is not None else None

        image: nib.filebasedimages.FileBasedImage | None
        image = nib.load(filename) if filename is not None else None

        affine: npt.NDArray[np.float64] | None
        affine = image.affine if image is not None else None

        masker: nilearn.maskers.NiftiMasker | None = None
        if image is not None and threshold is not None:
            masker = nilearn.maskers.NiftiMasker(
                nilearn.masking.compute_brain_mask(
                    target_img=image, threshold=threshold
                )
            )
            masker.fit()

        return masker, affine

    def get_template(
        self,
    ) -> tuple[
        nib.filebasedimages.FileBasedImage | None, npt.NDArray[np.float64] | None
    ]:
        """Read in template to serve as background for output images.  Returns `(None,
        None)` if no template is requested.

        Args:
            None

        Returns:
            nib.filebasedimages.FileBasedImage | None: Lazy-loaded template image.
            npt.NDArray[np.float64] | None: Affine transformation for template image.

        """

        d_raw: ConfigurationType | ConfigurationValue | None
        d_raw = self.config_get(["target_variables", "source_directory"])
        directory: pathlib.Path | None
        directory = pathlib.Path(cast(str, d_raw)) if d_raw is not None else None

        f_raw: ConfigurationType | ConfigurationValue | None
        f_raw = self.config_get(["target_variables", "template", "filename"])
        filename: pathlib.Path | None
        filename = pathlib.Path(cast(str, f_raw)) if f_raw is not None else None
        if directory is not None and filename is not None:
            filename = directory / filename

        voxels: nib.filebasedimages.FileBasedImage | None
        voxels = nib.load(filename) if filename is not None else None

        affine: npt.NDArray[np.float64] | None
        affine = voxels.affine if voxels is not None else None

        return voxels, affine

    def get_brain_segmentation(
        self,
    ) -> tuple[
        npt.NDArray[np.float64 | int] | None,
        npt.NDArray[np.float64] | None,
        collections.OrderedDict[str, Any] | None,
        dict[int, str] | None,
        int | None,
    ]:
        """Returns segmentation information for the brain, for use in annotating
        statistically significant voxels.

        Args:
            None

        Returns:
            npt.NDArray[np.float64 | int] | None: segmentation data organized by voxel.
                Voxels will be `np.float` with shape (num_segments, shapex, shapey,
                shapez) or will be `int` with shape (shapex, shapey, shapez).
            npt.NDArray[np.float64] | None: affine transformation for segmentation data
            collections.OrderedDict[str, Any] | None: raw header information from NRRD
                image file
            dict[int, str] | None: map from segment int to segment description
            int | None: segment index corresponding to background (no segment).

        """

        d_raw: ConfigurationType | ConfigurationValue | None
        d_raw = self.config_get(["target_variables", "source_directory"])
        directory: pathlib.Path | None
        directory = pathlib.Path(cast(str, d_raw)) if d_raw is not None else None

        f_raw: ConfigurationType | ConfigurationValue | None
        f_raw = self.config_get(["target_variables", "segmentation", "filename"])
        filename: pathlib.Path | None
        filename = pathlib.Path(cast(str, f_raw)) if f_raw is not None else None
        if directory is not None and filename is not None:
            filename = directory / filename

        b_raw: ConfigurationType | ConfigurationValue | None
        b_raw = self.config_get(
            ["target_variables", "segmentation", "background_index"]
        )
        background_index: int | None
        background_index = cast(int, b_raw) if b_raw is not None else None

        # Voxels will be `np.float` with shape (num_segments, shapex, shapey, shapez) or
        # will be `int` with shape (shapex, shapey, shapez).
        voxels: npt.NDArray[np.float64 | int] | None = None
        header: collections.OrderedDict[str, Any] | None = None
        affine: npt.NDArray[np.float64] | None = None
        segmentation_map: dict[int, str] | None = None
        if filename is not None:
            voxels, header = nrrd.read(filename)
            header = cast(collections.OrderedDict[str, Any], header)
            affine = self.construct_affine(header)
            segmentation_map = {
                int(value): str(header[name_key])
                for key, value in header.items()
                if key.endswith("_LabelValue")
                for name_key in [f'{key[:-len("_LabelValue")]}_Name']
                if name_key in header
            }

        return voxels, affine, header, segmentation_map, background_index

    def construct_affine(
        self, header: collections.OrderedDict[str, Any]
    ) -> npt.NDArray[np.float64]:
        """Construct 4-by-4 affine transformation matrix from NRRD image header
        information.

        Args:
            header (collections.OrderedDict[str, Any]): NNRD image header information.

        Returns:
            npt.NDArray[np.float64]: affine transformation matrix

        """

        affine: npt.NDArray[np.float64]

        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = header["space directions"][-3:, -3:]
        affine[:3, 3] = header["space origin"][-3:]
        if header["space"] == "left-posterior-superior":
            # Convert to right-anterior-superior
            affine[:2] *= -1.0

        return affine

    def allclose_affines(
        self, affines: list[npt.NDArray[np.float64]], mesg: str
    ) -> None:
        """Checks that all the supplied affine matrices are similar enough.

        Args:
            affines (list[npt.NDArray[np.float64]]): list of affine transformation
                matrices.
            mesg (str): message for ValueError exception when the matrices are not
                sufficiently similar.

        Returns:
            None

        """

        affines = [a for a in affines if a is not None]
        if len(affines) > 1 and not all(
            np.allclose(affines[0], a) for a in affines[1:]
        ):
            raise ValueError(mesg)

    def compute_significant_voxels(
        self,
        *,
        tested_vars: npt.NDArray[np.float64],
        target_images: list[nib.filebasedimages.FileBasedImage],
        confounding_vars: npt.NDArray[np.float64] | None,
        masker: nilearn.maskers.NiftiMasker | None,
    ) -> tuple[dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Invoke nilearn.permuted_ols and nilearn.glm_ols to compute statistically
        significant voxels and associated information.

        Args:
            tested_vars (npt.NDArray[np.float64]): tested variable data for each image.
            target_images (list[nib.filebasedimages.FileBasedImage]): Lazy-loadable
                target variable data for each image.
            confounding_vars (npt.NDArray[np.float64] | None): confounding variable data
                for each image.
            masker (nilearn.maskers.NiftiMasker | None): Mask indicating which voxels to
                compute statistical significance for.

        Returns:
            dict[str, npt.NDArray[np.float64]]: response from
                nilearn.mass_univariate.permuted_ols.
            npt.NDArray[np.float64]: response from
                nilearn.glm.OLSModel(...).fit(...).theta

        """

        # TODO: Verify that masker is doing what we hope it is doing

        # TODO: If masker is correct except for the number of channels (aka colors), can
        # we recover?

        # TODO: It *might* be the case that the `masker` parameter for permuted_ols can
        # be supplied and it will call fit_transform(target_images) and later
        # inverse_transform(logp_max_t) on our behalf; so we wouldn't do either of those
        # in our code.  However, a downside is that it may be the case that target_vars
        # would have to be supplied as all voxels of all images, in memory as a numpy
        # array rather than as a list of nib.filebasedimages.FileBasedImage that can be
        # loaded lazily / one at a time.
        target_vars: npt.NDArray[np.float64]
        target_vars = (
            masker.fit_transform(target_images)
            if masker is not None
            else np.stack(
                [np.asanyarray(img.dataobj).reshape((-1,)) for img in target_images]
            )
        )

        """
        Shapes of the numpy arrays are
          tested_vars.shape == (number_images, number_ksads)
          target_vars.shape == (number_images, number_of_good_voxels)
          confounding_vars.shape == (number_images, number_confounding_vars)
        """

        # Note that each can also be `None`
        perm_ols_spec: dict[str, type] = {
            "model_intercept": bool,
            "n_perm": int,
            "two_sided_test": bool,
            "random_state": int,
            "n_jobs": int,
            "verbose": int,
            "tfce": bool,
            "threshold": float,
            "output_type": str,
        }

        # Call nilearn.mass_univariate.permuted_ols.
        other_parameters: dict[str, Any] = {
            k: v
            for k, v in cast(
                dict[str, Any], self.config_get(["output", "permuted_ols"])
            ).items()
            if k in perm_ols_spec
        }
        permuted_ols_response: dict[str, npt.NDArray[np.float64]]
        permuted_ols_response = nilearn.mass_univariate.permuted_ols(
            tested_vars=tested_vars,  # ksads
            target_vars=target_vars,  # voxels
            confounding_vars=confounding_vars,  # e.g., interview_age
            **other_parameters,
        )
        # Use masker.inverse_transform on some individual values within
        # permuted_ols_response to restore the original shape.
        if masker is not None:
            # Two dimensional arrays where the second shape parameter is n_descriptors
            descriptors_keys: list[str]
            descriptors_keys = [
                "t",
                "logp_max_t",
                "tfce",
                "logp_max_tfce",
                "size",
                "logp_max_size",
                "mass",
                "logp_max_mass",
            ]
            # Note: masker.inverse_transform() returns an object that has the spatial
            # dimensions first and then the image index; we correct for that.

            # Note: permuted_ols_response[any_key].flags['F_CONTIGUOUS'] might be True
            # -- this is common for nibabel.nifti1.Nifti1Image.dataobj[:, :, :].  So, if
            # the data is to be sliced then it might be best as [:, :, z] (rather than
            # as [x, :, :]).
            permuted_ols_response = {
                **permuted_ols_response,
                **{
                    key: np.asanyarray(
                        masker.inverse_transform(value).dataobj, dtype=np.float64
                    ).transpose(3, 0, 1, 2)
                    for key, value in permuted_ols_response.items()
                    if key in descriptors_keys
                },
            }

        # In each case, compute the regression coefficient that yielded the t-stat and
        # p-value reported by permuted_ols
        glm_ols_response: npt.NDArray[np.float64]
        glm_ols_response = np.vstack(
            [
                nilearn.glm.OLSModel(
                    np.hstack((tested_var.reshape(-1, 1), confounding_vars))
                )
                .fit(target_vars)
                .theta[0, :]
                for tested_var in tested_vars.T
            ]
        )

        if masker is not None:
            # Note that masker.inverse_transform() returns an object that has the
            # spatial dimensions first and then the image index; we correct for that.
            glm_ols_response = np.asanyarray(
                masker.inverse_transform(glm_ols_response).dataobj, dtype=np.float64
            ).transpose(3, 0, 1, 2)

        return permuted_ols_response, glm_ols_response

    def compute_local_maxima(
        self,
        *,
        tested_names: list[str],
        logp_max_t: npt.NDArray[np.float64],
        segmentation_voxels: npt.NDArray[np.float64 | int] | None,
        segmentation_map: dict[int, str] | None,
        background_index: int | None,
    ) -> dict[str, list[tuple[list[int], str]]]:
        """For each tested variable, find local maxima in -log10(p-value) data and
        describe them.

        Args:
            logp_max_t (npt.NDArray[np.float64]): log10 p-values from permuted_ols for
                each tested variable and each voxel location.
            segmentation_voxels (npt.NDArray[np.float64 | int] | None): segmentation
                data organized by voxel.  Voxels will be `np.float` with shape
                (num_segments, shapex, shapey, shapez) or will be `int` with shape
                (shapex, shapey, shapez).
            segmentation_map (dict[int, str] | None): map from segment int to segment
                description
            background_index (int | None): segment index corresponding to background (no
                segment).

        Returns:
            list[list[tuple[list[int], str]]]: for each tested variable returns a list
                of local_maxima, where each local maxima is a 3-dimensional location
                with a description.

        """

        return {
            name: self.compute_local_maxima_for_variable(
                name,
                logp_max_t[index],
                segmentation_voxels,
                segmentation_map,
                background_index,
            )
            for index, name in enumerate(tested_names)
        }

    def compute_local_maxima_for_variable(
        self,
        name: str,
        variable: npt.NDArray[np.float64],
        segmentation_voxels: npt.NDArray[np.float64 | int] | None,
        segmentation_map: dict[int, str] | None,
        background_index: int | None,
    ) -> list[tuple[list[int], str]]:
        """For a specific tested variable, find local maxima in -log10(p-value) data and
        describe them.

        Args:
            name (str): name of tested variable being processed
            variable (npt.NDArray[np.float64]): log10 p-values from permuted_ols of a
                specific tested variable and each voxel location.
            segmentation_voxels (npt.NDArray[np.float64 | int] | None): segmentation
                data organized by voxel.  Voxels will be `np.float` with shape
                (num_segments, shapex, shapey, shapez) or will be `int` with shape
                (shapex, shapey, shapez).
            segmentation_map (dict[int, str] | None): map from segment int to segment
                description
            background_index (int | None): segment index corresponding to background (no
                segment).

        Returns:
            list[tuple[list[int], str]]: a list of local_maxima, where each local maxima
                is a 3-dimensional location with a description.

        """

        m_raw: ConfigurationType | ConfigurationValue | None
        m_raw = self.config_get(["output", "local_maxima", "minimum_negative_log10_p"])
        r_raw: ConfigurationType | ConfigurationValue | None
        r_raw = self.config_get(["output", "local_maxima", "cluster_radius"])

        mesgs: list[str]
        mesgs = [
            *(["minimum_negative_log10_p"] if m_raw is None else []),
            *(["cluster_radius"] if r_raw is None else []),
        ]
        if mesgs:
            mesg: str = (
                "Must supply output.local_maxima."
                + " and output.local_maxima.".join(mesgs)
                + "for abcdstats.workflow.Basic.compute_local_maxima."
            )
            raise ValueError(mesg)

        minimum: float
        minimum = cast(float, m_raw)
        radius: int
        radius = cast(int, r_raw)

        shapex: int
        shapey: int
        shapez: int
        shapex, shapey, shapez = variable.shape

        maxima: list[list[int]]
        maxima = [
            [x, y, z]
            for x, y, z in {
                (x, y, int(z))
                for x in range(shapex)
                for y in range(shapey)
                for z in scipy.signal.argrelextrema(
                    variable[x, y, :], np.greater_equal, order=radius
                )[0]
            }
            | {
                (x, int(y), z)
                for x in range(shapex)
                for z in range(shapez)
                for y in scipy.signal.argrelextrema(
                    variable[x, :, z], np.greater_equal, order=radius
                )[0]
            }
            | {
                (int(x), y, z)
                for y in range(shapey)
                for z in range(shapez)
                for x in scipy.signal.argrelextrema(
                    variable[:, y, z], np.greater_equal, order=radius
                )[0]
            }
            if variable[x, y, z] >= minimum
            for neighborhood in [
                variable[
                    max(0, x - radius) : min(shapex, x + radius + 1),
                    max(0, y - radius) : min(shapey, y + radius + 1),
                    max(0, z - radius) : min(shapez, z + radius + 1),
                ]
            ]
            if variable[x, y, z] == np.max(neighborhood)
        ]
        maxima.sort(key=lambda xyz: -variable[xyz[0], xyz[1], xyz[2]])
        return [
            (
                xyz,
                self.describe_maximum(
                    name,
                    xyz,
                    variable,
                    segmentation_voxels,
                    segmentation_map,
                    background_index,
                ),
            )
            for xyz in maxima
        ]

    def describe_maximum(
        self,
        name: str,
        xyz: list[int],
        log10_pvalue: npt.NDArray[np.float64],
        segmentation_voxels: npt.NDArray[np.float64 | int] | None,
        segmentation_map: dict[int, str] | None,
        background_index: int | None,
    ) -> str:
        """Describe a specific local maximum of a specific tested variable,

        Args:
            name (str): name of tested variable being described
            xyz (list[int]): location of voxel.
            log10_pvalue (npt.NDArray[np.float64]): log10 p-values from permuted_ols of
                a specific tested variable and each voxel location.
            segmentation_voxels (npt.NDArray[np.float64 | int] | None): segmentation
                data organized by voxel.  Voxels will be `np.float` with shape
                (num_segments, shapex, shapey, shapez) or will be `int` with shape
                (shapex, shapey, shapez).
            segmentation_map (dict[int, str] | None): map from segment int to segment
                description
            background_index (int | None): segment index corresponding to background (no
                segment).

        Returns:
            str: a local_maximum's description.

        """

        return (
            "No description."
            if segmentation_voxels is None
            or segmentation_map is None
            or background_index is None
            else self.describe_maximum_using_partition(
                name,
                xyz,
                log10_pvalue,
                segmentation_voxels,
                segmentation_map,
                background_index,
            )
            if len(segmentation_voxels.shape) == 3
            else self.describe_maximum_using_cloud(
                name,
                xyz,
                log10_pvalue,
                segmentation_voxels,
                segmentation_map,
                background_index,
            )
            if len(segmentation_voxels.shape) == 4
            else "segmentation_voxels shape error."
        )

    def describe_maximum_using_partition(
        self,
        name: str,  # noqa: ARG002
        xyz: list[int],
        log10_pvalue: npt.NDArray[np.float64],
        segmentation_voxels: npt.NDArray[int],
        segmentation_map: dict[int, str],
        background_index: int,
    ) -> str:
        """Describe a specific local maximum of a specific tested variable using a
        partition (i.e. a 3-dimensional segmentation_voxels).

        Args:
            name (str): name of tested variable being described
            xyz (list[int]): location of voxel.
            log10_pvalue (npt.NDArray[np.float64]): log10 p-values from permuted_ols of
                a specific tested variable and each voxel location.
            segmentation_voxels (npt.NDArray[np.float64 | int] | None): segmentation
                data organized by voxel.  Voxels will be `np.float` with shape
                (num_segments, shapex, shapey, shapez) or will be `int` with shape
                (shapex, shapey, shapez).
            segmentation_map (dict[int, str] | None): map from segment int to segment
                description
            background_index (int | None): segment index corresponding to background (no
                segment).

        Returns:
            str: a local_maximum's description.
        """

        # `radius` describes how far to look for a label.  It is distinct from the
        # cluster_radius that is used to define isolation of a peak.
        radius: int = 3
        x, y, z = xyz
        shapex, shapey, shapez = log10_pvalue.shape
        region_index: int = segmentation_voxels[x, y, z]
        where: str = "in"
        if region_index == background_index:
            # Instead of background, take the most common nearby brain region
            for distance in range(1, radius + 1):
                neighborhood: npt.NDArray[np.float64] = segmentation_voxels[
                    max(0, x - distance) : min(shapex, x + distance + 1),
                    max(0, y - distance) : min(shapey, y + distance + 1),
                    max(0, z - distance) : min(shapez, z + distance + 1),
                ].reshape(-1)
                neighborhood = neighborhood[neighborhood != background_index].astype(
                    int
                )
                if neighborhood.size:
                    values, counts = np.unique(neighborhood, return_counts=True)
                    # Ties go to the lower integer; oh well
                    region_index = values[np.argmax(counts)]
                    where = "near"
                    break
        return (
            f"-log_10 p(t-stat)={round(1000.0 * log10_pvalue[x, y, z]) / 1000.0} at"
            f" ({x}, {y}, {z}) {where} region {segmentation_map[region_index]}"
            f" ({region_index})."
        )

    def describe_maximum_using_cloud(
        self,
        name: str,  # noqa: ARG002
        xyz: list[int],
        log10_pvalue: npt.NDArray[np.float64],
        segmentation_voxels: npt.NDArray[np.float64],
        segmentation_map: dict[int, str],
        background_index: int,
    ) -> str:
        """Describe a specific local maximum of a specific tested variable using a cloud
        (i.e. a 4-dimensional segmentation_voxels).

        Args:
            name (str): name of tested variable being described
            xyz (list[int]): location of voxel.
            log10_pvalue (npt.NDArray[np.float64]): log10 p-values from permuted_ols of
                a specific tested variable and each voxel location.
            segmentation_voxels (npt.NDArray[np.float64 | int] | None): segmentation
                data organized by voxel.  Voxels will be `np.float` with shape
                (num_segments, shapex, shapey, shapez) or will be `int` with shape
                (shapex, shapey, shapez).
            segmentation_map (dict[int, str] | None): map from segment int to segment
                description
            background_index (int | None): segment index corresponding to background (no
                segment).

        Returns:
            str: a local_maximum's description.
        """

        mesg: str
        t_raw: ConfigurationType | ConfigurationValue | None
        t_raw = self.config_get(["output", "local_maxima", "label_threshold"])
        if t_raw is None:
            mesg = "output.local_maxima.label_threshold must be supplied."
            raise ValueError(mesg)
        threshold: float
        threshold = cast(float, t_raw)

        x, y, z = xyz
        shapex, shapey, shapez = log10_pvalue.shape
        cloud_here: npt.NDArray[np.float64] = segmentation_voxels[:, x, y, z]
        argsort: npt.NDArray[np.float64] = np.argsort(cloud_here)[::-1]
        # Show those that exceed threshold; showing at least 2 regions
        description: list[str]
        description = [
            "Found local maximum -log_10 p(t-stat)="
            f"{round(1000.0 * log10_pvalue[x, y, z]) / 1000.0} at ({x}, {y}, {z})"
            " in regions:",
            *[
                f"    {segmentation_map[region]} ({region}) confidence = "
                f"{round(100000.0 * cloud_here[r]) / 1000}%."
                for i, r in enumerate(argsort)
                if i < 2 or cloud_here[r] >= threshold
                for region in [r + background_index + 1]
            ],
        ]

        return "\n".join(description)

    def save_output(
        self,
        *,
        template_voxels: nib.filebasedimages.FileBasedImage | None,
        affine: npt.NDArray[np.float64] | None,
        segmentation_voxels: npt.NDArray[np.float64 | int] | None,  # noqa: ARG002
        segmentation_header: collections.OrderedDict[str, Any] | None,  # noqa: ARG002
        segmentation_map: dict[int, str] | None,  # noqa: ARG002
        permuted_ols: dict[str, npt.NDArray[np.float64]],
        glm_ols: npt.NDArray[np.float64],
        local_maxima_description: dict[str, list[tuple[list[int], str]]],
    ) -> None:
        """Write outputs to files as directed by the configuration

        Args:
            template_voxels (nib.filebasedimages.FileBasedImage | None): background
                image on which to display results.
            affine (npt.NDArray[np.float64] | None): affine transformation to use for
                saved images.
            segmentation_voxels (npt.NDArray[np.float64 | int] | None): information
                about the segmentation, as a partition (storing a segment index (int)
                across 3 dimensions) or as a cloud (storing a probability (np.float64)
                across 4 dimensions; segment index and three spatial dimensions.
            segmentation_header (collections.OrderedDict[str, Any] | None): the entire
                nifti header for the segmentation.
            segmentation_map (dict[int, str] | None): a map of segment index to segment
                label.
            permuted_ols (dict[str, npt.NDArray[np.float64]]): the output from
                nilearn.mass_univariate.permuted_ols after applying
                masker.inverse_transform().
            glm_ols (npt.NDArray[np.float64]): the beta for each tested var for each
                voxel.
            local_maxima_description (dict[str, list[tuple[list[int], str]]]): location
                and description for each tested variable, for each local maximum.

        Returns:
           None

        """

        gamma: float = cast(float, self.config_get(["output", "images", "gamma"]))
        brain_voxels: npt.NDArray[np.float64] | None = None
        if template_voxels is not None:
            brain_voxels = np.asanyarray(template_voxels.dataobj[:, :, :])
            # Scale max to 1, noting that -log10(10%)==1
            brain_voxels = brain_voxels / np.max(brain_voxels)
            # Change brain_voxels to gray in RGB.  Matplotlib expects color to be the
            # last dimension
            brain_voxels = np.stack((brain_voxels,) * 3, axis=-1)

        logp_max_t_all: npt.NDArray[np.float64]
        logp_max_t_all = permuted_ols["logp_max_t"]

        # We'll show three interesting slices for each tested variable, one each of
        # sagittal (x), coronal (y), and axial (z).  Find them for all tested variables
        # simultaneously.
        x_margins: npt.NDArray[np.float64]
        x_margins = logp_max_t_all.sum(axis=(2, 3))
        minX: npt.NDArray[int]
        bestX: npt.NDArray[int]
        maxX: npt.NDArray[int]
        minX, bestX, maxX = self.find_good_slice(x_margins)
        y_margins: npt.NDArray[np.float64]
        y_margins = logp_max_t_all.sum(axis=(1, 3))
        minY: npt.NDArray[int]
        bestY: npt.NDArray[int]
        maxY: npt.NDArray[int]
        minY, bestY, maxY = self.find_good_slice(y_margins)
        z_margins: npt.NDArray[np.float64]
        z_margins = logp_max_t_all.sum(axis=(1, 2))
        minZ: npt.NDArray[int]
        bestZ: npt.NDArray[int]
        maxZ: npt.NDArray[int]
        minZ, bestZ, maxZ = self.find_good_slice(z_margins)

        # TODO: Do we want the output to also include the `desired_modality`?

        index: int
        name: str
        _description: list[tuple[list[int], str]]
        for index, (name, _description) in enumerate(local_maxima_description.items()):
            logp_max_t: npt.NDArray[np.float64]
            logp_max_t = logp_max_t_all[index, :, :, :]
            glm_ols_for_name: npt.NDArray[np.float64]
            glm_ols_for_name = glm_ols[index, :, :, :]  # noqa: F841
            # TODO: Save this `logp_max_t` image for `name`
            # TODO: Save this `glm_ols_for_name` image for `name`
            # TODO: Save this text `description` for `name`, maybe in a text file?
            # RGB image
            brain_voxels = (
                np.zeros((*logp_max_t.shape, 3))
                if brain_voxels is None
                else brain_voxels
            )
            show_voxels = npt.NDArray[np.float64]
            show_voxels = brain_voxels.copy()
            # Add gamma corrected output to the green channel
            show_voxels[:, :, :, 1] += np.power(logp_max_t[:, :, :], gamma)
            show_voxels = np.clip(show_voxels, 0, 1)
            slices_voxels: dict[str, npt.NDArray[np.float64]]
            slices_voxels = self.orient_data_for_slices(affine, show_voxels)
            # TODO: Instead of printing and plotting, save these 2d color images to
            # files.
            # TODO: Remove import of matplotlib as mpl
            slice_2d: npt.NDArray[np.float64]

            print(f"X={bestX[index]} sagittal slice from L (A->P by I->S) for {name!r}")  # noqa: T201
            slice_2d = slices_voxels["sagittal"][bestX[index], :, :, :]
            # matplotlib.pyplot.imshow uses [row, column, color] (i.e., [y, x, color])
            mpl.pyplot.imshow(np.swapaxes(slice_2d, 0, 1), origin="lower")
            mpl.pyplot.show()

            print(f"Y={bestY[index]} coronal slice from A (R->L, I->S) for {name!r}")  # noqa: T201
            slice_2d = slices_voxels["coronal"][bestY[index], :, :, :]
            # matplotlib.pyplot.imshow uses [row, column, color] (i.e., [y, x, color])
            mpl.pyplot.imshow(np.swapaxes(slice_2d, 0, 1), origin="lower")
            mpl.pyplot.show()

            print(f"Z={bestZ[index]} axial slice from I (R->L, P->A) for {name!r}")  # noqa: T201
            slice_2d = slices_voxels["axial"][bestZ[index], :, :, :]
            # matplotlib.pyplot.imshow uses [row, column, color] (i.e., [y, x, color])
            mpl.pyplot.imshow(np.swapaxes(slice_2d, 0, 1), origin="lower")
            mpl.pyplot.show()

    def find_good_slice(
        self, margins: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[int], npt.NDArray[int], npt.NDArray[int]]:
        """
        For each tested variable, we want to show an interesting slice of the 3d-data.
        For example, we choose a slice in the X dimension (i.e., we choose a YZ plane)
            by designating its x coordinate and its lower and upper bounds for y and z.
        First step is summing out over the Y and Z dimensions to compute `margins`,
            which is done before calling find_good_slice().
        (`margins` is 2-dimensional; it is computed from 4-dimensional data with
            shape=(number_tested_variables, size_x, size_y, size_z))
        We then compute:
            for i in range(list_of_tested_variables):
                min_[i] = The lowest x for which margins[i, x] is non-zero
                max_[i] = One more than the largest x for which margins[i, x] is
                    non-zero
                best_[i] = The (first) value of x that maximizes margins[i, :]
        This routine works identically for slices in the Y or Z dimensions, so long as
            margins is supplied by summing out the remaining dimensions.

        Args:
           list_of_keys (list[str]): the location of the desired information, e.g.,
               ["tested_variables", "variable_default", "type"]

        Returns:
           The value of the configuration value, if available, otherwise `None`.
        """
        min_: npt.NDArray[int] = np.argmax(margins > 0.0, axis=-1)
        best_: npt.NDArray[int] = np.argmax(margins, axis=-1)
        max_: npt.NDArray[int] = (
            np.argmax(np.cumsum(margins > 0.0, axis=-1), axis=-1) + 1
        )
        return min_, best_, max_

    def orient_data_for_slices(
        self,
        transform_matrix: npt.NDArray[np.float64],
        data_matrix: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """We make three copies of the data, one for each of sagittal, coronal, or axial
        slices.

        The copy of the data for a kind of slice puts the normal dimension first.  It
        then puts the dimension usually viewed horizontally from left to right as next.
        It puts the dimension usually viewed vertically from bottom to top as
        next.  Finally is the dimension for color / channel.

        In the following L=left, R=right, P=posterior, A=Anterior, I=inferior, and
        S=superior.

        For sagittal slices, axis 0 is L->R or R->L, whichever way it arrives; axis 1 is
        A->P; axis 2 is I->S; axis 3 is color.

        For coronal slices, axis 0 is P->A or A->P, whichever way it arrives; axis 1 is
        R->L; axis 2 is I->S; axis 3 is color

        For axial slices, axis 0 is I->S or S->I, whichever way it arrives; axis 1 is
        R->L; axis 2 is P->A; axis 3 is color

        Args:
           transform_matrix (npt.NDArray[np.float64]): the 3-by-3 part of an affine
               matrix (or the entire 4-by-4 affine matrix) that is used to determine the
               current order of the axes and their orientations relative to
               right-anterior-superior (RAS) coordinates.
           data_matrix (npt.NDArray[np.float64]): the voxel data matrix with dimensions
               [i][j][k][c] where i,j,k, map to R,A,S via the transform_matrix and c is
               the color channel.

        Returns:
           dict[str, npt.NDArray[np.float64]]: for each of "sagittal", "coronal", and
           "axial" returns a view of the input data_matrix but in the standard
           orientation for that kind of slice.

        """

        best_perm: npt.NDArray[int]
        signs: npt.NDArray[bool]
        best_perm, signs = self.best_axis_alignment(transform_matrix)
        # Permute spatial axes to get RAS (or the closest we can get to it); though we
        # have to handle signs too, later.  Keep the color channel last.
        data_matrix = np.transpose(data_matrix, (*best_perm, 3))

        # Sagittal: Want axis 0 to be L->R or R->L, AS IS; axis 1 is A->P; axis 2 is
        # I->S; axis 3 is color
        sagittal_matrix: npt.NDArray[np.float64]
        sagittal_matrix = np.transpose(data_matrix, (0, 1, 2, 3))
        if signs[1]:
            # Switch P->A to A->P
            sagittal_matrix = sagittal_matrix[:, ::-1, :, :]
        if not signs[2]:
            # Switch S->I to I->S
            sagittal_matrix = sagittal_matrix[:, :, ::-1, :]

        # Coronal: Want axis 0 to be P->A or A->P, AS IS; axis 1 is R->L; axis 2 is
        # I->S; axis 3 is color
        coronal_matrix: npt.NDArray[np.float64]
        coronal_matrix = np.transpose(data_matrix, (1, 0, 2, 3))
        if signs[0]:
            # Switch L->R to R->L
            coronal_matrix = coronal_matrix[:, ::-1, :, :]
        if not signs[2]:
            # Switch S->I to I->S
            coronal_matrix = coronal_matrix[:, :, ::-1, :]

        # Axial: Want axis 0 to be I->S or S->I, AS IS; axis 1 is R->L; axis 2 is P->A;
        # axis 3 is color
        axial_matrix: npt.NDArray[np.float64]
        axial_matrix = np.transpose(data_matrix, (2, 0, 1, 3))
        if signs[0]:
            # Switch L->R to R->L
            axial_matrix = axial_matrix[:, ::-1, :, :]
        if not signs[1]:
            # Switch A->P to P->A
            axial_matrix = axial_matrix[:, :, ::-1, :]

        return {
            "sagittal": sagittal_matrix,
            "coronal": coronal_matrix,
            "axial": axial_matrix,
        }

    def best_axis_alignment(
        self, transform_matrix: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[int], npt.NDArray[bool]]:
        """Try all permutations of the spatial axes to see which produces a (permuted)
        transformation matrix most like the identity matrix, and return that combination
        of permutations and signs.

        Args:
           transform_matrix (npt.NDArray[np.float64]): the 3-by-3 part of an affine
               matrix (or the entire 4-by-4 affine matrix) that is used to determine the
               current order of the axes and their orientations relative to
               right-anterior-superior (RAS) coordinates.

        Returns:
           npt.NDArray[int]: a vector that converts axis order to be R (or L), A (or P),
               S (or I) if given to numpy.transpose():
           npt.NDArray[bool]: a vector that gives signs indicating whether the
               transposed axes become R, A, S (all True); or L, P, I (all False); or
               some mixture.

        """

        # The input matrix maps column vector (i,j,k) to column vector (R,A,S).  (It is
        # (i,j,k,1) to (R,A,S,1), if affine.)  In case it is an affine matrix, use just
        # the upper-left 3 by 3.
        transform_matrix = transform_matrix[0:3, 0:3]
        # Normalize the column vectors
        transform_matrix = transform_matrix / np.linalg.norm(transform_matrix, axis=0)
        # Compare permutations of the rows with the identity matrix
        best_perm: tuple[int, ...]
        best_perm_value: float = -100
        perm: tuple[int, ...]
        for perm in itertools.permutations(range(3)):
            new_perm_value: float
            new_perm_value = np.sum(np.abs(transform_matrix[perm, range(3)]))
            if best_perm_value < new_perm_value:
                best_perm_value = new_perm_value
                best_perm = perm
        # For permuted the data, signs == True means R, A, S (respectively); signs ==
        # False means L, P, I.
        signs: npt.NDArray[bool]
        signs = transform_matrix[best_perm, range(3)] > 0.0
        return np.array(best_perm), np.array(signs)

    def config_get(
        self, list_of_keys: list[str]
    ) -> ConfigurationType | ConfigurationValue | None:
        """Read a value from the workflow configuration.

        Args:
           list_of_keys (list[str]): the location of the desired information, e.g.,
               ["tested_variables", "variable_default", "type"]

        Returns:
           The value of the configuration value, if available, otherwise `None`.

        """

        my_dict: ConfigurationType = self.config
        key: str
        for key in list_of_keys[:-1]:
            my_dict = cast(ConfigurationType, my_dict.get(key, {}))
        my_value: ConfigurationType | ConfigurationValue | None
        my_value = my_dict.get(list_of_keys[-1])
        return my_value

    def merge_df_list(
        self, df_list: list[pd.core.frame.DataFrame]
    ) -> pd.core.frame.DataFrame:
        """Merge a list of pandas DataFrames using the self.join_keys

        Args:
            df_list (list[pd.core.frame.DataFrame]): list of dataframes to be merged.

        Returns:
            pd.core.frame.DataFrame: the merged dataframe

        """

        response: pd.core.frame.DataFrame
        if df_list:
            response = df_list[0]
            for next in df_list[1:]:
                response = response.merge(
                    next, on=self.join_keys, how="inner", validate="one_to_one"
                )
        else:
            response = pd.core.frame.DataFrame()
        return response

    def lint(self) -> list[str]:
        """Return messages that describe how the configuration is not well formed

        Args:
            None

        Returns:
            list[str]: a list of messages describing faults in the configuration

        """

        # Fields marked with `"required": True` must be present.

        # Note that some fields are required, but there are two places ("variable" or
        # "variable_default") where they could be supplied, and being missing from just
        # one of them is not an error.  These cases are marked with `"required": False`
        # in the below, which actually does nothing because False is the default.
        # However, this marking is a reminder that we should be doing something to
        # verify that at least one of the places supplies a value.

        # TODO: Enforce this "either or" appropriately

        tested_keys = {
            "filename": {"required": False},
            "convert": {},
            "handle_missing": {
                "values": {"by_value", "invalidate", "separately", "together"}
            },
            "is_missing": {},
            "type": {"required": False, "values": {"ordered", "unordered"}},
            "description": {},
            "internal_name": {},
        }
        confounding_keys = {
            **tested_keys,
            "longitudinal": {
                "required": False,
                "values": {"intercept", "slope", "time"},
            },
            "minimum_perplexity": {},
        }
        schema = {
            "keys": {
                "version": {"required": True, "values": {"1.0"}},
                "tested_variables": {
                    "required": True,
                    "keys": {
                        "source_directory": {},
                        "variable_default": {"keys": tested_keys},
                        "variable": {
                            "required": True,
                            "default_keys": {"keys": tested_keys},
                        },
                    },
                },
                "target_variables": {
                    "required": True,
                    "keys": {
                        "source_directory": {},
                        "desired_modality": {"required": True, "values": {"fa", "md"}},
                        "table_of_filenames_and_metadata": {},
                        "individual_filenames_and_metadata": {
                            "default_keys": {
                                "keys": {
                                    "filename": {"required": True},
                                    "src_subject_id": {"required": True},
                                    "eventname": {
                                        "required": True,
                                        "values": {
                                            "baseline_year_1_arm_1",
                                            "1_year_follow_up_y_arm_1",
                                            "2_year_follow_up_y_arm_1",
                                            "3_year_follow_up_y_arm_1",
                                            "4_year_follow_up_y_arm_1",
                                        },
                                    },
                                    "modality": {
                                        "required": True,
                                        "values": {"fa", "md"},
                                    },
                                    "description": {},
                                }
                            }
                        },
                        "mask": {
                            "keys": {"filename": {"required": True}, "threshold": {}}
                        },
                        "segmentation": {
                            "keys": {"filename": {"required": True}, "background": {}}
                        },
                        "template": {"keys": {"filename": {"required": True}}},
                    },
                },
                "confounding_variables": {
                    "required": True,
                    "keys": {
                        "source_directory": {},
                        "variable_default": {"keys": confounding_keys},
                        "variable": {
                            "required": True,
                            "default_keys": {"keys": confounding_keys},
                        },
                    },
                },
                "output": {
                    "required": True,
                    "keys": {
                        "destination_directory": {"required": True},
                        "local_maxima": {
                            "keys": {
                                "minimum_negative_log10_p": {},
                                "cluster_radius": {},
                            }
                        },
                    },
                },
            }
        }

        fields: list[str] = self.check_fields(schema)
        files: list[str] = self.check_files(schema)
        return [*fields, *files]

    def check_fields(self, schema: dict[str, Any]) -> list[str]:
        """Top-level check that used configuration keys are permitted and required keys
        are present.

        Args:
            schema (dict[str, Any]): Description of what keys are permitted and
            required.

        Returns:
            list[str]: a list of messages describing faults in the configuration

        """

        return self.recursive_check_fields([], self.config, schema)

    def recursive_check_fields(
        self,
        context: list[str],
        config: dict[str, Any] | list[Any],
        schema: dict[str, Any],
    ) -> list[str]:
        """Recursively called function check that used configuration keys are permitted
        and required keys are present.

        Args:
            context (list[str]): list of keys we've already descended through
            config (dict[str, Any] | list[Any]): sub-tree of the configuration
            schema (dict[str, Any]): sub-tree of the schema

        Returns:
            list[str]: a list of messages describing faults in the configuration

        The purpose of this routine is to check that the top-level keys in `config` and
            `schema` validate properly, and to recurse more deeply as appropriate.
        A `config` is a dict[str, Any], representing the entire YAML file or,
            recursively, a part of it.
        A `schema` is a `dict[str, Any]` with up to four keys: "required", "keys",
            "default_keys", and "values".
            * The value associated with the "required" key is a `bool`.  If missing it
                is interpreted as False.
            * The value associated with the "keys" key is a `dict[str, schema]`, which
                is one schema per key.
            * The value associated with the "default_keys" key is a `schema`.  If
                present, this is the schema applicable to any key that does not have a
                schema within schema["keys"]
            * The value associated with the "values" key is a `set[Any]`, which is the
                set of legal values for the key.  If the key is absent, all values are
                legal.
            * "Required" will already have been checked at the top-level, but we need to
                check it just before recursing to a schema from "keys" or "default_keys"

        """

        key: str
        value: Any
        new_context: list[str]
        new_values: set[Any] | None
        new_schema: dict[str, Any]
        response: list[str] = []
        config = (
            {"Entry_" + str(k): v for k, v in dict(enumerate(config)).items()}
            if isinstance(config, list)
            else config
        )
        for key, value in config.items():
            new_context = [*context, key]
            # If "values" is present then check that value is valid
            new_values = schema.get("values")
            if new_values is not None and value not in new_values:
                response = [
                    *response,
                    f'Value {value!r} of key {".".join(new_context)} must be one of'
                    f" {new_values!r}.",
                ]
            # If there is an applicable schema then recurse
            new_schema = (
                schema.get("keys", {})
                .get(key, {})
                .get("schema", schema.get("default_keys", {}).get("schema"))
            )
            if new_schema is not None:
                response = [
                    *response,
                    *self.recursive_check_fields(new_context, value, new_schema),
                ]
        for key, value in schema.items():
            new_context = [*context, key]
            # If a key is required but not present then report that
            if value.get("required", False) and key not in config:
                response = [
                    *response,
                    f'Key {".".join(new_context)} is required but was not provided.',
                ]

        return response

    def check_files(self, schema: dict[str, Any]) -> list[str]:
        """Check that files and directories in the configuration exist and can be
        accessed.

        Args:
            schema (dict[str, Any]): Description of what keys are permitted and
                required, including fields named "filename", etc.

        Returns:
            list[str]: a list of messages describing faults in the configuration

        """

        # For each source_directory, destination_directory, and filename check whether
        # it exists.  (Filenames may be relative to the specified directories.) For
        # table_of_filenames_and_metadata, check existence and validity.

        response: list[str] = []
        input_raw: ConfigurationType | ConfigurationValue | None

        # tested = ksads
        # target = voxels
        # confounding = interview_age, etc.
        for input in ["tested", "target", "confounding"]:
            # source_directory can be provided for tested, target, and confounding
            input_raw = self.config_get([input + "_variables", "source_directory"])
            input_source_directory: pathlib.Path | None
            input_source_directory = (
                pathlib.Path(cast(str, input_raw)) if input_raw is not None else None
            )

            if input != "target":
                response = [
                    *response,
                    *self.check_file_internal(
                        input_source_directory,
                        self.config_get(
                            [input + "_variables", "variable_default", "filename"]
                        ),
                    ),
                ]

                input_raw = self.config_get([input + "_variables", "variable"])
                for variable in cast(dict[str, Any], input_raw):
                    response = [
                        *response,
                        *self.check_file_internal(
                            input_source_directory,
                            self.config_get(
                                [input + "_variables", "variable", variable, "filename"]
                            ),
                        ),
                    ]

            if input == "target":
                input_raw = self.config_get(
                    ["target_variables", "individual_filenames_and_metadata"]
                )
                if input_raw is not None:
                    input_list: list[dict[str, Any]]
                    input_list = cast(list[dict[str, Any]], input_raw)
                    for file_dict in input_list:
                        response = [
                            *response,
                            *self.check_file_internal(
                                input_source_directory, file_dict.get("filename")
                            ),
                        ]
                input_raw = self.config_get(
                    ["target_variables", "table_of_filenames_and_metadata"]
                )
                response = [
                    *response,
                    *self.check_csv(input_source_directory, input_raw, schema),
                ]

        input_raw = self.config_get(["output", "destination_directory"])
        if input_raw is None:
            response = [*response, "output.destination_directory must be supplied."]
        else:
            destination_path: pathlib.Path
            destination_path = pathlib.Path(cast(str, input_raw))
            if not (destination_path.is_dir() and os.access(destination_path, os.W_OK)):
                response = [
                    *response,
                    f"output.destination_directory {destination_path} does not exist"
                    " or is not writable.",
                ]

        return response

    def check_file_internal(
        self,
        directory: pathlib.Path | None,
        filename_raw: ConfigurationType | ConfigurationValue | None,
    ) -> list[str]:
        """Check that a specific file exists and is readable

        Args:
            directory (pathlib.Path | None): Optional directory for relative paths
            filename_raw (ConfigurationType | ConfigurationValue | None): location of
                file, as a full path or a path relative to the directory.

        Returns:
        Returns:
            list[str]: a list of messages describing faults in the configuration

        """

        if filename_raw is None:
            return []
        filename: pathlib.Path = pathlib.Path(cast(str, filename_raw))
        if directory is not None:
            filename = directory / filename
        path: pathlib.Path = pathlib.Path(filename)
        if not (path.is_file() and os.access(path, os.R_OK)):
            return [f"Cannot read {filename}."]
        # All is good
        return []

    def check_csv(
        self,
        directory: pathlib.Path | None,
        filename_raw: ConfigurationType | ConfigurationValue | None,
        schema: dict[str, Any],
    ) -> list[str]:
        """Check that the CSV file is well formed and perform a similar check on any
        individual_filenames_and_metadata specified in the YAML file.

        Args:
            directory (pathlib.Path | None): Optional directory for relative paths
            filename_raw (ConfigurationType | ConfigurationValue | None): location of
                file, as a full path or a path relative to the directory.
            schema (dict[str, Any]): Description of what keys are permitted and
                required.

        Returns:
            list[str]: a list of messages describing faults in the configuration

        """

        if filename_raw is None:
            return []
        filename: pathlib.Path = pathlib.Path(cast(str, filename_raw))
        if directory is not None:
            filename = directory / filename
        path: pathlib.Path = pathlib.Path(filename)
        if not (path.is_file() and os.access(path, os.R_OK)):
            return [f"Cannot read {filename}."]
        # Sanity check contents
        csv: pd.core.frame.DataFrame = pd.read_csv(path)
        expected: list[str] = [
            "filename",
            "src_subject_id",
            "eventname",
            "modality",
            "description",
        ]
        if list(csv.columns) != expected:
            return [
                f"CSV file {filename} has columns {list(csv.columns)} but should have"
                f" columns {expected}."
            ]
        config: list[dict[str, Any]]
        config = csv.to_dict(orient="records")
        self.recursive_check_fields(
            ["target_variables", "table_of_filenames_and_metadata"],
            config,
            schema["keys"]["target_variables"]["keys"][
                "individual_filenames_and_metadata"
            ],
        )
        # All is good
        return []

    def pandas_replace(
        self, series: pd.core.series.Series, src: Any, dst: Any
    ) -> pd.core.series.Series:
        """Implement a panda Series.replace command such that replacement occurs if
        values are numerically equal, even if they are not of the same type.

        Args:
            series (pd.core.series.Series): The pandas series on which to perform
                the replace operation.
            src (Any): the value being replaced.  If it is numeric then any value that
                is numerically equivalent will be replaced regardless of its type in
                (str, bool, int, float, np.int64, np.float64).
            dst (Any): the replacement value

        Returns:
            pd.core.series.Series: the modified pandas Series.

        """

        series = series.replace(src, dst)
        # Convert even if the type is different
        for t in (str, bool, int, float, np.int64, np.float64):
            try:
                if float(src) == float(t(src)) or (
                    math.isnan(float(src)) and math.isnan(float(t(src)))
                ):
                    series = series.replace(t(src), dst)
            except (TypeError, ValueError):
                pass

        return series
