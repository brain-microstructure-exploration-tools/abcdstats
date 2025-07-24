from __future__ import annotations

import collections
import copy
import math
import os
import pathlib
from typing import Any, TypeAlias, Union, cast

import nibabel as nib  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.maskers  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.masking  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nrrd  # type: ignore[import-not-found,import-untyped,unused-ignore]
import numpy as np  # type: ignore[import-not-found,import-untyped,unused-ignore]
import numpy.typing as npt  # type: ignore[import-not-found,import-untyped,unused-ignore]
import pandas as pd  # type: ignore[import-not-found,import-untyped,unused-ignore]
import scipy.signal  # type: ignore[import-not-found,import-untyped,unused-ignore]
import yaml  # type: ignore[import-not-found,import-untyped,unused-ignore]

BasicValue: TypeAlias = bool | int | float | str | None
ConfigurationValue: TypeAlias = BasicValue | list[Any]
ConfigurationType: TypeAlias = dict[str, Union[ConfigurationValue, "ConfigurationType"]]

# TODO: Break up long functions if there are meaningful subfunctions


class Basic:
    """
    This is a working workflow that is configurable via a YAML file.  Changes not
    achievable via configuration can be made by subclassing this class and overriding
    relevant methods.
    """

    def __init__(self, *, yaml_file: str | pathlib.Path | None = None) -> None:
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
                    "n_perm": 10000,
                    "two_sided_test": True,
                    "random_state": None,
                    "n_jobs": -1,  # All available
                    "verbose": 1,
                    "tfce": False,
                    "threshold": None,
                    "output_type": "dict",
                },
            },
        }

        # First, initialize with defaults
        self.config: ConfigurationType
        self.configure(yaml_file=None)
        # Second, use user-supplied values from file, if any
        if yaml_file is not None:
            self.configure(yaml_file=yaml_file)

    def configure(self, *, yaml_file: str | pathlib.Path | None) -> None:
        if yaml_file is not None:
            with pathlib.Path(yaml_file).open("r", encoding="utf-8") as file:
                self.copy_keys_into(src=yaml.safe_load(file), dest=self.config)
        else:
            # The user requests the system defaults
            self.config = copy.deepcopy(self.config_default)

    def copy_keys_into(
        self, *, src: ConfigurationType, dest: ConfigurationType
    ) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and key in dest and isinstance(dest[key], dict):
                self.copy_keys_into(src=value, dest=cast(ConfigurationType, dest[key]))
            else:
                dest[key] = copy.deepcopy(value)

    def run(self) -> None:
        # Validate the supplied YAML file
        warnings: list[str] = self.lint()
        if warnings:
            # TODO: Log these instead of print?
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
        local_maxima_description: list[list[tuple[list[int], str]]]
        local_maxima_description = self.compute_local_maxima(  # noqa: F841
            logp_max_t=logp_max_t,
            segmentation_voxels=segmentation_voxels,
            segmentation_map=segmentation_map,
            background_index=background_index,
        )
        # TODO: Invoke matplotlib

    def get_target_data(self) -> pd.core.frame.DataFrame:
        table: pathlib.Path | None
        individuals: list[dict[str, Any]] | None
        table, individuals = self.get_target_data_table_and_individuals()

        target_frame: pd.core.frame.DataFrame
        target_frame = self.get_target_data_frame(table, individuals)

        return target_frame

    def get_target_data_table_and_individuals(
        self,
    ) -> tuple[pathlib.Path | None, list[dict[str, Any]] | None]:
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
        # Prepend directory to individual filenames if appropriate
        if individuals is not None and directory is not None:
            individuals = [
                i | {"filename": str(directory / pathlib.Path(i["filename"]))}
                for i in individuals
            ]

        return table, individuals

    def get_target_data_frame(
        self, table: pathlib.Path | None, individuals: list[dict[str, Any]] | None
    ) -> pd.core.frame.DataFrame:
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
        by_var: dict[str, tuple[ConfigurationType, pd.core.frame.DataFrame]]
        by_var = {
            variable: self.handle_variable_config(variable, var_config, df_by_file)
            for variable, var_config in variables.items()
        }
        variables = {
            **variables,
            **{
                variable: cast(dict[str, Any], var_config)
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
    ) -> tuple[ConfigurationType, pd.core.frame.DataFrame]:
        convert: dict[str, Any]
        convert = var_config["convert"]
        handle_missing: str
        handle_missing = var_config["handle_missing"]
        is_missing: list[Any]
        is_missing = var_config["is_missing"]
        var_type: str
        var_type = var_config["type"]
        df_var: pd.core.frame.DataFrame
        df_var = df_by_file[var_config["filename"]][[*self.join_keys, variable]]

        # Apply convert, handle_missing and is_missing
        for src, dst in convert.items():
            df_var[variable] = df_var[variable].replace(src, dst)

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
                df_var[variable] = df_var[variable].replace(val, is_missing[0])
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
            if var_type == "ordered":
                mesg = (
                    "We do not currently handle the case that"
                    ' handle_missing == "separately" and var_type == "ordered".'
                )
                raise ValueError(mesg)

        # Convert categorical data to a multicolumn one-hot representation.  Note that
        # setting drop_first=True will drop one of the categories; and in such a case,
        # we should drop the single, "missing" category if there is such a thing.

        if var_type == "unordered":
            # Note: if an unordered variable has high perplexity but its individual
            # categories (in one-hot representation) do not, the categories with low
            # perplexity will ultimately be removed.
            df_var_columns: set[str]
            df_var_columns = set(df_var.columns)
            df_var = pd.get_dummies(
                df_var, dummy_na=True, columns=[variable], drop_first=False
            )

        # Each newly created column, if any, is a new variable, so prepare an entry for
        # `variables` by copying from its origin variable.
        variables: ConfigurationType
        variables = {
            new_variable: {**var_config, "internal_name": new_variable}
            for new_variable in set(df_var.columns) - df_var_columns
        }

        return variables, df_var

    def enforce_perplexity(
        self, df_var: pd.core.frame.DataFrame, variables: dict[str, dict[str, Any]]
    ) -> pd.core.frame.DataFrame:
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
            df_intercept_by_var = {v: df_by_var[v] for v in has["intercept"]}

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
        confound_variables: dict[str, dict[str, Any]],
        confound_frame: pd.core.frame.DataFrame,
    ) -> tuple[
        pd.core.frame.DataFrame,
        pd.core.frame.DataFrame,
        pd.core.frame.DataFrame,
        npt.NDArray[np.float64],
        list[nib.filebasedimages.FileBasedImage],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        # Note that the join with target_frame is "one_to_many" because there could be
        # multiple images associated with a particular (src_subject_id, eventname) pair.
        # For example, they could be images with different modalities (such as FA
        # vs. MD).
        all_frame: pd.core.frame.DataFrame
        all_frame = tested_frame.merge(
            confound_frame, on=self.join_keys, how="inner", validate="one_to_one"
        ).merge(target_frame, on=self.join_keys, how="inner", validate="one_to_many")

        all_frame = all_frame.dropna()

        # Now that we have determined which rows are actually going to be processed,
        # let's remove columns that do not meet the perplexity requirement.
        variables: dict[str, dict[str, Any]]
        variables = {**tested_variables, **confound_variables}
        all_frame = self.enforce_perplexity(all_frame, variables)

        # Recreate the input frames, respecting any selection, any replication, and any
        # reordering of rows to produce all_frame
        tested_frame = all_frame[tested_frame.columns]
        tested_keys: set[str]
        tested_keys = set(tested_frame.columns) - set(self.join_keys)
        tested_array: npt.NDArray[np.float64]
        tested_array = tested_frame[tested_keys].to_numpy(dtype=np.float64)

        target_frame = all_frame[target_frame.columns]
        target_keys: set[str]
        target_keys = set(target_frame.columns) - set(self.join_keys)
        target_images: list[nib.filebasedimages.FileBasedImage]
        target_affine: npt.NDArray[np.float64]
        target_images, target_affine = self.get_target_data_voxels_and_affine(
            target_frame[target_keys]
        )

        confound_frame = all_frame[confound_frame.columns]
        confound_keys: set[str]
        confound_keys = set(confound_frame.columns) - set(self.join_keys)
        confound_array: npt.NDArray[np.float64]
        confound_array = confound_frame[confound_keys].to_numpy(dtype=np.float64)

        return (
            tested_frame,
            target_frame,
            confound_frame,
            tested_array,
            target_images,
            target_affine,
            confound_array,
        )

    def get_source_mask(
        self,
    ) -> tuple[nilearn.maskers.NiftiMasker | None, npt.NDArray[np.float64] | None]:
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

        voxels: npt.NDArray[np.float64 | int] | None = None
        header: collections.OrderedDict[str, Any] | None = None
        affine: npt.NDArray[np.float64] | None = None
        segmentation_map: dict[int, str] | None = None
        if filename is not None:
            voxels, header = nrrd.read(filename)  # shape = (71, 140, 140, 140)
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
        mesg: str
        """
        TODO: Maybe we'll need this later:
          number_values: int
          number_values = sum([math.prod(img.shape) for img in source_images_voxels])
          max_values_per_iteration: int = 1_000_000_000
          number_iterations = math.ceil(number_values / max_values_per_iteration)
        """

        target_vars: npt.NDArray[np.float64]
        # Avoid img.get_fdata() so that we have just one copy of the voxel data
        target_vars = np.stack([np.asanyarray(img.dataobj) for img in target_images])

        """
        Shapes of the numpy arrays are
          tested_vars.shape == (number_images, number_ksads)
          target_vars.shape == (number_images, *voxels.shape)
          confounding_vars.shape == (number_images, number_confounding_vars)
        """

        # TODO: If masker is correct except for the number of channels, can we recover?
        if masker is not None and target_vars.shape[1:] != masker.mask_img_.shape:
            mesg = (
                f"The shape of each target image {target_vars.shape[1:]} and the"
                f" shape of the mask {masker.mask_img_.shape} must match."
            )
            raise ValueError(mesg)

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
        # TODO: Verify that masker is doing what we hope it is doing
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
            masker=masker,
            **other_parameters,
        )

        # In each case, compute the regression coefficient that yielded the t-stat and
        # p-value reported by permuted_ols
        bool_mask: npt.NDArray[bool]
        bool_mask = (
            np.asanyarray(masker.mask_img_.dataobj, dtype=bool)
            if masker is not None
            else np.ones(target_vars.shape[1:], dtype=bool)
        )
        kept_target_vars: npt.NDArray[np.float64]
        kept_target_vars = target_vars[:, bool_mask]

        glm_ols_response: npt.NDArray = np.zeros(target_vars.shape, dtype=np.float64)
        glm_ols_response[:, bool_mask] = np.vstack(
            [
                nilearn.glm.OLSModel(
                    np.hstack((tested_var.reshape(-1, 1), confounding_vars))
                )
                .fit(kept_target_vars)
                .theta[0, :]
                for tested_var in tested_vars.T
            ]
        )

        return permuted_ols_response, glm_ols_response

    def compute_local_maxima(
        self,
        *,
        logp_max_t: npt.NDArray[np.float64],
        segmentation_voxels: npt.NDArray[np.float64 | int] | None,
        segmentation_map: dict[int, str] | None,
        background_index: int | None,
    ) -> list[list[tuple[list[int], str]]]:
        return [
            self.compute_local_maxima_for_variable(
                variable, segmentation_voxels, segmentation_map, background_index
            )
            for variable in logp_max_t
        ]

    def compute_local_maxima_for_variable(
        self,
        variable: npt.NDArray,
        segmentation_voxels: npt.NDArray[np.float64 | int] | None,
        segmentation_map: dict[int, str] | None,
        background_index: int | None,
    ) -> list[tuple[list[int], str]]:
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
        xyz: list[int],
        log10_pvalue: npt.NDArray,
        segmentation_voxels: npt.NDArray[np.float64 | int] | None,
        segmentation_map: dict[int, str] | None,
        background_index: int | None,
    ) -> str:
        return (
            "No description."
            if segmentation_voxels is None
            or segmentation_map is None
            or background_index is None
            else self.describe_maximum_using_partition(
                xyz,
                log10_pvalue,
                segmentation_voxels,
                segmentation_map,
                background_index,
            )
            if len(segmentation_voxels.shape) == 3
            else self.describe_maximum_using_cloud(
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
        xyz: list[int],
        log10_pvalue: npt.NDArray,
        segmentation_voxels: npt.NDArray[int],
        segmentation_map: dict[int, str],
        background_index: int,
    ) -> str:
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
                neighborhood: npt.NDArray = segmentation_voxels[
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
        xyz: list[int],
        log10_pvalue: npt.NDArray,
        segmentation_voxels: npt.NDArray[np.float64],
        segmentation_map: dict[int, str],
        background_index: int,
    ) -> str:
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
        cloud_here: npt.NDArray = segmentation_voxels[:, x, y, z]
        argsort: npt.NDArray = np.argsort(cloud_here)[::-1]
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

    def config_get(
        self, list_of_keys: list[str]
    ) -> ConfigurationType | ConfigurationValue | None:
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
        # Note that required tested_keys and confound_keys have some fields labeled as
        # `"required": False` (which does nothing because False is the default) because
        # they must be supplied as either a "variable" or "variable_default", but need
        # not be both.
        # TODO: Can we enforce this "either or" appropriately?
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
        confound_keys = {
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
                                    "event_name": {
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
                        "variable_default": {"keys": confound_keys},
                        "variable": {
                            "required": True,
                            "default_keys": {"keys": confound_keys},
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
        return self.recursive_check_fields([], self.config, schema)

    def recursive_check_fields(
        self,
        context: list[str],
        config: dict[str, Any] | list[Any],
        schema: dict[str, Any],
    ) -> list[str]:
        # The purpose of this routine is to check that the top-level keys in `config`
        #   and `schema` validate properly, and to recurse more deeply as appropriate.
        # A `config` is a dict[str, Any], representing the entire YAML file or,
        #   recursively, a part of it.
        # A `schema` is a `dict[str, Any]` with up to four keys: "required", "keys",
        #   "default_keys", and "values".
        #   * The value associated with the "required" key is a `bool`.  If missing it
        #     is interpreted as False.
        #   * The value associated with the "keys" key is a `dict[str, schema]`, which
        #     is one schema per key.
        #   * The value associated with the "default_keys" key is a `schema`.  If
        #     present, this is the schema applicable to any key that does not have a
        #     schema within schema["keys"]
        #   * The value associated with the "values" key is a `set[Any]`, which is the
        #     set of legal values for the key.  If the key is absent, all values are
        #     legal.
        #   * "Required" will already have been checked at the top-level, but we need to
        #     check it just before recursing to a schema from "keys" or "default_keys".

        # The return value is a list of failed checks, each expressed as a warning
        # message.

        key: str
        value: Any
        new_context: list[str]
        new_values: set[Any]
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
            new_values = schema.get("values", {value})
            if value not in new_values:
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
        elif not pathlib.Path(cast(str, input_raw)).is_dir():
            response = [
                *response,
                f"output.destination_directory {input_raw} does not exist.",
            ]

        return response

    def check_file_internal(
        self,
        directory: pathlib.Path | None,
        filename_raw: ConfigurationType | ConfigurationValue | None,
    ) -> list[str]:
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
            "event_name",
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
