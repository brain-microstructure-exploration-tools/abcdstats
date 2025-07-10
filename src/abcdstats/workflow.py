from __future__ import annotations

import collections
import copy
import math
import pathlib
from typing import Any, TypeAlias, Union, cast

import nibabel as nib  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.maskers  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.masking  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nrrd  # type: ignore[import-not-found,import-untyped,unused-ignore]
import numpy as np  # type: ignore[import-not-found,import-untyped,unused-ignore]
import numpy.typing as npt  # type: ignore[import-not-found,import-untyped,unused-ignore]
import pandas as pd  # type: ignore[import-not-found,import-untyped,unused-ignore]
import yaml  # type: ignore[import-not-found,import-untyped,unused-ignore]

BasicValue: TypeAlias = str | int | float
ConfigurationValue: TypeAlias = BasicValue | list[Any]
ConfigurationType: TypeAlias = dict[str, Union[ConfigurationValue, "ConfigurationType"]]


# TODO: Check that there are no extra fields in our YAML file

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
            "tested_variables": {
                "variable_default": variable_default,
            },
            "target_variables": {
                # TODO: Do we still use filename_pattern?
                "filename_pattern": r"^.*\.nii(\.gz)?$",
                "segmentation": {"background_index": 0},
            },
            "confounding_variables": {
                "variable_default": {**variable_default, "longitudinal": ["intercept"]},
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
        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        tested_frame: pd.core.frame.DataFrame
        # Assembly can include conversion to one-hot.
        tested_frame = self.get_tested_data()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        target_frame: pd.core.frame.DataFrame
        target_frame = self.get_target_data()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        confounding_frame: pd.core.frame.DataFrame
        # Assembly can include conversion to one-hot, as well as use as intercept or
        # slope random effects.
        confounding_frame = self.get_confounding_data()

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
        ) = self.make_arrays(tested_frame, target_frame, confounding_frame)

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        mask_masker: nilearn.maskers.NiftiMasker | None
        mask_affine: npt.NDArray[np.float64] | None
        mask_masker, mask_affine = self.get_source_mask()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        background_voxels: nib.filebasedimages.FileBasedImage | None
        background_affine: npt.NDArray[np.float64] | None
        background_voxels, background_affine = self.get_whole_brain()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        segmentation_voxels: npt.NDArray[np.float64 | np.int_] | None
        segmentation_affine: npt.NDArray[np.float64] | None
        segmentation_header: collections.OrderedDict[str, Any] | None
        # Can be a labelmap (all-or-none segmentation) or a cloud that gives
        # probabilities for each segment.
        segmentation_voxels, segmentation_affine, segmentation_header = (
            self.get_brain_segmentation()
        )

        self.allclose_affines(
            [target_affine, mask_affine, background_affine, segmentation_affine],
            "Supplied affine transformations for"
            " target, mask, background, and segmentation must match",
        )

        # Process inputs to compute statistically significant voxels.
        permuted_ols: dict[str, npt.NDArray[np.float64]]
        glm_ols: npt.NDArray[np.float64]
        permuted_ols, glm_ols = self.compute_significant_voxels(
            tested_vars=tested_array,
            target_vars=target_images,
            confounding_vars=confounding_array,
            masker=mask_masker,
        )
        logp_max_t: npt.NDArray[np.float64] = permuted_ols["logp_max_t"]

        # Process output to compute local maxima.
        # For each tested variable, for each local maximum, output coordinates and a
        # description
        local_maxima_description: list[list[tuple[list[int], str]]]
        local_maxima_description = self.compute_local_maxima(logp_max_t=logp_max_t)  # noqa: F841
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
                " or `individual_filenames_and_metadata`"
            )
            raise KeyError(mesg)

        # Prepend directory to table location if appropriate
        if table is not None and directory is not None:
            table = directory / table
        # Convert individual filenames to pathlib.Path and prepend directory
        if individuals is not None:
            individuals = [
                i | {"filename": pathlib.Path(i["filename"])} for i in individuals
            ]
            if directory is not None:
                individuals = [
                    i | {"filename": directory / i["filename"]} for i in individuals
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
            mesg = "The target images are not all the same shape"
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

    def get_tested_data(self) -> pd.core.frame.DataFrame:
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

        df_var: pd.core.frame.DataFrame
        variables, df_var = self.fetch_variables(directory, variable_default, variables)

        return df_var

    def get_confounding_data(self) -> pd.core.frame.DataFrame:
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

        df_var: pd.core.frame.DataFrame
        variables, df_var = self.fetch_variables(directory, variable_default, variables)
        df_var = self.handle_longitudinal(variables, df_var)

        return df_var

    def fetch_variables(
        self,
        directory: pathlib.Path | None,
        variable_default: dict[str, Any],
        variables: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], pd.core.frame.DataFrame]:
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
        dict_df_var: dict[str, pd.core.frame.DataFrame]
        dict_df_var = {
            fn: pd.read_csv(fn)[
                *self.join_keys,
                *{
                    var_config["internal_name"]
                    for var_config in variables.values()
                    if var_config["filename"] == fn
                },
            ]
            for fn in {var_config["filename"] for var_config in variables.values()}
        }

        # Switch to user-specified names.  Do this before merging tables in case of
        # internal_name collisions.
        dict_df_var = {
            fn: df_var.rename(
                columns={
                    var_config["internal_name"]: variable
                    for variable, var_config in variables.items()
                    if var_config["filename"] == fn
                    and var_config["internal_name"] != variable
                }
            )
            for fn, df_var in dict_df_var.items()
        }

        # Merge tables into one
        table_names: list[str] = list(dict_df_var.keys())
        df_var: pd.core.frame.DataFrame
        df_var = dict_df_var[table_names[0]]
        for t in table_names[1:]:
            df_var = df_var.merge(
                dict_df_var[t], on=self.join_keys, how="inner", validate="one_to_one"
            )
        del dict_df_var

        # Apply convert, handle_missing and is_missing
        for variable, var_config in variables.items():
            convert: dict[str, Any]
            convert = var_config["convert"]
            handle_missing: str
            handle_missing = var_config["handle_missing"]
            is_missing: list[Any]
            is_missing = var_config["is_missing"]
            var_type: str
            var_type = var_config["type"]
            minimum_perplexity: float
            minimum_perplexity = var_config["minimum_perplexity"]

            for src, dst in convert.items():
                df_var[variable] = df_var[variable].replace(src, dst)

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
                    # We will interpret missing as 0, but add a one-hot column so that
                    # it can effectively be any constant.  In particular, it will not be
                    # confounded with actual values of 0.
                    new_missing_name: str = variable + "_missing"
                    if new_missing_name in df_var.columns:
                        mesg = (
                            "Failed to get unique column name for"
                            f" {new_missing_name!r}."
                        )
                        raise ValueError(mesg)
                    df_var[new_missing_name] = (
                        df_var[variable] == is_missing[0]
                    ).astype(int)
                    df_var.loc[df_var[variable] == is_missing[0], variable] = 0
            if handle_missing == "by_value":
                if var_type == "unordered":
                    # There is nothing more to do.
                    pass
                if var_type == "ordered":
                    mesg = (
                        "We do not currently handle the case that"
                        ' handle_missing == "by_value" and var_type == "ordered"'
                    )
                    raise ValueError(mesg)
            if handle_missing == "separately":
                # We want to use unused distinct values for each type of "missing".  We
                # add unique values so the pd.get_dummies call creates a one-hot column
                # for each missing value.
                unused_numeric_value: int = 1 + max(
                    [int(x) for x in df_var[variable] if isinstance(x, int | float)]
                    + [0]
                )
                number_needed_values: int = (df_var[variable] == is_missing[0]).sum()
                if var_type == "unordered":
                    df_var.loc[df_var[variable] == is_missing[0], variable] = range(
                        unused_numeric_value,
                        unused_numeric_value + number_needed_values,
                    )
                if var_type == "ordered":
                    mesg = (
                        "We do not currently handle the case that"
                        ' handle_missing == "separately" and var_type == "ordered"'
                    )
                    raise ValueError(mesg)

            # Convert categorical data to a multicolumn one-hot representation
            if var_type == "unordered":
                df_var = pd.get_dummies(
                    df_var, dummy_na=True, columns=[variable], drop_first=False
                )

            # Check perplexity
            df_var = self.enforce_perplexity(df_var, variable, minimum_perplexity)
        return variables, df_var

    def enforce_perplexity(
        self, df_var: pd.core.frame.DataFrame, variable: str, min_perp: float
    ) -> pd.core.frame.DataFrame:
        counts: dict[Any, int]
        counts = dict(df_var[variable].value_counts(dropna=False).astype(int))
        total: int
        total = sum(counts.values())
        perp: float
        perp = math.prod(
            (total / count) ** (count / total) for count in counts.values() if count > 0
        )
        if perp < min_perp and not math.isclose(perp, min_perp):
            df_var = df_var.drop(variable, axis=1)

        return df_var

    def handle_longitudinal(
        self, variables: dict[str, dict[str, Any]], df_var: pd.core.frame.DataFrame
    ) -> pd.core.frame.DataFrame:
        kinds: list[str] = ["time", "intercept", "slope"]

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
            mesg: str = "\n".join(mesgs)
            raise ValueError(mesg)

        # TODO: If some columns have been transformed (e.g., via one-hot), make sure
        #       that we apply longitudinal considerations to them regardless.

        df_new: dict[str, pd.core.frame.DataFrame]
        df_new = {
            kind: df_var[[*self.join_keys, *has[kind]]].copy(deep=True)
            for kind in kinds
        }

        if len(has["time"]) > 0:
            new_time_name: str = "ubhzaeZTE3McmbxX"
            if any(new_time_name in df_new[kind].columns for kind in kinds):
                mesg = "Failed to get unique column name for df_time"
                raise ValueError(mesg)
            df_new["time"].rename(columns={has["time"][0]: new_time_name})
            # TODO: Rename non-join-key columns of df_new["slope"], including checking
            #       that they have not already been used
            df_new["slope"] = df_new["slope"].merge(
                df_new["time"], on=self.join_keys, how="inner", validate="one_to_one"
            )
            # TODO: Multiply new_time_name column into each slope column
            # TODO: Keep only slope columns (and join keys)
            # TODO: Merge slope columns into df_new["intercept"]

        # TODO: Write me
        return df_var

    def make_arrays(
        self,
        tested_frame: pd.core.frame.DataFrame,
        target_frame: pd.core.frame.DataFrame,
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

        # TODO: We should dropna sooner, e.g., before the perplexity test
        all_frame = all_frame.dropna()

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

    def get_whole_brain(
        self,
    ) -> tuple[
        nib.filebasedimages.FileBasedImage | None, npt.NDArray[np.float64] | None
    ]:
        d_raw: ConfigurationType | ConfigurationValue | None
        d_raw = self.config_get(["target_variables", "source_directory"])
        directory: pathlib.Path | None
        directory = pathlib.Path(cast(str, d_raw)) if d_raw is not None else None

        f_raw: ConfigurationType | ConfigurationValue | None
        f_raw = self.config_get(["target_variables", "background", "filename"])
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
        npt.NDArray[np.float64 | np.int_] | None,
        npt.NDArray[np.float64] | None,
        collections.OrderedDict[str, Any] | None,
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

        voxels: npt.NDArray[np.float64 | np.int_] | None = None
        header: collections.OrderedDict[str, Any] | None = None
        affine: npt.NDArray[np.float64] | None = None
        if filename is not None:
            voxels, header = nrrd.read(filename)  # shape = (71, 140, 140, 140)
            header = cast(collections.OrderedDict[str, Any], header)
            affine = self.construct_affine(header)

        return voxels, affine, header

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
        tested_vars: npt.NDArray[np.float64],  # noqa: ARG002
        target_vars: list[nib.filebasedimages.FileBasedImage],  # noqa: ARG002
        confounding_vars: npt.NDArray[np.float64],  # noqa: ARG002
        masker: nilearn.maskers.NiftiMasker,  # noqa: ARG002
    ) -> tuple[dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """
        Shapes of the numpy arrays are
          tested_vars.shape == (number_images, number_ksads)
          target_vars.shape == (number_images, *voxels.shape)
          confounding_vars.shape == (number_images, number_confounding_vars)

        Need this later
          number_values: int
          number_values = sum([math.prod(img.shape) for img in source_images_voxels])
          max_values_per_iteration: int = 1_000_000_000
          number_iterations = math.ceil(number_values / max_values_per_iteration)
        """
        # TODO: Should we handle multiple channel data too?  If so, how?
        # TODO: Verify that masker is doing what we hope it is doing
        # TODO: Write me
        return {}, np.zeros((), dtype=np.float64)

    def compute_local_maxima(
        self,
        *,
        logp_max_t: npt.NDArray[np.float64],  # noqa: ARG002
    ) -> list[list[tuple[list[int], str]]]:
        # TODO: Write me
        return []

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
