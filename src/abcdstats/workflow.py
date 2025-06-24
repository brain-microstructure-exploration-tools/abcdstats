from __future__ import annotations

import copy
import pathlib
import re
from typing import Any, TypeAlias, Union, cast

import nibabel as nib  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.maskers  # type: ignore[import-not-found,import-untyped,unused-ignore]
import numpy as np  # type: ignore[import-not-found,import-untyped,unused-ignore]
import numpy.typing as npt  # type: ignore[import-not-found,import-untyped,unused-ignore]
import pandas as pd  # type: ignore[import-not-found,import-untyped,unused-ignore]
import yaml  # type: ignore[import-not-found,import-untyped,unused-ignore]

BasicValue: TypeAlias = str | int | float
ConfigurationValue: TypeAlias = BasicValue | list[Any]
ConfigurationType: TypeAlias = dict[str, Union[ConfigurationValue, "ConfigurationType"]]


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
            "convert": [],
            "handle_missing": "invalidate",
            "is_missing": ["", np.nan],
        }
        self.config_default: ConfigurationType = {
            "tested_variables": {"variable_default": variable_default},
            "target_variables": {
                "filename_pattern": r"^.*\.nii(\.gz)?$",
                "segmentation": {"background_index": 0},
            },
            "confounding_variables": {
                "minimum_perplexity": 1.0,
                "variable_default": {**variable_default, "longitudinal": ["intercept"]},
            },
        }
        self.config: ConfigurationType
        self.configure(yaml_file=None)

        if yaml_file is not None:
            self.configure(yaml_file=yaml_file)

    def copy_keys_into(
        self, *, src: ConfigurationType, dest: ConfigurationType
    ) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and key in dest and isinstance(dest[key], dict):
                self.copy_keys_into(src=value, dest=cast(ConfigurationType, dest[key]))
            else:
                dest[key] = copy.deepcopy(value)

    def configure(self, *, yaml_file: str | pathlib.Path | None) -> None:
        if yaml_file is not None:
            with pathlib.Path(yaml_file).open("r", encoding="utf-8") as file:
                self.copy_keys_into(src=yaml.safe_load(file), dest=self.config)
        else:
            # The user requests the system defaults
            self.config = copy.deepcopy(self.config_default)

    def run(self) -> None:
        """
        TODO: Check that each affine_transform produced is np.all_close with the first
        TODO: Make sure that the images are in the same order for each numpy array
        """

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        # TODO: Note that YAML file will specify "fa" vs. "md" images
        source_images_voxels: list[nib.filebasedimages.FileBasedImage]
        source_images_affine_transform: npt.NDArray[np.float64]
        source_images_metadata: pd.core.frame.DataFrame
        source_images_voxels, source_images_affine_transform, source_images_metadata = (
            self.get_source_images()
        )

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        source_mask_voxels: npt.NDArray[np.bool_]
        source_mask_affine_transform: npt.NDArray[np.float64]
        source_mask_masker: nilearn.maskers.NiftiMasker
        source_mask_voxels, source_mask_affine_transform, source_mask_masker = (
            self.get_source_mask()
        )

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        whole_brain_voxels: npt.NDArray[np.float64]
        whole_brain_affine_transform: npt.NDArray[np.float64]
        whole_brain_voxels, whole_brain_affine_transform = self.get_whole_brain()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        brain_segmentation_voxels: npt.NDArray[np.float64 | np.int_]
        brain_segmentation_affine_transform: npt.NDArray[np.float64]
        # Can be a labelmap (all-or-none segmentation) or a cloud that gives
        # probabilities for each segment.
        brain_segmentation_voxels, brain_segmentation_affine_transform = (
            self.get_brain_segmentation()
        )

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        tested_data_frame: pd.core.frame.DataFrame
        tested_data_array: npt.NDArray[np.float64]
        # Assembly can include conversion to one-hot, as well as use as intercept or
        # slope random effects.
        tested_data_frame, tested_data_array = self.get_tested_data()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        confounding_data_frame: pd.core.frame.DataFrame
        confounding_data_array: npt.NDArray[np.float64]
        # Assembly can include conversion to one-hot, as well as use as intercept or
        # slope random effects.
        confounding_data_frame, confounding_data_array = self.get_confounding_data()

        # Process inputs to compute statistically significant voxels.
        # TODO: Don't forget to use the source_mask_voxels to mask target_vars.  (TODO:
        # Or use masker and threshold?)
        permuted_ols: dict[str, npt.NDArray[np.float64]]
        glm_ols: npt.NDArray[np.float64]
        permuted_ols, glm_ols = self.compute_significant_voxels(
            tested_vars=tested_data_array,
            target_vars=source_images_voxels,
            confounding_vars=confounding_data_array,
            masker=source_mask_masker,
        )
        logp_max_t: npt.NDArray[np.float64] = permuted_ols["logp_max_t"]

        # Process output to compute local maxima.
        # For each tested variable, for each local maximum, output coordinates and a
        # description
        local_maxima_description: list[list[tuple[list[int], str]]] = (  # noqa: F841
            self.compute_local_maxima(logp_max_t=logp_max_t)
        )
        # TODO: Invoke matplotlib

    def get_source_images(
        self,
    ) -> tuple[
        list[nib.filebasedimages.FileBasedImage],
        npt.NDArray[np.float64],
        pd.core.frame.DataFrame,
    ]:
        # TODO: Recognize whether filename is relative or absolute and treat it
        #       accordingly

        # Throw exception if these fields are needed but not supplied
        directory: pathlib.Path = pathlib.Path(
            cast(
                str,
                cast(ConfigurationType, self.config["target_variables"])[
                    "source_directory"
                ],
            )
        )
        pattern: str = cast(
            str,
            cast(ConfigurationType, self.config["target_variables"])[
                "filename_pattern"
            ],
        )
        filename_list: list[pathlib.Path] = [
            directory / filename
            for filename in directory.iterdir()
            if bool(re.match(pattern, str(filename)))
        ]
        # TODO: Build metadata from filenames???
        source_images_metadata: pd.core.frame.DataFrame = pd.core.frame.DataFrame()
        source_images_voxels: list[nib.filebasedimages.FileBasedImage] = [
            nib.load(path)  # type: ignore[attr-defined,unused-ignore]
            for path in filename_list  # type: ignore[attr-defined,unused-ignore]
        ]
        # TODO: Check that there is at least one image with at least one voxel
        # TODO: Check that shapes match
        # TODO: Check that affines match
        source_images_affine_transform: npt.NDArray[np.float64] = source_images_voxels[
            0
        ].affine  # type: ignore[attr-defined,unused-ignore]

        # Need this later
        # number_values: int = sum([math.prod(img.shape) for img in source_images_voxels])
        # max_values_per_iteration: int = 1_000_000_000
        # number_iterations = math.ceil(number_values / max_values_per_iteration)

        return (
            source_images_voxels,
            source_images_affine_transform,
            source_images_metadata,
        )

    def get_source_mask(
        self,
    ) -> tuple[
        npt.NDArray[np.bool_], npt.NDArray[np.float64], nilearn.maskers.NiftiMasker
    ]:
        # TODO: Write me
        return (
            np.zeros((), dtype=bool),
            np.zeros((), dtype=np.float64),
            nilearn.maskers.NiftiMasker(),
        )

    def get_whole_brain(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # TODO: Write me
        return np.zeros((), dtype=np.float64), np.zeros((), dtype=np.float64)

    def get_brain_segmentation(
        self,
    ) -> tuple[npt.NDArray[np.float64 | np.int_], npt.NDArray[np.float64]]:
        # TODO: Write me
        return np.zeros((), dtype=np.float64), np.zeros((), dtype=np.float64)

    def get_tested_data(
        self,
    ) -> tuple[pd.core.frame.DataFrame, npt.NDArray[np.float64]]:
        # TODO: Write me
        return pd.core.frame.DataFrame(), np.zeros((), dtype=np.float64)

    def get_confounding_data(
        self,
    ) -> tuple[pd.core.frame.DataFrame, npt.NDArray[np.float64]]:
        # TODO: Write me
        return pd.core.frame.DataFrame(), np.zeros((), dtype=np.float64)

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
          target_vars.shape == (number_images, number_voxels)
          confounding_vars.shape == (number_images, number_confounding_vars)
        """
        # TODO: Change target_vars to be a list of lazy-loaded nibabel nifti images.
        #       Once we run permuted_ols on some voxels, we'll want to release the
        #       memory for those voxels to make room for the next set of voxels.
        # TODO: Write me
        return {}, np.zeros((), dtype=np.float64)

    def compute_local_maxima(
        self,
        *,
        logp_max_t: npt.NDArray[np.float64],  # noqa: ARG002
    ) -> list[list[tuple[list[int], str]]]:
        # TODO: Write me
        return []
