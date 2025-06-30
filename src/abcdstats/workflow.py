from __future__ import annotations

import copy
import pathlib
from typing import Any, TypeAlias, Union, cast

import nibabel as nib  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.maskers  # type: ignore[import-not-found,import-untyped,unused-ignore]
import nilearn.masking  # type: ignore[import-not-found,import-untyped,unused-ignore]
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
        source_images_voxels: list[nib.filebasedimages.FileBasedImage]
        source_images_affine_transform: npt.NDArray[np.float64]
        source_images_metadata: pd.core.frame.DataFrame
        source_images_voxels, source_images_affine_transform, source_images_metadata = (
            self.get_source_images()
        )

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        source_mask_masker: nilearn.maskers.NiftiMasker | None
        source_mask_affine_transform: npt.NDArray[np.float64] | None
        source_mask_masker, source_mask_affine_transform = self.get_source_mask()

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
        table: pathlib.Path | None
        individuals: list[dict[str, Any]] | None
        table, individuals = self.get_source_images_table_and_individuals()

        metadata: pd.core.frame.Dataframe = self.get_source_images_metadata(
            table, individuals
        )

        voxels: list[nib.filebasedimages.FileBasedImage]
        affine_transform: npt.NDArray[np.float64]
        voxels, affine_transform = self.get_voxels_and_affine(metadata)

        return voxels, affine_transform, metadata

    def get_source_images_table_and_individuals(
        self,
    ) -> tuple[pathlib.Path | None, list[dict[str, Any]] | None]:
        mesg: str
        targets: ConfigurationType
        targets = cast(ConfigurationType, self.config["target_variables"])

        # Find source filenames and their metadata
        source_directory: pathlib.Path | None
        source_directory = (
            pathlib.Path(cast(str, targets["source_directory"]))
            if "source_directory" in targets
            else None
        )
        table: pathlib.Path | None
        table = (
            pathlib.Path(cast(str, targets["table_of_filenames_and_metadata"]))
            if "table_of_filenames_and_metadata" in targets
            else None
        )
        # We will be modifying and returning `individuals` so make a deepcopy
        individuals: list[dict[str, Any]] | None
        individuals = (
            cast(
                list[dict[str, Any]],
                copy.deepcopy(targets["individual_filenames_and_metadata"]),
            )
            if "individual_filenames_and_metadata" in targets
            else None
        )
        if table is None and individuals is None:
            mesg = (
                "Must supply at least one of `table_of_filenames_and_metadata`"
                " and `individual_filenames_and_metadata`"
            )
            raise KeyError(mesg)
        if source_directory is not None and table is not None:
            table = source_directory / table
        if source_directory is not None and individuals is not None:
            for entry in individuals:
                if "filename" in entry:
                    entry["filename"] = str(
                        source_directory / pathlib.Path(entry["filename"])
                    )
        return table, individuals

    def get_source_images_metadata(
        self, table: pathlib.Path | None, individuals: list[dict[str, Any]] | None
    ) -> pd.core.frame.DataFrame:
        metadata: pd.core.frame.Dataframe = pd.concat(
            [
                pd.read_csv(table) if table is not None else None,
                pd.Dataframe(individuals) if individuals is not None else None,
            ],
            ignore_index=True,
        )
        return metadata

    def get_voxels_and_affine(
        self, metadata: pd.core.frame.DataFrame
    ) -> tuple[list[nib.filebasedimages.FileBasedImage], npt.NDArray[np.float64]]:
        mesg: str
        # Create table of metadata to describe each input file

        # Create nib.filebasedimages.FileBasedImage for each input file.  Note that the
        # order of the elements in `voxels` must match the order of the rows in
        # `metadata`.
        voxels: list[nib.filebasedimages.FileBasedImage] = [
            nib.load(path)  # type: ignore[attr-defined,unused-ignore]
            for path in metadata["filename"].tolist()
        ]

        # Check that shapes match
        all_shapes: list[tuple[int, ...]] = [img.shape for img in voxels]
        if not all(all_shapes[0] == all_shapes[i] for i in range(1, len(all_shapes))):
            mesg = "The target images are not all the same shape"
            raise ValueError(mesg)

        # Find the common affine transformation
        all_affines: list[npt.NDArray[np.float64]] = [img.affine for img in voxels]
        if not all(
            np.allclose(all_affines[0], all_affines[i])
            for i in range(1, len(all_affines))
        ):
            mesg = (
                "The affine transformation matrices for the images do not all match;"
                " are the images registered?"
            )
            raise ValueError(mesg)
        affine_transform: npt.NDArray[np.float64] = (
            all_affines[0] if len(all_affines) > 0 else None
        )
        return voxels, affine_transform

    def get_source_mask(
        self,
    ) -> tuple[nilearn.maskers.NiftiMasker | None, npt.NDArray[np.float64] | None]:
        targets: ConfigurationType
        targets = cast(ConfigurationType, self.config["target_variables"])
        source_directory: pathlib.Path | None
        source_directory = (
            pathlib.Path(cast(str, targets["source_directory"]))
            if "source_directory" in targets
            else None
        )
        mask_config: ConfigurationType | None
        mask_config = (
            cast(ConfigurationType, targets["mask"]) if "mask" in targets else None
        )
        mask_filename: pathlib.Path | None
        mask_filename = (
            pathlib.Path(cast(str, mask_config["filename"]))
            if mask_config is not None and "filename" in mask_config
            else None
        )
        if source_directory is not None and mask_filename is not None:
            mask_filename = source_directory / mask_filename
        mask_image: nib.filebasedimages.FileBasedImage | None
        mask_image = nib.load(mask_filename) if mask_filename is not None else None
        mask_affine: npt.NDArray[np.float64] | None
        mask_affine = mask_image.affine if mask_image is not None else None
        mask_threshold: float | None
        mask_threshold = (
            cast(float, mask_config["threshold"])
            if mask_config is not None and "threshold" in mask_config
            else None
        )
        masker: nilearn.maskers.NiftiMasker | None
        masker = (
            nilearn.maskers.NiftiMasker(
                nilearn.masking.compute_brain_mask(
                    target_img=mask_image, threshold=mask_threshold
                )
            )
            if mask_image is not None and mask_threshold is not None
            else None
        )
        if masker is not None:
            masker.fit()

        return masker, mask_affine

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

        Need this later
          number_values: int = sum([math.prod(img.shape) for img in source_images_voxels])
          max_values_per_iteration: int = 1_000_000_000
          number_iterations = math.ceil(number_values / max_values_per_iteration)
        """
        # TODO: Should we handle multiple channel data too?
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
