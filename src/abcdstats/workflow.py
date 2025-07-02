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
import SimpleItK as sitk  # type: ignore[import-not-found,import-untyped,unused-ignore]
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
        whole_brain_voxels: nib.filebasedimages.FileBasedImage | None
        whole_brain_affine_transform: npt.NDArray[np.float64] | None
        whole_brain_voxels, whole_brain_affine_transform = self.get_whole_brain()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        brain_segmentation_voxels: npt.NDArray[np.float64 | np.int_] | None
        brain_segmentation_affine_transform: npt.NDArray[np.float64] | None
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
        local_maxima_description: list[list[tuple[list[int], str]]]
        local_maxima_description = self.compute_local_maxima(logp_max_t=logp_max_t)  # noqa: F841
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

        metadata: pd.core.frame.DataFrame
        metadata = self.get_source_images_metadata(table, individuals)

        voxels: list[nib.filebasedimages.FileBasedImage]
        affine_transform: npt.NDArray[np.float64]
        voxels, affine_transform = self.get_source_images_voxels_and_affine(metadata)

        return voxels, affine_transform, metadata

    def get_source_images_table_and_individuals(
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
        if directory is not None and table is not None:
            table = directory / table
        if directory is not None and individuals is not None:
            for entry in individuals:
                if "filename" in entry:
                    entry["filename"] = str(directory / pathlib.Path(entry["filename"]))
        return table, individuals

    def get_source_images_metadata(
        self, table: pathlib.Path | None, individuals: list[dict[str, Any]] | None
    ) -> pd.core.frame.DataFrame:
        metadata: pd.core.frame.DataFrame = pd.concat(
            [
                pd.read_csv(table) if table is not None else None,
                pd.DataFrame(individuals) if individuals is not None else None,
            ],
            ignore_index=True,
        )
        return metadata

    def get_source_images_voxels_and_affine(
        self, metadata: pd.core.frame.DataFrame
    ) -> tuple[list[nib.filebasedimages.FileBasedImage], npt.NDArray[np.float64]]:
        mesg: str
        # Create table of metadata to describe each input file

        # Create nib.filebasedimages.FileBasedImage for each input file.  Note that the
        # order of the elements in `voxels` must match the order of the rows in
        # `metadata`.
        voxels: list[nib.filebasedimages.FileBasedImage]
        voxels = [nib.load(path) for path in metadata["filename"].tolist()]

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
        affine: npt.NDArray[np.float64]
        affine = all_affines[0] if len(all_affines) > 0 else None
        return voxels, affine

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
        image: nib.filebasedimages.FileBasedImage | None
        image = nib.load(filename) if filename is not None else None
        affine: npt.NDArray[np.float64] | None
        affine = image.affine if image is not None else None
        t_raw: ConfigurationType | ConfigurationValue | None
        t_raw = self.config_get(["target_variables", "mask", "threshold"])
        threshold: float | None
        threshold = cast(float, t_raw) if t_raw is not None else None
        masker: nilearn.maskers.NiftiMasker | None
        masker = (
            nilearn.maskers.NiftiMasker(
                nilearn.masking.compute_brain_mask(
                    target_img=image, threshold=threshold
                )
            )
            if image is not None and threshold is not None
            else None
        )
        if masker is not None:
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
        npt.NDArray[np.float64 | np.int_] | None, npt.NDArray[np.float64] | None
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

        image: sitk.SimpleITK.Image | None
        image = sitk.ReadImage(filename) if filename is not None else None
        voxels: npt.NDArray[np.float64 | np.int_] | None
        # Index order of ITK is reverse of what we are using
        voxels = (
            sitk.GetArrayFromImage(image).transpose(2, 1, 0)
            if image is not None
            else None
        )

        affine: npt.NDArray[np.float64] | None
        if image is not None:
            # Verified on a test case that this calculation gives the same as slicerio's ijkToLPS
            origin: npt.NDArray[np.float64]
            origin = np.array(image.GetOrigin(), dtype=np.float64)
            spacing: npt.NDArray[np.float64]
            spacing = np.array(image.GetSpacing(), dtype=np.float64)
            direction: npt.NDArray[np.float64]
            direction = np.array(image.GetDirection(), dtype=np.float64)
            dim: int = image.GetDimension()
            direction_matrix: npt.NDArray[np.float64] = direction.reshape((dim, dim))
            affine = np.eye(dim + 1, dtype=np.float64)
            affine[:dim, :dim] = direction_matrix * spacing  # scale columns
            affine[:dim, dim] = origin
            # Convert LPS to RAS by changing sign of first two rows
            affine[:2] *= -1.0
        else:
            affine = None

        return voxels, affine

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
