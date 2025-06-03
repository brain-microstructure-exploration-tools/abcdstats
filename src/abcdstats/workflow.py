from __future__ import annotations

import pathlib

import nilearn.maskers
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml


class Basic:
    """
    This is a working workflow that is configurable via yaml file.  Changes not achievable via the YAML file can be made
    by subclassing this class and overriding relevant methods.
    """

    def __init__(self, *, yaml_file: str | pathlib.Path | None = None) -> None:
        if yaml_file is not None:
            self.configure(yaml_file=yaml_file)

    def configure(self, *, yaml_file: str | pathlib.Path) -> None:
        with pathlib.Path(yaml_file).open("r") as f:
            self.config = yaml.safe_load(f)

    def run(self) -> None:
        """
        TODO: Check that each affine_transform produced is np.all_close with the first
        TODO: Make sure that the images are in the same order for each numpy array
        tested_input.shape == (number_images, number_ksads)
        target_input.shape == (number_images, number_voxels)
        confounding_input.shape == (number_images, number_confounding_vars)
        """

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        # TODO: Note that YAML file will specifiy "fa" vs. "md" images
        source_images_voxels: npt.NDArray[np.float64]
        source_images_affine_transform: npt.NDArray[np.float64]
        source_images_metadata: pd.core.frame.DataFrame
        source_images_voxels, source_images_affine_transform, source_images_metadata = self.getSourceImages()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        source_mask_voxels: npt.NDArray[np.bool_]
        source_mask_affine_transform: npt.NDArray[np.float64]
        source_mask_masker: nilearn.maskers.NiftiMasker
        source_mask_voxels, source_mask_affine_transform, source_mask_masker = self.getSourceMask()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        whole_brain_voxels: npt.NDArray[np.float64]
        whole_brain_affine_transform: npt.NDArray[np.float64]
        whole_brain_voxels, whole_brain_affine_transform = self.getWholeBrain()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        brain_segmentation_voxels: npt.NDArray[np.float64 | np.int_]
        brain_segmentation_affine_transform: npt.NDArray[np.float64]
        # Can be a labelmap (all-or-none segmentation) or a cloud that gives probablities for each segment.
        brain_segmentation_voxels, brain_segmentation_affine_transform = self.getBrainSegmentation()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        tested_data_frame: pd.core.frame.DataFrame
        tested_data_array: npt.NDArray[np.float64]
        # Assembly can include conversion to one-hot, as well as use as intercept or slope random effects.
        tested_data_frame, tested_data_array = self.getTestedData()

        # Fetch, check, assemble, and clean the data as directed by the YAML file.
        confounding_data_frame: pd.core.frame.DataFrame
        confounding_data_array: npt.NDArray[np.float64]
        # Assembly can include conversion to one-hot, as well as use as intercept or slope random effects.
        confounding_data_frame, confounding_data_array = self.getConfoundingData()

        # Process inputs to compute statistically significant voxels.
        # TODO: Don't forget to use the source_mask_voxels to mask target_vars.  (TODO: Or use masker and threshold?)
        permuted_ols: dict[str, npt.NDArray[np.float64]]
        glm_ols: npt.NDArray[np.float64]
        permuted_ols, glm_ols = self.computeSignificantVoxels(
            tested_vars=tested_data_array,
            target_vars=source_images_voxels,
            confounding_vars=confounding_data_array,
            masker=source_mask_masker,
        )
        logp_max_t: npt.NDArray[np.float64] = permuted_ols["logp_max_t"]

        # Process output to compute local maxima.
        # For each tested variable, for each local maximum, output coordinates and a description
        local_maxima_description: list[list[tuple[list[int], str]]]
        local_maxima_description = self.computeLocalMaxima(logp_max_t=logp_max_t)

        # TODO: Invoke matplotlib

    def getSourceImages(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], pd.core.frame.DataFrame]:
        # TODO: Write me
        pass

    def getSourceMask(self) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64], nilearn.maskers.NiftiMasker]:
        # TODO: Write me
        pass

    def getWholeBrain(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # TODO: Write me
        pass

    def getBrainSegmentation(self) -> tuple[npt.NDArray[np.float64 | np.int_], npt.NDArray[np.float64]]:
        # TODO: Write me
        pass

    def getTestedData(self) -> tuple[pd.core.frame.DataFrame, npt.NDArray[np.float64]]:
        # TODO: Write me
        pass

    def getConfoundingData(self) -> tuple[pd.core.frame.DataFrame, npt.NDArray[np.float64]]:
        # TODO: Write me
        pass

    def computeSignificantVoxels(
        self,
        *,
        tested_vars: npt.NDArray[np.float64],
        target_vars: npt.NDArray[np.float64],
        confounding_vars: npt.NDArray[np.float64],
        masker: nilearn.maskers.NiftiMasker,
    ) -> tuple[dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        # TODO: Write me
        pass

    def computeLocalMaxima(self, *, logp_max_t: npt.NDArray[np.float64]) -> list[list[tuple[list[int], str]]]:
        # TODO: Write me
        pass
