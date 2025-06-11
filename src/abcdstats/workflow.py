from __future__ import annotations

import copy
import pathlib
from typing import Dict, List, Union

import nilearn.maskers
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from typing_extensions import TypeAlias

BasicValue: TypeAlias = Union[str, int, float]
ConfigurationValue: TypeAlias = Union[BasicValue, List[BasicValue]]
ConfigurationType: TypeAlias = Dict[str, Union[ConfigurationValue, "ConfigurationType"]]


class Basic:
    """
    This is a working workflow that is configurable via a YAML file.  Changes not achievable via configuration can be
    made by subclassing this class and overriding relevant methods.
    """

    def __init__(self, *, yaml_file: str | pathlib.Path | None = None) -> None:
        # globally useful values
        self.join_keys: list[str] = ["src_subject_id", "eventname"]
        self.default_config = {}
        """
        # If we want to supply defaults for the user, we could use the following:
        variable_default = {"convert": [], "handle_missing": "invalidate", "is_missing": ["", np.nan]}
        self.default_config = {
            "tested_variables": {
                "variable_default": variable_default,
            },
            "target_variables": {
                "segmentation": {"background_index": 0},
            },
            "confounding_variables": {
                "minimum_perplexity": 1.0,
                "variable_default": {**variable_default, "longitudinal": ["intercept"]},
            },
        }
        """
        self.config = copy.deepcopy(self.default_config)

        if yaml_file is not None:
            self.configure(yaml_file=yaml_file)

    def copy_keys_into(self, *, src: ConfigurationType, dest: ConfigurationType) -> None:
        # Note: this is not a deep copy
        for key, value in src.items():
            if isinstance(value, ConfigurationType) and key in dest and isinstance(dest[key], ConfigurationType):
                self.copy_keys_into(src=value, dest=dest[key])
            else:
                dest[key] = value

    def configure(self, *, yaml_file: str | pathlib.Path, clear: bool = False) -> None:
        if clear:
            self.config = copy.deepcopy(self.default_config)
        with pathlib.Path(yaml_file).open("r") as f:
            self.copy_keys_into(src=yaml.safe_load(f), dest=self.config)

    def run(self) -> None:
        """
        TODO: Check that each affine_transform produced is np.all_close with the first
        TODO: Make sure that the images are in the same order for each numpy array
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
        # TODO: Load them lazily, e.g. with nibabel.load (using nibabel.ArrayProxy), so that if we call permuted_ols for
        #       only some voxels at a time, not all voxels need to be read in each time.
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
        """
        Shapes of the numpy arrays are
          tested_vars.shape == (number_images, number_ksads)
          target_vars.shape == (number_images, number_voxels)
          confounding_vars.shape == (number_images, number_confounding_vars)
        """
        # TODO: Change target_vars to be a list of lazy-loaded nibabel nifti images.  Once we run permuted_ols on some
        #       voxels, we'll want to release the memory for those voxels to make room for the next set of voxels.
        # TODO: Write me
        pass

    def computeLocalMaxima(self, *, logp_max_t: npt.NDArray[np.float64]) -> list[list[tuple[list[int], str]]]:
        # TODO: Write me
        pass
