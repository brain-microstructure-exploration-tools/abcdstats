from __future__ import annotations

import pathlib
from typing import Dict, List, Union

import numpy as np
import yaml
from typing_extensions import TypeAlias

BasicValue: TypeAlias = Union[str, int, float]
ConfigurationValue: TypeAlias = Union[BasicValue, List[BasicValue]]
ConfigurationType: TypeAlias = Dict[str, Union[ConfigurationValue, "ConfigurationType"]]

sample_configuration: ConfigurationType = {
    "tested_variables": {
        "source_file": "/data2/ABCD/abcd-5.0-tabular-data-extracted/core/mental-health/mh_y_ksads_ss.csv",
        "variable_default": {
            "convert": {"555": np.nan, "888": 0},
            "handle_missing": "invalidate",
            "type": "unordered",
            "is_missing": ["", np.nan],
        },
        "variable": {
            "ksads_1_187_t": {
                "description": "Symptom - No two month symptom-free interval, Present",
                "internal_name": "ksads_1_187_t",
            },
            "ksads_1_188_t": {"description": "Symptom - No two month symptom-free interval, Past"},
            "ksads_22_142_t": {"description": "Symptom - Insomnia, Past"},
            "ksads_22_970_t": {"description": "Diagnosis - SLEEP PROBLEMS, Past"},
            "ksads_2_11_t": {"description": "Symptom - Explosive Irritability, Past"},
        },
    },
    "target_variables": {
        "source_directory": "/data2/ABCD/gor-images/coregistered-images",
        "subtype": "fa",
        "mask": {"file": "/data2/ABCD/gor-images/gortemplate0.nii.gz", "threshold": 0.7},
        "segmentation": {
            "file": "/home/local/KHQ/lee.newberg/git/brain-microstructure-exploration-tools/abcd-data-exploration/prototype/segmentation_data_mrtrix.seg.nrrd",
            "background_index": 0,
        },
    },
    "confounding_variables": {
        "source_directory": "/data2/ABCD/abcd-5.0-tabular-data-extracted/core",
        "minimum_perplexity": 1.1,
        "variable_default": {"handle_missing": "invalidate", "type": "unordered", "longitudinal": ["intercept"]},
        "variable": {
            "interview_age": {
                "file": "abcd-general/abcd_y_lt.csv",
                "internal_name": "interview_age",
                "type": "ordered",
                "longitudinal": ["intercept", "slope"],
            },
            "site_id_l": {"file": "abcd-general/abcd_y_lt.csv", "handle_missing": "together"},
            "demo_gender_id_v2": {
                "file": "gender-identity-sexual-health/gish_p_gi.csv",
                "convert": {"777": "", "999": ""},
            },
            "rel_family_id": {"file": "abcd-general/abcd_y_lt.csv", "handle_missing": "together"},
        },
    },
    "output": {"destination_directory": "TODO:", "local_maxima": {"minimum_peak": 0.1, "minimum_radius": 3}},
}

# Write it out
filename: str = "sample.yaml"
with pathlib.Path(filename).open("w") as file:
    yaml.dump(sample_configuration, file, default_flow_style=False, sort_keys=False)
