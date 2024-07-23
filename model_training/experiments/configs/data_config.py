import numpy as np
import ml_collections


ACT_MEAN = [
    1.9296819e-04,
    1.3667766e-04,
    -1.4583133e-04,
    -1.8390431e-04,
    -3.0808983e-04,
    2.7425270e-04,
    5.9716219e-01,
]

ACT_STD = [
    0.00912848,
    0.0127196,
    0.01229497,
    0.02606696,
    0.02875283,
    0.07807977,
    0.48710242,
]

ACT_MIN = [
    -0.0437546,
    -0.052831028,
    -0.035931006,
    -0.14489305,
    -0.15591072,
    -0.26039174,
    -0.780331,
]  # 0.1% quantile

ACT_MAX = [
    0.04158026,
    0.05223833,
    0.05382493,
    0.15559858,
    0.142592,
    0.25956747,
    0.79311615,
]  # 99.9% quantile

ACTION_PROPRIO_METADATA = {
    "action": {
        "mean": np.array(ACT_MEAN),
        "std": np.array(ACT_STD),
        "min": np.array(ACT_MIN),
        "max": np.array(ACT_MAX),
    },
    # TODO compute these
    "proprio": {
        "mean": np.array(ACT_MEAN),
        "std": np.array(ACT_STD),
        "min": np.array(ACT_MIN),
        "max": np.array(ACT_MAX),
    }
}

def get_config(config_string):
    possible_structures = {
        "all": ml_collections.ConfigDict(
            {
                "pretraining_data": [
                    "gs://rail-orca-central2/resize_256_256/bridge_dataset/1.0.0"
                ],
                "autonomous_data": [
                    "gs://autonomous-improvement/rlds_data/robot_4_drawer"
                ],
                "exclude": [],
                "sampling_weights": {
                    "pretraining_data": 0.0,
                    "autonomous_data_successes": 0.5,
                    "autonomous_data_failures": 0.5,
                },
            }
        ),
    }
    return possible_structures[config_string]
