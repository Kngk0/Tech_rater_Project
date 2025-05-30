import numpy as np

def generate_data():

    # 0D Array: Bias for the tech
    bias = np.array(1, dtype='i4') # 1: Positive, 0: Neutral, -1: Negative

    # 1D Array: labels for the tech
    techs_labels = np.array([0, 1, 2], dtype='u4') # 0: Basic, 1: Intermediate, 2: Advanced

    # 1D Array: a single tech's features
    tech = np.array([5, 1, 3, 2, 3], ndmin=2, dtype='u4') # [scalability, automation, real-time processing, interpretability, efficiency]
    weights = np.array([2, 4, 6, 8, 10], dtype='u4') # Importance of each feature

    # 2D Array: features of 3 techs
    techs = np.array([
        [5, 1, 3, 2, 3], # Tech 1
        [4, 2, 4, 3, 5], # Tech 2
        [7, 3, 2, 4, 6]  # Tech 3
    ], dtype='u4') # [scalability, automation, real-time processing, interpretability, efficiency]

    # 2D Array: labels for the tech groups
    tech_group_labels = np.array([
        [0, 1, 2], # Group 1 labels
        [0, 1, 2], # Group 2 labels
        [0, 1, 2] # Group 3 labels
    ], dtype='u4') # 0: Legacy, 1: Modern, 2: NextGen

    # 3D Array: 3 groups, each with 3 tech and 5 features
    tech_groups = np.array([
        [ # Group 1
            [5, 1, 3, 2, 3], # Tech 1
            [4, 2, 4, 3, 5], # Tech 2
            [7, 3, 2, 4, 6]  # Tech 3
        ],
        [ # Group 2
            [6, 1, 4, 3, 5], # Tech 1
            [5, 3, 5, 4, 9], # Tech 2
            [8, 4, 3, 5, 7]  # Tech 3
        ],
        [ # Group 3
            [7, 2, 5, 4, 6], # Tech 1
            [6, 4, 6, 5, 7], # Tech 2
            [9, 5, 4, 6, 8]  # Tech 3
        ]
    ], dtype='u4') # [scalability, automation, real-time processing, interpretability, efficiency]

    # 3D Array: labels for the tech locations
    tech_locations_labels = np.array([
        [ # Location 1
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]
        ],
        [ # Location 2
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]
        ]
    ], dtype='u4') # 0: Cloud, 1: Edge, 2: On-Premises

    # 4D Array: 2 locations, 3 groups, 3 techs, 5 characteristics
    tech_locations = np.array([
        [ # Location 1
            [ # Group 1
                [5, 1, 3, 2, 3], # Tech 1
                [4, 2, 4, 3, 5], # Tech 2
                [7, 3, 2, 4, 6]  # Tech 3
            ],
            [ # Group 2
                [6, 1, 4, 3, 5], # Tech 1
                [5, 3, 5, 4, 9], # Tech 2
                [8, 4, 3, 5, 7]  # Tech 3
            ],
            [ # Group 3
                [7, 2, 5, 4, 6], # Tech 1
                [6, 4, 6, 5, 7], # Tech 2
                [9, 5, 4, 6, 8]  # Tech 3
            ]
        ],
        [ # Location 2
            [ # Group 1
                [8, 1, 2, 3, 4], # Tech 1
                [7, 2, 3, 4, 5], # Tech 2
                [6, 3, 4, 5, 6]  # Tech 3
            ],
            [ # Group 2
                [9, 1, 2, 3, 4], # Tech 1
                [8, 2, 3, 4, 5], # Tech 2
                [7, 3, 4, 5, 6]  # Tech 3
            ],
            [ # Group 3
                [4, 1, 2, 3, 10], # Tech 1
                [9, 2, 3, 4, 5],  # Tech 2
                [8, 3, 4, 5, 6]   # Tech 3
            ]
        ]
    ], dtype='u4') # [scalability, automation, real-time processing, interpretability, efficiency]

    # print(bias, tech, weights, techs, techs_labels, tech_groups, tech_group_labels, tech_locations, tech_locations_labels)
    # print(type(bias), type(tech), type(weights), type(techs), type(techs_labels), type(tech_groups), type(tech_group_labels), type(tech_locations), type(tech_locations_labels))
    # print(bias.ndim, tech.ndim, weights.ndim, techs.ndim, techs_labels.ndim, tech_groups.ndim, tech_group_labels.ndim, tech_locations.ndim, tech_locations_labels.ndim)
    # print(f"{bias.dtype}\n{techs_labels.dtype}\n{tech.dtype}\n{weights.dtype}\n{techs.dtype}\n{tech_group_labels.dtype}\n{tech_groups.dtype}\n{tech_locations_labels.dtype}\n{tech_locations.dtype}")
    print(f"{bias.shape}\n{techs_labels.shape}\n{tech.shape}\n{weights.shape}\n{techs.shape}\n{tech_group_labels.shape}\n{tech_groups.shape}\n{tech_locations_labels.shape}\n{tech_locations.shape}")

    return (bias, techs_labels, tech, weights, techs, tech_group_labels, tech_groups, tech_locations, tech_locations_labels)


generate_data()
