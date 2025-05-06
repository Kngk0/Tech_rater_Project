import numpy as np

def generate_data():

    # 0D Array: Bias for the tech
    bias = np.array(1)

    # 1D Array: tech's characteristics (scalability, automation, real-time processing, interpretability, efficiency)
    tech = np.array([9, 10, 8, 7, 9]) # Tech with scalability, automation, real-time processing, interpretability, efficiency
    weights = np.array([2, 4, 6, 8, 10]) # Importance for each characteristic


    # 2D Array: tech's characteristics (3 techs, 5 characteristics)
    techs = np.array([
        [5, 1, 3, 2, 4], # Tech 1
        [4, 2, 4, 3, 5], # Tech 2
        [7, 3, 2, 4, 6]  # Tech 3
    ])

    # 3D Array: tech's characteristics (3 groups, 3 tech, 5 characteristics)
    tech_groups = np.array([
        [ # Group 1
            [5, 1, 3, 2, 3], # Tech 1
            [4, 2, 5, 3, 4], # Tech 2
            [7, 3, 2, 4, 5]  # Tech 3
        ],
        [ # Group 2
            [6, 1, 4, 3, 5], # Tech 1
            [5, 3, 5, 4, 6], # Tech 2
            [8, 4, 3, 5, 7]  # Tech 3
        ],
        [ # Group 3
            [7, 2, 5, 4, 6], # Tech 1
            [6, 4, 6, 5, 7], # Tech 2
            [9, 5, 4, 6, 8]  # Tech 3
        ]
    ])

    # 4D Array: tech's characteristics (2 locations, 3 groups, 3 techs, 5 characteristics)
    tech_locations = np.array([
        [ # Location 1
            [ # Group 1
                [5, 1, 3, 2, 3], # Tech 1
                [4, 2, 5, 3, 4], # Tech 2
                [7, 3, 2, 4, 5]  # Tech 3
            ],
            [ # Group 2
                [6, 1, 4, 3, 5], # Tech 1
                [5, 3, 5, 4, 6], # Tech 2
                [8, 4, 3, 5, 7]  # Tech 3
            ],
            [ # Group 3
                [7, 2, 5, 4, 6], # Tech 1
                [6, 4, 6, 5, 7], # Tech 2
                [9, 5, 4, 6, 8]  # Tech 3
            ]
        ],
        [ # Location 2
            [ # Group A
                [8, 1, 2, 3, 4], # Tech A
                [7, 2, 3, 4, 5], # Tech B
                [6, 3, 4, 5, 6]  # Tech C
            ],
            [ # Group B
                [9, 1, 2, 3, 4], # Tech A
                [8, 2, 3, 4, 5], # Tech B
                [7, 3, 4, 5, 6]  # Tech C
            ],
            [ # Group C
                [10, 1, 2, 3, 4], # Tech A
                [9, 2, 3, 4, 5],  # Tech B
                [8, 3, 4, 5, 6]   # Tech C
            ]
        ]
    ])

    print(bias, tech, weights, techs, tech_groups, tech_locations)
    print(type(bias), type(tech), type(weights), type(techs), type(tech_groups), type(tech_locations))
    print(bias.ndim, tech.ndim, weights.ndim, techs.ndim, tech_groups.ndim, tech_locations.ndim)


generate_data()
