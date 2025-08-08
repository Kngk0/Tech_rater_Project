import numpy as np
from data_generator import generate_data

def slice_1D(tech_labels, weights):
    """
    Slices 1D technology labels and weights for evaluation.
    Returns key labels and computed features.
    """
    # Labels for the tech
    basic = tech_labels[-3]
    intermediate = tech_labels[-2]
    advanced = tech_labels[-1]

    # Corresponding weights
    w_scalability = weights[-5]
    w_automation = weights[-4]
    w_real_time_processing = weights[-3]
    w_interpretability = weights[-2]
    w_efficiency = weights[-1]
    w_combined_score = w_scalability + w_automation + w_real_time_processing + w_interpretability + w_efficiency
    core_weights = weights[:3]
    quality_weights = weights[3:]

    return basic, intermediate, advanced, w_scalability, w_automation, w_real_time_processing, w_interpretability, w_efficiency, w_combined_score, core_weights, quality_weights

def slice_2D(tech, techs, tech_group_labels):
    """
    Slices 2D technology data, extracting key features and labels.
    """
    # Feature values
    scalability = tech[0, -5]
    automation = tech[0, -4]
    real_time_processing = tech[0, -3]
    interpretability = tech[0, -2]
    efficiency = tech[0, -1]
    combined_score = scalability + automation + real_time_processing + interpretability + efficiency
    #intelligent_features_tech = tech[0, 1:4]
    performance_features = tech[0, ::2]
    tech[0, 1:4:2]

    # Feature values for the second and third techs
    tech2 = techs[1]
    scalability2 = techs[1, 0]
    automation2 = techs[1, 1]
    real_time_processing2 = techs[1, 2]
    interpretability2 = techs[1, 3]
    efficiency2 = techs[1, 4]
    combined_score2 = scalability2 + automation2 + real_time_processing2 + interpretability2 + efficiency2
    tech3 = techs[2]
    scalability3 = techs[2, 0]
    automation3 = techs[2, 1]
    real_time_processing3 = techs[2, 2]
    interpretability3 = techs[2, 3]
    efficiency3 = techs[2, 4]
    combined_score3 = scalability3 + automation3 + real_time_processing3 + interpretability3 + efficiency3
    #efficiency_scores= techs[:, -1]
    #scalability_scores = techs[:, 0]
    core_features_scores = techs[:, 1::2]

    # Labels for the tech groups
    legacy = tech_group_labels[0, 0]
    modern = tech_group_labels[0, 1]
    nextgen = tech_group_labels[0, 2]

    return scalability, automation, real_time_processing, interpretability, efficiency, combined_score, performance_features, tech2, scalability2, automation2, real_time_processing2, interpretability2, efficiency2, combined_score2, tech3, scalability3, automation3, real_time_processing3, interpretability3, efficiency3, combined_score3, core_features_scores, legacy, modern, nextgen

def slice_3D(tech_groups, tech_locations_labels):
    """
    Slices 3D technology data, extracting key features and labels.
    """
    # Feature values for techs in the second and third groups
    tech_group2 = tech_groups[1]
    tech2_1 = tech_groups[1, 0]
    scalability2_1 = tech_groups[1, 0, 0]
    automation2_1 = tech_groups[1, 0, 1]
    real_time_processing2_1 = tech_groups[1, 0, 2]
    interpretability2_1 = tech_groups[1, 0, 3]
    efficiency2_1 = tech_groups[1, 0, 4]
    combined_score2_1 = scalability2_1 + automation2_1 + real_time_processing2_1 + interpretability2_1 + efficiency2_1
    tech2_2 = tech_groups[1, 1]
    scalability2_2 = tech_groups[1, 1, 0]
    automation2_2 = tech_groups[1, 1, 1]
    real_time_processing2_2 = tech_groups[1, 1, 2]
    interpretability2_2 = tech_groups[1, 1, 3]
    efficiency2_2 = tech_groups[1, 1, 4]
    combined_score2_2 = scalability2_2 + automation2_2 + real_time_processing2_2 + interpretability2_2 + efficiency2_2
    tech2_3 = tech_groups[1, 2]
    scalability2_3 = tech_groups[1, 2, 0]
    automation2_3 = tech_groups[1, 2, 1]
    real_time_processing2_3 = tech_groups[1, 2, 2]
    interpretability2_3 = tech_groups[1, 2, 3]
    efficiency2_3 = tech_groups[1, 2, 4]
    combined_score2_3 = scalability2_3 + automation2_3 + real_time_processing2_3 + interpretability2_3 + efficiency2_3
    tech_group3 = tech_groups[2]
    tech3_1 = tech_groups[2, 0]
    scalability3_1 = tech_groups[2, 0, 0]
    automation3_1 = tech_groups[2, 0, 1]
    real_time_processing3_1 = tech_groups[2, 0, 2]
    interpretability3_1 = tech_groups[2, 0, 3]
    efficiency3_1 = tech_groups[2, 0, 4]
    combined_score3_1 = scalability3_1 + automation3_1 + real_time_processing3_1 + interpretability3_1 + efficiency3_1
    tech3_2 = tech_groups[2, 1]
    scalability3_2 = tech_groups[2, 1, 0]
    automation3_2 = tech_groups[2, 1, 1]
    real_time_processing3_2 = tech_groups[2, 1, 2]
    interpretability3_2 = tech_groups[2, 1, 3]
    efficiency3_2 = tech_groups[2, 1, 4]
    combined_score3_2 = scalability3_2 + automation3_2 + real_time_processing3_2 + interpretability3_2 + efficiency3_2
    tech3_3 = tech_groups[2, 2]
    scalability3_3 = tech_groups[2, 2, 0]
    automation3_3 = tech_groups[2, 2, 1]
    real_time_processing3_3 = tech_groups[2, 2, 2]
    interpretability3_3 = tech_groups[2, 2, 3]
    efficiency3_3 = tech_groups[2, 2, 4]
    combined_score3_3 = scalability3_3 + automation3_3 + real_time_processing3_3 + interpretability3_3 + efficiency3_3
    group_1_automation = tech_groups[0, :, 1]
    #groups_scalability = tech_groups[:, :, 0]

    # Labels for the tech locations
    cloud = tech_locations_labels[0, 0, 0]
    edge = tech_locations_labels[0, 1, 1]
    on_premises = tech_locations_labels[0, 2, 2]

    return tech_group2, tech2_1, scalability2_1, automation2_1, real_time_processing2_1, interpretability2_1, efficiency2_1, combined_score2_1, tech2_2, scalability2_2, automation2_2, real_time_processing2_2, interpretability2_2, efficiency2_2, combined_score2_2, tech2_3, scalability2_3, automation2_3, real_time_processing2_3, interpretability2_3, efficiency2_3, combined_score2_3, tech_group3, tech3_1, scalability3_1, automation3_1, real_time_processing3_1, interpretability3_1, efficiency3_1, combined_score3_1, tech3_2, scalability3_2, automation3_2, real_time_processing3_2, interpretability3_2, efficiency3_2, combined_score3_2, tech3_3, scalability3_3, automation3_3, real_time_processing3_3, interpretability3_3, efficiency3_3, combined_score3_3, group_1_automation, cloud, edge, on_premises

def slice_4D(tech_locations):
    """
    Slices 4D technology data, extracting key features and labels.
    """
    tech_location2 = tech_locations[1]
    tech_location2_g1 = tech_locations[1, 0]
    tech_location2_g1_1 = tech_locations[1, 0, 0]
    scalability2_g1_1 = tech_locations[1, 0, 0, 0]
    automation2_g1_1 = tech_locations[1, 0, 0, 1]
    real_time_processing2_g1_1 = tech_locations[1, 0, 0, 2]
    interpretability2_g1_1 = tech_locations[1, 0, 0, 3]
    efficiency2_g1_1 = tech_locations[1, 0, 0, 4]
    combined_score2_g1_1 = scalability2_g1_1 + automation2_g1_1 + real_time_processing2_g1_1 + interpretability2_g1_1 + efficiency2_g1_1
    tech_location2_g1_2 = tech_locations[1, 0, 1]
    scalability2_g1_2 = tech_locations[1, 0, 1, 0]
    automation2_g1_2 = tech_locations[1, 0, 1, 1]
    real_time_processing2_g1_2 = tech_locations[1, 0, 1, 2]
    interpretability2_g1_2 = tech_locations[1, 0, 1, 3]
    efficiency2_g1_2 = tech_locations[1, 0, 1, 4]
    combined_score2_g1_2 = scalability2_g1_2 + automation2_g1_2 + real_time_processing2_g1_2 + interpretability2_g1_2 + efficiency2_g1_2
    tech_location2_g1_3 = tech_locations[1, 0, 2]
    scalability2_g1_3 = tech_locations[1, 0, 2, 0]
    automation2_g1_3 = tech_locations[1, 0, 2, 1]
    real_time_processing2_g1_3 = tech_locations[1, 0, 2, 2]
    interpretability2_g1_3 = tech_locations[1, 0, 2, 3]
    efficiency2_g1_3 = tech_locations[1, 0, 2, 4]
    combined_score2_g1_3 = scalability2_g1_3 + automation2_g1_3 + real_time_processing2_g1_3 + interpretability2_g1_3 + efficiency2_g1_3
    tech_location2_g2 = tech_locations[1, 1]
    tech_location2_g2_1 = tech_locations[1, 1, 0]
    scalability2_g2_1 = tech_locations[1, 1, 0, 0]
    automation2_g2_1 = tech_locations[1, 1, 0, 1]
    real_time_processing2_g2_1 = tech_locations[1, 1, 0, 2]
    interpretability2_g2_1 = tech_locations[1, 1, 0, 3]
    efficiency2_g2_1 = tech_locations[1, 1, 0, 4]
    combined_score2_g2_1 = scalability2_g2_1 + automation2_g2_1 + real_time_processing2_g2_1 + interpretability2_g2_1 + efficiency2_g2_1
    tech_location2_g2_2 = tech_locations[1, 1, 1]
    scalability2_g2_2 = tech_locations[1, 1, 1, 0]
    automation2_g2_2 = tech_locations[1, 1, 1, 1]
    real_time_processing2_g2_2 = tech_locations[1, 1, 1, 2]
    interpretability2_g2_2 = tech_locations[1, 1, 1, 3]
    efficiency2_g2_2 = tech_locations[1, 1, 1, 4]
    combined_score2_g2_2 = scalability2_g2_2 + automation2_g2_2 + real_time_processing2_g2_2 + interpretability2_g2_2 + efficiency2_g2_2
    tech_location2_g2_3 = tech_locations[1, 1, 2]
    scalability2_g2_3 = tech_locations[1, 1, 2, 0]
    automation2_g2_3 = tech_locations[1, 1, 2, 1]
    real_time_processing2_g2_3 = tech_locations[1, 1, 2, 2]
    interpretability2_g2_3 = tech_locations[1, 1, 2, 3]
    efficiency2_g2_3 = tech_locations[1, 1, 2, 4]
    combined_score2_g2_3 = scalability2_g2_3 + automation2_g2_3 + real_time_processing2_g2_3 + interpretability2_g2_3 + efficiency2_g2_3
    tech_location2_g3 = tech_locations[1, 2]
    tech_location2_g3_1 = tech_locations[1, 2, 0]
    scalability2_g3_1 = tech_locations[1, 2, 0, 0]
    automation2_g3_1 = tech_locations[1, 2, 0, 1]
    real_time_processing2_g3_1 = tech_locations[1, 2, 0, 2]
    interpretability2_g3_1 = tech_locations[1, 2, 0, 3]
    efficiency2_g3_1 = tech_locations[1, 2, 0, 4]
    combined_score2_g3_1 = scalability2_g3_1 + automation2_g3_1 + real_time_processing2_g3_1 + interpretability2_g3_1 + efficiency2_g3_1
    tech_location2_g3_2 = tech_locations[1, 2, 1]
    scalability2_g3_2 = tech_locations[1, 2, 1, 0]
    automation2_g3_2 = tech_locations[1, 2, 1, 1]
    real_time_processing2_g3_2 = tech_locations[1, 2, 1, 2]
    interpretability2_g3_2 = tech_locations[1, 2, 1, 3]
    efficiency2_g3_2 = tech_locations[1, 2, 1, 4]
    combined_score2_g3_2 = scalability2_g3_2 + automation2_g3_2 + real_time_processing2_g3_2 + interpretability2_g3_2 + efficiency2_g3_2
    tech_location2_g3_3 = tech_locations[1, 2, 2]
    scalability2_g3_3 = tech_locations[1, 2, 2, 0]
    automation2_g3_3 = tech_locations[1, 2, 2, 1]
    real_time_processing2_g3_3 = tech_locations[1, 2, 2, 2]
    interpretability2_g3_3 = tech_locations[1, 2, 2, 3]
    efficiency2_g3_3 = tech_locations[1, 2, 2, 4]
    combined_score2_g3_3 = scalability2_g3_3 + automation2_g3_3 + real_time_processing2_g3_3 + interpretability2_g3_3 + efficiency2_g3_3
    #location2_group3_tech1_features = tech_locations[1, 2, 0]
    #group2_scalability_by_location = tech_locations[:, 1, :, 0]
    efficiency_feature_all_techs = tech_locations[:, :, :, -1]
    #realtime_scores_loc1 = tech_locations[0, :, :, 2]

    return tech_location2, tech_location2_g1, tech_location2_g1_1, scalability2_g1_1, automation2_g1_1, real_time_processing2_g1_1, interpretability2_g1_1, efficiency2_g1_1, combined_score2_g1_1, tech_location2_g1_2, scalability2_g1_2, automation2_g1_2, real_time_processing2_g1_2, interpretability2_g1_2, efficiency2_g1_2, combined_score2_g1_2, tech_location2_g1_3, scalability2_g1_3, automation2_g1_3, real_time_processing2_g1_3, interpretability2_g1_3, efficiency2_g1_3, combined_score2_g1_3, tech_location2_g2, tech_location2_g2_1, scalability2_g2_1, automation2_g2_1, real_time_processing2_g2_1, interpretability2_g2_1, efficiency2_g2_1, combined_score2_g2_1, tech_location2_g2_2, scalability2_g2_2, automation2_g2_2, real_time_processing2_g2_2, interpretability2_g2_2, efficiency2_g2_2, combined_score2_g2_2, tech_location2_g2_3, scalability2_g2_3, automation2_g2_3, real_time_processing2_g2_3, interpretability2_g2_3, efficiency2_g2_3, combined_score2_g2_3, tech_location2_g3, tech_location2_g3_1, scalability2_g3_1, automation2_g3_1, real_time_processing2_g3_1, interpretability2_g3_1, efficiency2_g3_1, combined_score2_g3_1, tech_location2_g3_2, scalability2_g3_2, automation2_g3_2, real_time_processing2_g3_2, interpretability2_g3_2, efficiency2_g3_2, combined_score2_g3_2, tech_location2_g3_3, scalability2_g3_3, automation2_g3_3, real_time_processing2_g3_3, interpretability2_g3_3, efficiency2_g3_3, combined_score2_g3_3, efficiency_feature_all_techs

def copy_view(bias, sorted_feature_weights, performance_features, core_feature_scores, group_1_automation, efficiency_feature_all_techs):
    # Create a copy of the original arrays to avoid modifying the original data

    # Convert bias to float32 for compatibility
    bias_copy = bias.astype('f')
    # Bias: Shift opinion from Positive to Negative
    bias[...] = -1.0
    bias_copy_view = bias_copy.view()
    bias_copy[...] = 0.01

    # Convert sorted feature weights to float32 for compatibility
    sorted_feature_weights_copy = sorted_feature_weights.astype('f')
    # Modify the sorted feature weights
    sorted_feature_weights[1] = 9.0
    sorted_feature_weights_copy_view = sorted_feature_weights_copy.view()
    # Modify the sorted feature weights
    sorted_feature_weights_copy[1] = 3.0

    # Convert last 3 features to float32 for compatibility
    performance_features_copy = performance_features.astype('f')
    performance_features[0] = 4.0
    performance_features_copy_view = performance_features_copy.view()
    # Modify the last 3 features
    performance_features_copy[0] = 2.0

    # Convert core feature scores to float32 for compatibility
    core_feature_scores_copy = core_feature_scores.astype('f')
    core_feature_scores[1, 1] = 9.0
    core_feature_scores_copy_view = core_feature_scores_copy.view()
    # Modify the core feature scores
    core_feature_scores_copy[1, 1] = 3.0

    # Convert groups scalability to float32 for compatibility
    group_1_automation_copy = group_1_automation.astype('f')
    group_1_automation[1] = 6.0
    group_1_automation_copy_view = group_1_automation_copy.view()
    group_1_automation_copy[1] = 3.0

    # Convert last feature of all techs to float32 for compatibility
    efficiency_feature_all_techs_copy = efficiency_feature_all_techs.astype('f')
    # Modify the last feature of all techs
    efficiency_feature_all_techs[0, 0, 1] = 10.0
    efficiency_feature_all_techs_copy_view = efficiency_feature_all_techs_copy.view()
    efficiency_feature_all_techs_copy[0, 0, 1] = 7.0

    return bias_copy, bias_copy_view, sorted_feature_weights_copy, sorted_feature_weights_copy_view, performance_features_copy, performance_features_copy_view, core_feature_scores_copy, core_feature_scores_copy_view, group_1_automation_copy, group_1_automation_copy_view, efficiency_feature_all_techs_copy, efficiency_feature_all_techs_copy_view

def reshape_array(sorted_feature_weights_copy, performance_features_copy, core_feature_scores_copy, group_1_automation_copy, efficiency_feature_all_techs_copy):
    """
    Reshapes a set of arrays for further analysis and visualization.
    Returns reshaped arrays.
    """
    # Reshape sorted feature weights to 5 rows and 1 column
    reshaped_sorted_feature_weights = sorted_feature_weights_copy.reshape(5, 1)

    # Reshape performance features to 1 row and 3 columns
    reshaped_performance_features = performance_features_copy.reshape(1, -1)

    # Reshape core feature scores to 1 row and 6 columns
    reshaped_core_feature_scores = core_feature_scores_copy.reshape(-1)

    # Reshape group 1's automation to 3 row and 1 column
    reshaped_group_1_automation = group_1_automation_copy.reshape(3, 1)

    # Reshape efficiency feature from all techs to 2 row and 9 columns
    reshaped_efficiency_feature_all_techs = efficiency_feature_all_techs_copy.reshape(2, 9)

    # print(f"{reshaped_sorted_feature_weights.base}\n{reshaped_performance_features.base}\n{reshaped_core_feature_scores.base}\n{reshaped_group_1_automation.base}\n{reshaped_efficiency_feature_all_techs.base}\n")
    # print(f"{reshaped_sorted_feature_weights.shape}\n{reshaped_performance_features.shape}\n{reshaped_core_feature_scores.shape}\n{reshaped_group_1_automation.shape}\n{reshaped_efficiency_feature_all_techs.shape}\n")

    return reshaped_sorted_feature_weights, reshaped_performance_features, reshaped_core_feature_scores, reshaped_group_1_automation, reshaped_efficiency_feature_all_techs

def join():
    pass

def split(weights_copy, techs_copy, tech_groups_copy, tech_locations_copy):
    # Split the weights into 3 equal parts
    weight_batches = np.array_split(weights_copy, 3)

    # Split the techs into 3 equal parts
    tech_batches = np.array_split(techs_copy, 3, axis=1)

    # Split the techs in the other 2 groups into 3 equal parts
    tech_groups_batches = np.array_split(tech_groups_copy[1:], 3, axis=2)

    # Split the techs in the other 2 locations into 3 equal parts
    tech_locations_batches = np.array_split(tech_locations_copy[1:], 3, axis=3)

    return weight_batches, tech_batches, tech_groups_batches, tech_locations_batches

def search(tech_locations_copy):
    # Search for passing tech features in the tech locations
    passing_features = np.where(tech_locations_copy >= 7)
    failing_features = np.where(tech_locations_copy < 7)
    top_techs = tech_locations_copy[np.where(tech_locations_copy > 8)]

    return passing_features, failing_features, top_techs

def sort(tech_locations_copy):
    # Sort the tech locations
    sorted_locations = np.sort(tech_locations_copy)
    
    return sorted_locations