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
    performance_features = tech[0, ::-2]

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
    automation_features = techs[:, -4]

    # Labels for the tech groups
    legacy = tech_group_labels[0, 0]
    modern = tech_group_labels[0, 1]
    nextgen = tech_group_labels[0, 2]

    return scalability, automation, real_time_processing, interpretability, efficiency, combined_score, performance_features, tech2, scalability2, automation2, real_time_processing2, interpretability2, efficiency2, combined_score2, tech3, scalability3, automation3, real_time_processing3, interpretability3, efficiency3, combined_score3, automation_features, legacy, modern, nextgen

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
    interpretability_features = tech_groups[:, :, -2]

    # Labels for the tech locations
    cloud = tech_locations_labels[0, 0, 0]
    edge = tech_locations_labels[0, 1, 1]
    on_premises = tech_locations_labels[0, 2, 2]

    return tech_group2, tech2_1, scalability2_1, automation2_1, real_time_processing2_1, interpretability2_1, efficiency2_1, combined_score2_1, tech2_2, scalability2_2, automation2_2, real_time_processing2_2, interpretability2_2, efficiency2_2, combined_score2_2, tech2_3, scalability2_3, automation2_3, real_time_processing2_3, interpretability2_3, efficiency2_3, combined_score2_3, tech_group3, tech3_1, scalability3_1, automation3_1, real_time_processing3_1, interpretability3_1, efficiency3_1, combined_score3_1, tech3_2, scalability3_2, automation3_2, real_time_processing3_2, interpretability3_2, efficiency3_2, combined_score3_2, tech3_3, scalability3_3, automation3_3, real_time_processing3_3, interpretability3_3, efficiency3_3, combined_score3_3, interpretability_features, cloud, edge, on_premises

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
    efficiency_feature_all_techs = tech_locations[:, :, :, -1]
    rt_processing_feature_all_techs = tech_locations[:, :, :, 2]

    return tech_location2, tech_location2_g1, tech_location2_g1_1, scalability2_g1_1, automation2_g1_1, real_time_processing2_g1_1, interpretability2_g1_1, efficiency2_g1_1, combined_score2_g1_1, tech_location2_g1_2, scalability2_g1_2, automation2_g1_2, real_time_processing2_g1_2, interpretability2_g1_2, efficiency2_g1_2, combined_score2_g1_2, tech_location2_g1_3, scalability2_g1_3, automation2_g1_3, real_time_processing2_g1_3, interpretability2_g1_3, efficiency2_g1_3, combined_score2_g1_3, tech_location2_g2, tech_location2_g2_1, scalability2_g2_1, automation2_g2_1, real_time_processing2_g2_1, interpretability2_g2_1, efficiency2_g2_1, combined_score2_g2_1, tech_location2_g2_2, scalability2_g2_2, automation2_g2_2, real_time_processing2_g2_2, interpretability2_g2_2, efficiency2_g2_2, combined_score2_g2_2, tech_location2_g2_3, scalability2_g2_3, automation2_g2_3, real_time_processing2_g2_3, interpretability2_g2_3, efficiency2_g2_3, combined_score2_g2_3, tech_location2_g3, tech_location2_g3_1, scalability2_g3_1, automation2_g3_1, real_time_processing2_g3_1, interpretability2_g3_1, efficiency2_g3_1, combined_score2_g3_1, tech_location2_g3_2, scalability2_g3_2, automation2_g3_2, real_time_processing2_g3_2, interpretability2_g3_2, efficiency2_g3_2, combined_score2_g3_2, tech_location2_g3_3, scalability2_g3_3, automation2_g3_3, real_time_processing2_g3_3, interpretability2_g3_3, efficiency2_g3_3, combined_score2_g3_3, efficiency_feature_all_techs, rt_processing_feature_all_techs

def copy_view(bias, performance_features, automation_features, interpretability_features, efficiency_feature_all_techs, rt_processing_feature_all_techs):
    # Create a copy of the original arrays to avoid modifying the original data

    # Convert bias to float32 for compatibility
    bias_copy = bias.astype('f')
    bias_view = bias.view()
    # Bias: Shift opinion from Positive to Negative
    bias_copy[...] = -1.0

    # Convert efficiency, real-time processing, and scalability features to float32 for compatibility
    performance_features_copy = performance_features.astype('f')
    performance_features_view = performance_features.view()
    # Modify the efficiency features
    performance_features_copy[0] = 8.5

    # Convert automation features to float32 for compatibility
    automation_features_copy = automation_features.astype('f')
    automation_features_view = automation_features.view()
    # Modify the quality feature scores
    automation_features_copy[1] = 7.5

    # Convert interpretability features per tech group to float32 for compatibility
    interpretability_features_copy = interpretability_features.astype('f')
    interpretability_features_view = interpretability_features.view()
    interpretability_features_copy[0, 1] = 8.5

    # Convert efficiency feature of all techs to float32 for compatibility
    efficiency_feature_all_techs_copy = efficiency_feature_all_techs.copy()
    efficiency_feature_all_techs_view = efficiency_feature_all_techs.view()
    efficiency_feature_all_techs_copy[0, 0, 1] = 7

    # Convert real-time processing feature of all techs to float32 for compatibility
    rt_processing_feature_all_techs_copy = rt_processing_feature_all_techs.copy()
    rt_processing_feature_all_techs_view = rt_processing_feature_all_techs.view()
    rt_processing_feature_all_techs_copy[0, 0, 1] = 6.5

    return bias_copy, bias_view, performance_features_copy, performance_features_view, automation_features_copy, automation_features_view, interpretability_features_copy, interpretability_features_view, efficiency_feature_all_techs_copy, efficiency_feature_all_techs_view, rt_processing_feature_all_techs_copy, rt_processing_feature_all_techs_view

def reshape_array(performance_features, performance_features_copy, automation_features, automation_features_copy, interpretability_features, interpretability_features_copy, efficiency_feature_all_techs,efficiency_feature_all_techs_copy, rt_processing_feature_all_techs, rt_processing_feature_all_techs_copy):
    """
    Reshapes a set of arrays for further analysis and visualization.
    Returns reshaped arrays.
    """
    # Reshape performance features and its copy to 3 row and 1 column
    performance_features_reshaped = performance_features.reshape(-1, 1)
    performance_features_copy_reshaped = performance_features_copy.reshape(-1, 1)

    # Reshape automation feature scores to 1 layer 3 row and 1 column
    automation_features_reshaped = automation_features.reshape(1, 3, -1)
    automation_features_copy_reshaped = automation_features_copy.reshape(1, 3, -1)

    # Reshape groups interpretability features to 3 layers 3 techs and 1 column
    interpretability_features_reshaped = interpretability_features.reshape(3, 3, -1)
    interpretability_features_copy_reshaped = interpretability_features_copy.reshape(3, 3, -1)

    # Reshape efficiency feature from all group to 18 columns
    efficiency_feature_all_techs_reshaped = efficiency_feature_all_techs.reshape(-1)
    efficiency_feature_all_techs_copy_reshaped = efficiency_feature_all_techs_copy.reshape(-1)

    # Reshape real-time processing feature from all groups to 6 rows, 3 columns
    rt_processing_feature_all_techs_reshaped = rt_processing_feature_all_techs.reshape(-1, 3)
    rt_processing_feature_all_techs_copy_reshaped = rt_processing_feature_all_techs_copy.reshape(-1, 3)

    # print(f"{performance_features_reshaped.base}\n{performance_features_copy_reshaped.base}\n{automation_features_reshaped.base}\n{automation_features_copy_reshaped.base}\n{groups_quality_features_reshaped.base}\n{groups_quality_features_copy_reshaped.base}\n{efficiency_feature_all_techs_reshaped.base}\n{efficiency_feature_all_techs_copy_reshaped.base}\n{rt_processing_feature_all_techs_reshaped.base}\n{rt_processing_feature_all_techs_copy_reshaped.base}")
    # print(f"{performance_features_reshaped.shape}\n{performance_features_copy_reshaped.shape}\n{automation_features_reshaped.shape}\n{automation_features_copy_reshaped.shape}\n{groups_quality_features_reshaped.shape}\n{groups_quality_features_copy_reshaped.shape}\n{efficiency_feature_all_techs_reshaped.shape}\n{efficiency_feature_all_techs_copy_reshaped.shape}\n{rt_processing_feature_all_techs_reshaped.shape}\n{rt_processing_feature_all_techs_copy_reshaped.shape}")

    return performance_features_reshaped, performance_features_copy_reshaped, automation_features_reshaped, automation_features_copy_reshaped, interpretability_features_reshaped, interpretability_features_copy_reshaped, efficiency_feature_all_techs_reshaped, efficiency_feature_all_techs_copy_reshaped, rt_processing_feature_all_techs_reshaped, rt_processing_feature_all_techs_copy_reshaped

def join(performance_features_reshaped, performance_features_copy_reshaped, automation_features_reshaped, automation_features_copy_reshaped, interpretability_features_reshaped, interpretability_features_copy_reshaped, efficiency_feature_all_techs_reshaped, efficiency_feature_all_techs_copy_reshaped, rt_processing_feature_all_techs_reshaped, rt_processing_feature_all_techs_copy_reshaped):

    performance_features_comparison = np.concatenate((performance_features_reshaped, performance_features_copy_reshaped), axis=1)

    automation_features_comparison = np.stack((automation_features_reshaped, automation_features_copy_reshaped), axis=1)

    interpretability_features_comparison = np.hstack((interpretability_features_reshaped, interpretability_features_copy_reshaped))

    efficiency_feature_all_techs_comparison = np.vstack((efficiency_feature_all_techs_reshaped, efficiency_feature_all_techs_copy_reshaped))

    rt_processing_feature_all_techs_comparison = np.dstack((rt_processing_feature_all_techs_reshaped, rt_processing_feature_all_techs_copy_reshaped))

    return performance_features_comparison, automation_features_comparison, interpretability_features_comparison, efficiency_feature_all_techs_comparison, rt_processing_feature_all_techs_comparison

def split(performance_features_comparison, automation_features_comparison, interpretability_features_comparison, efficiency_feature_all_techs_comparison, rt_processing_feature_all_techs_comparison):
    # Split features into batches for analysis
    performance_batches = np.array_split(performance_features_comparison, 3)

    # Split features into batches for analysis
    automation_batches = np.array_split(automation_features_comparison, 3, axis=2)

    # Split features into batches for analysis
    interpretability_batches = np.hsplit(interpretability_features_comparison, 6)

    # Split features into batches for analysis
    efficiency_batches = np.split(efficiency_feature_all_techs_comparison, 18, axis=1)

    # Split features into batches for analysis
    rt_processing_batches = np.vsplit(rt_processing_feature_all_techs_comparison, 6)

    return performance_batches, automation_batches, interpretability_batches, efficiency_batches, rt_processing_batches

def search_sort(performance_batches, automation_batches, interpretability_batches, efficiency_batches, rt_processing_batches):
    # Search for critical performance values (e.g., greater than 5) and sort the batches  
    performance_batches = np.concatenate(performance_batches, axis=0)
    critical_performance_indices = np.where(performance_batches%2 == 1)
    sorted_performance_batches = np.sort(performance_batches)
    sorted_performance_flat = np.sort(sorted_performance_batches.flatten())
    insert_performance_indices = np.searchsorted(sorted_performance_flat, 7)

    automation_batches = np.concatenate(automation_batches, axis=0)
    critical_automation_indices = np.where(automation_batches%2 == 1)
    sorted_automation_batches = np.sort(automation_batches)
    sorted_automation_flat = np.sort(sorted_automation_batches.flatten())
    insert_automation_indices = np.searchsorted(sorted_automation_flat, 7)

    interpretability_batches = np.concatenate(interpretability_batches, axis=0)
    critical_interpretability_indices = np.where(interpretability_batches%2 == 1)
    sorted_interpretability_batches = np.sort(interpretability_batches)
    sorted_interpretability_flat = np.sort(sorted_interpretability_batches.flatten())
    insert_interpretability_indices = np.searchsorted(sorted_interpretability_flat, 7)

    efficiency_batches = np.concatenate(efficiency_batches, axis=0)
    critical_efficiency_indices = np.where(efficiency_batches%2 == 1)
    sorted_efficiency_batches = np.sort(efficiency_batches)
    sorted_efficiency_flat = np.sort(sorted_efficiency_batches.flatten())
    insert_efficiency_indices = np.searchsorted(sorted_efficiency_flat, 7)

    rt_processing_batches = np.concatenate(rt_processing_batches, axis=0)
    critical_rt_processing_indices = np.where(rt_processing_batches%2 == 1)
    sorted_rt_processing_batches = np.sort(rt_processing_batches)
    sorted_rt_processing_flat = np.sort(sorted_rt_processing_batches.flatten())
    insert_rt_processing_indices = np.searchsorted(sorted_rt_processing_flat, 7)

    return critical_performance_indices, sorted_performance_batches, sorted_performance_flat, insert_performance_indices, critical_automation_indices, sorted_automation_batches, sorted_automation_flat, insert_automation_indices, critical_interpretability_indices, sorted_interpretability_batches, sorted_interpretability_flat, insert_interpretability_indices, critical_efficiency_indices, sorted_efficiency_batches, sorted_efficiency_flat, insert_efficiency_indices, critical_rt_processing_indices, sorted_rt_processing_batches, sorted_rt_processing_flat, insert_rt_processing_indices

def filter(sorted_performance_flat, sorted_automation_flat, sorted_interpretability_flat, sorted_efficiency_flat, sorted_rt_processing_flat):
    # Filter array that will return only specific values
    sorted_performance_flat_filters = []
    for element in sorted_performance_flat:
        if element >= 5:
            sorted_performance_flat_filters.append(True)
        else:
            sorted_performance_flat_filters.append(False)

    filtered_sorted_performance_flat = sorted_performance_flat[sorted_performance_flat_filters]

    sorted_automation_flat_filters = []
    for element in sorted_automation_flat:
        if element >= 3:
            sorted_automation_flat_filters.append(True)
        else:
            sorted_automation_flat_filters.append(False)
    
    filtered_sorted_automation_flat = sorted_automation_flat[sorted_automation_flat_filters]

    sorted_interpretability_flat_filters = []
    for element in sorted_interpretability_flat:
        if element >= 5:
            sorted_interpretability_flat_filters.append(True)
        else:
            sorted_interpretability_flat_filters.append(False)

    filtered_sorted_interpretability_flat = sorted_interpretability_flat[sorted_interpretability_flat_filters]

    sorted_efficiency_flat_filters = sorted_efficiency_flat >= 6
    filtered_sorted_efficiency_flat = sorted_efficiency_flat[sorted_efficiency_flat_filters]

    sorted_rt_processing_flat_filters = sorted_rt_processing_flat >= 5
    filtered_sorted_rt_processing_flat = sorted_rt_processing_flat[sorted_rt_processing_flat_filters]

    return sorted_performance_flat_filters, filtered_sorted_performance_flat, sorted_automation_flat_filters, filtered_sorted_automation_flat, sorted_interpretability_flat_filters, filtered_sorted_interpretability_flat, sorted_efficiency_flat_filters, filtered_sorted_efficiency_flat, sorted_rt_processing_flat_filters, filtered_sorted_rt_processing_flat