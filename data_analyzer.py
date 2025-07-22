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
    top_3_features = weights[-3:]
    reversed_alternate_weights = weights[::-2]

    return basic, intermediate, advanced, w_scalability, w_automation, w_real_time_processing, w_interpretability, w_efficiency, w_combined_score, top_3_features, reversed_alternate_weights

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
    last_3_features = tech[0, -3:]
    reverse_order = tech[0, ::-1]

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
    last_2_techs = techs[-2:]
    last_feature = techs[:, -1]
    reverse_order_techs = techs[:, ::-1]

    # Labels for the tech groups
    legacy = tech_group_labels[0, 0]
    modern = tech_group_labels[0, 1]
    nextgen = tech_group_labels[0, 2]

    return scalability, automation, real_time_processing, interpretability, efficiency, combined_score, last_3_features, reverse_order, tech2, scalability2, automation2, real_time_processing2, interpretability2, efficiency2, combined_score2, tech3, scalability3, automation3, real_time_processing3, interpretability3, efficiency3, combined_score3, last_2_techs, last_feature, reverse_order_techs, legacy, modern, nextgen
    
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
    last_group = tech_groups[-1]
    last_2_features = tech_groups[:, :, -2:]
    
    # Labels for the tech locations
    cloud = tech_locations_labels[0, 0, 0]
    edge = tech_locations_labels[0, 1, 1]
    on_premises = tech_locations_labels[0, 2, 2]

    return tech_group2, tech2_1, scalability2_1, automation2_1, real_time_processing2_1, interpretability2_1, efficiency2_1, combined_score2_1, tech2_2, scalability2_2, automation2_2, real_time_processing2_2, interpretability2_2, efficiency2_2, combined_score2_2, tech2_3, scalability2_3, automation2_3, real_time_processing2_3, interpretability2_3, efficiency2_3, combined_score2_3, tech_group3, tech3_1, scalability3_1, automation3_1, real_time_processing3_1, interpretability3_1, efficiency3_1, combined_score3_1, tech3_2, scalability3_2, automation3_2, real_time_processing3_2, interpretability3_2, efficiency3_2, combined_score3_2, tech3_3, scalability3_3, automation3_3, real_time_processing3_3, interpretability3_3, efficiency3_3, combined_score3_3, last_group, last_2_features, cloud, edge, on_premises

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
    last_feature_all_techs = tech_locations[:, :, :, -1]
    reversed_techs = tech_locations[:, :, ::-1, :]

    return tech_location2, tech_location2_g1, tech_location2_g1_1, scalability2_g1_1, automation2_g1_1, real_time_processing2_g1_1, interpretability2_g1_1, efficiency2_g1_1, combined_score2_g1_1, tech_location2_g1_2, scalability2_g1_2, automation2_g1_2, real_time_processing2_g1_2, interpretability2_g1_2, efficiency2_g1_2, combined_score2_g1_2, tech_location2_g1_3, scalability2_g1_3, automation2_g1_3, real_time_processing2_g1_3, interpretability2_g1_3, efficiency2_g1_3, combined_score2_g1_3, tech_location2_g2, tech_location2_g2_1, scalability2_g2_1, automation2_g2_1, real_time_processing2_g2_1, interpretability2_g2_1, efficiency2_g2_1, combined_score2_g2_1, tech_location2_g2_2, scalability2_g2_2, automation2_g2_2, real_time_processing2_g2_2, interpretability2_g2_2, efficiency2_g2_2, combined_score2_g2_2, tech_location2_g2_3, scalability2_g2_3, automation2_g2_3, real_time_processing2_g2_3, interpretability2_g2_3, efficiency2_g2_3, combined_score2_g2_3, tech_location2_g3, tech_location2_g3_1, scalability2_g3_1, automation2_g3_1, real_time_processing2_g3_1, interpretability2_g3_1, efficiency2_g3_1, combined_score2_g3_1, tech_location2_g3_2, scalability2_g3_2, automation2_g3_2, real_time_processing2_g3_2, interpretability2_g3_2, efficiency2_g3_2, combined_score2_g3_2, tech_location2_g3_3, scalability2_g3_3, automation2_g3_3, real_time_processing2_g3_3, interpretability2_g3_3, efficiency2_g3_3, combined_score2_g3_3, last_feature_all_techs, reversed_techs

def copy_view(bias, top_3_features, reversed_alternate_weights, last_3_features, reverse_order, last_2_techs, last_feature, reverse_order_techs, last_group, last_2_features, last_feature_all_techs, reversed_techs):
    # Create a copy of the original arrays to avoid modifying the original data

    # Convert bias to float32 for compatibility
    bias_copy = bias.astype('f')

    bias_copy_view = bias_copy.view()

    # Bias: Shift opinion from Positive to Negative
    bias_copy_view[...] = -1.0  # Example modification to show that the original will not change

    # Convert top 3 features to float32 for compatibility
    top_3_features_copy = top_3_features.astype('f')

    top_3_features_copy_view = top_3_features_copy.view()

    # Modify the top 3 features
    top_3_features_copy_view[0] = 10.0  # Example modification

    # Convert reversed alternate weights to float32 for compatibility
    reversed_alternate_weights_copy = reversed_alternate_weights.astype('f')

    reversed_alternate_weights_copy_view = reversed_alternate_weights_copy.view()

    # Modify the reversed alternate weights
    reversed_alternate_weights_copy_view[1] = 9.0

    # Convert last 3 features to float32 for compatibility
    last_3_features_copy = last_3_features.astype('f')

    last_3_features_copy_view = last_3_features_copy.view()

    # Modify the last 3 features
    last_3_features_copy_view[0] = 5.0

    # Convert reverse order to float32 for compatibility
    reverse_order_copy = reverse_order.astype('f')

    reverse_order_copy_view = reverse_order_copy.view()

    # Modify the reverse order
    reverse_order_copy_view[0] = 5.0

    # Convert last 2 techs to float32 for compatibility
    last_2_techs_copy = last_2_techs.astype('f')

    last_2_techs_copy_view = last_2_techs_copy.view()

    # Modify the last 2 techs
    last_2_techs_copy_view[0, 4] = 6.0

    # Convert last feature from each tech to float32 for compatibility
    last_feature_copy = last_feature.astype('f')

    last_feature_copy_view = last_feature_copy.view()

    # Modify the last feature
    last_feature_copy_view[1] = 6.0

    # Convert reverse feature order of techs to float32 for compatibility
    reverse_order_techs_copy = reverse_order_techs.astype('f')

    reverse_order_techs_copy_view = reverse_order_techs_copy.view()

    # Modify the reverse order of techs
    reverse_order_techs_copy_view[1, 0] = 6.0

    # Convert last group of techs to float32 for compatibility
    last_group_copy = last_group.astype('f')

    last_group_copy_view = last_group_copy.view()

    # Modify the last group of techs
    last_group_copy_view[2, 2] = 9.0

    # Convert last 2 features of each tech to float32 for compatibility
    last_2_features_copy = last_2_features.astype('f')

    last_2_features_copy_view = last_2_features_copy.view()

    # Modify the last 2 features of each tech
    last_2_features_copy_view[0, 1, 1] = 6.0

    # Convert last feature of all techs to float32 for compatibility
    last_feature_all_techs_copy = last_feature_all_techs.astype('f')

    last_feature_all_techs_copy_view = last_feature_all_techs_copy.view()

    # Modify the last feature of all techs
    last_feature_all_techs_copy_view[0, 0, 1] = 6.0

    # Convert reversed techs to float32 for compatibility
    reversed_techs_copy = reversed_techs.astype('f')

    reversed_techs_copy_view = reversed_techs_copy.view()

    # Modify the reversed techs
    reversed_techs_copy_view[0, 0, 1, 4] = 6.0

    return bias_copy, top_3_features_copy, reversed_alternate_weights_copy, last_3_features_copy, reverse_order_copy, last_2_techs_copy, last_feature_copy, reverse_order_techs_copy, last_group_copy, last_2_features_copy, last_feature_all_techs_copy, reversed_techs_copy

def reshape_array(top_3_features_copy, last_3_features_copy, reverse_order_copy, reversed_alternate_weights_copy, last_feature_copy, last_2_techs_copy, reverse_order_techs_copy, last_group_copy):
    """
    Reshapes a set of arrays for further analysis and visualization.
    Returns reshaped arrays.
    """
    # Reshape top 3 features to a 2D array with 3 rows and 1 column
    reshaped_top_3_features = top_3_features_copy.reshape(-1, 1)

    # Reshape last 3 features to a 2D array with 1 row and 3 columns
    reshaped_last_3_features = last_3_features_copy.reshape(1, 3)

    # Reshape reversed order of tech 1 features to a 2D array with 1 row and 3 columns
    reshaped_reverse_order = reverse_order_copy.reshape(1, 5)

    # Reshape reversed alternate weights to a 3D array with 1 depth, 3 rows, and 1 column
    reshaped_reversed_alternate_weights = reversed_alternate_weights_copy.reshape(1, 3, 1)

    # Reshape last feature to a 3D array with 3 depth 1 row and 1 column
    reshaped_last_feature = last_feature_copy.reshape(3, 1, 1)

    # Reshape last 2 techs into a 1D array
    reshaped_last_2_techs = last_2_techs_copy.reshape(-1)

    # Reshape reversed order of techs into a 1D array
    reshaped_reverse_order_techs = reverse_order_techs_copy.reshape(-1)

    # Reshape last group of techs into a 1D array
    reshaped_last_group = last_group_copy.reshape(-1)

    # print(f"{reshaped_top_3_features.base}\n{reshaped_last_3_features.base}\n{reshaped_reverse_order.base}\n{reshaped_reversed_alternate_weights.base}\n{reshaped_last_feature.base}\n{reshaped_last_2_techs.base}\n{reshaped_reverse_order_techs.base}\n{reshaped_last_group.base}")
    # print(f"{reshaped_top_3_features.shape}\n{reshaped_last_3_features.shape}\n{reshaped_reverse_order.shape}\n{reshaped_reversed_alternate_weights.shape}\n{reshaped_last_feature.shape}\n{reshaped_last_2_techs.shape}\n{reshaped_reverse_order_techs.shape}\n{reshaped_last_group.shape}")
    # print(f"{top_3_features_copy.shape}\n{last_3_features_copy.shape}\n{reverse_order_copy.shape}\n{reversed_alternate_weights_copy.shape}\n{last_feature_copy.shape}\n{last_2_techs_copy.shape}\n{reverse_order_techs_copy.shape}\n{last_group_copy.shape}")

    return reshaped_top_3_features, reshaped_last_3_features, reshaped_reverse_order, reshaped_reversed_alternate_weights, reshaped_last_feature, reshaped_last_2_techs, reshaped_reverse_order_techs, reshaped_last_group

def join(techs_copy, weights_copy, reshaped_labels, tech_group_labels, tech_group_labels_copy, tech, tech_copy):
    # Join the last feature and reversed alternate weights into a single 1D array
    efficiency_vs_weights = np.concatenate((techs_copy[:, -1], weights_copy[::-2]))

    # Join the techs and reshaped labels into a 2D array
    techs_with_labels = np.concatenate((techs_copy, reshaped_labels), axis=1)

    # Join the last 3 features of tech 1 and top 3 features into a 2D array
    top_features_and_weights = np.stack((tech_copy[0, -3:], weights_copy[-3:]), axis=1)

    # Join the complexity labels of the techs with the generation labels
    complexity_labels = np.hstack((tech_group_labels, tech_group_labels_copy))

    # Join the reversed features of tech with the weights
    reversed_features_and_weights = np.vstack((tech_copy[0, ::-1], weights_copy))

    # Join the tech features with the modified tech features
    original_vs_modified = np.dstack((tech, tech_copy))

    return efficiency_vs_weights, techs_with_labels, top_features_and_weights, complexity_labels, reversed_features_and_weights, original_vs_modified

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