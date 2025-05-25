import numpy as np

def slice_1D(tech_labels, tech, weights):
    # Labels for the tech
    basic = tech_labels[-3]
    intermediate = tech_labels[-2]
    advanced = tech_labels[-1]

    # Feature values
    scalability = tech[-5]
    automation = tech[-4]
    real_time_processing = tech[-3]
    interpretability = tech[-2]
    efficiency = tech[-1]
    combined_score = scalability + automation + real_time_processing + interpretability + efficiency
    last_3_features = tech[-3:]
    reverse_order = tech[::-1]

    # Corresponding weights
    w_scalability = weights[-5]
    w_automation = weights[-4]
    w_real_time_processing = weights[-3]
    w_interpretability = weights[-2]
    w_efficiency = weights[-1]
    w_combined_score = w_scalability + w_automation + w_real_time_processing + w_interpretability + w_efficiency
    top_3_features = weights[-3:]
    reversed_alternate_weights = weights[::-2]

    return basic, intermediate, advanced, scalability, automation, real_time_processing, interpretability, efficiency, combined_score, last_3_features, reverse_order, w_scalability, w_automation, w_real_time_processing, w_interpretability, w_efficiency, w_combined_score, top_3_features, reversed_alternate_weights

def slice_2D(techs, tech_group_labels):
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

    return tech2, scalability2, automation2, real_time_processing2, interpretability2, efficiency2, combined_score2, tech3, scalability3, automation3, real_time_processing3, interpretability3, efficiency3, combined_score3, last_2_techs, last_feature, reverse_order_techs, legacy, modern, nextgen
    
def slice_3D(tech_groups, tech_locations_labels):
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

def copy(bias, tech, weights, techs, tech_groups, tech_locations):
    # Copying the arrays to ensure they are not modified
    # Convert bias to int8
    float_bias = bias.astype('f')

    # Convert tech to float32
    float_tech = tech.astype('f')

    # Convert weights to float32
    float_weights = weights.astype('f')

    # Convert techs to float32
    float_techs = techs.astype('f')

    # Convert tech groups to float32
    float_tech_groups = tech_groups.astype('f')

    # Convert tech locations to float32
    float_tech_locations = tech_locations.astype('f')

    return float_bias, float_tech, float_weights, float_techs, float_tech_groups, float_tech_locations