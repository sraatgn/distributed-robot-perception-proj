import numpy as np

def calculate_rmse(drones):
    mse = 0
    for drone in drones:
        mse += np.sum((np.array(drone.positions) - np.array(drone.true_positions)) ** 2)
    mse /= len(drones)
    return np.sqrt(mse)

def calculate_detection_metrics(drones, fire_position):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for drone in drones:
        if drone.detect_fire(fire_position):
            if np.linalg.norm(fire_position - drone.x[:2]) < 5:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if np.linalg.norm(fire_position - drone.x[:2]) < 5:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


"""
Precision: The ratio of correctly detected fires (true positives) to all detected fires (true positives + false positives).
Recall: The ratio of correctly detected fires (true positives) to all actual fires (true positives + false negatives).
F1 Score: The harmonic mean of precision and recall, providing a single metric that balances both.
"""