import os
import csv
import random
from cDDPM_v2.config import CONFIG

def generate_acquisition_configs(
    output_csv_path="./cDDPM_v2/dataset/sirt_configurations.csv", 
    num_objects=CONFIG["data"]["train_samples"] + CONFIG["data"]["val_samples"], 
    views_per_object=CONFIG["acquisition"]["views_per_object"],
    tilt_bounds=CONFIG["acquisition"]["tilt_bounds"],
    proj_bounds=CONFIG["acquisition"]["projection_bounds"]
):
    # Setup CSV headers: object_id, cfg_0, cfg_1, cfg_2, cfg_3, cfg_4
    header = ["object_id"] + [f"cfg_{i}" for i in range(views_per_object)]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for obj_idx in range(num_objects):
            cfgs = []
            
            # Pure random sampling across all views based on the config bounds
            for _ in range(views_per_object):
                t_val = round(random.uniform(tilt_bounds[0], tilt_bounds[1]), 2)
                p_val = random.randint(proj_bounds[0], proj_bounds[1])
                cfgs.append(f"{t_val}_{p_val}")
            
            # Write to CSV
            row = [f"sino_{obj_idx:04d}"] + cfgs
            writer.writerow(row)
            
    print(f"--> Saved {output_csv_path}")
    print(f"--> Total Objects: {num_objects} | Configurations per object: {views_per_object}")

if __name__ == "__main__":
    generate_acquisition_configs()