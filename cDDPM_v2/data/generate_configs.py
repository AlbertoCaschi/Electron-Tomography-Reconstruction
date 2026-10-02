import os
import csv
import random
from cDDPM_v2.config import CONFIG

def generate_acquisition_configs(
    output_csv_path="./cDDPM_v2/dataset/sirt_configurations.csv", 
    num_objects=CONFIG["data"]["train_samples"] + CONFIG["data"]["val_samples"], 
    views_per_object=CONFIG["acquisition"]["views_per_object"],
    worst_case_tilt=min(CONFIG["acquisition"]["tilt_bounds"]),
    worst_case_proj=min(CONFIG["acquisition"]["projection_bounds"]),
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
            
            # 1. Force the worst-case configuration
            worst_cfg = f"{worst_case_tilt}_{worst_case_proj}"
            cfgs.append(worst_cfg)
            
            # 2. Randomly sample the remaining configurations
            for _ in range(views_per_object - 1):
                t_val = round(random.uniform(tilt_bounds[0], tilt_bounds[1]), 2)
                p_val = random.randint(proj_bounds[0], proj_bounds[1])
                cfgs.append(f"{t_val}_{p_val}")
            
            # Shuffle so the worst-case isn't always locked to cfg_0
            random.shuffle(cfgs)
            
            # Write to CSV
            row = [f"sino_{obj_idx:04d}"] + cfgs
            writer.writerow(row)
            
    print(f"--> Saved {output_csv_path}")
    print(f"--> Total Objects: {num_objects} | Configurations per object: {views_per_object}")

if __name__ == "__main__":
    generate_acquisition_configs()