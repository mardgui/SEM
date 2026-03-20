import json
import re
import pandas as pd
from pathlib import Path

# Directories and file paths
config_dir = Path("./config")
output_dir = Path("./main_eval_output")
output_feather = "all_results.feather"

all_records = []

# 1. Iterate over config files in ./config
for config_path in config_dir.glob("*.yml"):
    if not config_path.is_file():
        continue
        
    basename = config_path.stem  
    
    # 2. Open the matching output JSON
    json_path = output_dir / f"{basename}.json"
    
    if not json_path.exists():
        print(f"Warning: Missing output for config {basename}. Skipping.")
        continue
        
    with open(json_path, "r") as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not read JSON in {json_path}. Skipping.")
            continue

    # Parse filename to isolate the method and seed
    if "_clip-" in basename:
        _, right_half = basename.split("_clip-", 1)
        if "_" in right_half:
            backbone_str, method_and_seed = right_half.split("_", 1)
        else:
            method_and_seed = right_half
    else:
        method_and_seed = basename.split("_")[-1]

    # 5. Extract seed if present
    seed_match = re.search(r'-seed(\d+)$', method_and_seed)
    if seed_match:
        seed_val = int(seed_match.group(1))
        true_method = method_and_seed[:seed_match.start()]
    else:
        seed_val = None
        true_method = method_and_seed

    # Process each dictionary in the JSON
    for record in json_data:
        
        # 4a. Update the method if it's incomplete or mismatched
        if record.get('method') != true_method:
            record['method'] = true_method
            
        # 5. Inject the seed
        record['seed'] = seed_val

        # 3 & 4b. Validate basic keys
        keys_to_check = ['dataset', 'task', 'backbone']
        for key in keys_to_check:
            if key in record:
                val = str(record[key])
                # Normalization fixes: Handle slashes, underscores, and the @336px suffix
                val_norm1 = val.replace('/', '-').replace('@336px', '') 
                val_norm2 = val.replace('_', '-')  
                
                if (val not in basename and val_norm1 not in basename and val_norm2 not in basename):
                    raise ValueError(f"CRITICAL MISMATCH: {key}='{val}' not found in filename '{basename}'.")

        # Validate the 'setting' key based on the task type
        if 'setting' in record:
            task_type = record.get('task')
            
            if task_type == 'retrieval':
                # Parse "query=stereotype|bias=gender|K=500"
                setting_str = str(record['setting'])
                parts = setting_str.split('|')
                for part in parts:
                    if '=' in part:
                        k, v = part.split('=', 1)
                        # We only expect 'query' and 'bias' values to be in the filename
                        if k in ['query', 'bias']:
                            if v not in basename:
                                raise ValueError(f"CRITICAL MISMATCH: Retrieval {k}='{v}' not found in filename '{basename}'.")
                                
            else: # zs_classification
                val = str(record['setting'])
                val_norm = val.replace('_', '-')
                if val not in basename and val_norm not in basename:
                    raise ValueError(f"CRITICAL MISMATCH: setting='{val}' not found in filename '{basename}'.")
        
        # 6. Add updated dict to the master list
        all_records.append(record)

# 7. Transform into pandas dataframe
if all_records:
    df = pd.DataFrame.from_records(all_records)
    
    # 8. Save as feather file
    df.to_feather(output_feather)
    print(f"Success! Processed {len(all_records)} total records and saved to '{output_feather}'.")
else:
    print("No records processed. Feather file was not created.")
