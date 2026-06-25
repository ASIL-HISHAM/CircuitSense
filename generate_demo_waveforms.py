#!/usr/bin/env python
"""Generate demo waveforms from training data for frontend."""
import pandas as pd
import json
import numpy as np
from pathlib import Path

def main():
    # Load training data
    df = pd.read_csv('data/raw/waveforms.csv')
    meta = json.load(open('data/raw/metadata.json'))
    
    # Get class names from metadata
    class_names = meta.get('class_names', [f'CLASS_{i}' for i in range(8)])
    
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(df)}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()[:10]}...")
    
    # Identify which columns are waveform data (numeric) vs labels
    waveform_cols = [col for col in df.columns if col.startswith('t')][:1024]
    label_col = 'label' if 'label' in df.columns else None
    fault_name_col = 'fault_name' if 'fault_name' in df.columns else None
    
    print(f"Waveform columns: {len(waveform_cols)}")
    print(f"Label column: {label_col}")
    print(f"Fault name column: {fault_name_col}")
    
    if len(waveform_cols) >= 1024 and label_col:
        # Get first sample for each class
        demo_waveforms = {}
        
        for class_idx, class_name in enumerate(class_names):
            # Find samples with this class
            class_mask = (df[label_col] == class_idx)
            if class_mask.any():
                sample_idx = class_mask.idxmax()
                waveform = df.loc[sample_idx, waveform_cols].values.astype(float).tolist()
                # Round to 4 decimals to save space
                demo_waveforms[class_name] = [round(v, 4) for v in waveform[:1024]]
                fault_name = df.loc[sample_idx, fault_name_col] if fault_name_col else 'Unknown'
                print(f"✓ {class_name} ({fault_name}): {len(waveform)} samples")
        
        # Save to file
        with open('demo_waveforms.json', 'w') as f:
            json.dump(demo_waveforms, f)
        print(f"\n✓ Saved {len(demo_waveforms)} demo waveforms to demo_waveforms.json")
        
        # Print JSON for use in HTML
        print("\nJSON for HTML (first 150 chars):")
        json_str = json.dumps(demo_waveforms)
        print(json_str[:150] + "...")

    else:
        print(f"ERROR: Not enough waveform columns. Found {len(waveform_cols)}, expected 1024")

if __name__ == '__main__':
    main()
