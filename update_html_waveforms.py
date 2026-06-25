#!/usr/bin/env python
"""Update HTML with full-length demo waveforms."""
import json
import re

# Read the full demo waveforms
with open('demo_waveforms.json', 'r') as f:
    demo_data = json.load(f)

# Map class names to expected names in HTML
class_mapping = {
    'CLASS_0': 'NORMAL',
    'CLASS_1': 'CAP_DEGRADED',
    'CLASS_2': 'CAP_SHORT',
    'CLASS_3': 'RES_DRIFT',
    'CLASS_4': 'TRANSISTOR_SATURATION',
    'CLASS_5': 'DIODE_OPEN',
    'CLASS_6': 'INDUCTOR_CORE_SAT',
    'CLASS_7': 'POWER_RAIL_NOISE'
}

# Rename keys
renamed_data = {v: demo_data[k] for k, v in class_mapping.items() if k in demo_data}

# Create the JSON string for HTML
json_str = json.dumps(renamed_data)

# Read the HTML file
with open('frontend/index_minimal.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Find and replace the demoWaveformsData script
pattern = r'<script id="demoWaveformsData" type="application/json">.*?</script>'
replacement = f'<script id="demoWaveformsData" type="application/json">{json_str}</script>'

html_updated = re.sub(pattern, replacement, html_content, flags=re.DOTALL)

# Write back
with open('frontend/index_minimal.html', 'w', encoding='utf-8') as f:
    f.write(html_updated)

print('✓ HTML updated with full-length demo waveforms')
print(f'✓ JSON size: {len(json_str)} chars')
print(f'✓ Classes: {list(renamed_data.keys())}')
