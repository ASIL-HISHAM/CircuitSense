# CircuitSense

CircuitSense is a waveform-based fault detection project.  
It uses a machine learning model to predict circuit faults from signal data.

## What it does
- Takes waveform values as input
- Predicts the most likely fault type
- Returns confidence and class probabilities

## Supported classes
- NORMAL
- CAP_DEGRADED
- CAP_SHORT
- RES_DRIFT
- TRANSISTOR_SATURATION
- DIODE_OPEN
- INDUCTOR_CORE_SAT
- POWER_RAIL_NOISE

## How to test
1. Start the FastAPI server.
2. Open `/docs` in your browser.
3. Use the `/predict` endpoint.
4. Paste the waveform values and run the request.

## Example result
The model can identify faults such as resistor drift or capacitor short from waveform shape.
