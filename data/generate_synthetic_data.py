"""
data/generate_synthetic_data.py
────────────────────────────────────────────────────────────────────────────
Generates all synthetic data needed for the CAE Platform:

  1. NVH vibration sensor readings  → data/synthetic/sensor_readings.json
  2. Motor simulation dataset       → data/synthetic/motor_simulation.csv
  3. Engineering standards KB       → data/synthetic/nvh_knowledge_base.json
  4. Golden Q&A eval set            → data/synthetic/golden_qa.json

Run:  python data/generate_synthetic_data.py
────────────────────────────────────────────────────────────────────────────
"""

import json
import csv
import random
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("data/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 1.  NVH VIBRATION SENSOR READINGS
# ══════════════════════════════════════════════════════════════════════════

def generate_nvh_signal(
    dominant_freq_hz: float,
    amplitude_db: float,
    rpm: int,
    component: str,
    duration: float = 0.5,
    fs: int = 10_000,
) -> dict:
    """
    Simulates a vibration time-series, applies FFT, and returns
    the structured features an LLM agent can reason about.
    """
    t = np.linspace(0, duration, int(fs * duration))

    # Build signal: fundamental + 2nd and 3rd harmonics + noise
    fundamental   = amplitude_db / 100 * np.sin(2 * np.pi * dominant_freq_hz * t)
    second_harm   = (amplitude_db / 100) * 0.35 * np.sin(2 * np.pi * 2 * dominant_freq_hz * t)
    third_harm    = (amplitude_db / 100) * 0.15 * np.sin(2 * np.pi * 3 * dominant_freq_hz * t)
    noise         = 0.05 * np.random.randn(len(t))
    signal        = fundamental + second_harm + third_harm + noise

    # FFT → frequency domain features
    fft_mag    = np.abs(np.fft.fft(signal))[: len(t) // 2]
    freqs      = np.fft.fftfreq(len(t), 1 / fs)[: len(t) // 2]
    peak_idx   = np.argsort(fft_mag)[-3:][::-1]           # top-3 peaks
    peak_freqs = [round(float(freqs[i]), 1) for i in peak_idx]
    peak_amps  = [round(float(20 * np.log10(fft_mag[i] + 1e-9)), 2) for i in peak_idx]

    # Classify frequency range
    if dominant_freq_hz < 200:
        freq_range = "low"
    elif dominant_freq_hz < 1000:
        freq_range = "mid"
    else:
        freq_range = "high"

    # Classify severity
    if amplitude_db < 50:
        severity = 1
    elif amplitude_db < 65:
        severity = 2
    elif amplitude_db < 75:
        severity = 3
    elif amplitude_db < 85:
        severity = 4
    else:
        severity = 5

    return {
        "rpm":             rpm,
        "component":       component,
        "dominant_freq_hz": round(dominant_freq_hz, 1),
        "amplitude_db":    round(amplitude_db, 2),
        "freq_range":      freq_range,
        "severity":        severity,
        "peak_frequencies_hz": peak_freqs,
        "peak_amplitudes_db":  peak_amps,
        "harmonic_ratio":  round(float(second_harm.max() / (fundamental.max() + 1e-9)), 3),
        "signal_rms":      round(float(np.sqrt(np.mean(signal ** 2))), 5),
        "signal_quality":  "good" if np.std(signal) > 0.01 else "noisy",
    }


# Define realistic NVH scenarios
SCENARIOS = [
    # (component, freq_hz, amp_db, rpm, label)
    ("electric_motor",  847.0, 82, 3000, "BPF_resonance_eNVH"),
    ("electric_motor",  500.0, 78, 2000, "electromagnetic_whine"),
    ("electric_motor", 1200.0, 72, 4000, "high_freq_eNVH"),
    ("blower",          340.0, 70, 1500, "blade_pass_frequency"),
    ("blower",          680.0, 75, 3000, "BPF_second_harmonic"),
    ("sheet_metal",     120.0, 65, 1000, "panel_resonance"),
    ("sheet_metal",      85.0, 60,  800, "low_freq_structural"),
    ("chassis",         210.0, 68, 1800, "structural_resonance"),
    ("chassis",          55.0, 58,  600, "body_bending_mode"),
    ("gearbox",         420.0, 80, 2800, "gear_mesh_frequency"),
    ("gearbox",         840.0, 74, 2800, "GMF_second_harmonic"),
    ("bearing",        2400.0, 69, 3600, "BPFI_defect"),
    ("bearing",        1800.0, 66, 3600, "BPFO_defect"),
    ("exhaust",          95.0, 62, 1200, "exhaust_resonance"),
    ("mount",           180.0, 71, 2400, "mount_isolation_failure"),
]

sensor_readings = []
for comp, freq, amp, rpm, label in SCENARIOS:
    # Generate 4 variants per scenario (slight variations)
    for i in range(4):
        freq_var  = freq  * (1 + random.uniform(-0.05, 0.05))
        amp_var   = amp   + random.uniform(-3, 3)
        rpm_var   = rpm   + random.randint(-100, 100)
        reading   = generate_nvh_signal(freq_var, amp_var, rpm_var, comp)
        reading["scenario_label"] = label
        reading["sample_id"]      = f"{comp}_{label}_{i+1}"
        sensor_readings.append(reading)

out_path = OUTPUT_DIR / "sensor_readings.json"
with open(out_path, "w") as f:
    json.dump(sensor_readings, f, indent=2)

print(f"✓ Sensor readings  → {out_path}  ({len(sensor_readings)} samples)")


# ══════════════════════════════════════════════════════════════════════════
# 2.  MOTOR SIMULATION DATASET  (for surrogate model in Step 3)
# ══════════════════════════════════════════════════════════════════════════

def simulate_nvh_output(
    rpm, load_nm, temp_c, stator_slots, rotor_poles, air_gap_mm
) -> float:
    """
    Physics-inspired surrogate: NVH noise level in dB.
    Captures known NVH trends: higher RPM/load → more noise,
    larger air gap → lower noise, temperature affects material damping.
    """
    bpf_component   = 0.008  * rpm * stator_slots / rotor_poles
    load_component  = 0.15   * load_nm
    temp_effect     = -0.05  * (temp_c - 25)          # damping increases with temp
    gap_effect      = -3.5   * air_gap_mm              # larger gap = less EM force
    base_noise      = 55.0
    interaction     = 0.0003 * rpm * load_nm / 100     # RPM×load cross-term
    noise           = (base_noise + bpf_component + load_component
                       + temp_effect + gap_effect + interaction
                       + random.gauss(0, 1.5))         # measurement noise
    return round(max(40.0, min(noise, 100.0)), 2)


rows = []
for _ in range(600):
    rpm        = random.randint(500, 6000)
    load       = round(random.uniform(0.5, 120.0), 1)
    temp       = round(random.uniform(20, 120), 1)
    slots      = random.choice([12, 18, 24, 36, 48])
    poles      = random.choice([4, 6, 8, 10])
    air_gap    = round(random.uniform(0.3, 1.5), 2)
    noise_db   = simulate_nvh_output(rpm, load, temp, slots, poles, air_gap)
    rows.append({
        "rpm":           rpm,
        "load_nm":       load,
        "temperature_c": temp,
        "stator_slots":  slots,
        "rotor_poles":   poles,
        "air_gap_mm":    air_gap,
        "nvh_db":        noise_db,          # ← target variable
    })

out_path = OUTPUT_DIR / "motor_simulation.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Motor simulation → {out_path}  ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════════════════
# 3.  ENGINEERING KNOWLEDGE BASE  (50 NVH case documents)
# ══════════════════════════════════════════════════════════════════════════

KB_CASES = [
    # ── Electric motor / e-NVH ────────────────────────────────────────────
    {
        "case_id": "eNVH_001",
        "title": "Blade Pass Frequency resonance in BLDC motor at 847 Hz",
        "resonance_type": "eNVH",
        "component": "electric_motor",
        "freq_range": "high",
        "severity": 4,
        "description": (
            "A BLDC motor operating at 3000 RPM produces a prominent tonal noise at 847 Hz, "
            "corresponding to the Blade Pass Frequency (BPF = RPM/60 × stator_slots). "
            "Electromagnetic forces acting on the stator teeth at BPF cause radial vibration "
            "of the stator housing. The frequency coincides with a structural resonance of the "
            "motor casing, amplifying the radiated noise to 82 dB."
        ),
        "root_cause": "BPF coincidence with stator housing structural resonance mode.",
        "corrective_action": (
            "Detune the stator resonance by modifying housing wall thickness (+2 mm). "
            "Alternatively, change stator slot count to shift BPF away from structural mode. "
            "Apply constrained-layer damping treatment to motor housing exterior."
        ),
        "standards_ref": "IEC 60034-9: Permissible noise levels for rotating machinery",
        "verified_reduction_db": 6.5,
    },
    {
        "case_id": "eNVH_002",
        "title": "Electromagnetic whine at 500 Hz due to Maxwell stress harmonics",
        "resonance_type": "eNVH",
        "component": "electric_motor",
        "freq_range": "mid",
        "severity": 4,
        "description": (
            "Maxwell stress forces on the stator teeth generate radial electromagnetic forces "
            "at 500 Hz and harmonics (1000 Hz, 1500 Hz). The 500 Hz component excites a "
            "bending mode of the motor end-shield, generating a distinct tonal whine audible "
            "in the passenger cabin. Altair Flux simulation confirmed force amplitude of 380 N/m²."
        ),
        "root_cause": "Maxwell stress force at electrical fundamental frequency exciting end-shield bending mode.",
        "corrective_action": (
            "Stiffen end-shield with radial ribs to shift bending mode above 800 Hz. "
            "Alternatively, use skewed rotor slots to reduce harmonic force content by ~40%."
        ),
        "standards_ref": "ISO 1680: Acoustics — Test code for airborne noise emitted by rotating electrical machinery",
        "verified_reduction_db": 4.0,
    },
    {
        "case_id": "eNVH_003",
        "title": "High-frequency e-NVH at 1200 Hz in EV traction motor",
        "resonance_type": "eNVH",
        "component": "electric_motor",
        "freq_range": "high",
        "severity": 3,
        "description": (
            "An EV traction motor at 4000 RPM produces high-frequency noise at 1200 Hz. "
            "Frequency analysis shows this corresponds to 3rd harmonic of the fundamental "
            "electromagnetic force. The noise is particularly noticeable during acceleration "
            "ramp-up between 2500 and 4500 RPM."
        ),
        "root_cause": "3rd harmonic electromagnetic excitation — insufficient PWM switching frequency.",
        "corrective_action": (
            "Increase inverter PWM switching frequency from 8 kHz to 16 kHz to shift "
            "switching harmonics above audible range. Add passive damping treatment to "
            "motor mounting brackets."
        ),
        "standards_ref": "ISO 10816-3: Vibration severity evaluation for motors above 15 kW",
        "verified_reduction_db": 7.0,
    },
    # ── Blower / aeroacoustics ─────────────────────────────────────────────
    {
        "case_id": "AERO_001",
        "title": "Blower blade pass frequency noise at 340 Hz",
        "resonance_type": "aeroacoustic",
        "component": "blower",
        "freq_range": "mid",
        "severity": 3,
        "description": (
            "A 9-blade centrifugal blower at 1500 RPM generates tonal noise at 340 Hz "
            "(BPF = 9 × 1500/60 = 225 Hz → measured at 340 Hz due to acoustic cavity resonance "
            "amplification). ESI VA One aero-vibroacoustic simulation confirmed acoustic cavity "
            "coupling with the blower scroll housing."
        ),
        "root_cause": "BPF excitation coupled with acoustic resonance of blower scroll housing cavity.",
        "corrective_action": (
            "Modify scroll housing volume to detune acoustic cavity resonance. "
            "Change blade count from 9 to 11 blades to shift BPF. "
            "Add acoustic absorber lining inside scroll housing."
        ),
        "standards_ref": "AMCA 300: Reverberant room method for sound testing of fans",
        "verified_reduction_db": 4.0,
    },
    {
        "case_id": "AERO_002",
        "title": "BPF second harmonic resonance at 680 Hz causing appliance NVH",
        "resonance_type": "aeroacoustic",
        "component": "blower",
        "freq_range": "mid",
        "severity": 4,
        "description": (
            "A washing machine blower at 3000 RPM produces strong tonal noise at 680 Hz "
            "corresponding to the second harmonic of BPF. Aero-vibroacoustic analysis in "
            "ESI VA One showed the 680 Hz harmonic coincides with a structural resonance "
            "of the blower housing panel, amplifying radiated noise by 9 dB."
        ),
        "root_cause": "2nd harmonic BPF at 680 Hz coupled with housing panel resonance.",
        "corrective_action": (
            "Add stiffening beads to blower housing panel to shift structural resonance above 900 Hz. "
            "Apply constrained-layer damping to panel exterior. Achieved verified 4 dBA reduction."
        ),
        "standards_ref": "IEC 60704-2-6: Household appliance noise measurement — washing machines",
        "verified_reduction_db": 4.0,
    },
    # ── Structural / sheet metal ───────────────────────────────────────────
    {
        "case_id": "STRUCT_001",
        "title": "Sheet metal panel resonance at 120 Hz — body boom",
        "resonance_type": "structural",
        "component": "sheet_metal",
        "freq_range": "low",
        "severity": 3,
        "description": (
            "Vehicle body panel resonance at 120 Hz causes low-frequency 'body boom' in cabin. "
            "Modal analysis (MSC Nastran) identified a door panel bending mode at 118 Hz. "
            "Excitation from road input at 120 Hz causes panel vibration, radiating low-frequency "
            "airborne noise into the cabin."
        ),
        "root_cause": "Door panel bending resonance mode at 118 Hz coinciding with road excitation frequency.",
        "corrective_action": (
            "Add 1.5 kg bitumen damping patch (400×300 mm) to inner door panel. "
            "Alternatively, add stiffening ribs to shift panel resonance above 160 Hz. "
            "Both methods verified to reduce boom by 5–8 dB."
        ),
        "standards_ref": "ISO 16940: Glass in building — Glazing and airborne sound insulation",
        "verified_reduction_db": 6.0,
    },
    {
        "case_id": "STRUCT_002",
        "title": "Low-frequency structural resonance at 85 Hz — chassis flex mode",
        "resonance_type": "structural",
        "component": "sheet_metal",
        "freq_range": "low",
        "severity": 2,
        "description": (
            "Commercial vehicle chassis exhibits first bending mode at 85 Hz. At engine idle "
            "(750 RPM → 12.5 Hz firing frequency × 7 = 87.5 Hz), chassis resonance is excited "
            "causing whole-body vibration. FE model in MSC Nastran confirmed mode shape is "
            "first-order chassis bending."
        ),
        "root_cause": "Chassis first bending mode at 85 Hz coinciding with engine firing harmonic.",
        "corrective_action": (
            "Add cross-member reinforcement at chassis mid-span to raise bending mode above 110 Hz. "
            "Alternatively, tune engine mounts to provide better isolation at 85–90 Hz range."
        ),
        "standards_ref": "ISO 2631-1: Evaluation of human exposure to whole-body vibration",
        "verified_reduction_db": 5.0,
    },
    # ── Gearbox ───────────────────────────────────────────────────────────
    {
        "case_id": "GEAR_001",
        "title": "Gear mesh frequency noise at 420 Hz in transmission",
        "resonance_type": "structural",
        "component": "gearbox",
        "freq_range": "mid",
        "severity": 4,
        "description": (
            "Transmission gearbox at 2800 RPM produces tonal noise at 420 Hz. "
            "Gear mesh frequency GMF = RPM/60 × number_of_teeth = 2800/60 × 9 = 420 Hz. "
            "The GMF excites a torsional mode of the output shaft, causing structure-borne "
            "noise transmission to the vehicle body."
        ),
        "root_cause": "GMF at 420 Hz coinciding with shaft torsional resonance — inadequate gear profile modification.",
        "corrective_action": (
            "Apply tip relief profile modification to reduce dynamic transmission error. "
            "Increase gear lead crowning to reduce edge loading. "
            "Install tuned vibration absorber on output shaft tuned to 420 Hz."
        ),
        "standards_ref": "ISO 6336-1: Calculation of load capacity of spur and helical gears",
        "verified_reduction_db": 5.5,
    },
    # ── Bearing ───────────────────────────────────────────────────────────
    {
        "case_id": "BEAR_001",
        "title": "Bearing inner race defect frequency (BPFI) at 2400 Hz",
        "resonance_type": "structural",
        "component": "bearing",
        "freq_range": "high",
        "severity": 5,
        "description": (
            "Motor bearing at 3600 RPM produces impulsive noise at 2400 Hz with characteristic "
            "BPFI signature (inner race defect). BPFI = RPM/60 × (N_balls/2) × (1 + ball_dia/PCD × cos α). "
            "Early-stage bearing pitting detected — if not addressed, will lead to catastrophic failure "
            "within approximately 200 operating hours."
        ),
        "root_cause": "Inner race surface fatigue pitting causing periodic impact excitation at BPFI.",
        "corrective_action": (
            "IMMEDIATE: Replace bearing within next 50 operating hours. "
            "Root cause investigation: check shaft alignment (< 0.05 mm runout), "
            "lubrication condition, and load rating adequacy."
        ),
        "standards_ref": "ISO 15243: Rolling bearings — Damage and failures — Terms, characteristics and causes",
        "verified_reduction_db": 18.0,
    },
    {
        "case_id": "BEAR_002",
        "title": "Bearing outer race defect frequency (BPFO) at 1800 Hz",
        "resonance_type": "structural",
        "component": "bearing",
        "freq_range": "high",
        "severity": 4,
        "description": (
            "Outer race defect producing periodic impulses at BPFO = 1800 Hz. "
            "Envelope analysis of vibration signal reveals clear sidebands around BPFO "
            "indicating progressed outer race spalling. Current severity: moderate — "
            "bearing replacement recommended within 500 hours."
        ),
        "root_cause": "Outer race spalling due to inadequate load distribution — possible misalignment.",
        "corrective_action": (
            "Replace bearing within next planned maintenance interval (< 500 hours). "
            "Check housing bore for out-of-roundness. Verify preload and axial clearance."
        ),
        "standards_ref": "ISO 15243: Rolling bearings — Damage and failures",
        "verified_reduction_db": 12.0,
    },
    # ── Mount / isolation ─────────────────────────────────────────────────
    {
        "case_id": "MOUNT_001",
        "title": "Engine mount isolation failure causing structure-borne noise at 180 Hz",
        "resonance_type": "structural",
        "component": "mount",
        "freq_range": "low",
        "severity": 4,
        "description": (
            "Degraded engine mount rubber causing poor vibration isolation above 150 Hz. "
            "At 2400 RPM, engine 2nd-order firing frequency (80 Hz) and higher harmonics "
            "pass through mount with less than 6 dB insertion loss. Expected isolation should "
            "be >15 dB. Stiffness measurement confirms mount dynamic stiffness has increased "
            "3× from new due to rubber hardening."
        ),
        "root_cause": "Rubber mount aging/hardening — dynamic stiffness increase from 150 N/mm to 470 N/mm.",
        "corrective_action": (
            "Replace engine mounts immediately. Use hydro-mounts for improved high-frequency isolation. "
            "Specify maximum service life of 80,000 km or 5 years for rubber mounts."
        ),
        "standards_ref": "ISO 10846-1: Acoustics and vibration — Laboratory measurement of vibro-acoustic transfer",
        "verified_reduction_db": 9.0,
    },
    # ── Compliance cases ──────────────────────────────────────────────────
    {
        "case_id": "COMP_001",
        "title": "ISO 362 drive-by noise limit compliance — passenger car",
        "resonance_type": "compliance",
        "component": "vehicle_system",
        "freq_range": "broadband",
        "severity": 3,
        "description": (
            "ISO 362-1:2015 sets maximum drive-by noise limits for M1 passenger cars at 72 dB(A). "
            "Measurement procedure: vehicle accelerates through 20m test section at 50 km/h, "
            "microphone at 7.5m lateral distance, 1.2m height. A-weighting applied. "
            "Results must be within 72 dB(A) for type approval."
        ),
        "root_cause": "N/A — compliance standard definition",
        "corrective_action": (
            "Primary contributors to drive-by noise: engine/intake/exhaust (30%), "
            "tyre-road interaction (45%), aerodynamic noise (25%). "
            "Address highest contributor first — typically tyre noise at highway speeds."
        ),
        "standards_ref": "ISO 362-1:2015 — Measurement of noise emitted by accelerating road vehicles",
        "verified_reduction_db": 0,
    },
    {
        "case_id": "COMP_002",
        "title": "IEC 60704 household appliance noise limit compliance",
        "resonance_type": "compliance",
        "component": "appliance_system",
        "freq_range": "broadband",
        "severity": 2,
        "description": (
            "IEC 60704 series defines noise measurement procedures for household appliances. "
            "For washing machines: maximum declared noise 72 dB(A) wash / 77 dB(A) spin. "
            "Measurement in anechoic room, microphone 1m from machine surface, 5 positions averaged. "
            "Energy label requirement: noise ≤ 72/72 dB(A) for A-rating."
        ),
        "root_cause": "N/A — compliance standard definition",
        "corrective_action": (
            "Primary contributors: drain pump (high freq), motor (mid freq), "
            "drum bearing (broadband), water inlet valve (impulsive). "
            "Tonal components above 6 dB(A) above background are most annoying and prioritised."
        ),
        "standards_ref": "IEC 60704-2-6:2011 — Household and similar appliances, washing machines",
        "verified_reduction_db": 0,
    },
]

# Add 38 more cases to reach 50 total
EXTRA_CASES = [
    ("eNVH_004", "Torque ripple at 6× electrical frequency in PMSM", "eNVH", "electric_motor", "mid", 3,
     "Permanent Magnet Synchronous Motor exhibits torque ripple at 6× electrical frequency (300 Hz at 3000 RPM). "
     "The ripple is caused by 5th and 7th current harmonics producing pulsating torque. Vibration transmitted "
     "through driveshaft to body structure.",
     "5th and 7th harmonic current injection from inverter producing 6× torque ripple.",
     "Implement feed-forward harmonic current injection cancellation in motor controller. "
     "Alternatively use fractional-slot winding design to reduce cogging."),
    ("eNVH_005", "Cogging torque noise at low speed (under 500 RPM)", "eNVH", "electric_motor", "low", 2,
     "At low rotational speeds below 500 RPM, cogging torque produces perceptible vibration and acoustic click. "
     "Cogging frequency = RPM/60 × LCM(stator_slots, rotor_poles). Particularly noticeable during "
     "start-stop cycles in EV applications.",
     "Cogging torque due to magnetic reluctance variation between rotor and stator teeth.",
     "Use rotor skewing (1 slot pitch) to reduce cogging by 60–80%. "
     "Alternatively use concentrated winding with fractional slot/pole combination."),
    ("AERO_003", "Fan unbalance noise at 1× rotational frequency", "aeroacoustic", "blower", "low", 2,
     "Fan shows 1× rotational frequency vibration at 25 Hz (1500 RPM). Unbalance force = m × e × ω². "
     "Residual unbalance after production balancing exceeds ISO 21940-11 Grade G2.5 tolerance.",
     "Residual mass unbalance after production balancing — tolerance grade G6.3 used instead of G2.5.",
     "Re-balance fan assembly to ISO 21940-11 Grade G2.5. "
     "Check for asymmetric blade fouling or manufacturing defect."),
    ("STRUCT_003", "Dashboard resonance at 145 Hz causing buzz-squeak-rattle", "structural", "sheet_metal", "low", 3,
     "Instrument panel exhibits rattling noise at 145 Hz when excited by road input. "
     "Multiple loose attachment points and plastic panel resonances contribute to broadband rattle. "
     "Customer complaint rate 2.3% — above 1.5% internal quality threshold.",
     "Multiple IP attachment point looseness combined with plastic panel resonance at 143 Hz.",
     "Apply BSR (Buzz-Squeak-Rattle) foam tape at all IP-to-body interface points. "
     "Retorque all screws to specification. Add anti-squeak flocking on mating plastic surfaces."),
    ("GEAR_002", "Whining noise from differential at highway speed", "structural", "gearbox", "mid", 3,
     "Rear differential produces whining noise at 55–80 km/h corresponding to GMF of 380–550 Hz. "
     "Gear contact pattern inspection shows uneven load distribution — heavy toe contact.",
     "Incorrect gear lapping pattern — uneven contact due to assembly shimming error.",
     "Adjust differential pinion shim to achieve uniform heel-to-toe contact pattern. "
     "Verify gear backlash within 0.10–0.18 mm specification."),
]

for i, (cid, title, rtype, comp, freq_r, sev, desc, cause, fix) in enumerate(EXTRA_CASES):
    KB_CASES.append({
        "case_id": cid, "title": title, "resonance_type": rtype,
        "component": comp, "freq_range": freq_r, "severity": sev,
        "description": desc, "root_cause": cause, "corrective_action": fix,
        "standards_ref": "See relevant ISO/IEC standard for component type",
        "verified_reduction_db": round(random.uniform(2, 8), 1),
    })

# Pad to 50
while len(KB_CASES) < 50:
    idx = len(KB_CASES)
    KB_CASES.append({
        "case_id": f"GEN_{idx:03d}",
        "title": f"NVH case study {idx} — mixed frequency vibration",
        "resonance_type": random.choice(["structural", "aeroacoustic", "eNVH"]),
        "component": random.choice(["electric_motor","blower","sheet_metal","chassis","bearing","mount"]),
        "freq_range": random.choice(["low","mid","high"]),
        "severity": random.randint(1, 5),
        "description": (
            f"Case {idx}: Component exhibits vibration at multiple frequencies. "
            "Detailed multi-physics simulation required for root cause identification. "
            "Combined structural and acoustic contribution requires BEM/FEM coupled analysis."
        ),
        "root_cause": "Multi-source excitation — requires dedicated measurement campaign.",
        "corrective_action": "Perform Operational Deflection Shape (ODS) analysis to identify dominant path.",
        "standards_ref": "ISO 10816 series — Vibration severity evaluation",
        "verified_reduction_db": round(random.uniform(1.5, 6.0), 1),
    })

out_path = OUTPUT_DIR / "nvh_knowledge_base.json"
with open(out_path, "w") as f:
    json.dump(KB_CASES[:50], f, indent=2)

print(f"✓ NVH knowledge base → {out_path}  ({len(KB_CASES[:50])} cases)")


# ══════════════════════════════════════════════════════════════════════════
# 4.  GOLDEN Q&A EVALUATION SET  (20 pairs for Pytest LLM-as-judge)
# ══════════════════════════════════════════════════════════════════════════

GOLDEN_QA = [
    {
        "id": "qa_001",
        "question": "What is the root cause of blade pass frequency noise at 340 Hz in a blower?",
        "expected_topics": ["BPF", "blade pass frequency", "acoustic cavity", "resonance"],
        "reference_case": "AERO_001",
        "expected_answer_contains": [
            "blade pass frequency",
            "acoustic cavity resonance",
            "scroll housing",
        ],
        "category": "root_cause",
    },
    {
        "id": "qa_002",
        "question": "My electric motor at 3000 RPM shows 82 dB noise at 847 Hz. What is causing this and how do I fix it?",
        "expected_topics": ["BPF", "stator resonance", "housing", "damping"],
        "reference_case": "eNVH_001",
        "expected_answer_contains": [
            "blade pass frequency",
            "stator",
            "housing",
            "damping",
        ],
        "category": "diagnosis_and_fix",
    },
    {
        "id": "qa_003",
        "question": "What does ISO 362-1 specify for drive-by noise limits of passenger cars?",
        "expected_topics": ["ISO 362", "72 dB", "drive-by", "type approval"],
        "reference_case": "COMP_001",
        "expected_answer_contains": ["72 dB", "ISO 362", "passenger", "50 km/h"],
        "category": "compliance",
    },
    {
        "id": "qa_004",
        "question": "A bearing shows BPFI signature at 2400 Hz with severity 5. What action should I take?",
        "expected_topics": ["bearing", "inner race", "BPFI", "replacement", "urgent"],
        "reference_case": "BEAR_001",
        "expected_answer_contains": ["replace", "50 hours", "inner race", "alignment"],
        "category": "urgent_action",
    },
    {
        "id": "qa_005",
        "question": "How do I reduce cogging torque noise in a PMSM motor at low speeds?",
        "expected_topics": ["cogging", "skewing", "fractional slot", "rotor"],
        "reference_case": "eNVH_005",
        "expected_answer_contains": ["skew", "cogging", "fractional"],
        "category": "design_fix",
    },
    {
        "id": "qa_006",
        "question": "What is gear mesh frequency and why does it cause noise in a gearbox?",
        "expected_topics": ["GMF", "gear mesh", "torsional", "teeth"],
        "reference_case": "GEAR_001",
        "expected_answer_contains": ["gear mesh frequency", "teeth", "RPM"],
        "category": "root_cause",
    },
    {
        "id": "qa_007",
        "question": "Dashboard rattling noise at 145 Hz — what is BSR and how to fix it?",
        "expected_topics": ["BSR", "buzz squeak rattle", "foam", "attachment"],
        "reference_case": "STRUCT_003",
        "expected_answer_contains": ["buzz", "squeak", "rattle", "foam tape"],
        "category": "diagnosis_and_fix",
    },
    {
        "id": "qa_008",
        "question": "Engine mount dynamic stiffness has increased 3x from new. What does this mean for NVH?",
        "expected_topics": ["mount", "stiffness", "isolation", "aging"],
        "reference_case": "MOUNT_001",
        "expected_answer_contains": ["stiffness", "isolation", "hardening", "replace"],
        "category": "diagnosis_and_fix",
    },
    {
        "id": "qa_009",
        "question": "What is Maxwell stress tensor and how does it cause electromagnetic NVH?",
        "expected_topics": ["Maxwell stress", "electromagnetic force", "stator", "radial force"],
        "reference_case": "eNVH_002",
        "expected_answer_contains": ["Maxwell stress", "electromagnetic", "stator", "radial"],
        "category": "concept_explanation",
    },
    {
        "id": "qa_010",
        "question": "IEC 60704 standard — what is the noise limit for washing machine in spin cycle?",
        "expected_topics": ["IEC 60704", "77 dB", "spin", "washing machine"],
        "reference_case": "COMP_002",
        "expected_answer_contains": ["77 dB", "IEC 60704", "spin"],
        "category": "compliance",
    },
    {
        "id": "qa_011",
        "question": "How do I calculate blade pass frequency for a 9-blade fan at 1500 RPM?",
        "expected_topics": ["BPF", "blade count", "RPM", "calculation"],
        "reference_case": "AERO_001",
        "expected_answer_contains": ["225 Hz", "1500", "9"],
        "category": "calculation",
    },
    {
        "id": "qa_012",
        "question": "What is the difference between BPFI and BPFO in bearing diagnostics?",
        "expected_topics": ["BPFI", "BPFO", "inner race", "outer race", "defect"],
        "reference_case": "BEAR_001",
        "expected_answer_contains": ["inner race", "outer race", "BPFI", "BPFO"],
        "category": "concept_explanation",
    },
    {
        "id": "qa_013",
        "question": "What structural modification eliminates sheet metal panel resonance at 120 Hz?",
        "expected_topics": ["panel resonance", "damping patch", "stiffening", "bitumen"],
        "reference_case": "STRUCT_001",
        "expected_answer_contains": ["damping", "stiffening", "resonance"],
        "category": "design_fix",
    },
    {
        "id": "qa_014",
        "question": "PWM switching frequency affects high-frequency motor noise — explain this.",
        "expected_topics": ["PWM", "switching frequency", "harmonics", "inverter"],
        "reference_case": "eNVH_003",
        "expected_answer_contains": ["PWM", "switching frequency", "harmonic"],
        "category": "concept_explanation",
    },
    {
        "id": "qa_015",
        "question": "A sensor reading shows dominant frequency 500 Hz, amplitude 78 dB, component electric_motor. Diagnose.",
        "expected_topics": ["electromagnetic whine", "Maxwell stress", "end-shield", "500 Hz"],
        "reference_case": "eNVH_002",
        "expected_answer_contains": ["electromagnetic", "500", "motor"],
        "category": "sensor_diagnosis",
    },
    {
        "id": "qa_016",
        "question": "What is torque ripple and at what frequency does it occur in a 6-pole PMSM at 3000 RPM?",
        "expected_topics": ["torque ripple", "6x electrical", "PMSM", "300 Hz"],
        "reference_case": "eNVH_004",
        "expected_answer_contains": ["torque ripple", "300 Hz", "6"],
        "category": "calculation",
    },
    {
        "id": "qa_017",
        "question": "How does rotor skewing reduce cogging torque in a permanent magnet motor?",
        "expected_topics": ["skewing", "cogging", "reluctance", "slot pitch"],
        "reference_case": "eNVH_005",
        "expected_answer_contains": ["skew", "cogging", "slot pitch", "reluctance"],
        "category": "design_fix",
    },
    {
        "id": "qa_018",
        "question": "Differential whining noise at highway speed — what is the diagnostic procedure?",
        "expected_topics": ["differential", "GMF", "contact pattern", "shimming"],
        "reference_case": "GEAR_002",
        "expected_answer_contains": ["contact pattern", "backlash", "shim"],
        "category": "diagnosis_and_fix",
    },
    {
        "id": "qa_019",
        "question": "ISO 2631-1 — what does it evaluate and what are the comfort threshold values?",
        "expected_topics": ["ISO 2631", "whole body vibration", "comfort", "acceleration"],
        "reference_case": "STRUCT_002",
        "expected_answer_contains": ["ISO 2631", "whole-body", "vibration"],
        "category": "compliance",
    },
    {
        "id": "qa_020",
        "question": "A fan has residual unbalance causing 1× vibration at 25 Hz. What ISO standard applies and what is the correction?",
        "expected_topics": ["ISO 21940", "unbalance", "G2.5", "balancing"],
        "reference_case": "AERO_003",
        "expected_answer_contains": ["ISO 21940", "balance", "G2.5"],
        "category": "compliance",
    },
]

out_path = OUTPUT_DIR / "golden_qa.json"
with open(out_path, "w") as f:
    json.dump(GOLDEN_QA, f, indent=2)

print(f"✓ Golden Q&A set    → {out_path}  ({len(GOLDEN_QA)} pairs)")
print("\n✅  All synthetic data generated successfully!")
print(f"   Output directory: {OUTPUT_DIR.resolve()}")
