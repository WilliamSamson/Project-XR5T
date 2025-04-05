import copy
import random
import time
import json
import math
import threading
import logging
import sys

# Configuration
THRESHOLD_BASE = 0.15
SIMULATION_STEPS = 15
ERROR_MARGIN = 0.04
LOG_FILE = "../system_log.txt"
MEMORY_FILE = "adaptive_memory.json"
SIM_INTERVAL = 0.3
VERBOSE = True

# Logging Setup
LOG_COLORS = {
    "STATE_UPDATE": "\033[94m",  # Blue
    "TIPPING_POINT_DETECTED": "\033[91m",  # Red
    "ADAPTIVE_LEARN": "\033[92m",  # Green
    "PUNISH": "\033[93m",  # Yellow
    "THRESHOLD_ADJUST": "\033[96m",  # Cyan
    "SIMULATION_RESULT": "\033[90m",  # Gray
    "RESET_TRIGGERED": "\033[95m",  # Magenta
    "DEFAULT": "\033[0m",
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        event_type = getattr(record, 'event_type', 'DEFAULT')
        color = LOG_COLORS.get(event_type, LOG_COLORS['DEFAULT'])
        message = super().format(record)
        return f"{color}{message}\033[0m"


# Logger Configuration
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(event_type)s: %(message)s'))

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(ColorFormatter('[%(asctime)s] %(event_type)s: %(message)s'))

logger = logging.getLogger("SimulationLogger")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def log_event(event_type, data, verbose=True):
    if not VERBOSE and event_type == "STATE_UPDATE" and not verbose:
        return
    logger.debug(str(data), extra={"event_type": event_type})


# ---------------- Core Classes ----------------

class SystemState:
    def __init__(self, data):
        self.data = data
        self.timestamp = time.time()

    def update(self, modifiers=None):
        for key in self.data:
            x = self.data[key]
            noise = random.gauss(0, 0.02)
            modifier = modifiers.get(key, 0) if modifiers else 0
            nonlinear_term = math.sin(x * math.pi) * 0.2
            self.data[key] = x + noise + modifier + nonlinear_term - 0.1 * (x**2)

    def diff(self, other):
        return sum(abs(self.data[k] - other.data[k]) for k in self.data) / len(self.data)

    def signature(self):
        return "|".join([f"{k}={round(self.data[k], 3)}" for k in sorted(self.data)])

    def clone(self):
        return copy.deepcopy(self)


# ---------------- Memory Handling ----------------

def load_memory():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except:
        return {"valid_patterns": {}, "threshold": THRESHOLD_BASE}


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)


# ---------------- Simulation Engine ----------------

def run_simulation(initial_state, steps, modifiers=None):
    sim_state = initial_state.clone()
    trajectory = []
    for _ in range(steps):
        sim_state.update(modifiers)
        trajectory.append(sim_state.clone())
    return trajectory


# ---------------- Learning Engine ----------------

def learn_from_success(memory, pattern_key):
    if pattern_key not in memory["valid_patterns"]:
        memory["valid_patterns"][pattern_key] = {"score": 1, "hits": 1}
    else:
        memory["valid_patterns"][pattern_key]["score"] += 2
        memory["valid_patterns"][pattern_key]["hits"] += 1

    memory["threshold"] = max(0.03, memory["threshold"] - 0.005)
    log_event("ADAPTIVE_LEARN", f"Learned: {pattern_key}, Score: {memory['valid_patterns'][pattern_key]['score']}")
    log_event("THRESHOLD_ADJUST", f"New Threshold: {memory['threshold']:.4f}")


def punish_false_prediction(memory, pattern_key):
    if pattern_key in memory["valid_patterns"]:
        memory["valid_patterns"][pattern_key]["score"] = max(0, memory["valid_patterns"][pattern_key]["score"] - 1)

    memory["threshold"] = min(0.3, memory["threshold"] + 0.005)
    log_event("PUNISH", f"Punished: {pattern_key}")
    log_event("THRESHOLD_ADJUST", f"New Threshold: {memory['threshold']:.4f}")


# ---------------- Prediction Evaluation ----------------

def evaluate_prediction(current_state, memory):
    pattern_key = current_state.signature()
    sim_trajectory = run_simulation(current_state, SIMULATION_STEPS)
    future_real = current_state.clone()

    for _ in range(SIMULATION_STEPS):
        future_real.update()

    deviation = future_real.diff(sim_trajectory[-1])
    log_event("SIMULATION_RESULT", f"Deviation: {deviation:.4f}")

    if deviation <= ERROR_MARGIN:
        learn_from_success(memory, pattern_key)
    else:
        punish_false_prediction(memory, pattern_key)

    save_memory(memory)


# ---------------- Main Monitor ----------------

def monitor_system():
    memory = load_memory()
    current_state = SystemState({"x": 0.5, "y": 0.3})
    last_state = current_state.clone()

    while True:
        time.sleep(SIM_INTERVAL)
        current_state.update()
        delta = current_state.diff(last_state)

        log_event("STATE_UPDATE", f"Delta: {delta:.4f}, Data: {current_state.data}")

        if delta > memory["threshold"]:
            log_event("TIPPING_POINT_DETECTED", f"Delta: {delta:.4f}")
            threading.Thread(target=evaluate_prediction, args=(current_state.clone(), memory)).start()

        last_state = current_state.clone()


# ---------------- Optional: Simulation Demo ----------------

def simulate_dynamic_process():
    start_time = time.time()

    MAX_STEPS = 5000
    DELTA_THRESHOLD = 1e-4
    CONVERGENCE_COUNT_REQUIRED = 30
    MAX_DURATION_SECONDS = 300

    convergence_counter = 0
    current_state = {"x": 0.0, "y": 0.0}

    for step in range(MAX_STEPS):
        delta = random.uniform(0.001, 0.2)
        current_state["x"] += random.uniform(-delta, delta)
        current_state["y"] += random.uniform(-delta, delta)

        log_event("STATE_UPDATE", f"Delta: {delta:.4f}, Data: {json.dumps(current_state)}")

        if step % 50 == 0:
            print(f"[Step {step}] Δ: {delta:.4f} | X: {current_state['x']:.4f}, Y: {current_state['y']:.4f}")

        if delta < DELTA_THRESHOLD:
            convergence_counter += 1
        else:
            convergence_counter = 0

        if convergence_counter >= CONVERGENCE_COUNT_REQUIRED:
            print(f"\n[INFO] Convergence detected after {step+1} steps.")
            break

        if (time.time() - start_time) >= MAX_DURATION_SECONDS:
            print(f"\n[INFO] Max duration of {MAX_DURATION_SECONDS} seconds reached.")
            break

        time.sleep(0.05)

    print("\n[INFO] Simulation finished.\n")


# ---------------- Entry ----------------

if __name__ == "__main__":
    monitor_system()
