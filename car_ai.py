"""
Shared code for the racing game and AI trainer.
Pure Python, no external libraries.
"""
import math
import random

WIDTH, HEIGHT = 900, 600
TRACK_WIDTH = 90

waypoints = [
    (150,500),(120,350),(150,180),(280,90),
    (500,80),(680,100),(800,220),(800,380),
    (680,480),(500,520),(320,500),(150,500)
]

SENSOR_ANGLES = [-70, -35, 0, 35, 70]   # degrees, relative to car heading
SENSOR_MAX_RANGE = 140

# Checkpoints: one per waypoint (excluding the duplicate closing point).
# A car must reach these IN ORDER for progress/laps to count - this stops
# the "spin in circles near the finish line" exploit, since that exploit
# relied on nearest-segment distance rather than actually visiting each
# checkpoint's real location in sequence.
checkpoints = waypoints[:-1]
CHECKPOINT_RADIUS = 45


def set_track(new_waypoints, width=None):
    """Swap the active track. Updates waypoints, checkpoints, and (optionally) track width."""
    global waypoints, checkpoints, TRACK_WIDTH
    waypoints = list(new_waypoints)
    checkpoints = waypoints[:-1]
    if width is not None:
        TRACK_WIDTH = width


def closest_point_info(x, y):
    best_dist = float('inf')
    best_seg = 0
    for i in range(len(waypoints)-1):
        ax, ay = waypoints[i]
        bx, by = waypoints[i+1]
        dx, dy = bx-ax, by-ay
        len2 = dx*dx + dy*dy
        t = 0 if len2 == 0 else ((x-ax)*dx + (y-ay)*dy) / len2
        t = max(0, min(1, t))
        px, py = ax + t*dx, ay + t*dy
        d = math.hypot(x-px, y-py)
        if d < best_dist:
            best_dist = d
            best_seg = i
    return best_dist, best_seg


def sense(x, y, angle):
    """Cast rays outward from the car; return normalized distance-to-edge for each ray."""
    readings = []
    for offset_deg in SENSOR_ANGLES:
        ray_angle = angle + math.radians(offset_deg)
        dx, dy = math.cos(ray_angle), math.sin(ray_angle)
        dist_found = SENSOR_MAX_RANGE
        step = 6
        d = 0
        while d < SENSOR_MAX_RANGE:
            px, py = x + dx*d, y + dy*d
            off_dist, _ = closest_point_info(px, py)
            if off_dist > TRACK_WIDTH/2:
                dist_found = d
                break
            d += step
        readings.append(dist_found / SENSOR_MAX_RANGE)  # normalized 0..1
    return readings


# ---------------- Neural network (from scratch) ----------------
class NeuralNet:
    """Feedforward net: inputs -> N hidden layers (tanh) -> outputs (tanh).

    hidden_sizes is a list, e.g. [10] for one hidden layer of 10 neurons,
    or [12, 8] for two hidden layers. `layers` holds one (W, b) pair per
    connection between consecutive layers, so len(layers) == len(hidden_sizes) + 1.
    """
    def __init__(self, n_in, hidden_sizes, n_out, weights=None):
        if isinstance(hidden_sizes, int):  # back-compat: a bare int means one hidden layer
            hidden_sizes = [hidden_sizes]
        self.n_in, self.n_out = n_in, n_out

        if weights is not None:
            self.layer_sizes = list(weights['layers'])
            self.layers = [(w, b) for w, b in weights['weights']]
            self.hidden_sizes = self.layer_sizes[1:-1]
        else:
            self.hidden_sizes = list(hidden_sizes)
            self.layer_sizes = [n_in] + self.hidden_sizes + [n_out]
            self.layers = []
            for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
                w = [[random.uniform(-1, 1) for _ in range(fan_in)] for _ in range(fan_out)]
                b = [random.uniform(-1, 1) for _ in range(fan_out)]
                self.layers.append((w, b))

        # last activations per layer (inputs, each hidden layer, outputs), for a live "mind" view
        self.activations = [[0.0] * n for n in self.layer_sizes]

    @property
    def n_hidden_layers(self):
        return len(self.hidden_sizes)

    def get_weights(self):
        return {'layers': self.layer_sizes, 'weights': [[w, b] for w, b in self.layers]}

    def forward(self, inputs):
        values = list(inputs)
        self.activations[0] = values
        for li, (w, b) in enumerate(self.layers):
            next_values = []
            for o in range(len(b)):
                s = b[o] + sum(w[o][i] * values[i] for i in range(len(values)))
                next_values.append(math.tanh(s))
            values = next_values
            self.activations[li + 1] = values
        return values

    @property
    def last_inputs(self):
        return self.activations[0]

    @property
    def last_outputs(self):
        return self.activations[-1]

    def mutate(self, rate=0.15, strength=0.5):
        def m(v):
            return v + random.gauss(0, strength) if random.random() < rate else v
        self.layers = [
            ([[m(v) for v in row] for row in w], [m(v) for v in b])
            for w, b in self.layers
        ]

    @staticmethod
    def crossover(a, b):
        """Mix two parent networks into a child (from scratch, no numpy). Assumes same architecture."""
        def mix_matrix(m1, m2):
            return [[random.choice([x, y]) for x, y in zip(r1, r2)] for r1, r2 in zip(m1, m2)]
        def mix_vec(v1, v2):
            return [random.choice([x, y]) for x, y in zip(v1, v2)]
        mixed_layers = [
            (mix_matrix(wa, wb), mix_vec(ba, bb))
            for (wa, ba), (wb, bb) in zip(a.layers, b.layers)
        ]
        child = NeuralNet(a.n_in, a.hidden_sizes, a.n_out,
                           weights={'layers': a.layer_sizes, 'weights': mixed_layers})
        return child


# ---------------- Car ----------------
class Car:
    def __init__(self, x, y, angle):
        self.x, self.y, self.angle = x, y, angle
        self.speed = 0
        self.max_speed = 4.2
        self.accel = 0.15
        self.friction = 0.03
        self.turn_rate = 0.045
        self.lap = 0
        self.alive = True
        self.off_track_steps = 0
        self.off_track_penalty = 0.0  # accumulates while off track, dragged down fitness
        self.total_progress = 0.0  # fitness accumulator = checkpoints passed, minus off-track penalty
        self.checkpoints_passed = 0
        self.next_checkpoint = 1  # checkpoint 0 is the start, already "passed" at spawn

    def step_ai(self, net):
        readings = sense(self.x, self.y, self.angle)
        inputs = readings + [self.speed / self.max_speed]
        steer, throttle = net.forward(inputs)
        self.angle += steer * self.turn_rate
        self.speed = max(-self.max_speed/1.6, min(self.max_speed, self.speed + throttle*self.accel))
        self._physics()

    def _physics(self):
        dist, seg = closest_point_info(self.x, self.y)
        off_track = dist > TRACK_WIDTH/2 - 8
        grip = 0.18 if off_track else 1.0  # off-road handling is much worse, not just a minor slip
        self.x += math.cos(self.angle) * self.speed * grip
        self.y += math.sin(self.angle) * self.speed * grip
        self.speed *= (1 - self.friction)
        if abs(self.speed) < 0.01:
            self.speed = 0

        if off_track:
            self.off_track_steps += 1
            self.off_track_penalty += 0.05  # every off-road step drags fitness down, even if it survives
            if self.off_track_steps > 40:  # crashes much sooner than before (was 90)
                self.alive = False
        else:
            self.off_track_steps = 0

        # Checkpoint progress: must reach checkpoints[next_checkpoint] within
        # CHECKPOINT_RADIUS, strictly in order. Only then does it advance.
        cx, cy = checkpoints[self.next_checkpoint]
        if math.hypot(self.x - cx, self.y - cy) < CHECKPOINT_RADIUS:
            self.checkpoints_passed += 1
            self.next_checkpoint = (self.next_checkpoint + 1) % len(checkpoints)

        self.lap = self.checkpoints_passed // len(checkpoints)
        # Two cars that reach the same checkpoints are no longer tied in fitness - the
        # one that stayed on the road the whole time ranks higher for evolution/breeding.
        self.total_progress = self.checkpoints_passed - self.off_track_penalty
