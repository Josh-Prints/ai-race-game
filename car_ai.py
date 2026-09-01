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
    """Simple feedforward net: inputs -> hidden (tanh) -> outputs (tanh)."""
    def __init__(self, n_in, n_hidden, n_out, weights=None):
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        if weights is not None:
            self.w1, self.b1, self.w2, self.b2 = weights
        else:
            self.w1 = [[random.uniform(-1, 1) for _ in range(n_in)] for _ in range(n_hidden)]
            self.b1 = [random.uniform(-1, 1) for _ in range(n_hidden)]
            self.w2 = [[random.uniform(-1, 1) for _ in range(n_hidden)] for _ in range(n_out)]
            self.b2 = [random.uniform(-1, 1) for _ in range(n_out)]
        # last activations, kept around so a UI can visualize "what the AI is thinking"
        self.last_inputs = [0.0] * n_in
        self.last_hidden = [0.0] * n_hidden
        self.last_outputs = [0.0] * n_out

    def get_weights(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, inputs):
        hidden = []
        for h in range(self.n_hidden):
            s = self.b1[h] + sum(self.w1[h][i] * inputs[i] for i in range(self.n_in))
            hidden.append(math.tanh(s))
        outputs = []
        for o in range(self.n_out):
            s = self.b2[o] + sum(self.w2[o][h] * hidden[h] for h in range(self.n_hidden))
            outputs.append(math.tanh(s))
        self.last_inputs = list(inputs)
        self.last_hidden = hidden
        self.last_outputs = outputs
        return outputs

    def mutate(self, rate=0.15, strength=0.5):
        def m(v):
            return v + random.gauss(0, strength) if random.random() < rate else v
        self.w1 = [[m(v) for v in row] for row in self.w1]
        self.b1 = [m(v) for v in self.b1]
        self.w2 = [[m(v) for v in row] for row in self.w2]
        self.b2 = [m(v) for v in self.b2]

    @staticmethod
    def crossover(a, b):
        """Mix two parent networks into a child (from scratch, no numpy)."""
        def mix_matrix(m1, m2):
            return [[random.choice([x, y]) for x, y in zip(r1, r2)] for r1, r2 in zip(m1, m2)]
        def mix_vec(v1, v2):
            return [random.choice([x, y]) for x, y in zip(v1, v2)]
        child = NeuralNet(a.n_in, a.n_hidden, a.n_out, weights=[
            mix_matrix(a.w1, b.w1), mix_vec(a.b1, b.b1),
            mix_matrix(a.w2, b.w2), mix_vec(a.b2, b.b2)
        ])
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
        self.total_progress = 0.0  # fitness accumulator = checkpoints passed (monotonic)
        self.checkpoints_passed = 0
        self.next_checkpoint = 1  # checkpoint 0 is the start, already "passed" at spawn

    def step_ai(self, net):
        readings = sense(self.x, self.y, self.angle)
        inputs = readings + [self.speed / self.max_speed]
        steer, throttle = net.forward(inputs)
        self.angle += steer * self.turn_rate
        self.speed = max(-self.max_speed/1.6, min(self.max_speed, self.speed + throttle*self.accel))
        self._physics()

    def step_manual(self, up, down, left, right):
        if up: self.speed = min(self.max_speed, self.speed + self.accel)
        if down: self.speed = max(-self.max_speed/1.6, self.speed - self.accel)
        if left: self.angle -= self.turn_rate * (self.speed/self.max_speed)
        if right: self.angle += self.turn_rate * (self.speed/self.max_speed)
        self._physics()

    def _physics(self):
        dist, seg = closest_point_info(self.x, self.y)
        off_track = dist > TRACK_WIDTH/2 - 8
        grip = 0.35 if off_track else 1.0
        self.x += math.cos(self.angle) * self.speed * grip
        self.y += math.sin(self.angle) * self.speed * grip
        self.speed *= (1 - self.friction)
        if abs(self.speed) < 0.01:
            self.speed = 0

        if off_track:
            self.off_track_steps += 1
            if self.off_track_steps > 90:  # off track too long -> "crash"
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
        self.total_progress = self.checkpoints_passed
