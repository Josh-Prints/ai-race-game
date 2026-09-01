"""
Race vs AI - choose or draw a track, train live in the same window, then race it.

Needs car_ai.py and tracks.py in the same folder.

Run:
    python race_trainer.py

Controls once racing: Arrow keys to drive. Press R during training to skip
ahead and race the best AI found so far.
"""
import tkinter as tk
import math
import time
import json
import os
import random
import re

import car_ai
import tracks

POP_SIZE = 24
GENERATIONS = 60
MAX_STEPS_PER_GEN = 700
SUBSTEPS_PER_FRAME = 4
N_IN, N_OUT = 6, 2
DEFAULT_HIDDEN_LAYERS = 1
DEFAULT_NEURONS_PER_LAYER = 10
CUSTOM_TRACK_PATH = 'custom_track.json'
LAPS_TO_WIN = 3
COUNTDOWN_SECONDS = 3
INPUT_LABELS = ['S1', 'S2', 'S3', 'S4', 'S5', 'Spd']
OUTPUT_LABELS = ['Steer', 'Gas']

BG = '#1b1f2b'
PANEL_BG = '#242a3a'
ACCENT = '#8ecae6'
TEXT = '#eef1f8'
MUTED = '#9aa3b8'


def safe_name(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name)


def weights_path(track_name):
    return f"weights_{safe_name(track_name)}.json"


def evolve(population, fitnesses):
    ranked = sorted(zip(population, fitnesses), key=lambda p: p[1], reverse=True)
    survivors = [net for net, fit in ranked[:max(4, POP_SIZE // 5)]]
    next_gen = survivors[:2]
    while len(next_gen) < POP_SIZE:
        a, b = random.choice(survivors), random.choice(survivors)
        child = car_ai.NeuralNet.crossover(a, b)
        child.mutate(rate=0.15, strength=0.4)
        next_gen.append(child)
    return next_gen


class App:
    def __init__(self, root):
        self.root = root
        root.configure(bg=BG)

        title = tk.Label(root, text="RACE vs AI", font=('Arial', 22, 'bold'), fg=ACCENT, bg=BG)
        title.pack(pady=(10, 2))

        self.canvas = tk.Canvas(root, width=900, height=600, bg='#3a5a3a', highlightthickness=2,
                                 highlightbackground='#000')
        self.canvas.pack(padx=10)

        self.hud = tk.Label(root, text="", font=('Arial', 13), justify='left', fg=TEXT, bg=BG)
        self.hud.pack(pady=(6, 0))
        self.controls = tk.Frame(root, bg=BG)
        self.controls.pack(pady=8)

        self.mode = 'menu'
        self.track_width_var = tk.IntVar(value=90)
        self.generations_var = tk.IntVar(value=GENERATIONS)
        self.steps_var = tk.IntVar(value=MAX_STEPS_PER_GEN)
        self.hidden_layers_var = tk.IntVar(value=DEFAULT_HIDDEN_LAYERS)
        self.neurons_var = tk.IntVar(value=DEFAULT_NEURONS_PER_LAYER)
        self.editor_points = []
        self.best_lap_time = None
        self._particles = []
        self.track_name = "Oval Classic"
        self.show_mind = False
        self.hidden_sizes = [DEFAULT_NEURONS_PER_LAYER] * DEFAULT_HIDDEN_LAYERS

        self.keys_down = set()
        root.bind('<KeyPress>', self.on_key_down)
        root.bind('<KeyRelease>', lambda e: self.keys_down.discard(e.keysym))
        self.canvas.bind('<Button-1>', self.on_canvas_click)

        self.build_menu()
        self.loop()

    # ---------------- menu ----------------
    def clear_controls(self):
        for w in self.controls.winfo_children():
            w.destroy()

    def make_button(self, parent, text, command, bg=PANEL_BG, fg=TEXT, width=16):
        return tk.Button(parent, text=text, width=width, command=command,
                          bg=bg, fg=fg, activebackground=ACCENT, relief='flat',
                          font=('Arial', 11, 'bold'), padx=4, pady=6, cursor='hand2')

    def build_menu(self):
        self.mode = 'menu'
        self.canvas.delete('all')
        self.clear_controls()
        self.draw_menu_background()
        self.hud.config(text="Choose a track to train your AI on, or race a saved AI instantly:",
                         font=('Arial', 15, 'bold'))

        for name in tracks.PRESET_TRACKS:
            col = tk.Frame(self.controls, bg=BG)
            col.pack(side='left', padx=4)
            self.make_button(col, name, lambda n=name: self.choose_preset(n),
                              bg='#33415c').pack(side='top')
            if os.path.exists(weights_path(name)):
                self.make_button(col, "⚡ Race Saved AI", lambda n=name: self.race_saved_ai(n),
                                  bg='#ffd166', fg='#10131c', width=16).pack(side='top', pady=(4, 0))

        custom_col = tk.Frame(self.controls, bg=BG)
        custom_col.pack(side='left', padx=4)
        self.make_button(custom_col, "Draw Custom Track", self.start_editor,
                          bg='#8ecae6', fg='#10131c', width=18).pack(side='top')
        if os.path.exists(CUSTOM_TRACK_PATH):
            self.make_button(custom_col, "Load Saved Track", self.load_custom_track,
                              bg='#8ecae6', fg='#10131c', width=18).pack(side='top', pady=(4, 0))
            if os.path.exists(weights_path("Custom")):
                self.make_button(custom_col, "⚡ Race Saved AI", lambda: self.race_saved_ai("Custom"),
                                  bg='#ffd166', fg='#10131c', width=18).pack(side='top', pady=(4, 0))

        settings_frame = tk.Frame(self.root, bg=BG)
        settings_frame.pack(pady=(0, 8))
        width_frame = tk.Frame(settings_frame, bg=BG)
        width_frame.pack(side='left', padx=10)
        tk.Label(width_frame, text="Track width:", fg=MUTED, bg=BG).pack(side='left')
        tk.Scale(width_frame, from_=50, to=140, orient='horizontal',
                 variable=self.track_width_var, length=160, bg=BG, fg=TEXT,
                 troughcolor=PANEL_BG, highlightthickness=0).pack(side='left')

        gen_frame = tk.Frame(settings_frame, bg=BG)
        gen_frame.pack(side='left', padx=10)
        tk.Label(gen_frame, text="Generations (training rounds):", fg=MUTED, bg=BG).pack(side='left')
        tk.Scale(gen_frame, from_=10, to=300, orient='horizontal',
                 variable=self.generations_var, length=160, bg=BG, fg=TEXT,
                 troughcolor=PANEL_BG, highlightthickness=0).pack(side='left')

        steps_frame = tk.Frame(settings_frame, bg=BG)
        steps_frame.pack(side='left', padx=10)
        tk.Label(steps_frame, text="Time per round (steps):", fg=MUTED, bg=BG).pack(side='left')
        tk.Scale(steps_frame, from_=200, to=4000, orient='horizontal',
                 variable=self.steps_var, length=160, bg=BG, fg=TEXT,
                 troughcolor=PANEL_BG, highlightthickness=0).pack(side='left')

        layers_frame2 = tk.Frame(self.root, bg=BG)
        layers_frame2.pack(pady=(0, 8))
        hl_frame = tk.Frame(layers_frame2, bg=BG)
        hl_frame.pack(side='left', padx=10)
        tk.Label(hl_frame, text="Hidden layers:", fg=MUTED, bg=BG).pack(side='left')
        tk.Scale(hl_frame, from_=1, to=4, orient='horizontal',
                 variable=self.hidden_layers_var, length=120, bg=BG, fg=TEXT,
                 troughcolor=PANEL_BG, highlightthickness=0).pack(side='left')

        neurons_frame = tk.Frame(layers_frame2, bg=BG)
        neurons_frame.pack(side='left', padx=10)
        tk.Label(neurons_frame, text="Neurons per hidden layer:", fg=MUTED, bg=BG).pack(side='left')
        tk.Scale(neurons_frame, from_=4, to=24, orient='horizontal',
                 variable=self.neurons_var, length=160, bg=BG, fg=TEXT,
                 troughcolor=PANEL_BG, highlightthickness=0).pack(side='left')

        self._width_frame = settings_frame
        self._layers_frame = layers_frame2

    def draw_menu_background(self):
        self.canvas.configure(bg='#20242f')
        w, h = 900, 600
        # faint preview of the first preset track as decoration
        preview = tracks.PRESET_TRACKS.get("Oval Classic")
        if preview:
            flat = [c for p in preview for c in p]
            self.canvas.create_line(*flat, width=70, fill='#2a2f3d', capstyle=tk.ROUND, joinstyle=tk.ROUND)
            self.canvas.create_line(*flat, width=2, fill='#3a4054', dash=(10, 8))
        self.canvas.create_text(w/2, h/2 - 40, text="RACE vs AI", font=('Arial', 40, 'bold'), fill='#333a4d')
        self.canvas.create_text(w/2, h/2 + 10, text="pick a track below to start training",
                                 font=('Arial', 14), fill='#454d63')

    def choose_preset(self, name):
        self.track_name = name
        car_ai.set_track(tracks.PRESET_TRACKS[name], self.track_width_var.get())
        self.begin_training()

    def load_custom_track(self):
        self.track_name = "Custom"
        with open(CUSTOM_TRACK_PATH) as f:
            pts = json.load(f)
        car_ai.set_track(pts, self.track_width_var.get())
        self.begin_training()

    def race_saved_ai(self, name):
        """Load a previously saved AI for this track and jump straight to racing it."""
        self.track_name = name
        if name == "Custom":
            with open(CUSTOM_TRACK_PATH) as f:
                pts = json.load(f)
            car_ai.set_track(pts, self.track_width_var.get())
        else:
            car_ai.set_track(tracks.PRESET_TRACKS[name], self.track_width_var.get())

        with open(weights_path(name)) as f:
            weights = json.load(f)
        net = car_ai.NeuralNet(N_IN, [], N_OUT, weights=weights)

        wps = car_ai.waypoints
        self.start_x, self.start_y = wps[0]
        self.start_angle = math.atan2(wps[1][1]-wps[0][1], wps[1][0]-wps[0][0])
        self.best_ever_net = net
        self.start_race()

    # ---------------- track editor ----------------
    def start_editor(self):
        self.mode = 'editor'
        self.editor_points = []
        self.canvas.delete('all')
        self.canvas.configure(bg='#20242f')
        self.clear_controls()
        self._width_frame.pack_forget()
        self._layers_frame.pack_forget()
        self.hud.config(text="Click to place track points (at least 4). Click points in order around the loop.")

        self.make_button(self.controls, "Undo last point", self.editor_undo).pack(side='left', padx=4)
        self.make_button(self.controls, "Finish Track", self.finish_editor,
                          bg='#90e0af', fg='#10131c').pack(side='left', padx=4)
        self.make_button(self.controls, "Cancel", self.build_menu).pack(side='left', padx=4)

    def on_canvas_click(self, event):
        if self.mode != 'editor':
            return
        self.editor_points.append((event.x, event.y))
        self.redraw_editor()

    def editor_undo(self):
        if self.editor_points:
            self.editor_points.pop()
            self.redraw_editor()

    def redraw_editor(self):
        self.canvas.delete('all')
        if len(self.editor_points) >= 2:
            flat = [c for p in self.editor_points for c in p]
            self.canvas.create_line(*flat, width=self.track_width_var.get(), fill='#3a4054',
                                     capstyle=tk.ROUND, joinstyle=tk.ROUND)
        for i, (x, y) in enumerate(self.editor_points):
            color = '#90e0af' if i == 0 else 'yellow'
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color, outline='#10131c')
            self.canvas.create_text(x, y - 14, text=str(i+1), fill=TEXT, font=('Arial', 9))
            if i > 0:
                px, py = self.editor_points[i-1]
                self.canvas.create_line(px, py, x, y, fill='white', dash=(4, 3))
        self.hud.config(text=f"Points placed: {len(self.editor_points)} (need at least 4). "
                              f"Click 'Finish Track' when your loop looks right.")

    def finish_editor(self):
        if len(self.editor_points) < 4:
            self.hud.config(text="Need at least 4 points before finishing.")
            return
        pts = self.editor_points[:]
        pts.append(pts[0])  # close the loop
        with open(CUSTOM_TRACK_PATH, 'w') as f:
            json.dump(pts, f)
        self.track_name = "Custom"
        car_ai.set_track(pts, self.track_width_var.get())
        self._width_frame.pack_forget()
        self._layers_frame.pack_forget()
        self.begin_training()

    # ---------------- shared drawing ----------------
    def draw_track(self):
        self.canvas.configure(bg='#3a5a3a')
        # subtle grass texture
        for gx in range(0, 900, 40):
            for gy in range(0, 600, 40):
                if (gx // 40 + gy // 40) % 2 == 0:
                    self.canvas.create_rectangle(gx, gy, gx+40, gy+40, fill='#375537', outline='')

        wps = car_ai.waypoints
        flat = [c for p in wps for c in p]
        # asphalt with a lighter inner line for depth
        self.canvas.create_line(*flat, width=car_ai.TRACK_WIDTH + 6, fill='#111318',
                                 capstyle=tk.ROUND, joinstyle=tk.ROUND)
        self.canvas.create_line(*flat, width=car_ai.TRACK_WIDTH, fill='#4a4f5c',
                                 capstyle=tk.ROUND, joinstyle=tk.ROUND)
        self.canvas.create_line(*flat, width=3, fill='#f4f4f4', dash=(12, 10))

        ax, ay = wps[0]
        bx, by = wps[1]
        ang = math.atan2(by - ay, bx - ax) + math.pi / 2
        hw = car_ai.TRACK_WIDTH / 2
        # checkered start/finish line
        n_check = 6
        for i in range(n_check):
            t0 = -hw + (2*hw)*i/n_check
            t1 = -hw + (2*hw)*(i+1)/n_check
            color = '#ffffff' if i % 2 == 0 else '#111318'
            x0 = ax + math.cos(ang)*t0
            y0 = ay + math.sin(ang)*t0
            x1 = ax + math.cos(ang)*t1
            y1 = ay + math.sin(ang)*t1
            self.canvas.create_line(x0, y0, x1, y1, width=6, fill=color)

        for cx, cy in car_ai.checkpoints[1:]:
            self.canvas.create_oval(cx - car_ai.CHECKPOINT_RADIUS, cy - car_ai.CHECKPOINT_RADIUS,
                                     cx + car_ai.CHECKPOINT_RADIUS, cy + car_ai.CHECKPOINT_RADIUS,
                                     outline='#ffffff', width=1, dash=(3, 3))

    def draw_car(self, car, color, size=10, label=None, dim=False):
        cos_a, sin_a = math.cos(car.angle), math.sin(car.angle)

        def to_world(cx, cy):
            return (car.x + cx*cos_a - cy*sin_a, car.y + cx*sin_a + cy*cos_a)

        # soft shadow
        shadow = []
        for cx, cy in [(-size, -6), (size, -6), (size, 6), (-size, 6)]:
            wx, wy = to_world(cx, cy)
            shadow.extend([wx + 3, wy + 3])
        self.canvas.create_polygon(shadow, fill='#000000', outline='', stipple='gray50')

        outline = '#10131c'
        body_fill = color if not dim else '#5a5f6b'

        # wheels
        for cx, cy in [(-size*0.55, -7), (-size*0.55, 7), (size*0.55, -7), (size*0.55, 7)]:
            wx, wy = to_world(cx, cy)
            self.canvas.create_oval(wx-2.5, wy-2.5, wx+2.5, wy+2.5, fill='#111318', outline='')

        corners = [(-size, -6), (size*0.7, -6), (size, 0), (size*0.7, 6), (-size, 6)]
        pts = []
        for cx, cy in corners:
            wx, wy = to_world(cx, cy)
            pts.extend([wx, wy])
        self.canvas.create_polygon(pts, fill=body_fill, outline=outline, width=1)

        # windshield + headlight
        wx1, wy1 = to_world(size*0.15, -4)
        wx2, wy2 = to_world(size*0.55, -3)
        wx3, wy3 = to_world(size*0.55, 3)
        wx4, wy4 = to_world(size*0.15, 4)
        self.canvas.create_polygon(wx1, wy1, wx2, wy2, wx3, wy3, wx4, wy4, fill='#bfe9ff', outline='')
        hx, hy = to_world(size, 0)
        self.canvas.create_oval(hx-2, hy-2, hx+2, hy+2, fill='#fff6c9', outline='')

        if label:
            self.canvas.create_text(car.x, car.y - size - 10, text=label, fill=TEXT, font=('Arial', 9, 'bold'))

    def draw_progress_bar(self, x, y, w, h, frac, color):
        frac = max(0.0, min(1.0, frac))
        self.canvas.create_rectangle(x, y, x+w, y+h, fill='#10131c', outline='#000')
        if frac > 0:
            self.canvas.create_rectangle(x, y, x + w*frac, y+h, fill=color, outline='')
        self.canvas.create_rectangle(x, y, x+w, y+h, outline='#000')

    def save_weights(self, net):
        """Persist a net's weights to a per-track file so it can be raced later without retraining."""
        if net is None:
            return
        try:
            with open(weights_path(self.track_name), 'w') as f:
                json.dump(net.get_weights(), f)
            with open('best_weights.json', 'w') as f:
                json.dump(net.get_weights(), f)
        except Exception:
            pass

    def _activation_color(self, v):
        v = max(-1.0, min(1.0, v))
        if v >= 0:
            r, g, b = 60, int(60 + v * 195), 90
        else:
            r, g, b = int(60 + -v * 195), 60, 70
        return f'#{r:02x}{g:02x}{b:02x}'

    def draw_ai_mind(self, net):
        """Live view of the leading AI's neural network: nodes lit by activation, edges by weight.

        Works for any number of hidden layers - one column per layer in net.layer_sizes,
        spaced evenly across the panel.
        """
        x0, y0, w, h = 630, 65, 250, 470
        self.canvas.create_rectangle(x0, y0, x0+w, y0+h, fill='#0e1119', outline=ACCENT, width=2)
        self.canvas.create_text(x0 + w/2, y0 + 16, text="AI's Mind", fill=ACCENT, font=('Arial', 12, 'bold'))

        n_layers = len(net.layer_sizes)
        top, bot = y0 + 40, y0 + h - 20
        left, right = x0 + 30, x0 + w - 30

        def layer_x(li):
            if n_layers == 1:
                return (left + right) / 2
            return left + (right - left) * li / (n_layers - 1)

        def positions(n):
            if n == 1:
                return [(top + bot) / 2]
            return [top + (bot - top) * i / (n - 1) for i in range(n)]

        node_y = [positions(n) for n in net.layer_sizes]
        node_x = [layer_x(li) for li in range(n_layers)]
        node_radius = [8] + [6] * (n_layers - 2) + [8] if n_layers > 1 else [8]

        # edges, drawn from each (w, b) layer connecting consecutive columns
        for li, (w_mat, b_vec) in enumerate(net.layers):
            xa, xb = node_x[li], node_x[li + 1]
            ys_a, ys_b = node_y[li], node_y[li + 1]
            for j in range(len(b_vec)):
                for i in range(len(ys_a)):
                    weight = w_mat[j][i]
                    line_w = min(3, 0.3 + abs(weight))
                    color = '#4caf7d' if weight >= 0 else '#c94f4f'
                    self.canvas.create_line(xa, ys_a[i], xb, ys_b[j], fill=color, width=line_w)

        # nodes, colored by the net's last activation for that layer
        for li in range(n_layers):
            xs = node_x[li]
            r = node_radius[li]
            values = net.activations[li] if li < len(net.activations) else [0.0] * net.layer_sizes[li]
            for i, ys in enumerate(node_y[li]):
                val = values[i] if i < len(values) else 0.0
                self.canvas.create_oval(xs-r, ys-r, xs+r, ys+r, fill=self._activation_color(val), outline='#000')
            if li == 0:
                for i, ys in enumerate(node_y[li]):
                    label = INPUT_LABELS[i] if i < len(INPUT_LABELS) else str(i)
                    self.canvas.create_text(xs - r - 16, ys, text=label, fill=MUTED, font=('Arial', 8), anchor='e')
            elif li == n_layers - 1:
                for i, ys in enumerate(node_y[li]):
                    label = OUTPUT_LABELS[i] if i < len(OUTPUT_LABELS) else str(i)
                    self.canvas.create_text(xs + r + 16, ys, text=label, fill=MUTED, font=('Arial', 8), anchor='w')

    # ---------------- training setup ----------------
    def begin_training(self):
        self.mode = 'training'
        self.clear_controls()
        self.canvas.delete('all')
        self.make_button(self.controls, "Back to Menu", self.build_menu, bg='#33415c').pack(side='left', padx=4)
        self.mind_btn = self.make_button(self.controls, "Show AI Mind", self.toggle_mind,
                                          bg='#33415c', width=16)
        self.mind_btn.pack(side='left', padx=4)

        wps = car_ai.waypoints
        self.start_x, self.start_y = wps[0]
        self.start_angle = math.atan2(wps[1][1]-wps[0][1], wps[1][0]-wps[0][0])

        self.generations_target = self.generations_var.get()
        self.max_steps_per_gen = self.steps_var.get()
        self.hidden_sizes = [self.neurons_var.get()] * self.hidden_layers_var.get()
        self.generation = 1
        self.steps_this_gen = 0
        self.best_ever_net = None
        self.best_ever_fit = -1
        self.leader_net = None
        self.population = [car_ai.NeuralNet(N_IN, self.hidden_sizes, N_OUT) for _ in range(POP_SIZE)]
        self.cars = [car_ai.Car(self.start_x, self.start_y, self.start_angle) for _ in self.population]

    def toggle_mind(self):
        self.show_mind = not self.show_mind
        self.mind_btn.config(text="Hide AI Mind" if self.show_mind else "Show AI Mind")

    # ---------------- training loop ----------------
    def training_step(self):
        if self.generation > self.generations_target:
            self.save_weights(self.best_ever_net)
            self.start_race()
            return

        for _ in range(SUBSTEPS_PER_FRAME):
            any_alive = False
            for car, net in zip(self.cars, self.population):
                if car.alive:
                    car.step_ai(net)
                    any_alive = True
            self.steps_this_gen += 1
            if not any_alive or self.steps_this_gen >= self.max_steps_per_gen:
                break

        # track whichever car is currently in the lead, so "Show AI Mind" has something live to show
        alive_cars = [(c, n) for c, n in zip(self.cars, self.population) if c.alive]
        if alive_cars:
            self.leader_net = max(alive_cars, key=lambda cn: cn[0].total_progress)[1]

        alive_count = sum(c.alive for c in self.cars)
        if alive_count == 0 or self.steps_this_gen >= self.max_steps_per_gen:
            fitnesses = [c.total_progress for c in self.cars]
            best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            gen_best_fit = fitnesses[best_idx]
            gen_best_net = self.population[best_idx]
            if gen_best_fit > self.best_ever_fit:
                self.best_ever_fit = gen_best_fit
                self.best_ever_net = gen_best_net
                self.save_weights(self.best_ever_net)

            self.population = evolve(self.population, fitnesses)
            self.cars = [car_ai.Car(self.start_x, self.start_y, self.start_angle) for _ in self.population]
            self.generation += 1
            self.steps_this_gen = 0

        self.draw_track()
        for car in self.cars:
            if car.alive:
                self.draw_car(car, '#7fd0ff', size=8)

        if self.show_mind and self.leader_net:
            self.draw_ai_mind(self.leader_net)

        alive_count = sum(c.alive for c in self.cars)
        max_possible = len(car_ai.checkpoints) * 10
        frac = min(1.0, self.best_ever_fit / max_possible) if max_possible else 0
        self.draw_progress_bar(20, 20, 250, 14, self.generation / self.generations_target, '#8ecae6')
        self.canvas.create_text(20, 42, anchor='w', text=f"Gen {self.generation}/{self.generations_target}",
                                 fill=TEXT, font=('Arial', 10, 'bold'))
        self.draw_progress_bar(20, 55, 250, 10, frac, '#90e0af')

        self.hud.config(text=(
            f"TRAINING - Generation {self.generation}/{self.generations_target}   "
            f"Alive: {alive_count}/{POP_SIZE}   Best ever fitness: {self.best_ever_fit:.1f}\n"
            f"Press R to skip ahead and race the best AI found so far."
        ), font=('Arial', 13))

    # ---------------- racing ----------------
    def start_race(self):
        self.mode = 'countdown'
        self.clear_controls()
        self.mind_btn = self.make_button(self.controls, "Show AI Mind", self.toggle_mind,
                                          bg='#33415c', width=16)
        self.mind_btn.pack(side='left', padx=4)
        self.mind_btn.config(text="Hide AI Mind" if self.show_mind else "Show AI Mind")
        net = self.best_ever_net if self.best_ever_net else car_ai.NeuralNet(N_IN, self.hidden_sizes, N_OUT)
        self.ai_net = net
        self.save_weights(net)
        self.player = car_ai.Car(self.start_x, self.start_y, self.start_angle)
        self.ai_car = car_ai.Car(self.start_x, self.start_y, self.start_angle)
        self.race_over = False
        self.end_buttons_shown = False
        self.countdown_start = time.time()

    def countdown_step(self):
        elapsed = time.time() - self.countdown_start
        remaining = COUNTDOWN_SECONDS - elapsed
        self.draw_track()
        self.draw_car(self.ai_car, '#3aa0ff', label="AI")
        self.draw_car(self.player, '#ff4d4d', label="YOU")
        if self.show_mind:
            self.draw_ai_mind(self.ai_net)
        if remaining > 0:
            text = str(int(remaining) + 1)
        else:
            text = "GO!"
        self.canvas.create_text(450, 300, text=text, font=('Arial', 64, 'bold'), fill='#ffffff')
        self.hud.config(text="Get ready! Arrow keys to drive once the race starts.", font=('Arial', 13))
        if remaining <= -0.6:
            self.mode = 'race'
            self.start_time = time.time()

    def race_step(self):
        if not self.race_over:
            up = 'Up' in self.keys_down
            down = 'Down' in self.keys_down
            left = 'Left' in self.keys_down
            right = 'Right' in self.keys_down
            self.player.step_manual(up, down, left, right)
            if self.ai_car.alive:
                self.ai_car.step_ai(self.ai_net)

            player_won = self.player.lap >= LAPS_TO_WIN
            ai_won = self.ai_car.lap >= LAPS_TO_WIN
            if player_won or ai_won:
                self.race_over = True
                self.race_end_time = time.time()

        self.draw_track()
        self.draw_car(self.ai_car, '#3aa0ff', label="AI")
        self.draw_car(self.player, '#ff4d4d', label="YOU")
        if self.show_mind:
            self.draw_ai_mind(self.ai_net)

        # HUD panel: lap progress bars + speedometer
        self.draw_progress_bar(20, 20, 260, 14, min(1.0, self.player.lap / LAPS_TO_WIN), '#ff4d4d')
        self.canvas.create_text(285, 27, anchor='w', text="YOU", fill=TEXT, font=('Arial', 10, 'bold'))
        self.draw_progress_bar(20, 40, 260, 14, min(1.0, self.ai_car.lap / LAPS_TO_WIN), '#3aa0ff')
        self.canvas.create_text(285, 47, anchor='w', text="AI", fill=TEXT, font=('Arial', 10, 'bold'))

        speed_frac = abs(self.player.speed) / self.player.max_speed
        self.draw_progress_bar(20, 62, 140, 8, speed_frac, '#ffd166')
        self.canvas.create_text(165, 66, anchor='w', text="speed", fill=MUTED, font=('Arial', 8))

        if self.race_over:
            player_won = self.player.lap >= LAPS_TO_WIN
            ai_won = self.ai_car.lap >= LAPS_TO_WIN
            if player_won and not ai_won:
                result = "YOU WIN!"
                result_color = '#90e0af'
            elif ai_won and not player_won:
                result = "AI WINS!"
                result_color = '#ff6b6b'
            else:
                result = "TIE!"
                result_color = '#ffd166'
            elapsed = self.race_end_time - self.start_time
            self.canvas.create_rectangle(200, 220, 700, 380, fill='#10131cdd', outline=ACCENT, width=2)
            self.canvas.create_text(450, 270, text=result, font=('Arial', 32, 'bold'), fill=result_color)
            self.canvas.create_text(450, 310, text=f"Race time: {elapsed:.1f}s", font=('Arial', 13), fill=TEXT)
            self.hud.config(text="Race finished.", font=('Arial', 15, 'bold'))
            if not self.end_buttons_shown:
                self.end_buttons_shown = True
                self.make_button(self.controls, "Race Again", self.start_race,
                                  bg='#90e0af', fg='#10131c', width=16).pack(side='left', padx=4)
                self.make_button(self.controls, "Back to Menu", self.build_menu,
                                  bg='#33415c', width=16).pack(side='left', padx=4)
        else:
            elapsed = time.time() - self.start_time
            self.hud.config(text=(
                f"RACE - You: Lap {min(self.player.lap,LAPS_TO_WIN)}/{LAPS_TO_WIN}   "
                f"AI: Lap {min(self.ai_car.lap,LAPS_TO_WIN)}/{LAPS_TO_WIN}   Time: {elapsed:.1f}s"
            ), font=('Arial', 13))

    def on_key_down(self, event):
        self.keys_down.add(event.keysym)
        if event.keysym.lower() == 'r' and self.mode == 'training':
            self.start_race()

    # ---------------- main loop ----------------
    def loop(self):
        if self.mode == 'training':
            self.canvas.delete('all')
            self.training_step()
        elif self.mode == 'countdown':
            self.canvas.delete('all')
            self.countdown_step()
        elif self.mode == 'race':
            self.canvas.delete('all')
            self.race_step()
        # 'menu' and 'editor' modes draw themselves on click/build, not every frame
        self.root.after(16, self.loop)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Race vs AI - choose your track")
    App(root)
    root.mainloop()
