"""
AI Racing Bets - choose or draw a track, train an AI live in the same window,
then bet coins on it against a rival AI and watch the race play out.

Needs car_ai.py and tracks.py in the same folder.

Run:
    python race_trainer.py

There's no manual driving - once training finishes, pick which AI you think
will win and place a bet. Press R during training to skip ahead early.
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
COINS_PATH = 'coins.json'
STARTING_COINS = 100
MIN_BET, MAX_BET = 5, 200
DEFAULT_BET = 10
LAPS_TO_WIN = 3
COUNTDOWN_SECONDS = 3
INPUT_LABELS = ['S1', 'S2', 'S3', 'S4', 'S5', 'Spd']
OUTPUT_LABELS = ['Steer', 'Gas']

RED = '#ff4d4d'
BLUE = '#3aa0ff'

BG = '#1b1f2b'
PANEL_BG = '#242a3a'
ACCENT = '#8ecae6'
TEXT = '#eef1f8'
MUTED = '#9aa3b8'


def safe_name(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', name)


def weights_path(track_name):
    return f"weights_{safe_name(track_name)}.json"


def load_coins():
    try:
        with open(COINS_PATH) as f:
            return int(json.load(f).get('coins', STARTING_COINS))
    except Exception:
        return STARTING_COINS


def save_coins(coins):
    try:
        with open(COINS_PATH, 'w') as f:
            json.dump({'coins': coins}, f)
    except Exception:
        pass


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

        # Everything lives inside a scrollable body, so on a short screen (laptop,
        # small display) nothing is ever unreachably off-screen - the window just
        # shows a scrollbar instead of clipping content. Only the outer container
        # is scrolled; the game canvas itself keeps its fixed size and coordinates.
        outer = tk.Frame(root, bg=BG)
        outer.pack(fill='both', expand=True)

        self.scroll_canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vscroll = tk.Scrollbar(outer, orient='vertical', command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=vscroll.set)
        self.scroll_canvas.pack(side='left', fill='both', expand=True)
        vscroll.pack(side='right', fill='y')

        self.body = tk.Frame(self.scroll_canvas, bg=BG)
        body_window = self.scroll_canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.body.bind('<Configure>', lambda e: self.scroll_canvas.configure(
            scrollregion=self.scroll_canvas.bbox('all')))
        self.scroll_canvas.bind('<Configure>', lambda e: self.scroll_canvas.itemconfig(
            body_window, width=e.width))

        def on_mousewheel(event):
            delta = -1 if event.num == 4 else 1 if event.num == 5 else int(-event.delta / 120)
            self.scroll_canvas.yview_scroll(delta, 'units')
        self.scroll_canvas.bind_all('<MouseWheel>', on_mousewheel)   # Windows/macOS
        self.scroll_canvas.bind_all('<Button-4>', on_mousewheel)     # Linux scroll up
        self.scroll_canvas.bind_all('<Button-5>', on_mousewheel)     # Linux scroll down

        title = tk.Label(self.body, text="AI RACING BETS", font=('Arial', 22, 'bold'), fg=ACCENT, bg=BG)
        title.pack(pady=(10, 2))

        self.canvas = tk.Canvas(self.body, width=900, height=600, bg='#3a5a3a', highlightthickness=2,
                                 highlightbackground='#000')
        self.canvas.pack(padx=10)

        self.hud = tk.Label(self.body, text="", font=('Arial', 13), justify='left', fg=TEXT, bg=BG)
        self.hud.pack(pady=(6, 0))
        self.controls = tk.Frame(self.body, bg=BG)
        self.controls.pack(pady=8)

        # Cap the window to comfortably fit the screen; the scrollbar handles
        # anything below that, so content is never simply cut off and unreachable.
        root.update_idletasks()
        screen_h = root.winfo_screenheight()
        win_h = min(800, screen_h - 80)
        root.geometry(f"940x{win_h}")
        root.minsize(700, 400)

        self.mode = 'menu'
        self.track_width_var = tk.IntVar(value=90)
        self.generations_var = tk.IntVar(value=GENERATIONS)
        self.steps_var = tk.IntVar(value=MAX_STEPS_PER_GEN)
        self.hidden_layers_var = tk.IntVar(value=DEFAULT_HIDDEN_LAYERS)
        self.neurons_var = tk.IntVar(value=DEFAULT_NEURONS_PER_LAYER)
        self.bet_amount_var = tk.IntVar(value=DEFAULT_BET)
        self.editor_points = []
        self.track_name = "Oval Classic"
        self.show_mind = False
        self.hidden_sizes = [DEFAULT_NEURONS_PER_LAYER] * DEFAULT_HIDDEN_LAYERS
        self.coins = load_coins()

        root.bind('<KeyPress>', self.on_key_down)
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
        self.hud.config(text=f"💰 Coins: {self.coins}   -   Choose a track to train an AI on, "
                              f"then bet on it against a rival:",
                         font=('Arial', 15, 'bold'))

        for name in tracks.PRESET_TRACKS:
            col = tk.Frame(self.controls, bg=BG)
            col.pack(side='left', padx=4)
            self.make_button(col, name, lambda n=name: self.choose_preset(n),
                              bg='#33415c').pack(side='top')
            if os.path.exists(weights_path(name)):
                self.make_button(col, "⚡ Bet on Saved AI", lambda n=name: self.bet_on_saved_ai(n),
                                  bg='#ffd166', fg='#10131c', width=16).pack(side='top', pady=(4, 0))

        custom_col = tk.Frame(self.controls, bg=BG)
        custom_col.pack(side='left', padx=4)
        self.make_button(custom_col, "Draw Custom Track", self.start_editor,
                          bg='#8ecae6', fg='#10131c', width=18).pack(side='top')
        if os.path.exists(CUSTOM_TRACK_PATH):
            self.make_button(custom_col, "Load Saved Track", self.load_custom_track,
                              bg='#8ecae6', fg='#10131c', width=18).pack(side='top', pady=(4, 0))
            if os.path.exists(weights_path("Custom")):
                self.make_button(custom_col, "⚡ Bet on Saved AI", lambda: self.bet_on_saved_ai("Custom"),
                                  bg='#ffd166', fg='#10131c', width=18).pack(side='top', pady=(4, 0))

        if self.coins < MIN_BET:
            self.make_button(self.controls, "💸 Out of coins - reset bankroll", self.reset_bankroll,
                              bg='#c94f4f', width=24).pack(side='left', padx=4)

        # Settings sliders are collapsed by default (they'd otherwise push the window
        # taller than most screens, below a 600px canvas) - toggle to reveal them.
        toggle_col = tk.Frame(self.controls, bg=BG)
        toggle_col.pack(side='left', padx=4)
        self.settings_toggle_btn = self.make_button(
            toggle_col, "⚙ Training Settings ▾", self.toggle_settings_panel,
            bg='#2c3346', width=20)
        self.settings_toggle_btn.pack(side='top')

        self._settings_panel = tk.Frame(self.body, bg=PANEL_BG)
        self._settings_visible = False

        def slider(parent, label, var, lo, hi, r, c):
            cell = tk.Frame(parent, bg=PANEL_BG)
            cell.grid(row=r, column=c, padx=12, pady=6, sticky='w')
            tk.Label(cell, text=label, fg=MUTED, bg=PANEL_BG).pack(side='left')
            tk.Scale(cell, from_=lo, to=hi, orient='horizontal', variable=var, length=150,
                     bg=PANEL_BG, fg=TEXT, troughcolor=BG, highlightthickness=0).pack(side='left')

        slider(self._settings_panel, "Track width:", self.track_width_var, 50, 140, 0, 0)
        slider(self._settings_panel, "Generations:", self.generations_var, 10, 300, 0, 1)
        slider(self._settings_panel, "Time per round (steps):", self.steps_var, 200, 4000, 0, 2)
        slider(self._settings_panel, "Hidden layers:", self.hidden_layers_var, 1, 4, 1, 0)
        slider(self._settings_panel, "Neurons per hidden layer:", self.neurons_var, 4, 24, 1, 1)

        if self._settings_visible:
            self._settings_panel.pack(pady=(0, 8))

    def toggle_settings_panel(self):
        self._settings_visible = not self._settings_visible
        if self._settings_visible:
            self._settings_panel.pack(pady=(0, 8))
            self.settings_toggle_btn.config(text="⚙ Training Settings ▴")
        else:
            self._settings_panel.pack_forget()
            self.settings_toggle_btn.config(text="⚙ Training Settings ▾")

    def reset_bankroll(self):
        self.coins = STARTING_COINS
        save_coins(self.coins)
        self.build_menu()

    def draw_menu_background(self):
        self.canvas.configure(bg='#20242f')
        w, h = 900, 600
        # faint preview of the first preset track as decoration
        preview = tracks.PRESET_TRACKS.get("Oval Classic")
        if preview:
            flat = [c for p in preview for c in p]
            self.canvas.create_line(*flat, width=70, fill='#2a2f3d', capstyle=tk.ROUND, joinstyle=tk.ROUND)
            self.canvas.create_line(*flat, width=2, fill='#3a4054', dash=(10, 8))
        self.canvas.create_text(w/2, h/2 - 40, text="AI RACING BETS", font=('Arial', 40, 'bold'), fill='#333a4d')
        self.canvas.create_text(w/2, h/2 + 10, text="pick a track below to train an AI, then bet on the race",
                                 font=('Arial', 14), fill='#454d63')

    def _load_saved_net(self, track_name):
        """Load a previously saved AI for a track, or None if there isn't one."""
        path = weights_path(track_name)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                weights = json.load(f)
            return car_ai.NeuralNet(N_IN, [], N_OUT, weights=weights)
        except Exception:
            return None

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

    def bet_on_saved_ai(self, name):
        """Load a previously saved AI ('Champion') and pit it against a fresh rookie AI to bet on."""
        self.track_name = name
        if name == "Custom":
            with open(CUSTOM_TRACK_PATH) as f:
                pts = json.load(f)
            car_ai.set_track(pts, self.track_width_var.get())
        else:
            car_ai.set_track(tracks.PRESET_TRACKS[name], self.track_width_var.get())

        champion = self._load_saved_net(name)
        rookie = car_ai.NeuralNet(champion.n_in, champion.hidden_sizes, champion.n_out)

        wps = car_ai.waypoints
        self.start_x, self.start_y = wps[0]
        self.start_angle = math.atan2(wps[1][1]-wps[0][1], wps[1][0]-wps[0][0])
        self.start_betting(champion, "Champion", rookie, "Rookie")

    # ---------------- track editor ----------------
    def start_editor(self):
        self.mode = 'editor'
        self.editor_points = []
        self.canvas.delete('all')
        self.canvas.configure(bg='#20242f')
        self.clear_controls()
        self._settings_panel.pack_forget()
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
        self._settings_panel.pack_forget()
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
        """Persist a net's weights to a per-track file so it can be bet on later without retraining."""
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
        """Live view of an AI's neural network: nodes lit by activation, edges by weight.

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

        # Whatever AI was previously saved for this track becomes the rival to bet
        # against once training finishes - captured now, before training overwrites it.
        self.reigning_champion = self._load_saved_net(self.track_name)

        self.generations_target = self.generations_var.get()
        self.max_steps_per_gen = self.steps_var.get()
        self.hidden_sizes = [self.neurons_var.get()] * self.hidden_layers_var.get()
        self.generation = 1
        self.steps_this_gen = 0
        self.best_ever_net = None
        self.best_ever_fit = float('-inf')  # fitness can go negative now (off-track penalty), so -1 wasn't low enough
        self.leader_net = None
        self.population = [car_ai.NeuralNet(N_IN, self.hidden_sizes, N_OUT) for _ in range(POP_SIZE)]
        self.cars = [car_ai.Car(self.start_x, self.start_y, self.start_angle) for _ in self.population]

    def toggle_mind(self):
        self.show_mind = not self.show_mind
        self.mind_btn.config(text="Hide AI Mind" if self.show_mind else "Show AI Mind")

    # ---------------- training loop ----------------
    def training_step(self):
        if self.generation > self.generations_target:
            # If training was skipped (R key) before any generation finished evaluating,
            # best_ever_net can still be None - fall back to a fresh net rather than crash.
            trained_net = self.best_ever_net or car_ai.NeuralNet(N_IN, self.hidden_sizes, N_OUT)
            self.save_weights(trained_net)
            if self.reigning_champion is not None:
                self.start_betting(trained_net, "New Challenger", self.reigning_champion, "Reigning Champion")
            else:
                rookie = car_ai.NeuralNet(N_IN, self.hidden_sizes, N_OUT)
                self.start_betting(trained_net, "Your AI", rookie, "Untrained Rookie")
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
            f"Press R to skip ahead once you're ready to bet on the best AI found so far."
        ), font=('Arial', 13))

    # ---------------- betting ----------------
    def start_betting(self, net_a, name_a, net_b, name_b):
        """Show the two contestants and let the user pick one to bet on before racing."""
        self.mode = 'betting'
        self.contestant_a, self.name_a = net_a, name_a
        self.contestant_b, self.name_b = net_b, name_b
        self.bet_side = None
        self.clear_controls()

        tk.Label(self.controls, text="Bet amount:", fg=TEXT, bg=BG).pack(side='left', padx=(4, 0))
        tk.Scale(self.controls, from_=MIN_BET, to=min(MAX_BET, max(MIN_BET, self.coins)), orient='horizontal',
                 variable=self.bet_amount_var, length=140, bg=BG, fg=TEXT,
                 troughcolor=PANEL_BG, highlightthickness=0).pack(side='left', padx=(0, 10))
        self.make_button(self.controls, f"🔴 Bet on {name_a}", lambda: self.place_bet('a'),
                          bg=RED, fg='#10131c', width=20).pack(side='left', padx=4)
        self.make_button(self.controls, f"🔵 Bet on {name_b}", lambda: self.place_bet('b'),
                          bg=BLUE, fg='#10131c', width=20).pack(side='left', padx=4)
        self.make_button(self.controls, "Back to Menu", self.build_menu, bg='#33415c').pack(side='left', padx=4)

        self.betting_car_a = car_ai.Car(self.start_x, self.start_y, self.start_angle)
        self.betting_car_b = car_ai.Car(self.start_x, self.start_y, self.start_angle)

    def betting_step(self):
        self.draw_track()
        self.draw_car(self.betting_car_a, RED, label=self.name_a)
        self.draw_car(self.betting_car_b, BLUE, label=self.name_b)
        self.hud.config(text=(
            f"💰 Coins: {self.coins}   -   Place your bet: who wins this race, "
            f"🔴 {self.name_a} or 🔵 {self.name_b}?"
        ), font=('Arial', 15, 'bold'))

    def place_bet(self, side):
        if self.coins < MIN_BET:
            return
        self.bet_amount = min(self.bet_amount_var.get(), self.coins)
        self.bet_side = side
        self.start_race()

    # ---------------- racing (AI vs AI - the outcome of what you bet on) ----------------
    def start_race(self):
        self.mode = 'countdown'
        self.clear_controls()
        self.mind_btn = self.make_button(self.controls, "Show AI Mind", self.toggle_mind,
                                          bg='#33415c', width=16)
        self.mind_btn.pack(side='left', padx=4)
        self.mind_btn.config(text="Hide AI Mind" if self.show_mind else "Show AI Mind")
        self.car_a = car_ai.Car(self.start_x, self.start_y, self.start_angle)
        self.car_b = car_ai.Car(self.start_x, self.start_y, self.start_angle)
        self.race_over = False
        self.end_buttons_shown = False
        self.countdown_start = time.time()

    def _picked_net(self):
        return self.contestant_a if self.bet_side == 'a' else self.contestant_b

    def countdown_step(self):
        elapsed = time.time() - self.countdown_start
        remaining = COUNTDOWN_SECONDS - elapsed
        self.draw_track()
        self.draw_car(self.car_a, RED, label=self.name_a)
        self.draw_car(self.car_b, BLUE, label=self.name_b)
        if self.show_mind:
            self.draw_ai_mind(self._picked_net())
        text = str(int(remaining) + 1) if remaining > 0 else "GO!"
        self.canvas.create_text(450, 300, text=text, font=('Arial', 64, 'bold'), fill='#ffffff')
        picked_name = self.name_a if self.bet_side == 'a' else self.name_b
        self.hud.config(text=f"You bet {self.bet_amount} coins on {picked_name}. Get ready!",
                         font=('Arial', 13))
        if remaining <= -0.6:
            self.mode = 'race'
            self.start_time = time.time()

    def race_step(self):
        if not self.race_over:
            if self.car_a.alive:
                self.car_a.step_ai(self.contestant_a)
            if self.car_b.alive:
                self.car_b.step_ai(self.contestant_b)

            a_won = self.car_a.lap >= LAPS_TO_WIN
            b_won = self.car_b.lap >= LAPS_TO_WIN
            if a_won or b_won:
                # This block runs exactly once, the frame the race ends - it settles the bet
                # here rather than in the drawing code below, which re-runs every frame
                # afterwards (until the player clicks a button) and would otherwise pay out
                # or charge the bet again on every single one of those frames.
                self.race_over = True
                self.race_end_time = time.time()
                tie = a_won and b_won
                winner_side = None if tie else ('a' if a_won else 'b')
                winner_name = "Nobody" if tie else (self.name_a if winner_side == 'a' else self.name_b)
                if tie:
                    self.result_text = "IT'S A TIE - bet refunded"
                    self.result_color = '#ffd166'
                elif winner_side == self.bet_side:
                    self.coins += self.bet_amount
                    self.result_text = f"{winner_name} WINS! You won {self.bet_amount} coins!"
                    self.result_color = '#90e0af'
                else:
                    self.coins -= self.bet_amount
                    self.result_text = f"{winner_name} WINS! You lost {self.bet_amount} coins."
                    self.result_color = '#ff6b6b'
                save_coins(self.coins)

        self.draw_track()
        self.draw_car(self.car_a, RED, label=self.name_a)
        self.draw_car(self.car_b, BLUE, label=self.name_b)
        if self.show_mind:
            self.draw_ai_mind(self._picked_net())

        # HUD panel: lap progress bars for both contestants
        self.draw_progress_bar(20, 20, 260, 14, min(1.0, self.car_a.lap / LAPS_TO_WIN), RED)
        self.canvas.create_text(285, 27, anchor='w', text=self.name_a, fill=TEXT, font=('Arial', 10, 'bold'))
        self.draw_progress_bar(20, 40, 260, 14, min(1.0, self.car_b.lap / LAPS_TO_WIN), BLUE)
        self.canvas.create_text(285, 47, anchor='w', text=self.name_b, fill=TEXT, font=('Arial', 10, 'bold'))

        if self.race_over:
            elapsed = self.race_end_time - self.start_time
            self.canvas.create_rectangle(150, 210, 750, 390, fill='#10131c', outline=ACCENT, width=2)
            self.canvas.create_text(450, 260, text=self.result_text, font=('Arial', 22, 'bold'), fill=self.result_color)
            self.canvas.create_text(450, 300, text=f"Race time: {elapsed:.1f}s", font=('Arial', 13), fill=TEXT)
            self.canvas.create_text(450, 330, text=f"💰 Coins: {self.coins}", font=('Arial', 16, 'bold'), fill=TEXT)
            self.hud.config(text="Race finished.", font=('Arial', 15, 'bold'))
            if not self.end_buttons_shown:
                self.end_buttons_shown = True
                self.make_button(self.controls, "Bet Again (same AIs)",
                                  lambda: self.start_betting(self.contestant_a, self.name_a,
                                                              self.contestant_b, self.name_b),
                                  bg='#90e0af', fg='#10131c', width=20).pack(side='left', padx=4)
                self.make_button(self.controls, "Back to Menu", self.build_menu,
                                  bg='#33415c', width=16).pack(side='left', padx=4)
        else:
            elapsed = time.time() - self.start_time
            self.hud.config(text=(
                f"RACING - {self.name_a}: Lap {min(self.car_a.lap,LAPS_TO_WIN)}/{LAPS_TO_WIN}   "
                f"{self.name_b}: Lap {min(self.car_b.lap,LAPS_TO_WIN)}/{LAPS_TO_WIN}   Time: {elapsed:.1f}s"
            ), font=('Arial', 13))

    def on_key_down(self, event):
        if event.keysym.lower() == 'r' and self.mode == 'training':
            self.training_step_skip_to_betting()

    def training_step_skip_to_betting(self):
        self.generation = self.generations_target + 1
        self.training_step()

    # ---------------- main loop ----------------
    def loop(self):
        if self.mode == 'training':
            self.canvas.delete('all')
            self.training_step()
        elif self.mode == 'betting':
            self.canvas.delete('all')
            self.betting_step()
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
    root.title("AI Racing Bets - choose your track")
    App(root)
    root.mainloop()
