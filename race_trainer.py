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

import car_ai
import tracks

POP_SIZE = 24
GENERATIONS = 60
MAX_STEPS_PER_GEN = 700
SUBSTEPS_PER_FRAME = 4
N_IN, N_HIDDEN, N_OUT = 6, 10, 2
CUSTOM_TRACK_PATH = 'custom_track.json'


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
        self.canvas = tk.Canvas(root, width=900, height=600, bg='#222')
        self.canvas.pack()
        self.hud = tk.Label(root, text="", font=('Arial', 13), justify='left')
        self.hud.pack()
        self.controls = tk.Frame(root)
        self.controls.pack(pady=6)

        self.mode = 'menu'
        self.track_width_var = tk.IntVar(value=90)
        self.editor_points = []

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

    def build_menu(self):
        self.mode = 'menu'
        self.canvas.delete('all')
        self.clear_controls()
        self.hud.config(text="Choose a track:", font=('Arial', 15, 'bold'))

        for name in tracks.PRESET_TRACKS:
            tk.Button(self.controls, text=name, width=16,
                      command=lambda n=name: self.choose_preset(n)).pack(side='left', padx=4)

        tk.Button(self.controls, text="Draw Custom Track", width=18, bg='#8ecae6',
                  command=self.start_editor).pack(side='left', padx=4)

        if os.path.exists(CUSTOM_TRACK_PATH):
            tk.Button(self.controls, text="Load Saved Custom Track", width=20, bg='#ffd166',
                      command=self.load_custom_track).pack(side='left', padx=4)

        width_frame = tk.Frame(self.root)
        width_frame.pack()
        tk.Label(width_frame, text="Track width:").pack(side='left')
        tk.Scale(width_frame, from_=50, to=140, orient='horizontal',
                 variable=self.track_width_var, length=200).pack(side='left')
        self._width_frame = width_frame

    def choose_preset(self, name):
        car_ai.set_track(tracks.PRESET_TRACKS[name], self.track_width_var.get())
        self.begin_training()

    def load_custom_track(self):
        with open(CUSTOM_TRACK_PATH) as f:
            pts = json.load(f)
        car_ai.set_track(pts, self.track_width_var.get())
        self.begin_training()

    # ---------------- track editor ----------------
    def start_editor(self):
        self.mode = 'editor'
        self.editor_points = []
        self.canvas.delete('all')
        self.clear_controls()
        self._width_frame.pack_forget()
        self.hud.config(text="Click to place track points (at least 4). Click points in order around the loop.")

        tk.Button(self.controls, text="Undo last point", command=self.editor_undo).pack(side='left', padx=4)
        tk.Button(self.controls, text="Finish Track", bg='#90e0af',
                  command=self.finish_editor).pack(side='left', padx=4)
        tk.Button(self.controls, text="Cancel", command=self.build_menu).pack(side='left', padx=4)

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
        for i, (x, y) in enumerate(self.editor_points):
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill='yellow')
            if i > 0:
                px, py = self.editor_points[i-1]
                self.canvas.create_line(px, py, x, y, fill='white')
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
        car_ai.set_track(pts, self.track_width_var.get())
        self._width_frame.pack_forget()
        self.begin_training()

    # ---------------- shared drawing ----------------
    def draw_track(self):
        wps = car_ai.waypoints
        flat = [c for p in wps for c in p]
        self.canvas.create_line(*flat, width=car_ai.TRACK_WIDTH, fill='#555',
                                 capstyle=tk.ROUND, joinstyle=tk.ROUND)
        self.canvas.create_line(*flat, width=3, fill='white', dash=(12, 10))
        ax, ay = wps[0]
        bx, by = wps[1]
        ang = math.atan2(by - ay, bx - ax) + math.pi / 2
        hw = car_ai.TRACK_WIDTH / 2
        self.canvas.create_line(ax + math.cos(ang) * hw, ay + math.sin(ang) * hw,
                                 ax - math.cos(ang) * hw, ay - math.sin(ang) * hw,
                                 width=4, fill='yellow')
        for cx, cy in car_ai.checkpoints[1:]:
            self.canvas.create_oval(cx - car_ai.CHECKPOINT_RADIUS, cy - car_ai.CHECKPOINT_RADIUS,
                                     cx + car_ai.CHECKPOINT_RADIUS, cy + car_ai.CHECKPOINT_RADIUS,
                                     outline='#ffffff', width=1, dash=(3, 3))

    def draw_car(self, car, color, size=10):
        cos_a, sin_a = math.cos(car.angle), math.sin(car.angle)
        corners = [(-size, -6), (size, -6), (size, 6), (-size, 6)]
        pts = []
        for cx, cy in corners:
            rx = car.x + cx * cos_a - cy * sin_a
            ry = car.y + cx * sin_a + cy * cos_a
            pts.extend([rx, ry])
        self.canvas.create_polygon(pts, fill=color, outline='')

    # ---------------- training setup ----------------
    def begin_training(self):
        self.mode = 'training'
        self.clear_controls()
        self.canvas.delete('all')

        wps = car_ai.waypoints
        self.start_x, self.start_y = wps[0]
        self.start_angle = math.atan2(wps[1][1]-wps[0][1], wps[1][0]-wps[0][0])

        self.generation = 1
        self.steps_this_gen = 0
        self.best_ever_net = None
        self.best_ever_fit = -1
        self.population = [car_ai.NeuralNet(N_IN, N_HIDDEN, N_OUT) for _ in range(POP_SIZE)]
        self.cars = [car_ai.Car(self.start_x, self.start_y, self.start_angle) for _ in self.population]

    # ---------------- training loop ----------------
    def training_step(self):
        if self.generation > GENERATIONS:
            self.start_race()
            return

        for _ in range(SUBSTEPS_PER_FRAME):
            any_alive = False
            for car, net in zip(self.cars, self.population):
                if car.alive:
                    car.step_ai(net)
                    any_alive = True
            self.steps_this_gen += 1
            if not any_alive or self.steps_this_gen >= MAX_STEPS_PER_GEN:
                break

        alive_count = sum(c.alive for c in self.cars)
        if alive_count == 0 or self.steps_this_gen >= MAX_STEPS_PER_GEN:
            fitnesses = [c.total_progress for c in self.cars]
            best_idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            gen_best_fit = fitnesses[best_idx]
            gen_best_net = self.population[best_idx]
            if gen_best_fit > self.best_ever_fit:
                self.best_ever_fit = gen_best_fit
                self.best_ever_net = gen_best_net

            self.population = evolve(self.population, fitnesses)
            self.cars = [car_ai.Car(self.start_x, self.start_y, self.start_angle) for _ in self.population]
            self.generation += 1
            self.steps_this_gen = 0

        self.draw_track()
        for car in self.cars:
            if car.alive:
                self.draw_car(car, '#7fd0ff', size=8)

        alive_count = sum(c.alive for c in self.cars)
        self.hud.config(text=(
            f"TRAINING - Generation {self.generation}/{GENERATIONS}   "
            f"Alive: {alive_count}/{POP_SIZE}   Best ever fitness: {self.best_ever_fit:.1f}\n"
            f"Press R to skip ahead and race the best AI found so far."
        ), font=('Arial', 13))

    # ---------------- racing ----------------
    def start_race(self):
        self.mode = 'race'
        net = self.best_ever_net if self.best_ever_net else car_ai.NeuralNet(N_IN, N_HIDDEN, N_OUT)
        self.ai_net = net
        try:
            with open('best_weights.json', 'w') as f:
                json.dump(net.get_weights(), f)
        except Exception:
            pass
        self.player = car_ai.Car(self.start_x, self.start_y, self.start_angle)
        self.ai_car = car_ai.Car(self.start_x, self.start_y, self.start_angle)
        self.race_over = False
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

            player_won = self.player.lap >= 3
            ai_won = self.ai_car.lap >= 3
            if player_won or ai_won:
                self.race_over = True

        self.draw_track()
        self.draw_car(self.ai_car, '#39f')
        self.draw_car(self.player, '#e33')

        if self.race_over:
            player_won = self.player.lap >= 3
            ai_won = self.ai_car.lap >= 3
            result = "YOU WIN!" if player_won and not ai_won else "AI WINS!" if ai_won and not player_won else "TIE!"
            self.hud.config(text=result + "   (close window and re-run to pick a new track)", font=('Arial', 20, 'bold'))
        else:
            elapsed = time.time() - self.start_time
            self.hud.config(text=(
                f"RACE - You: Lap {min(self.player.lap,3)}/3   "
                f"AI: Lap {min(self.ai_car.lap,3)}/3   Time: {elapsed:.1f}s"
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
