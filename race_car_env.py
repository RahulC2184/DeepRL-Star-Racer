# import pygame
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces

# class RaceCarEnv(gym.Env):
#     def __init__(self):
#         super(RaceCarEnv, self).__init__()
#         self.width = 800
#         self.height = 600
        
#         self.outer_track_points = [(50, 50), (750, 50), (750, 550), (50, 550), (50, 50)]
#         self.inner_track_points = [(150, 150), (650, 150), (650, 450), (150, 450), (150, 150)]
#         self.track_center_points = [(100, 100), (700, 100), (700, 500), (100, 500), (100, 100)]
        
#         self.car_size = 20
#         self.max_speed = 5
#         self.max_turn_angle = np.pi / 30

#         self.screen = None
#         self.clock = None
#         self.car_pos = None
#         self.car_angle = None
#         self.speed = None
#         self.current_track_segment = 0
#         self.max_steps = 500
#         self.steps_taken = 0
#         self.laps_completed = 0
#         self.episode_reward = 0
        
#         self.stagnation_counter = 0
#         self.last_pos = None

#         self.action_space = spaces.Discrete(4)
        
#         self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

#         self.last_reward = 0
#         self.last_action_str = "None"
#         self.last_observation = np.zeros(self.observation_space.shape, dtype=np.float32)

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.car_pos = np.array([100.0, 100.0])
#         self.car_angle = np.pi / 2
#         self.speed = 0.0
#         self.current_track_segment = 0
#         self.steps_taken = 0
#         self.laps_completed = 0
#         self.episode_reward = 0
        
#         self.stagnation_counter = 0
#         self.last_pos = self.car_pos.copy()
        
#         observation = self._get_observation()
#         info = {}
        
#         self.last_reward = 0
#         self.last_action_str = "None"
#         self.last_observation = observation.copy()
        
#         return observation, info

#     def step(self, action):
#         self.steps_taken += 1
#         prev_pos = self.car_pos.copy()

#         if action == 0:
#             self.speed = min(self.max_speed, self.speed + 0.2)
#             self.last_action_str = "Accelerate"
#         elif action == 1:
#             self.speed = max(0, self.speed - 0.4)
#             self.last_action_str = "Brake"
#         elif action == 2:
#             self.car_angle += self.max_turn_angle
#             self.last_action_str = "Turn Left"
#         elif action == 3:
#             self.car_angle -= self.max_turn_angle
#             self.last_action_str = "Turn Right"
            
#         self.car_pos[0] += self.speed * np.cos(self.car_angle)
#         self.car_pos[1] += self.speed * np.sin(self.car_angle)

#         reward = 0
#         done = False
#         truncated = False
        
#         dist_moved = np.linalg.norm(self.car_pos - self.last_pos)
#         if dist_moved < 1.0:
#             self.stagnation_counter += 1
#         else:
#             self.stagnation_counter = 0
#         self.last_pos = self.car_pos.copy()

#         if self._is_off_track():
#             reward = -200
#             done = True
#         else:
#             # We now separate the unconditional lap completion reward from the others
#             lap_reward, lap_completed = self._get_lap_completion_status()
#             progress_reward = self._get_progress_reward(prev_pos)
#             centerline_reward = self._get_centerline_reward()
#             speed_reward = self._get_speed_reward()

#             reward = progress_reward + centerline_reward + speed_reward
#             if lap_completed:
#                 self.laps_completed += 1
#                 reward += lap_reward
#                 done = True # End episode on lap completion

#         stagnation_penalty = 0
#         if self.stagnation_counter > 50:
#             stagnation_penalty = -10.0
#         if self.stagnation_counter > 100:
#             stagnation_penalty = -20.0
#         reward += stagnation_penalty
            
#         reward -= 0.1

#         # The episode now ends if a lap is completed or a crash occurs
#         # The max_steps condition is removed
#         # Removed redundant done check here; it's handled in the lap reward logic

#         self.episode_reward += reward
#         self.last_reward = reward

#         observation = self._get_observation()
#         info = {"laps": self.laps_completed, "episode_reward": self.episode_reward}
        
#         self.last_observation = observation.copy()

#         return observation, reward, done, truncated, info

#     def _get_lap_completion_status(self):
#         # A simple check: if the car is at the finish line and has previously been at the other side of the track
#         # This is a very basic "unconditional" lap check.
#         x, y = self.car_pos
#         # We'll consider the car to have crossed the finish line if it passes a certain X-coordinate
#         # on the last leg of the track
#         if self.current_track_segment == len(self.track_center_points) - 2: # At the final segment
#              if x > 100: # Has passed the x=100 line from the bottom left
#                  return 500, True # Award a large reward and signal lap completion
#         return 0, False

#     def _is_off_track(self):
#         x, y = self.car_pos
#         is_outside_outer = not (50 <= x <= 750 and 50 <= y <= 550)
#         is_inside_inner = (150 <= x <= 650 and 150 <= y <= 450)
#         return is_outside_outer or is_inside_inner

#     def _get_progress_reward(self, prev_pos):
#         reward = 0
#         target_point = self.track_center_points[self.current_track_segment + 1]
        
#         dist_to_target = np.linalg.norm(self.car_pos - target_point)
#         dist_from_prev_to_target = np.linalg.norm(prev_pos - target_point)

#         reward += dist_from_prev_to_target - dist_to_target

#         if dist_to_target < 50:
#             self.current_track_segment = (self.current_track_segment + 1) % (len(self.track_center_points) - 1)
#             reward += 50
#         return reward

#     def _get_centerline_reward(self):
#         x, y = self.car_pos
#         if 150 < x < 650 and 150 < y < 450:
#             return 0
#         if 50 > x or x > 750 or 50 > y or y > 550:
#             return 0
        
#         dist_x = min(x - 50, 750 - x)
#         dist_y = min(y - 50, 550 - y)
#         dist_to_outer = min(dist_x, dist_y)
        
#         dist_x_inner = min(x - 150, 650 - x)
#         dist_y_inner = min(y - 150, 450 - y)
#         dist_to_inner = min(dist_x_inner, dist_y_inner)

#         center_dist = abs(dist_to_outer - dist_to_inner)
#         max_dist_from_center = (750-150)/2
#         normalized_center_dist = center_dist / max_dist_from_center
#         return 1.0 - normalized_center_dist

#     def _get_speed_reward(self):
#         return self.speed / self.max_speed * 0.5

#     def _get_observation(self):
#         sensor_angles = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
#         sensor_readings = [self._get_distance_to_wall(self.car_angle + angle) for angle in sensor_angles]
        
#         normalized_pos_x = self.car_pos[0] / self.width
#         normalized_pos_y = self.car_pos[1] / self.height
#         normalized_angle = (self.car_angle % (2 * np.pi) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
#         normalized_speed = self.speed / self.max_speed
        
#         normalized_sensors = [s / self.width for s in sensor_readings]
        
#         normalized_stagnation = self.stagnation_counter / self.max_steps
        
#         return np.array([normalized_pos_x, normalized_pos_y, normalized_angle, normalized_speed] + normalized_sensors + [normalized_stagnation], dtype=np.float32)

#     def _get_distance_to_wall(self, angle):
#         x, y = self.car_pos
#         dx, dy = np.cos(angle), np.sin(angle)
        
#         max_ray_length = max(self.width, self.height)
        
#         for i in range(1, int(max_ray_length)):
#             test_x = x + dx * i
#             test_y = y + dy * i
            
#             is_outside_outer = not (50 <= test_x <= 750 and 50 <= test_y <= 550)
#             is_inside_inner = (150 <= test_x <= 650 and 150 <= test_y <= 450)
            
#             if is_outside_outer or is_inside_inner:
#                 return float(i)
#         return float(max_ray_length)

#     def render(self):
#         if self.screen is None:
#             pygame.init()
#             pygame.display.set_caption("Speedy AI Racer")
#             self.screen = pygame.display.set_mode((self.width, self.height))
#             self.clock = pygame.time.Clock()
#             self.font = pygame.font.Font(None, 24)
            
#         self.screen.fill((0, 0, 0))

#         pygame.draw.lines(self.screen, (255, 255, 255), True, self.outer_track_points, 5)
#         pygame.draw.lines(self.screen, (255, 255, 255), True, self.inner_track_points, 5)

#         for i in range(len(self.track_center_points) - 1):
#             color = (0, 150, 0)
#             if i == self.current_track_segment:
#                 color = (0, 255, 0)
            
#             p1 = self.track_center_points[i]
#             p2 = self.track_center_points[i+1]
#             pygame.draw.line(self.screen, color, p1, p2, 2)

#         car_points = []
#         car_points.append((self.car_pos[0] + self.car_size * np.cos(self.car_angle),
#                            self.car_pos[1] + self.car_size * np.sin(self.car_angle)))
#         car_points.append((self.car_pos[0] + self.car_size * 0.7 * np.cos(self.car_angle + 2*np.pi/3),
#                            self.car_pos[1] + self.car_size * 0.7 * np.sin(self.car_angle + 2*np.pi/3)))
#         car_points.append((self.car_pos[0] + self.car_size * 0.7 * np.cos(self.car_angle - 2*np.pi/3),
#                            self.car_pos[1] + self.car_size * 0.7 * np.sin(self.car_angle - 2*np.pi/3)))
        
#         pygame.draw.polygon(self.screen, (0, 255, 0), car_points)
#         pygame.draw.circle(self.screen, (255, 0, 0), (int(self.car_pos[0]), int(self.car_pos[1])), 5)

#         text_color = (255, 255, 255)
#         self._display_text(f"Episode: {self.current_episode}", 10, 10, text_color)
#         self._display_text(f"Last Reward: {self.last_reward:.2f}", 10, 40, text_color)
#         self._display_text(f"Ep. Total Reward: {self.episode_reward:.2f}", 10, 70, text_color)
#         self._display_text(f"Action: {self.last_action_str}", self.width - 200, 10, text_color)
#         self._display_text(f"Speed: {self.speed:.2f}", self.width - 200, 40, text_color)
#         self._display_text(f"Laps: {self.laps_completed}", self.width - 200, 70, text_color)
        
#         self._display_text(f"Stagnation: {self.stagnation_counter}", self.width - 200, 100, text_color)
        
#         normalized_sensors_text = [f"{s:.2f}" for s in self.last_observation[4:9]]
#         obs_text = f"Sensors: F:{normalized_sensors_text[2]} FL:{normalized_sensors_text[1]} L:{normalized_sensors_text[0]} R:{normalized_sensors_text[3]} FR:{normalized_sensors_text[4]}"
#         self._display_text(obs_text, 10, self.height - 30, text_color)
        
#         pygame.display.flip()

#     def _display_text(self, text, x, y, color):
#         text_surface = self.font.render(text, True, color)
#         self.screen.blit(text_surface, (x, y))

#     def close(self):
#         if self.screen is not None:
#             pygame.display.quit()
#             pygame.quit()
#             self.screen = None
#             self.clock = None
#             self.font = None


# import pygame
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces

# class RaceCarEnv(gym.Env):
#     def __init__(self):
#         super(RaceCarEnv, self).__init__()
#         self.width = 800
#         self.height = 600
        
#         self.outer_track_points = self._generate_star_points(300, 300, 300, 150)
#         self.inner_track_points = self._generate_star_points(300, 300, 200, 100)
        
#         self.track_center_points = self._generate_star_points(300, 300, 250, 125)
        
#         self.car_size = 20
#         self.max_speed = 5
#         self.max_turn_angle = np.pi / 30

#         self.screen = None
#         self.clock = None
#         self.car_pos = None
#         self.car_angle = None
#         self.speed = None
#         self.current_track_segment = 0
#         self.max_steps = 500
#         self.steps_taken = 0
#         self.laps_completed = 0
#         self.episode_reward = 0
        
#         self.stagnation_counter = 0
#         self.last_pos = None

#         self.action_space = spaces.Discrete(4)
        
#         self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

#         self.last_reward = 0
#         self.last_action_str = "None"
#         self.last_observation = np.zeros(self.observation_space.shape, dtype=np.float32)

#     def _generate_star_points(self, center_x, center_y, outer_radius, inner_radius):
#         points = []
#         for i in range(10):
#             if i % 2 == 0:
#                 radius = outer_radius
#             else:
#                 radius = inner_radius
#             angle = np.pi / 2 - (i / 10) * 2 * np.pi
#             x = center_x + radius * np.cos(angle)
#             y = center_y - radius * np.sin(angle)
#             points.append((x, y))
#         return points

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.car_pos = np.array(self.track_center_points[0])
#         self.car_angle = np.pi / 2
#         self.speed = 0.0
#         self.current_track_segment = 0
#         self.steps_taken = 0
#         self.laps_completed = 0
#         self.episode_reward = 0
        
#         self.stagnation_counter = 0
#         self.last_pos = self.car_pos.copy()
        
#         observation = self._get_observation()
#         info = {}
        
#         self.last_reward = 0
#         self.last_action_str = "None"
#         self.last_observation = observation.copy()
        
#         return observation, info

#     def step(self, action):
#         self.steps_taken += 1
#         prev_pos = self.car_pos.copy()

#         if action == 0:
#             self.speed = min(self.max_speed, self.speed + 0.2)
#             self.last_action_str = "Accelerate"
#         elif action == 1:
#             self.speed = max(0, self.speed - 0.4)
#             self.last_action_str = "Brake"
#         elif action == 2:
#             self.car_angle += self.max_turn_angle
#             self.last_action_str = "Turn Left"
#         elif action == 3:
#             self.car_angle -= self.max_turn_angle
#             self.last_action_str = "Turn Right"
            
#         self.car_pos[0] += self.speed * np.cos(self.car_angle)
#         self.car_pos[1] += self.speed * np.sin(self.car_angle)

#         reward = 0
#         done = False
#         truncated = False
        
#         dist_moved = np.linalg.norm(self.car_pos - self.last_pos)
#         if dist_moved < 1.0:
#             self.stagnation_counter += 1
#         else:
#             self.stagnation_counter = 0
#         self.last_pos = self.car_pos.copy()

#         if not self._is_inside_track(self.car_pos):
#             reward = -200
#             done = True
#         else:
#             lap_reward, lap_completed = self._get_lap_completion_status()
#             progress_reward = self._get_progress_reward(prev_pos)
#             centerline_reward = self._get_centerline_reward()
#             speed_reward = self._get_speed_reward()

#             reward = progress_reward + centerline_reward + speed_reward
#             if lap_completed:
#                 self.laps_completed += 1
#                 reward += lap_reward
#                 done = True

#         stagnation_penalty = 0
#         if self.stagnation_counter > 50:
#             stagnation_penalty = -10.0
#         if self.stagnation_counter > 100:
#             stagnation_penalty = -20.0
#         reward += stagnation_penalty
            
#         reward -= 0.1

#         self.episode_reward += reward
#         self.last_reward = reward

#         observation = self._get_observation()
#         info = {"laps": self.laps_completed, "episode_reward": self.episode_reward}
        
#         self.last_observation = observation.copy()

#         return observation, reward, done, truncated, info

#     def _is_inside_track(self, point):
#         def is_inside_polygon(point, polygon):
#             x, y = point
#             n = len(polygon)
#             inside = False
#             p1x, p1y = polygon[0]
#             for i in range(n + 1):
#                 p2x, p2y = polygon[i % n]
#                 if y > min(p1y, p2y) and y <= max(p1y, p2y):
#                     if x <= max(p1x, p2x):
#                         if p1y != p2y:
#                             xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#                         if p1x == p2x or x <= xinters:
#                             inside = not inside
#                 p1x, p1y = p2x, p2y
#             return inside

#         is_inside_outer = is_inside_polygon(point, self.outer_track_points)
#         is_inside_inner = is_inside_polygon(point, self.inner_track_points)
        
#         return is_inside_outer and not is_inside_inner

#     def _get_lap_completion_status(self):
#         distances = np.linalg.norm(np.array(self.track_center_points) - self.car_pos, axis=1)
#         if distances[0] < 50:
#             if self.laps_completed == 0:
#                 if self.steps_taken > 100:
#                     return 500, True
#             else:
#                 return 500, True
#         return 0, False

#     def _get_progress_reward(self, prev_pos):
#         reward = 0
#         target_point = np.array(self.track_center_points[(self.current_track_segment + 1) % len(self.track_center_points)])
        
#         dist_to_target = np.linalg.norm(self.car_pos - target_point)
#         dist_from_prev_to_target = np.linalg.norm(prev_pos - target_point)

#         reward += dist_from_prev_to_target - dist_to_target

#         if dist_to_target < 50:
#             self.current_track_segment = (self.current_track_segment + 1) % len(self.track_center_points)
#             reward += 50
#         return reward
    
#     def _get_centerline_reward(self):
#         distances = np.linalg.norm(np.array(self.track_center_points) - self.car_pos, axis=1)
#         dist_to_center = np.min(distances)
        
#         normalized_dist = dist_to_center / (self.height/2)
#         return 1.0 - normalized_dist

#     def _get_speed_reward(self):
#         return self.speed / self.max_speed * 0.5
        
#     def _get_observation(self):
#         sensor_angles = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
#         sensor_readings = [self._get_distance_to_wall(self.car_angle + angle) for angle in sensor_angles]
        
#         normalized_pos_x = self.car_pos[0] / self.width
#         normalized_pos_y = self.car_pos[1] / self.height
#         normalized_angle = (self.car_angle % (2 * np.pi) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
#         normalized_speed = self.speed / self.max_speed
        
#         normalized_sensors = [s / self.width for s in sensor_readings]
#         normalized_stagnation = self.stagnation_counter / self.max_steps
        
#         return np.array([normalized_pos_x, normalized_pos_y, normalized_angle, normalized_speed] + normalized_sensors + [normalized_stagnation], dtype=np.float32)

#     def _get_distance_to_wall(self, angle):
#         x, y = self.car_pos
#         dx, dy = np.cos(angle), np.sin(angle)
        
#         max_ray_length = max(self.width, self.height)
        
#         for i in range(1, int(max_ray_length)):
#             test_x = x + dx * i
#             test_y = y + dy * i
            
#             if not self._is_inside_track((test_x, test_y)):
#                 return float(i)
#         return float(max_ray_length)

#     def render(self):
#         if self.screen is None:
#             pygame.init()
#             pygame.display.set_caption("Speedy AI Racer")
#             self.screen = pygame.display.set_mode((self.width, self.height))
#             self.clock = pygame.time.Clock()
#             self.font = pygame.font.Font(None, 24)
            
#         self.screen.fill((0, 0, 0))

#         pygame.draw.lines(self.screen, (255, 255, 255), True, self.outer_track_points, 5)
#         pygame.draw.lines(self.screen, (255, 255, 255), True, self.inner_track_points, 5)

#         for i in range(len(self.track_center_points)):
#             color = (0, 150, 0)
#             if i == self.current_track_segment:
#                 color = (0, 255, 0)
            
#             p1 = self.track_center_points[i]
#             p2 = self.track_center_points[(i + 1) % len(self.track_center_points)]
#             pygame.draw.line(self.screen, color, p1, p2, 2)

#         start_line_points = [
#             self.inner_track_points[0], 
#             self.outer_track_points[0]
#         ]
#         pygame.draw.line(self.screen, (255, 0, 0), start_line_points[0], start_line_points[1], 5)

#         car_points = []
#         car_points.append((self.car_pos[0] + self.car_size * np.cos(self.car_angle),
#                            self.car_pos[1] + self.car_size * np.sin(self.car_angle)))
#         car_points.append((self.car_pos[0] + self.car_size * 0.7 * np.cos(self.car_angle + 2*np.pi/3),
#                            self.car_pos[1] + self.car_size * 0.7 * np.sin(self.car_angle + 2*np.pi/3)))
#         car_points.append((self.car_pos[0] + self.car_size * 0.7 * np.cos(self.car_angle - 2*np.pi/3),
#                            self.car_pos[1] + self.car_size * 0.7 * np.sin(self.car_angle - 2*np.pi/3)))
        
#         pygame.draw.polygon(self.screen, (0, 255, 0), car_points)
#         pygame.draw.circle(self.screen, (255, 0, 0), (int(self.car_pos[0]), int(self.car_pos[1])), 5)

#         text_color = (255, 255, 255)
#         self._display_text(f"Episode: {self.current_episode}", 10, 10, text_color)
#         self._display_text(f"Last Reward: {self.last_reward:.2f}", 10, 40, text_color)
#         self._display_text(f"Ep. Total Reward: {self.episode_reward:.2f}", 10, 70, text_color)
#         self._display_text(f"Action: {self.last_action_str}", self.width - 200, 10, text_color)
#         self._display_text(f"Speed: {self.speed:.2f}", self.width - 200, 40, text_color)
#         self._display_text(f"Laps: {self.laps_completed}", self.width - 200, 70, text_color)
        
#         self._display_text(f"Stagnation: {self.stagnation_counter}", self.width - 200, 100, text_color)
        
#         # Display the car's position
#         self._display_text(f"Pos: ({self.car_pos[0]:.0f}, {self.car_pos[1]:.0f})", self.width - 200, 130, text_color)
        
#         normalized_sensors_text = [f"{s:.2f}" for s in self.last_observation[4:9]]
#         obs_text = f"Sensors: F:{normalized_sensors_text[2]} FL:{normalized_sensors_text[1]} L:{normalized_sensors_text[0]} R:{normalized_sensors_text[3]} FR:{normalized_sensors_text[4]}"
#         self._display_text(obs_text, 10, self.height - 30, text_color)
        
#         pygame.display.flip()

#     def _display_text(self, text, x, y, color):
#         text_surface = self.font.render(text, True, color)
#         self.screen.blit(text_surface, (x, y))

#     def close(self):
#         if self.screen is not None:
#             pygame.display.quit()
#             pygame.quit()
#             self.screen = None
#             self.clock = None
#             self.font = None




# import pygame
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces

# class RaceCarEnv(gym.Env):
#     def __init__(self):
#         super(RaceCarEnv, self).__init__()
#         self.width = 800
#         self.height = 600
        
#         self.outer_track_points = self._generate_star_points(300, 300, 300, 150)
#         self.inner_track_points = self._generate_star_points(300, 300, 200, 100)
        
#         self.track_center_points = self._generate_star_points(300, 300, 250, 125)
        
#         self.car_size = 20
#         self.max_speed = 5
#         self.max_turn_angle = np.pi / 30

#         self.screen = None
#         self.clock = None
#         self.car_pos = None
#         self.car_angle = None
#         self.speed = None
#         self.current_track_segment = 0
#         self.max_steps = 500
#         self.steps_taken = 0
#         self.laps_completed = 0
#         self.episode_reward = 0
        
#         self.stagnation_counter = 0
#         self.last_pos = None

#         self.action_space = spaces.Discrete(4)
        
#         # Observation space is now 10 + 2 (opponent's pos)
#         self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)

#         self.last_reward = 0
#         self.last_action_str = "None"
#         self.last_observation = np.zeros(self.observation_space.shape, dtype=np.float32)

#         # New opponent variables
#         self.opponent_pos = None
#         self.opponent_speed = 3.0
#         self.opponent_segment = 0
#         self.opponent_car_size = 20

#     def _generate_star_points(self, center_x, center_y, outer_radius, inner_radius):
#         points = []
#         for i in range(10):
#             if i % 2 == 0:
#                 radius = outer_radius
#             else:
#                 radius = inner_radius
#             angle = np.pi / 2 - (i / 10) * 2 * np.pi
#             x = center_x + radius * np.cos(angle)
#             y = center_y - radius * np.sin(angle)
#             points.append((x, y))
#         return points

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.car_pos = np.array(self.track_center_points[0])
#         self.car_angle = np.pi / 2
#         self.speed = 0.0
#         self.current_track_segment = 0
#         self.steps_taken = 0
#         self.laps_completed = 0
#         self.episode_reward = 0
        
#         self.stagnation_counter = 0
#         self.last_pos = self.car_pos.copy()
        
#         # Reset opponent position and state
#         self.opponent_pos = np.array(self.track_center_points[3])
#         self.opponent_segment = 3
        
#         observation = self._get_observation()
#         info = {}
        
#         self.last_reward = 0
#         self.last_action_str = "None"
#         self.last_observation = observation.copy()
        
#         return observation, info

#     def step(self, action):
#         self.steps_taken += 1
#         prev_pos = self.car_pos.copy()

#         # Update player car
#         if action == 0:
#             self.speed = min(self.max_speed, self.speed + 0.2)
#             self.last_action_str = "Accelerate"
#         elif action == 1:
#             self.speed = max(0, self.speed - 0.4)
#             self.last_action_str = "Brake"
#         elif action == 2:
#             self.car_angle += self.max_turn_angle * (1 - self.speed/self.max_speed) # Physics: harder to turn at high speed
#             self.last_action_str = "Turn Left"
#         elif action == 3:
#             self.car_angle -= self.max_turn_angle * (1 - self.speed/self.max_speed) # Physics: harder to turn at high speed
#             self.last_action_str = "Turn Right"
            
#         self.car_pos[0] += self.speed * np.cos(self.car_angle)
#         self.car_pos[1] += self.speed * np.sin(self.car_angle)
        
#         # Update opponent car
#         self._update_opponent()

#         reward = 0
#         done = False
#         truncated = False
        
#         dist_moved = np.linalg.norm(self.car_pos - self.last_pos)
#         if dist_moved < 1.0:
#             self.stagnation_counter += 1
#         else:
#             self.stagnation_counter = 0
#         self.last_pos = self.car_pos.copy()
        
#         # Check for collisions with walls and opponent
#         if not self._is_inside_track(self.car_pos) or self._check_opponent_collision():
#             reward = -200
#             done = True
#         else:
#             lap_reward, lap_completed = self._get_lap_completion_status()
#             progress_reward = self._get_progress_reward(prev_pos)
#             centerline_reward = self._get_centerline_reward()
#             speed_reward = self._get_speed_reward()

#             reward = progress_reward + centerline_reward + speed_reward
#             if lap_completed:
#                 self.laps_completed += 1
#                 reward += lap_reward
#                 done = True

#         stagnation_penalty = 0
#         if self.stagnation_counter > 50:
#             stagnation_penalty = -10.0
#         if self.stagnation_counter > 100:
#             stagnation_penalty = -20.0
#         reward += stagnation_penalty
            
#         reward -= 0.1

#         self.episode_reward += reward
#         self.last_reward = reward

#         observation = self._get_observation()
#         info = {"laps": self.laps_completed, "episode_reward": self.episode_reward}
        
#         self.last_observation = observation.copy()

#         return observation, reward, done, truncated, info

#     def _update_opponent(self):
#         target_point = np.array(self.track_center_points[(self.opponent_segment + 1) % len(self.track_center_points)])
#         direction = target_point - self.opponent_pos
#         direction = direction / np.linalg.norm(direction)
#         self.opponent_pos += direction * self.opponent_speed
        
#         dist_to_target = np.linalg.norm(self.opponent_pos - target_point)
#         if dist_to_target < 50:
#             self.opponent_segment = (self.opponent_segment + 1) % len(self.track_center_points)

#     def _check_opponent_collision(self):
#         dist = np.linalg.norm(self.car_pos - self.opponent_pos)
#         return dist < (self.car_size + self.opponent_car_size) / 2

#     def _is_inside_track(self, point):
#         def is_inside_polygon(point, polygon):
#             x, y = point
#             n = len(polygon)
#             inside = False
#             p1x, p1y = polygon[0]
#             for i in range(n + 1):
#                 p2x, p2y = polygon[i % n]
#                 if y > min(p1y, p2y) and y <= max(p1y, p2y):
#                     if x <= max(p1x, p2x):
#                         if p1y != p2y:
#                             xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#                         if p1x == p2x or x <= xinters:
#                             inside = not inside
#                 p1x, p1y = p2x, p2y
#             return inside

#         is_inside_outer = is_inside_polygon(point, self.outer_track_points)
#         is_inside_inner = is_inside_polygon(point, self.inner_track_points)
        
#         return is_inside_outer and not is_inside_inner

#     def _get_lap_completion_status(self):
#         distances = np.linalg.norm(np.array(self.track_center_points) - self.car_pos, axis=1)
#         if distances[0] < 50:
#             if self.laps_completed == 0:
#                 if self.steps_taken > 100:
#                     return 500, True
#             else:
#                 return 500, True
#         return 0, False

#     def _get_progress_reward(self, prev_pos):
#         reward = 0
#         target_point = np.array(self.track_center_points[(self.current_track_segment + 1) % len(self.track_center_points)])
        
#         dist_to_target = np.linalg.norm(self.car_pos - target_point)
#         dist_from_prev_to_target = np.linalg.norm(prev_pos - target_point)

#         reward += dist_from_prev_to_target - dist_to_target

#         if dist_to_target < 50:
#             self.current_track_segment = (self.current_track_segment + 1) % len(self.track_center_points)
#             reward += 50
#         return reward
    
#     def _get_centerline_reward(self):
#         distances = np.linalg.norm(np.array(self.track_center_points) - self.car_pos, axis=1)
#         dist_to_center = np.min(distances)
        
#         normalized_dist = dist_to_center / (self.height/2)
#         return 1.0 - normalized_dist

#     def _get_speed_reward(self):
#         return self.speed / self.max_speed * 0.5
        
#     def _get_observation(self):
#         sensor_angles = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
#         sensor_readings = [self._get_distance_to_wall(self.car_angle + angle) for angle in sensor_angles]
        
#         normalized_pos_x = self.car_pos[0] / self.width
#         normalized_pos_y = self.car_pos[1] / self.height
#         normalized_angle = (self.car_angle % (2 * np.pi) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
#         normalized_speed = self.speed / self.max_speed
        
#         normalized_sensors = [s / self.width for s in sensor_readings]
#         normalized_stagnation = self.stagnation_counter / self.max_steps
        
#         # Add opponent position to the observation
#         normalized_opponent_pos_x = self.opponent_pos[0] / self.width
#         normalized_opponent_pos_y = self.opponent_pos[1] / self.height
        
#         return np.array([normalized_pos_x, normalized_pos_y, normalized_angle, normalized_speed] + normalized_sensors + [normalized_stagnation, normalized_opponent_pos_x, normalized_opponent_pos_y], dtype=np.float32)

#     def _get_distance_to_wall(self, angle):
#         x, y = self.car_pos
#         dx, dy = np.cos(angle), np.sin(angle)
        
#         max_ray_length = max(self.width, self.height)
        
#         for i in range(1, int(max_ray_length)):
#             test_x = x + dx * i
#             test_y = y + dy * i
            
#             if not self._is_inside_track((test_x, test_y)):
#                 return float(i)
#         return float(max_ray_length)

#     def render(self):
#         if self.screen is None:
#             pygame.init()
#             pygame.display.set_caption("Speedy AI Racer")
#             self.screen = pygame.display.set_mode((self.width, self.height))
#             self.clock = pygame.time.Clock()
#             self.font = pygame.font.Font(None, 24)
            
#         self.screen.fill((0, 0, 0))

#         pygame.draw.lines(self.screen, (255, 255, 255), True, self.outer_track_points, 5)
#         pygame.draw.lines(self.screen, (255, 255, 255), True, self.inner_track_points, 5)

#         for i in range(len(self.track_center_points)):
#             color = (0, 150, 0)
#             if i == self.current_track_segment:
#                 color = (0, 255, 0)
            
#             p1 = self.track_center_points[i]
#             p2 = self.track_center_points[(i + 1) % len(self.track_center_points)]
#             pygame.draw.line(self.screen, color, p1, p2, 2)

#         start_line_points = [
#             self.inner_track_points[0], 
#             self.outer_track_points[0]
#         ]
#         pygame.draw.line(self.screen, (255, 0, 0), start_line_points[0], start_line_points[1], 5)

#         car_points = []
#         car_points.append((self.car_pos[0] + self.car_size * np.cos(self.car_angle),
#                            self.car_pos[1] + self.car_size * np.sin(self.car_angle)))
#         car_points.append((self.car_pos[0] + self.car_size * 0.7 * np.cos(self.car_angle + 2*np.pi/3),
#                            self.car_pos[1] + self.car_size * 0.7 * np.sin(self.car_angle + 2*np.pi/3)))
#         car_points.append((self.car_pos[0] + self.car_size * 0.7 * np.cos(self.car_angle - 2*np.pi/3),
#                            self.car_pos[1] + self.car_size * 0.7 * np.sin(self.car_angle - 2*np.pi/3)))
        
#         pygame.draw.polygon(self.screen, (0, 255, 0), car_points)
#         pygame.draw.circle(self.screen, (255, 0, 0), (int(self.car_pos[0]), int(self.car_pos[1])), 5)
        
#         # Draw opponent car
#         pygame.draw.circle(self.screen, (255, 255, 0), (int(self.opponent_pos[0]), int(self.opponent_pos[1])), self.opponent_car_size // 2)

#         text_color = (255, 255, 255)
#         self._display_text(f"Episode: {self.current_episode}", 10, 10, text_color)
#         self._display_text(f"Last Reward: {self.last_reward:.2f}", 10, 40, text_color)
#         self._display_text(f"Ep. Total Reward: {self.episode_reward:.2f}", 10, 70, text_color)
#         self._display_text(f"Action: {self.last_action_str}", self.width - 200, 10, text_color)
#         self._display_text(f"Speed: {self.speed:.2f}", self.width - 200, 40, text_color)
#         self._display_text(f"Laps: {self.laps_completed}", self.width - 200, 70, text_color)
        
#         self._display_text(f"Stagnation: {self.stagnation_counter}", self.width - 200, 100, text_color)
#         self._display_text(f"Pos: ({self.car_pos[0]:.0f}, {self.car_pos[1]:.0f})", self.width - 200, 130, text_color)
        
#         normalized_sensors_text = [f"{s:.2f}" for s in self.last_observation[4:9]]
#         obs_text = f"Sensors: F:{normalized_sensors_text[2]} FL:{normalized_sensors_text[1]} L:{normalized_sensors_text[0]} R:{normalized_sensors_text[3]} FR:{normalized_sensors_text[4]}"
#         self._display_text(obs_text, 10, self.height - 30, text_color)
        
#         pygame.display.flip()

#     def _display_text(self, text, x, y, color):
#         text_surface = self.font.render(text, True, color)
#         self.screen.blit(text_surface, (x, y))

#     def close(self):
#         if self.screen is not None:
#             pygame.display.quit()
#             pygame.quit()
#             self.screen = None
#             self.clock = None
#             self.font = None








# import pygame
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces

# class RaceCarEnv(gym.Env):
#     def __init__(self):
#         super(RaceCarEnv, self).__init__()
#         self.width = 800
#         self.height = 600
        
#         self.outer_track_points = self._generate_star_points(300, 300, 300, 150)
#         self.inner_track_points = self._generate_star_points(300, 300, 200, 100)
        
#         self.track_center_points = self._generate_star_points(300, 300, 250, 125)
        
#         self.car_size = 20
#         self.max_speed = 5
#         self.max_turn_angle = np.pi / 30

#         self.screen = None
#         self.clock = None
#         self.car_pos = None
#         self.car_angle = None
#         self.speed = None
#         self.current_track_segment = 0
#         self.max_steps = 500  
#         self.steps_taken = 0
#         self.laps_completed = 0
#         self.episode_reward = 0
        
#         self.stagnation_counter = 0
#         self.last_pos = None

#         self.action_space = spaces.Discrete(4)
        
#         self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

#         self.last_reward = 0
#         self.last_action_str = "None"
#         self.last_observation = np.zeros(self.observation_space.shape, dtype=np.float32)

#         self.car_image_orig = None
#         # --- NEW: Define initial offset for your UPWARD-pointing sprite ---
#         # Your sprite points UP (90 degrees from RIGHT). Pygame expects RIGHT (0 degrees).
#         # So we subtract 90 degrees from the angle we pass to Pygame.
#         self.initial_sprite_angle_offset = 90 
#         # --- END NEW ---

#     def _generate_star_points(self, center_x, center_y, outer_radius, inner_radius):
#         points = []
#         for i in range(10):
#             if i % 2 == 0:
#                 radius = outer_radius
#             else:
#                 radius = inner_radius
#             angle = np.pi / 2 - (i / 10) * 2 * np.pi
#             x = center_x + radius * np.cos(angle)
#             y = center_y - radius * np.sin(angle)
#             points.append((x, y))
#         return points

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.car_pos = np.array(self.track_center_points[0])
#         self.car_angle = np.pi / 2
#         self.speed = 0.0
#         self.current_track_segment = 0
#         self.steps_taken = 0
#         self.laps_completed = 0
#         self.episode_reward = 0
        
#         self.stagnation_counter = 0
#         self.last_pos = self.car_pos.copy()
        
#         observation = self._get_observation()
#         info = {}
        
#         self.last_reward = 0
#         self.last_action_str = "None"
#         self.last_observation = observation.copy()
        
#         return observation, info

#     def step(self, action):
#         self.steps_taken += 1
#         prev_pos = self.car_pos.copy()

#         if action == 0:
#             self.speed = min(self.max_speed, self.speed + 0.2)
#             self.last_action_str = "Accelerate"
#         elif action == 1:
#             self.speed = max(0, self.speed - 0.4)
#             self.last_action_str = "Brake"
#         elif action == 2:
#             self.car_angle += self.max_turn_angle
#             self.last_action_str = "Turn Left"
#         elif action == 3:
#             self.car_angle -= self.max_turn_angle
#             self.last_action_str = "Turn Right"
            
#         self.car_pos[0] += self.speed * np.cos(self.car_angle)
#         self.car_pos[1] += self.speed * np.sin(self.car_angle)

#         reward = 0
#         done = False
#         truncated = False
        
#         dist_moved = np.linalg.norm(self.car_pos - self.last_pos)
#         if dist_moved < 1.0:
#             self.stagnation_counter += 1
#         else:
#             self.stagnation_counter = 0
#         self.last_pos = self.car_pos.copy()

#         if not self._is_inside_track(self.car_pos):
#             reward = -200
#             done = True
#         else:
#             # Checkpoint-based lap completion
#             lap_reward, lap_completed = self._get_lap_completion_status()
#             progress_reward = self._get_progress_reward(prev_pos)
#             centerline_reward = self._get_centerline_reward()
#             speed_reward = self._get_speed_reward()

#             reward = progress_reward + centerline_reward + speed_reward
#             if lap_completed:
#                 self.laps_completed += 1
#                 reward += lap_reward
#                 done = True

#         stagnation_penalty = 0
#         if self.stagnation_counter > 50:
#             stagnation_penalty = -10.0
#         if self.stagnation_counter > 100:
#             stagnation_penalty = -20.0
#         reward += stagnation_penalty
            
#         reward -= 0.1

#         self.episode_reward += reward
#         self.last_reward = reward

#         observation = self._get_observation()
#         info = {"laps": self.laps_completed, "episode_reward": self.episode_reward}
        
#         self.last_observation = observation.copy()

#         return observation, reward, done, truncated, info

#     def _is_inside_track(self, point):
#         def is_inside_polygon(point, polygon):
#             x, y = point
#             n = len(polygon)
#             inside = False
#             p1x, p1y = polygon[0]
#             for i in range(n + 1):
#                 p2x, p2y = polygon[i % n]
#                 if y > min(p1y, p2y) and y <= max(p1y, p2y):
#                     if x <= max(p1x, p2x):
#                         if p1y != p2y:
#                             xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#                         if p1x == p2x or x <= xinters:
#                             inside = not inside
#                 p1x, p1y = p2x, p2y
#             return inside

#         is_inside_outer = is_inside_polygon(point, self.outer_track_points)
#         is_inside_inner = is_inside_polygon(point, self.inner_track_points)
        
#         return is_inside_outer and not is_inside_inner

#     def _get_lap_completion_status(self):
#         # We will check if the car has visited all segments of the track
#         if self.current_track_segment == len(self.track_center_points) - 1:
#             distances = np.linalg.norm(np.array(self.track_center_points) - self.car_pos, axis=1)
#             # Check if the car is at the starting point of the track
#             if distances[0] < 50:
#                  return 500, True
#         return 0, False

#     def _get_progress_reward(self, prev_pos):
#         reward = 0
#         target_point = np.array(self.track_center_points[(self.current_track_segment + 1) % len(self.track_center_points)])
        
#         dist_to_target = np.linalg.norm(self.car_pos - target_point)
#         dist_from_prev_to_target = np.linalg.norm(prev_pos - target_point)

#         reward += dist_from_prev_to_target - dist_to_target

#         if dist_to_target < 50:
#             self.current_track_segment = (self.current_track_segment + 1) % len(self.track_center_points)
#             reward += 50
#         return reward
    
#     def _get_centerline_reward(self):
#         distances = np.linalg.norm(np.array(self.track_center_points) - self.car_pos, axis=1)
#         dist_to_center = np.min(distances)
        
#         normalized_dist = dist_to_center / (self.height/2)
#         return 1.0 - normalized_dist

#     def _get_speed_reward(self):
#         return self.speed / self.max_speed * 0.5
        
#     def _get_observation(self):
#         sensor_angles = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
#         sensor_readings = [self._get_distance_to_wall(self.car_angle + angle) for angle in sensor_angles]
        
#         normalized_pos_x = self.car_pos[0] / self.width
#         normalized_pos_y = self.car_pos[1] / self.height
#         normalized_angle = (self.car_angle % (2 * np.pi) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
#         normalized_speed = self.speed / self.max_speed
        
#         normalized_sensors = [s / self.width for s in sensor_readings]
#         normalized_stagnation = self.stagnation_counter / self.max_steps
        
#         return np.array([normalized_pos_x, normalized_pos_y, normalized_angle, normalized_speed] + normalized_sensors + [normalized_stagnation], dtype=np.float32)

#     def _get_distance_to_wall(self, angle):
#         x, y = self.car_pos
#         dx, dy = np.cos(angle), np.sin(angle)
        
#         max_ray_length = max(self.width, self.height)
        
#         for i in range(1, int(max_ray_length)):
#             test_x = x + dx * i
#             test_y = y + dy * i
            
#             if not self._is_inside_track((test_x, test_y)):
#                 return float(i)
#         return float(max_ray_length)

#     def render(self):
#         if self.screen is None:
#             pygame.init()
#             pygame.display.set_caption("Speedy AI Racer")
#             self.screen = pygame.display.set_mode((self.width, self.height))
#             self.clock = pygame.time.Clock()
#             self.font = pygame.font.Font(None, 24)
            
#             try:
#                 self.car_image_orig = pygame.image.load("car.png")
#                 self.car_image_orig = self.car_image_orig.convert_alpha()
                
#                 # --- This is the new, better way ---
#                 # Your 'car.png' sprite points UP.
#                 # So, the first value is BREADTH (width) and the second is LENGTH (height).
                
#                 CAR_BREADTH = 65  # <-- Adjust this number for car BREADTH
#                 CAR_LENGTH = 65   # <-- Adjust this number for car LENGTH
                
#                 # We now pass those independent values to scale
#                 self.car_image_orig = pygame.transform.scale(self.car_image_orig, (CAR_BREADTH, CAR_LENGTH))
#                 # --- End of fix ---

#             except pygame.error as e:
#                 print(f"Error loading 'car.png' in render: {e}")
#                 print("Please make sure 'car.png' is in the same directory.")
#                 self.car_image_orig = None

#         self.screen.fill((0, 0, 0))

#         pygame.draw.lines(self.screen, (255, 255, 255), True, self.outer_track_points, 5)
#         pygame.draw.lines(self.screen, (255, 255, 255), True, self.inner_track_points, 5)

#         for i in range(len(self.track_center_points)):
#             color = (0, 150, 0)
#             if i == self.current_track_segment:
#                 color = (0, 255, 0)
            
#             p1 = self.track_center_points[i]
#             p2 = self.track_center_points[(i + 1) % len(self.track_center_points)]
#             pygame.draw.line(self.screen, color, p1, p2, 2)

#         start_line_points = [
#             self.inner_track_points[0], 
#             self.outer_track_points[0]
#         ]
#         pygame.draw.line(self.screen, (255, 0, 0), start_line_points[0], start_line_points[1], 5)

#         if self.car_image_orig:
#             # --- MODIFIED: Adjust for initial sprite orientation ---
#             # Pygame rotates counter-clockwise.
#             # self.car_angle is in radians, so convert to degrees.
#             # We subtract the car_angle from the offset because the car's angle increases counter-clockwise,
#             # but pygame.transform.rotate rotates the image *itself* counter-clockwise.
#             # An easier way to think about it: if the car points 0 deg (right), we want the sprite rotated 90 deg CCW (up).
#             # If the car points 90 deg (up), we want the sprite rotated 0 deg (right).
#             # So, the rotation applied is initial_offset - current_car_angle_in_degrees
#             rotation_angle_for_pygame = self.initial_sprite_angle_offset - np.degrees(self.car_angle)

#             rotated_car = pygame.transform.rotate(self.car_image_orig, rotation_angle_for_pygame)
            
#             car_rect = rotated_car.get_rect(center = self.car_pos)
#             self.screen.blit(rotated_car, car_rect)
#         else:
#             pygame.draw.circle(self.screen, (0, 150, 255), (int(self.car_pos[0]), int(self.car_pos[1])), self.car_size // 2)

        
#         text_color = (255, 255, 255)
        
#         episode_text = "N/A"
#         if hasattr(self, 'current_episode'):
#             episode_text = str(self.current_episode)

#         self._display_text(f"Episode: {episode_text}", 10, 10, text_color)
#         self._display_text(f"Last Reward: {self.last_reward:.2f}", 10, 40, text_color)
#         self._display_text(f"Ep. Total Reward: {self.episode_reward:.2f}", 10, 70, text_color)
#         self._display_text(f"Action: {self.last_action_str}", self.width - 200, 10, text_color)
#         self._display_text(f"Speed: {self.speed:.2f}", self.width - 200, 40, text_color)
#         self._display_text(f"Laps: {self.laps_completed}", self.width - 200, 70, text_color)
        
#         self._display_text(f"Stagnation: {self.stagnation_counter}", self.width - 200, 100, text_color)
#         self._display_text(f"Steps: {self.steps_taken}/{self.max_steps}", self.width - 200, 130, text_color) # <-- Added this back
        
#         normalized_sensors_text = [f"{s:.2f}" for s in self.last_observation[4:9]]
#         obs_text = f"Sensors: F:{normalized_sensors_text[2]} FL:{normalized_sensors_text[1]} L:{normalized_sensors_text[0]} R:{normalized_sensors_text[3]} FR:{normalized_sensors_text[4]}"
#         self._display_text(obs_text, 10, self.height - 30, text_color)
        
#         pygame.display.flip()

# # Clear the loaded image

#     def _display_text(self, text, x, y, color):
#         text_surface = self.font.render(text, True, color)
#         self.screen.blit(text_surface, (x, y))

#     def close(self):
#         if self.screen is not None:
#             pygame.display.quit()
#             pygame.quit()
#             self.screen = None
#             self.clock = None
#             self.font = None
#             self.car_image_orig = None 

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RaceCarEnv(gym.Env):
    def __init__(self):
        super(RaceCarEnv, self).__init__()
        self.width = 800
        self.height = 600
        
        self.outer_track_points = self._generate_star_points(300, 300, 300, 150)
        self.inner_track_points = self._generate_star_points(300, 300, 200, 100)
        
        self.track_center_points = self._generate_star_points(300, 300, 250, 125)
        
        # --- FIX: Force Anti-Clockwise Direction ---
        # The points are generated clockwise. Reversing the list
        # makes the "next" point the anti-clockwise one.
        self.track_center_points.reverse()
        # --- END FIX ---
        
        self.car_size = 20
        self.max_speed = 5
        self.max_turn_angle = np.pi / 30

        self.screen = None
        self.clock = None
        self.car_pos = None
        self.car_angle = None
        self.speed = None
        self.current_track_segment = 0
        
        # --- FIX: Add a hard time limit ---
        # This prevents the agent from looping forever in one episode.
        self.max_steps = 2000 # Give it 2000 steps to complete a lap
        # --- END FIX ---
          
        self.steps_taken = 0
        self.laps_completed = 0
        self.episode_reward = 0
        
        self.stagnation_counter = 0
        self.last_pos = None

        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

        self.last_reward = 0
        self.last_action_str = "None"
        self.last_observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        self.car_image_orig = None
        # Sprite points UP (90 deg), but 0 deg is RIGHT.
        self.initial_sprite_angle_offset = 90 

    def _generate_star_points(self, center_x, center_y, outer_radius, inner_radius):
        points = []
        for i in range(10):
            if i % 2 == 0:
                radius = outer_radius
            else:
                radius = inner_radius
            angle = np.pi / 2 - (i / 10) * 2 * np.pi
            x = center_x + radius * np.cos(angle)
            y = center_y - radius * np.sin(angle)
            points.append((x, y))
        return points

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.car_pos = np.array(self.track_center_points[0])
        
        # --- FIX: Update start angle ---
        # The new start point is (Up-Left). The next point is (Left-Up).
        # We need to face left to start. (pi radians = 180 degrees)
        self.car_angle = np.pi
        # --- END FIX ---
        
        self.speed = 0.0
        self.current_track_segment = 0
        self.steps_taken = 0
        self.laps_completed = 0
        self.episode_reward = 0
        
        self.stagnation_counter = 0
        self.last_pos = self.car_pos.copy()
        
        observation = self._get_observation()
        info = {}
        
        self.last_reward = 0
        self.last_action_str = "None"
        self.last_observation = observation.copy()
        
        return observation, info

    def step(self, action):
        self.steps_taken += 1
        prev_pos = self.car_pos.copy()

        if action == 0:
            self.speed = min(self.max_speed, self.speed + 0.2)
            self.last_action_str = "Accelerate"
        elif action == 1:
            self.speed = max(0, self.speed - 0.4)
            self.last_action_str = "Brake"
        elif action == 2:
            self.car_angle += self.max_turn_angle
            self.last_action_str = "Turn Left"
        elif action == 3:
            self.car_angle -= self.max_turn_angle
            self.last_action_str = "Turn Right"
            
        self.car_pos[0] += self.speed * np.cos(self.car_angle)
        self.car_pos[1] += self.speed * np.sin(self.car_angle)

        reward = 0
        done = False
        truncated = False
        
        dist_moved = np.linalg.norm(self.car_pos - self.last_pos)
        if dist_moved < 1.0:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_pos = self.car_pos.copy()

        if not self._is_inside_track(self.car_pos):
            reward = -200
            done = True
        else:
            lap_reward, lap_completed = self._get_lap_completion_status()
            progress_reward = self._get_progress_reward(prev_pos) # <-- Uses fixed function
            centerline_reward = self._get_centerline_reward()
            speed_reward = self._get_speed_reward()

            reward = progress_reward + centerline_reward + speed_reward
            if lap_completed:
                self.laps_completed += 1
                reward += lap_reward
                done = True

        stagnation_penalty = 0
        if self.stagnation_counter > 50:
            stagnation_penalty = -10.0
        if self.stagnation_counter > 100:
            stagnation_penalty = -20.0
        reward += stagnation_penalty
            
        # --- FIX: Increased time penalty ---
        # Make "living" more expensive, so looping is unprofitable.
        reward -= 0.5
        # --- END FIX ---

        self.episode_reward += reward
        self.last_reward = reward

        observation = self._get_observation()
        info = {"laps": self.laps_completed, "episode_reward": self.episode_reward}
        
        self.last_observation = observation.copy()

        # --- FIX: Add truncation logic ---
        # If the agent runs out of time, end the episode
        if self.steps_taken >= self.max_steps:
            truncated = True  # Signal that the episode ended due to time
            done = True       # End the episode
        # --- END FIX ---

        return observation, reward, done, truncated, info

    def _is_inside_track(self, point):
        def is_inside_polygon(point, polygon):
            x, y = point
            n = len(polygon)
            inside = False
            p1x, p1y = polygon[0]
            for i in range(n + 1):
                p2x, p2y = polygon[i % n]
                if y > min(p1y, p2y) and y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
                p1x, p1y = p2x, p2y
            return inside

        is_inside_outer = is_inside_polygon(point, self.outer_track_points)
        is_inside_inner = is_inside_polygon(point, self.inner_track_points)
        
        return is_inside_outer and not is_inside_inner

    def _get_lap_completion_status(self):
        # Check if the car is at the final segment
        if self.current_track_segment == len(self.track_center_points) - 1:
            # Check distance to the *first* checkpoint (the start/finish line)
            dist_to_start = np.linalg.norm(self.car_pos - self.track_center_points[0])
            if dist_to_start < 50:
                 return 500, True # Give reward and end episode
        return 0, False

    def _get_progress_reward(self, prev_pos):
        # --- FIX: Discrete Checkpoint Reward ---
        reward = 0
        # Get the next checkpoint the car needs to hit
        target_point = np.array(self.track_center_points[(self.current_track_segment + 1) % len(self.track_center_points)])
        
        dist_to_target = np.linalg.norm(self.car_pos - target_point)
        
        # REMOVED: The continuous, "hackable" reward for just getting closer
        # dist_from_prev_to_target = np.linalg.norm(prev_pos - target_point)
        # reward += dist_from_prev_to_target - dist_to_target

        # The *only* progress reward is for hitting the checkpoint
        if dist_to_target < 50:
            self.current_track_segment = (self.current_track_segment + 1) % len(self.track_center_points)
            reward += 100 # Large, discrete reward for success
        return reward
        # --- END FIX ---
    
    def _get_centerline_reward(self):
        distances = np.linalg.norm(np.array(self.track_center_points) - self.car_pos, axis=1)
        dist_to_center = np.min(distances)
        
        # Give a small reward for being close to the center
        normalized_dist = dist_to_center / (self.height/2) # Normalize by a large distance
        return 0.1 * (1.0 - normalized_dist) # Make this reward smaller

    def _get_speed_reward(self):
        # Only reward speed if the agent is actually moving
        if self.speed > 1.0:
            return self.speed / self.max_speed * 0.5 # Small reward for speed
        return 0
        
    def _get_observation(self):
        sensor_angles = [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]
        sensor_readings = [self._get_distance_to_wall(self.car_angle + angle) for angle in sensor_angles]
        
        normalized_pos_x = self.car_pos[0] / self.width
        normalized_pos_y = self.car_pos[1] / self.height
        normalized_angle = (self.car_angle % (2 * np.pi) + 2 * np.pi) % (2 * np.pi) / (2 * np.pi)
        normalized_speed = self.speed / self.max_speed
        
        normalized_sensors = [s / self.width for s in sensor_readings]
        
        # Normalize stagnation counter by new max_steps
        normalized_stagnation = self.stagnation_counter / self.max_steps
        
        return np.array([normalized_pos_x, normalized_pos_y, normalized_angle, normalized_speed] + normalized_sensors + [normalized_stagnation], dtype=np.float32)

    def _get_distance_to_wall(self, angle):
        x, y = self.car_pos
        dx, dy = np.cos(angle), np.sin(angle)
        
        max_ray_length = max(self.width, self.height)
        
        for i in range(1, int(max_ray_length)):
            test_x = x + dx * i
            test_y = y + dy * i
            
            if not self._is_inside_track((test_x, test_y)):
                return float(i)
        return float(max_ray_length)

    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Speedy AI Racer")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            
            try:
                self.car_image_orig = pygame.image.load("car.png")
                self.car_image_orig = self.car_image_orig.convert_alpha()
                
                # --- Set car size independently ---
                CAR_BREADTH = 35  # Adjust this number for car BREADTH
                CAR_LENGTH = 45   # Adjust this number for car LENGTH
                
                self.car_image_orig = pygame.transform.scale(self.car_image_orig, (CAR_BREADTH, CAR_LENGTH))
                # --- End ---

            except pygame.error as e:
                print(f"Error loading 'car.png' in render: {e}")
                print("Please make sure 'car.png' is in the same directory.")
                self.car_image_orig = None

        self.screen.fill((0, 0, 0))

        pygame.draw.lines(self.screen, (255, 255, 255), True, self.outer_track_points, 5)
        pygame.draw.lines(self.screen, (255, 255, 255), True, self.inner_track_points, 5)

        for i in range(len(self.track_center_points)):
            color = (0, 150, 0)
            if i == self.current_track_segment:
                color = (0, 255, 0)
            
            p1 = self.track_center_points[i]
            p2 = self.track_center_points[(i + 1) % len(self.track_center_points)]
            pygame.draw.line(self.screen, color, p1, p2, 2)

        start_line_points = [
            self.inner_track_points[0], 
            self.outer_track_points[0]
        ]
        pygame.draw.line(self.screen, (255, 0, 0), start_line_points[0], start_line_points[1], 5)

        if self.car_image_orig:
            # Adjust angle for UP-pointing sprite
            rotation_angle_for_pygame = self.initial_sprite_angle_offset - np.degrees(self.car_angle)
            rotated_car = pygame.transform.rotate(self.car_image_orig, rotation_angle_for_pygame)
            
            car_rect = rotated_car.get_rect(center = self.car_pos)
            self.screen.blit(rotated_car, car_rect)
        else:
            pygame.draw.circle(self.screen, (0, 150, 255), (int(self.car_pos[0]), int(self.car_pos[1])), self.car_size // 2)

        
        text_color = (255, 255, 255)
        
        episode_text = "N/A"
        if hasattr(self, 'current_episode'):
            episode_text = str(self.current_episode)

        self._display_text(f"Episode: {episode_text}", 10, 10, text_color)
        self._display_text(f"Last Reward: {self.last_reward:.2f}", 10, 40, text_color)
        self._display_text(f"Ep. Total Reward: {self.episode_reward:.2f}", 10, 70, text_color)
        self._display_text(f"Action: {self.last_action_str}", self.width - 200, 10, text_color)
        self._display_text(f"Speed: {self.speed:.2f}", self.width - 200, 40, text_color)
        self._display_text(f"Laps: {self.laps_completed}", self.width - 200, 70, text_color)
        
        self._display_text(f"Stagnation: {self.stagnation_counter}", self.width - 200, 100, text_color)
        self._display_text(f"Steps: {self.steps_taken}/{self.max_steps}", self.width - 200, 130, text_color)
        
        normalized_sensors_text = [f"{s:.2f}" for s in self.last_observation[4:9]]
        obs_text = f"Sensors: F:{normalized_sensors_text[2]} FL:{normalized_sensors_text[1]} L:{normalized_sensors_text[0]} R:{normalized_sensors_text[3]} FR:{normalized_sensors_text[4]}"
        self._display_text(obs_text, 10, self.height - 30, text_color)
        
        pygame.display.flip()

    def _display_text(self, text, x, y, color):
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
            self.car_image_orig = None