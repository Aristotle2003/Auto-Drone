import numpy as np
import random


class DroneDeliveryEnv:
    def __init__(self):
        self.area_size = (10, 10)  # Size of the area (10x10 grid)
        self.target_zones = {'red': (9, 9), 'yellow': (8, 8)}  # Target zones
        self.position = (0, 0)  # Drone's starting position
        self.goal = 'red'  # Change to 'yellow' to set different goals
        self.obstacles = [(5, 5), (6, 5), (5, 6)]  # Example obstacles

    def reset(self):
        self.position = (0, 0)  # Reset position to start
        return self.position

    def step(self, action):
        # Define movement actions
        if action == 0:  # Up
            self.position = (max(self.position[0] - 1, 0), self.position[1])
        elif action == 1:  # Down
            self.position = (min(self.position[0] + 1, self.area_size[0] - 1), self.position[1])
        elif action == 2:  # Left
            self.position = (self.position[0], max(self.position[1] - 1, 0))
        elif action == 3:  # Right
            self.position = (self.position[0], min(self.position[1] + 1, self.area_size[1] - 1))

        # Check if the drone hits an obstacle
        if self.position in self.obstacles:
            reward = -100
            done = True
        # Check if the drone reaches the target zone
        elif self.position == self.target_zones[self.goal]:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        return self.position, reward, done


env = DroneDeliveryEnv()


def train_drone(env, episodes = 500, learning_rate = 0.8, discount_factor = 0.95, explore_rate = 1.0,
                max_explore_rate = 1.0, min_explore_rate = 0.01, decay_rate = 0.01):
    q_table = np.zeros((env.area_size[0], env.area_size[1], 4))
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if random.uniform(0, 1) < explore_rate:
                action = random.randint(0, 3)  # Explore action space
            else:
                action = np.argmax(q_table[state[0], state[1]])  # Exploit learned values

            new_state, reward, done = env.step(action)
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[new_state[0], new_state[1]])

            # Update Q-value
            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[state[0], state[1], action] = new_value

            state = new_state

        # Update exploration rate
        explore_rate = min_explore_rate + (max_explore_rate - min_explore_rate) * np.exp(-decay_rate * episode)

    return q_table


trained_q_table = train_drone(env)


def test_drone(env, q_table):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state[0], state[1]])  # Choose best action
        state, reward, done = env.step(action)
        print("Move to:", state, "Reward:", reward)


test_drone(env, trained_q_table)
