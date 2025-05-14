# ===== agent.py =====
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from env_mgu import MGUEnvironment
import matplotlib.pyplot as plt

# ─── Reproducibility Seeds ───────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

class QLearningAgent:
    def __init__(self, action_size, alpha=0.1, gamma=0.9,
                 epsilon=0.8, epsilon_decay=0.99, epsilon_min=0.1):
        self.q_table = np.zeros((1, action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_size = action_size

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[0]))

    def update_q(self, action, reward):
        best_future = np.max(self.q_table[0])
        td_error = reward + self.gamma * best_future - self.q_table[0, action]
        self.q_table[0, action] += self.alpha * td_error
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# ─── Main Loop ────────────────────────────────────────────────────────────────
df = pd.read_excel("Bitcoin-price-USDnew.xlsx", sheet_name="Bitcoin-price-USD")
env = MGUEnvironment(df)

# Compute and save baseline MGU (action 0) for comparison
base_reward, base_mape, base_mae, base_rmse, base_model, base_ytr, base_ypr = env.step(0, return_model=True)
base_model.save("mgu_base_model_for_comparison.keras")
np.save("base_actual_testset.npy", np.array(base_ytr))
np.save("base_predicted_testset.npy", np.array(base_ypr))
print(f"Base MGU | Reward: {base_reward:.2f} | MAPE: {base_mape:.2f}% | MAE: {base_mae:.2f} | RMSE: {base_rmse:.2f}")

# Initialize Q-learning agent and tracking
best_reward = base_reward
agent = QLearningAgent(action_size=len(env.actions))
n_episodes = 8

# Start results with the base MGU as Episode 0
results = [{
    "Episode": 0,
    "Action": 0,
    "MAPE": base_mape,
    "MAE": base_mae,
    "RMSE": base_rmse,
    "Reward": base_reward,
    "Epsilon": None
}]

for episode in range(n_episodes):
    action = agent.choose_action()
    reward, mape, mae, rmse, model, ytr, ypr = env.step(action, return_model=True)
    agent.update_q(action, reward)

    # Save a new “best” model whenever reward improves
    if reward > best_reward:
        best_reward = reward
        best_actuals = ytr
        best_predictions = ypr
        model.save("best_mgu_model.keras")

    # Log this episode’s metrics
    results.append({
        "Episode": episode + 1,
        "Action": action,
        "MAPE": mape,
        "MAE": mae,
        "RMSE": rmse,
        "Reward": reward,
        "Epsilon": agent.epsilon
    })

    print(
        f"Episode {episode+1} | Action: {action} | Reward: {reward:.2f} | "
        f"MAPE: {mape:.2f}% | MAE: {mae:.2f} | RMSE: {rmse:.2f} | "
        f"Epsilon: {agent.epsilon:.2f}"
    )

# Save best prediction & actuals after all episodes
np.save("best_actual_testset.npy", np.array(best_actuals))
np.save("best_predicted_testset.npy", np.array(best_predictions))

# Save the log, including the base as Episode 0, with 2-decimal formatting
df_log = pd.DataFrame(results)
df_log.to_csv("log_rl_results.csv", index=False, float_format="%.2f")

# Plot reward evolution
plt.plot(df_log["Episode"], df_log["Reward"])
plt.title("Reward over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Final Q-Table:")
print(agent.q_table)
