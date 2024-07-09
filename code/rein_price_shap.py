import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
import shap

class SalesPredictionEnv(gym.Env):
    def __init__(self, initial_price, initial_discount, true_sales_function):
        super(SalesPredictionEnv, self).__init__()
        self.initial_price = initial_price
        self.initial_discount = initial_discount
        self.true_sales_function = true_sales_function

        # Action space: change in price and discount (continuous)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        # Observation space: current price, discount, and predicted sales
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

        self.price = self.initial_price
        self.discount = self.initial_discount
        self.sales = self.true_sales_function(self.price, self.discount)
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.price = self.initial_price
        self.discount = self.initial_discount
        self.sales = self.true_sales_function(self.price, self.discount)
        return np.array([self.price, self.discount, self.sales], dtype=np.float32), {}

    def step(self, action):
        self.price += action[0]
        self.discount += action[1]
        new_sales = self.true_sales_function(self.price, self.discount)

        # Reward is the negative absolute error of the prediction
        reward = -abs(self.sales - new_sales)

        self.sales = new_sales
        self.done = False

        return np.array([self.price, self.discount, self.sales], dtype=np.float32), reward, False, False, {}

    def render(self, mode='human'):
        print(f'Price: {self.price}, Discount: {self.discount}, Sales: {self.sales}')

# Define the true sales function
def true_sales_function(price, discount):
    return -0.5 * price ** 2 + price + 11 + 2 * discount

# Initialize environment
env = SalesPredictionEnv(initial_price=5.0, initial_discount=1.0, true_sales_function=true_sales_function)

# Verify the environment
check_env(env)

# Set up the RL agent
model = DDPG('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Reset the environment
# Reset the environment
obs, _ = env.reset()

# Collect states and actions for SHAP analysis
states = []
actions = []
for _ in range(10):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, _ = env.step(action)
    env.render()
    states.append(obs)
    actions.append(action)

# Convert lists to numpy arrays for SHAP
states = np.array(states)

# SHAP analysis
# Here we define a wrapper function to ensure the correct output format for SHAP
def predict_wrapper(observations):
    predictions = []
    for obs in observations:
        action, _states = model.predict(obs)
        predictions.append(action.flatten())
    return np.array(predictions)

# Create a DataFrame to store predictions
predictions = {
    'ID': list(range(len(states))),
    'price': states[:, 0],
    'discount': states[:, 1],
    'sales': states[:, 2],
    'predicted_action_0': [None] * len(states),
    'predicted_action_1': [None] * len(states)
}

# Collect predictions for each state
for idx, state in enumerate(states):
    action, _states = model.predict(state)
    predictions['predicted_action_0'][idx] = action[0]
    predictions['predicted_action_1'][idx] = action[1]

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions)

# Save to Excel file
predictions_df.to_excel("reinforcement_learning_predictions.xlsx", index=False)

print(predictions_df.head(10))

explainer = shap.Explainer(predict_wrapper, states)
shap_values = explainer(states)

# Since beeswarm plot requires a single output dimension, we will reshape appropriately
shap_values_price = shap_values[..., 0]  # Extracting SHAP values for price prediction

# Visualize the SHAP values for the collected states
shap.plots.beeswarm(shap_values_price)

# Local bar plot for a specific state
sample_index = 0  # replace with the actual index if needed
shap.plots.bar(shap_values_price[sample_index])

# Extract the top influential features of each state
data = {
    'ID': list(range(len(states))),
    'price': states[:, 0],
    'discount': states[:, 1],
    'sales': states[:, 2],
    'top_feature1': [None] * len(states), 'top_feature2': [None] * len(states),
    'importance1': [None] * len(states), 'importance2': [None] * len(states)
}

# Ensure iteration within bounds of shap_values
features = ['price', 'discount', 'sales']
for i in range(len(states)):
    sorted_indices = np.argsort(-np.abs(shap_values.values[i][:, 0]))
    data['top_feature1'][i] = features[sorted_indices[0]]
    data['importance1'][i] = shap_values.values[i][sorted_indices[0], 0]
    
    if len(sorted_indices) > 1:
        data['top_feature2'][i] = features[sorted_indices[1]]
        data['importance2'][i] = shap_values.values[i][sorted_indices[1], 0]

# Convert the collected data into a DataFrame for easy analysis
reason_df = pd.DataFrame(data)

print(reason_df.head(10))