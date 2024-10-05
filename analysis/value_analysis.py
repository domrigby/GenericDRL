import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.decomposition import PCA
import torch

from RL.networks import ValueNetwork # type: ignore
from RL.agent import RLAgent    

class ValueFuncAnalysis:
        
    def __init__(self):

        self.env = gym.make("BipedalWalker-v3", render_mode="human", hardcore=True)

        self.value_net = ValueNetwork(state_dim=24, name='value')
        self.value_net.to('cuda')
        self.value_net.load_network(folder="working_networks/bipedal_walker")

        self.observation_range = self.env.observation_space.high - self.env.observation_space.low
        self.obs_space_bias = self.env.observation_space.low

        self.len_obs_space = len(self.env.observation_space.high)\
        
        self.gradient_sensitivity()

    def plot_weights(self):
        weights = self.value_net.fc1.weight.data.cpu().numpy()

        # Compute the sum of weights for each column
        column_sums = np.sum(weights, axis=0)

        # Create a figure with a gridspec layout
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

        # Plotting the heatmap
        ax0 = plt.subplot(gs[0])
        cax = ax0.imshow(weights, cmap='viridis', aspect='auto')

        # Adding colorbar for reference
        plt.colorbar(cax, ax=ax0, label='Weight Value')

        # Adding labels
        ax0.set_title('Heatmap of fc1 Weights')
        ax0.set_xlabel('Neurons in the First Layer')
        ax0.set_ylabel('Input Features')

        # Prepare the table data: Neuron index and corresponding column sum
        table_data = [[f'N{i+1}', f'{sum_val:.2f}'] for i, sum_val in enumerate(column_sums)]

        # Create a table to the right of the heatmap to display column sums
        ax1 = plt.subplot(gs[1])
        ax1.axis('off')  # Hide the axis

        # Plot the table with larger text
        table = ax1.table(cellText=table_data, colLabels=['Neuron', 'Sum'], cellLoc='center', loc='center', colWidths=[0.2, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 1.5)  # Adjust the scale of the table for better readability

        # Show the plot
        plt.tight_layout()
        plt.show()


    def generate_data(self, num=256, requires_grad=False):

        rand_state = self.observation_range[:, np.newaxis] * np.random.random((self.len_obs_space, num)) + self.obs_space_bias[:, np.newaxis]

        state_tensor = torch.tensor(rand_state.T, device=self.value_net.device, dtype=torch.float32, requires_grad=requires_grad)

        values = self.value_net(state_tensor)

        return state_tensor, values
    
    def real_data(self, num_episode=10):

        agent = RLAgent(self.env, max_steps=5000, load_networks_dir='working_networks/bipedal_walker', learning=False) 

        for i in range(num_episode):
            agent.explore_one_episode()

        states = agent.replay.state_memory[:agent.replay.write_num]
        state_tensor = torch.tensor(states, device=self.value_net.device, dtype=torch.float32)
        values = self.value_net(state_tensor)
        return state_tensor, values
    
    def plot_principle_components(self):

        states, values = self.real_data(5)

        states = states.cpu().detach().numpy()
        values = values.cpu().detach().numpy()

        # Perform PCA on the states matrix
        pca = PCA(n_components=10)
        pca_result = pca.fit_transform(states)

        # Create the figure with the first two dimensions as the first two principal components
        # and the third dimension as the result values.
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot using the first two principal components as x and y, and the results as z
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], values, c=values, cmap='viridis', alpha=0.8)

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Result values')
        ax.set_title('PCA - First 2 Components and Results as 3rd Dimension')

        # Add color bar
        fig.colorbar(scatter, ax=ax, label='Result values')

        plt.tight_layout()
        breakpoint()
        plt.show()

    def gradient_sensitivity(self):
        states, values = self.generate_data(4096, requires_grad=True)

        # Backward pass: Compute the gradient of the output with respect to the input

        value_total = values.sum()
        value_total.backward()

        # Access the gradients
        input_gradients = states.grad

        mean_gradients = torch.mean(torch.abs(input_gradients), dim=0).cpu().numpy()

        plt.bar(np.arange(len(mean_gradients)), mean_gradients)
        plt.xlabel('Input Features')
        plt.ylabel('Mean Gradient Magnitude')
        plt.title('Mean Gradient Sensitivity Across Batch')
        plt.show()