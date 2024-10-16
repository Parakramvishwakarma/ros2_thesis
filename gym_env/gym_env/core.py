import numpy as np
import matplotlib.pyplot as plt
import torch



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def plot_learning_curve(x, filename, save_plot=True):
    avg_x = [np.mean(x[np.max([0, i - 100]):i]) for i in range(len(x))]
    plt.figure(dpi=200)
    plt.title('Learning Curve')
    plt.plot(range(len(x)), x, label='score', alpha=0.3)
    plt.plot(range(len(avg_x)), avg_x, label='average score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    if save_plot:
        plt.savefig(filename + '.png')
    plt.show()


def process_observation(flat_observation):
    # Deconstruct the flat observation back into lidar and other_obs components
    map_size = 663 * 730
    batch_size = flat_observation.shape[0]
    lidar = torch.as_tensor(flat_observation[:, :map_size].reshape(batch_size, 663, 730), dtype=torch.float32)
    other_obs = torch.as_tensor(flat_observation[:, map_size:], dtype=torch.float32)
        
    return lidar, other_obs