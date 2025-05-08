import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio.v2 as imageio
import os

# Parameters for the simulation
num_krill = 10
num_iterations = 30
initial_position_range = (-10, 10)  # Initial range for the krill positions
target_range = (-8, 8)  # Range for target positions

# Initialize krill positions randomly within the specified range
positions = np.random.uniform(low=initial_position_range[0], high=initial_position_range[1], size=(num_krill, 2))

# Speed parameters for Krill Herd algorithm
social_influence = 0.15  # Influence of other krill (cohesion)
attraction_strength = 0.1  # How much krill are attracted to the target
random_walk_strength = 0.15  # Random movement to simulate natural foraging
target_distance_threshold = 0.8  # Threshold distance for krill to "reach" the target

# Function to update krill positions based on KH algorithm logic
def update_positions(positions, target, social_influence, attraction_strength, random_walk_strength):
    # Calculate the movement vector for each krill
    movements = np.zeros_like(positions)

    # Cohesion: krill are attracted to the average position of others (but with reduced strength)
    average_position = np.mean(positions, axis=0)
    cohesion = (average_position - positions) * social_influence

    # Attraction: krill are attracted to the target
    attraction = (target - positions) * attraction_strength

    # Random movement (simulating foraging)
    random_walk = np.random.uniform(-random_walk_strength, random_walk_strength, size=positions.shape)

    # Update positions: Each krill's movement is the sum of the three components
    movements = cohesion + attraction + random_walk
    positions += movements
    return positions

# Function to check if all krill have reached the target
def all_krill_reached_target(positions, target, threshold):
    distances = np.linalg.norm(positions - target, axis=1)
    return np.all(distances < threshold)

# Function to generate a new target randomly
def generate_new_target():
    return np.random.uniform(low=target_range[0], high=target_range[1], size=2)

# Plotting setup
fig, ax = plt.subplots()
ax.set_xlim(initial_position_range[0]-2, initial_position_range[1]+2)
ax.set_ylim(initial_position_range[0]-2, initial_position_range[1]+2)
ax.set_title('Krill Herd Algorithm')

# Initialize krill plot
krill_scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue', label='Krill', s=100)

# Plot target (it will be updated later in the animation)
target_plot, = ax.plot([], [], 'rx', markersize=10, label='Target')

# Directory to save frames
frames_dir = "krill_herd_frames"
os.makedirs(frames_dir, exist_ok=True)

# Function to update the animation
def animate(frame):
    global positions, target

    # Update krill positions
    positions = update_positions(positions, target, social_influence, attraction_strength, random_walk_strength)
    
    # Check if all krill have reached the target
    if all_krill_reached_target(positions, target, target_distance_threshold):
        # Generate new target if all krill reached the current one
        target = generate_new_target()
    
    # Update krill scatter plot
    krill_scatter.set_offsets(positions)

    # Update target plot
    target_plot.set_data(target[0], target[1])

    # Save frame
    frame_filename = os.path.join(frames_dir, f"frame_{frame:03d}.png")
    plt.savefig(frame_filename)

    return krill_scatter, target_plot

# Initial target
target = generate_new_target()

# Animation setup
ani = FuncAnimation(fig, animate, frames=num_iterations, interval=100, blit=True)

# Show plot with animation
plt.legend(loc='best')

# Run the animation and save the frames
plt.show()

# Create GIF
all_frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
images = [imageio.imread(frame) for frame in all_frames]
imageio.mimsave("krill_herd_algorithm.gif", images, duration=0.3)

# Clean up
for f in all_frames:
    os.remove(f)
os.rmdir(frames_dir)

print("GIF saved as 'krill_herd_algorithm.gif'")
