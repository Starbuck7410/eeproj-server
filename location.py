import numpy as np

class Location:

    def __init__(self, size = 20, jump_threshold = 20.0, persistence_threshold = 15, cluster_radius = 15.0, invalidate_counter_max = 3):
        coords = np.dtype([('x', 'f4'), ('y', 'f4')])
        self.size = size
        self.jump_threshold = jump_threshold
        self.persistence_threshold = persistence_threshold
        self.cluster_radius = cluster_radius  # Added parameter for density check
        self.arr_coords = np.zeros(size, dtype = coords)
        self.coords_valid = False
        self.coords = (0, 0)
        self.outlier_buffer = []              # Replaced integer counter with a list buffer
        self.arr_idx = 0
        self.invalidate_counter = 0
        self.invalidate_counter_max = invalidate_counter_max

    def update(self, new_coords):
        """
        Updates the rolling average with protection against sudden false jumps.
        Updates and returns the current trusted coordinates.
        """
        new_x, new_y = new_coords

        # Calculate Euclidean distance from the current running average
        dist = np.sqrt((new_x - self.coords[0])**2 + (new_y - self.coords[1])**2)
        
        
        if dist <= self.jump_threshold:
            # Signal is close enough; normal tracking behavior
            self.outlier_buffer.clear()  # Clear transient noise spikes
            self._push_to_buffer(new_x, new_y)
            self.validate()
        else:
            # Signal jumped too far. Accumulate sample points for voting
            self.outlier_buffer.append((new_x, new_y))
            
            if len(self.outlier_buffer) >= self.persistence_threshold:
                self.validate()
                # Confirmed target shift: Find the densest cluster among the samples
                best_x, best_y = self._get_densest_sample()
                self._reset_buffer(best_x, best_y)
                self.outlier_buffer.clear()

        # Compute the updated mean
        self.coords = (self.arr_coords['x'].mean(), self.arr_coords['y'].mean())
        return self.coords

    def invalidate(self):
        if(self.invalidate_counter < self.invalidate_counter_max):
            self.invalidate_counter += 1
        else:
            self.invalidate_counter = 0
            self.coords_valid = False
            self.outlier_buffer.clear()

    def validate(self):
        self.invalidate_counter = 0
        self.coords_valid = True

    def _push_to_buffer(self, x, y):
        self.arr_coords[self.arr_idx] = (x, y)
        self.arr_idx = (self.arr_idx + 1) % self.size

    def _reset_buffer(self, x, y):
        self.arr_coords['x'] = x
        self.arr_coords['y'] = y
        self.arr_idx = 0

    def get_coords(self):
        return self.coords
    
    def get_valid(self):
        return self.coords_valid

    def _get_densest_sample(self):
        """
        Finds the sample point within the outlier buffer that has 
        the highest number of neighbors within `cluster_radius`.
        """
        pts = np.array(self.outlier_buffer)  # Shape: (N, 2)
        
        # Matrix broadcasting to find pairwise distances between all collected samples
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # Shape: (N, N, 2)
        dists = np.sqrt(np.sum(diff**2, axis=-1))             # Shape: (N, N)
        
        # Count how many neighbors each point has within your cluster radius
        neighbor_counts = np.sum(dists <= self.cluster_radius, axis=1)
        
        # Pick the index of the point with the highest local density
        best_idx = np.argmax(neighbor_counts)
        return pts[best_idx]