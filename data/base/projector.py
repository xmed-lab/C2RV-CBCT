import tigre
import numpy as np
import matplotlib.pyplot as plt



def visualize_projections(path, projections, angles, figs_per_row=10):
    """
    Visualize and save a grid of projection images.
    
    Args:
        path (str): File path to save the visualization.
        projections (np.ndarray): Array of projection images.
        angles (np.ndarray): Array of projection angles in radians.
        figs_per_row (int): Number of images per row in the grid.
    """
    
    # Calculate the number of rows needed
    n_row = (len(projections) + figs_per_row - 1) // figs_per_row
    
    # Normalize projections to [0, 1]
    projections = projections.copy()
    projections = (projections - projections.min()) / (projections.max() - projections.min())
    
    # Create a figure for the projections
    plt.figure(figsize=(figs_per_row * 2, n_row * 2))
    for i in range(len(projections)):
        angle_deg = int((angles[i] / np.pi) * 180)  # Convert angle to degrees
        plt.subplot(n_row, figs_per_row, i + 1)
        plt.imshow(projections[i], cmap='gray', vmin=0, vmax=1)
        plt.title(f'{angle_deg}°')
        plt.axis('off')
    
    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=300)
    plt.close()



class ConeGeometry_special(tigre.utilities.geometry.Geometry):
    """
    Custom geometry class for TIGRE to define a cone-beam CT scanner.
    
    This class sets up the scanner geometry based on the provided configuration.
    """
    
    def __init__(self, config):
        """
        Initialize the geometry with the given configuration.
        
        Args:
            config (dict): Configuration dictionary containing geometry parameters.
        """
        super().__init__()
        
        # Distances in meters (converted from mm)
        self.DSD = config['DSD'] / 1000  # Distance from source to detector
        self.DSO = config['DSO'] / 1000  # Distance from source to origin
        
        # Detector parameters
        self.nDetector = np.array(config['nDetector'])          # Number of detector pixels [u, v]
        self.dDetector = np.array(config['dDetector']) / 1000   # Size of each detector pixel [u, v] in meters
        self.sDetector = self.nDetector * self.dDetector        # Total size of the detector [u, v] in meters
        
        # Image parameters (note the reversal to [z, y, x])
        self.nVoxel = np.array(config['nVoxel'][::-1])          # Number of voxels [z, y, x]
        self.dVoxel = np.array(config['dVoxel'][::-1]) / 1000   # Size of each voxel [z, y, x] in meters
        self.sVoxel = self.nVoxel * self.dVoxel                 # Total size of the image [z, y, x] in meters
        
        # Offsets
        self.offOrigin = np.array(config['offOrigin'][::-1]) / 1000  # Image offset [z, y, x] in meters
        self.offDetector = np.array(
            [config['offDetector'][1], config['offDetector'][0], 0]) / 1000  # Detector offset [v, u, 0] in meters
        
        # Auxiliary parameters
        self.accuracy = config['accuracy']  # Accuracy of forward projection (voxels per sample)
        
        # Mode and filter
        self.mode = config['mode']        # 'parallel' or 'cone'
        self.filter = config['filter']    # Filter type for reconstruction algorithms



class Projector:
    """
    Projector class to simulate 2D projections of 3D images using TIGRE.
    
    Attributes:
        _geo (tigre.utilities.geometry.Geometry): Scanner geometry.
        _angles (np.ndarray): Array of projection angles in radians.
    """
    
    def __init__(self, config):
        """
        Initialize the projector with the given configuration.
        
        Args:
            config (dict): Configuration dictionary containing projector parameters.
        """
        self._geo = ConeGeometry_special(config)
        
        # Generate equally spaced projection angles
        self._angles = np.linspace(
            config['start_angle'] / 180 * np.pi,                # Start angle in radians
            (config['start_angle'] + config['total_angle']) / 180 * np.pi,  # End angle in radians
            config['n_porjections'],
            endpoint=False
        )
    
    
    def __call__(self, image):
        """
        Simulate 2D projections of the given 3D image.
        
        Args:
            image (np.ndarray): 3D image array [x, y, z].
        
        Returns:
            dict: Dictionary containing 'projs' (projections) and 'angles'.
        """
        # Transpose image to [z, y, x] as expected by TIGRE
        image_tigre = image.transpose(2, 1, 0).copy()
        # Perform forward projection
        projections = tigre.Ax(
            image_tigre,
            self._geo,
            self._angles
        )
        # Flip projections vertically to match coordinate system if needed
        projections = projections[:, ::-1, :]
        
        return {
            'projs': projections,
            'angles': self._angles
        }
    
#cloner174