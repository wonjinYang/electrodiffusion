from dataclasses import dataclass

@dataclass
class DefaultConfig:
    """
    Default configuration parameters for the electrodiffusion library.
    This dataclass groups settings into logical categories for noise scheduling, physical geometry,
    particle properties, simulation hyperparameters, and multi-scale dynamics. Values are chosen based
    on typical biophysical scenarios (e.g., K+ channel with 100 residues) and can be overridden as needed.
    All parameters are immutable by default but can be modified post-instantiation if required.
    """

    # Noise scheduling parameters (used in NoiseSchedule for VP-SDE)
    # BETA_MIN: Minimum noise level at t=0 (mild perturbation for data integrity)
    BETA_MIN: float = 0.1
    # BETA_MAX: Maximum noise level at t=1 (strong noise for Gaussian prior)
    BETA_MAX: float = 20.0
    # COSINE_S: Smoothing factor for cosine schedule to avoid abrupt noise changes
    COSINE_S: float = 0.008

    # Channel geometry parameters (in meters; typical ion channel scales ~50 Å length, 5 Å radius)
    # CHANNEL_LENGTH: Axial length of the channel pore
    CHANNEL_LENGTH: float = 50e-10  # 50 Angstrom
    # CHANNEL_RADIUS: Effective radius for cylindrical approximation
    CHANNEL_RADIUS: float = 5e-10   # 5 Angstrom

    # Ion properties (for single monovalent ion like K+)
    # ION_CHARGE: Charge in elementary units (e for positive ions)
    ION_CHARGE: float = 1.0
    # DIFFUSION_COEFF: Base diffusion coefficient in water (m²/s)
    DIFFUSION_COEFF: float = 1e-9

    # Protein parameters (for coarse-grained model)
    # NUM_RESIDUES: Default number of amino acid residues in the protein chain
    NUM_RESIDUES: int = 100
    # BACKBONE_DIMENSIONS: Per-residue dims for backbone (e.g., CA:3, CB:3, normal:3)
    BACKBONE_DIMENSIONS: int = 9
    # SIDECHAIN_DIMENSIONS: Per-residue dims for side chains (e.g., pos:3, charge:1, dipole:3)
    SIDECHAIN_DIMENSIONS: int = 7

    # Simulation and training parameters
    # TIME_STEPS: Default number of steps for integrations or denoising chains
    TIME_STEPS: int = 1000
    # BATCH_SIZE: Default batch size for training/sampling
    BATCH_SIZE: int = 64
    # LEARNING_RATE: Default optimizer learning rate (e.g., for Adam in score network training)
    LEARNING_RATE: float = 1e-4
    # NUM_EPOCHS: Default training epochs for models like score networks
    NUM_EPOCHS: int = 500

    # Multi-scale and memory parameters
    # FAST_DT: Timestep for fast dynamics (e.g., side chains, ions; in seconds)
    FAST_DT: float = 1e-6  # 1 ps
    # SLOW_DT: Timestep for slow dynamics (e.g., backbone; in seconds)
    SLOW_DT: float = 1e-4  # 100 ps
    # MEMORY_TIME: Default memory kernel decay timescale τ (in seconds)
    MEMORY_TIME: float = 1e-6  # 1 ps, matching fast scale

    # Additional parameters (e.g., for data augmentation or validation)
    # NOISE_LEVEL: Default noise std for data augmentation
    NOISE_LEVEL: float = 0.01
    # TOLERANCE: Default error tolerance for adaptive solvers
    TOLERANCE: float = 1e-4
    # MAX_LAG: Default max lag for autocorrelation computations
    MAX_LAG: int = 100

    def __post_init__(self):
        """
        Post-initialization hook to validate parameters.
        Ensures logical constraints (e.g., FAST_DT < SLOW_DT) and positive values.
        """
        if self.FAST_DT >= self.SLOW_DT:
            raise ValueError(f"FAST_DT ({self.FAST_DT}) must be less than SLOW_DT ({self.SLOW_DT})")
        if self.BETA_MIN >= self.BETA_MAX:
            raise ValueError(f"BETA_MIN ({self.BETA_MIN}) must be less than BETA_MAX ({self.BETA_MAX})")
        # Ensure positive values for scales and rates
        positive_params = [
            self.CHANNEL_LENGTH, self.CHANNEL_RADIUS, self.DIFFUSION_COEFF,
            self.LEARNING_RATE, self.FAST_DT, self.SLOW_DT, self.MEMORY_TIME,
            self.NOISE_LEVEL, self.TOLERANCE
        ]
        for param in positive_params:
            if param <= 0:
                raise ValueError("All scale and rate parameters must be positive")

    def to_dict(self) -> dict:
        """
        Converts the config to a dictionary for easy serialization or logging.
        
        Returns:
            dict: Dictionary representation of the config attributes.
        """
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
