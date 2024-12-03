TIMESTEP = 0.01
MAX_IMPLICIT_ITERS = 30
MAX_IMPLICIT_ERR = 1e4
MIN_IMPLICIT_ERR = 1e-4
IMPLICIT_RATIO = 0.5
WEIGHT_EPSILON = 1e-4

# YOUNGS_MODULUS = 1.5e5		# Young's modulus (springiness) (1.4e5)
# POISSONS_RATIO = 0.2		# Poisson's ratio (transverse/axial strain ratio) (0.2)
# LAMBDA = YOUNGS_MODULUS*POISSONS_RATIO/((1+POISSONS_RATIO)*(1-2*POISSONS_RATIO))
# MU = YOUNGS_MODULUS/(2+2*POISSONS_RATIO)
# ALPHA = 10
# GRID_SPACE = 1
# CRIT_COMPRESS = 2.5e-2
# CRIT_STRETCH = 7.5e-3

YOUNGS_MODULUS = 4e4       # Reduce stiffness
POISSONS_RATIO = 0.1       # Lower transverse-to-axial strain ratio
LAMBDA = YOUNGS_MODULUS*POISSONS_RATIO/((1+POISSONS_RATIO)*(1-2*POISSONS_RATIO))
MU = YOUNGS_MODULUS/(2+2*POISSONS_RATIO)
ALPHA = 5                  # Reduce hardening
CRIT_COMPRESS = 5e-2       # Increase compression threshold
CRIT_STRETCH = 2e-2        # Increase stretch threshold
GRID_SPACE = 0.1           # Finer grid resolution