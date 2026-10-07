"""Physics core: a faithful NumPy port of the reentry_simulator translational
models (US-1976 atmosphere, rotating-J2 gravity, WB001 polynomial aero,
Sutton-Graves heating) plus the 3-DOF equations of motion.

Every model here is validated node-for-node against the MATLAB benchmark
trajectories before any RL is trained (see reentry_rl/validation)."""
