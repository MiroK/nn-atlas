# Solvers (test implementation/stability)
- [ ] Harmonic (add distance^q smoothing) - variable coefs http://dx.doi.org/10.1016/j.compstruc.2011.02.019
- [ ] Biharmonic
- [ ] Elasticity
- [ ] p-Laplacian*
- [ ] time*

# Domains (with mapping from reference)
- [ ] Square with cusp
- [ ] Circle with cusp (parametrized such that we recover smooth)
- [ ] Cake

# Networks (with reg loss, as **autoencoder**)
- [ ] FFFF
- [ ] CNN?
- [ ] ResNet
- [ ] determinant constraint vs mesh based cross product
- [ ] hard bcs vs in loss

# Theory
- [ ] Conformal mappings - maybe use rational activation functions
- [ ] Circle <-> disc
- [ ] Convolution in a disk

# Analysis/Visualization
- [ ] Sparse regression of the network to see if we can find a PDE it solved

# Ultimate
- [ ] Something like operator network (reference, bdry) --> NN(x_ref) that
      is the extension network