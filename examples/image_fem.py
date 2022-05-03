# Reordering the FEM data (on a structured mesh) into an image
from nn_atlas.nn_extensions.cnn_utils import set_image, set_function
import matplotlib.pyplot as plt    
import dolfin as df
import numpy as np


mesh = df.UnitSquareMesh(32, 30)

# Scalars 
V = df.FunctionSpace(mesh, 'CG', 1)
u = df.interpolate(df.Expression('-3*x[0]+4*x[1]', degree=1), V)

vals0 = set_image(u)
    
fig, ax = plt.subplots(1, 2)
ax[0].imshow(vals0)
df.plot(u)

# The other way around    
u.vector()[:] *= 0
set_function(vals0, u)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(vals0)
df.plot(u)
    
plt.show()
    
# Vectors
V = df.VectorFunctionSpace(mesh, 'CG', 1)
u = df.interpolate(df.Expression(('-3*x[0]+4*x[1]', '-5*x[0]+2*x[1]'), degree=1), V)
    
vals0 = set_image(u)
    
fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.linalg.norm(vals0, 2, axis=2))
df.plot(df.sqrt(df.inner(u, u)))

# The other way around    
u.vector()[:] *= 0
set_function(vals0, u)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(np.linalg.norm(vals0, 2, axis=2))
df.plot(df.sqrt(df.inner(u, u)))

plt.show()
