import math
import numpy as np
import tbplas as tb
import matplotlib.pyplot as plt

def make_empty_cell():
    """Generate lattice vectors of monolayer graphene."""
    a = 2.46  # cell length in Angstrom
    c = 10.0  # arbitrary z-dimension
    vectors = np.array([
        [a, 0, 0],
        [a/2, a*np.sqrt(3)/2, 0],
        [0, 0, c]
    ])
    return tb.PrimitiveCell(vectors, unit=tb.ANG)

def add_orbitals(cell):
    """Add orbitals to the model."""
    cell.add_orbital((1./3, 1./3, 0), energy=0.0, label="pz")
    cell.add_orbital((2./3, 2./3, 0), energy=0.0, label="pz")

def add_hopping_terms(cell):
    """Add hopping terms to the model."""
    cell.add_hopping((0, 0, 0), 0, 1, -2.7)  # Intra-cell hopping
    cell.add_hopping((1, 0, 0), 1, 0, -2.7)  # Inter-cell hopping
    cell.add_hopping((0, 1, 0), 1, 0, -2.7)  # Inter-cell hopping

# Create and configure the cell
cell = make_empty_cell()
add_orbitals(cell)
add_hopping_terms(cell)

# Visualize the cell
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Get atom positions in Cartesian coordinates
positions = []
for orb in cell.orbitals:
    # Convert fractional to Cartesian coordinates
    pos = np.dot(orb.frac_coords, cell.lat_vec)
    positions.append(pos)

positions = np.array(positions)

# Plot atoms
ax.scatter(positions[:,0], positions[:,1], positions[:,2], c='k', s=100)

# Plot hopping terms
for hop in cell.hoppings:
    # Get start and end positions
    rn = hop.rn
    i = hop.orb_i
    j = hop.orb_j
    
    start_pos = np.dot(cell.orbitals[i].frac_coords, cell.lat_vec)
    end_pos = np.dot(cell.orbitals[j].frac_coords + rn, cell.lat_vec)
    
    # Draw line
    ax.plot([start_pos[0], end_pos[0]],
            [start_pos[1], end_pos[1]],
            [start_pos[2], end_pos[2]], 'r-')

# Set labels and title
ax.set_xlabel('x (Å)')
ax.set_ylabel('y (Å)')
ax.set_zlabel('z (Å)')
ax.set_title('Graphene Primitive Cell')
ax.grid(True)

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 0.1])  # Flatten z-axis for 2D materials

plt.tight_layout()
plt.savefig("tbplas_test.png")