"""
Uses TBPLaS to compute the single particle DOS of an hBN sample using an (s,px,py,pz) Slater Koster model.
TB parameters from Galvani et al 2024 J. Phys. Mater. 7 035003.
TBPLaS: 
Reference: Y. Li, Z. Zhan, X. Kuang, Y. Li and S. Yuan, TBPLaS: a Tight-Binding Package for Large-scale Simulation, Comput. Phys. Commun. 285, 108632 (2023)
Documentation: https://www.tbplas.net/
"""

from math import exp

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

import tbplas as tb

from scipy.spatial import cKDTree
from scipy import integrate


# %matplotlib inline


#Structure handling
class Structure():
    def __init__(self, N_atoms,types,coords,A_lat):
        self.N_atoms=N_atoms
        self.types=types
        self.coords=coords
        self.A_lat=A_lat
        
    def unpack(self):
        return self.N_atoms,self.types,self.coords,self.A_lat


def ReadXSFGeom(coords_data_adress, verbose=True):
    '''
    Read XSF data file at the given adress, return N_atoms, types, coords and lattice vectors.
    Will copnvert letter types as B,b->1; N,n->2. Other letters will lead to undefined behavior later.
    Comments in the XSF file will break the function.
    '''

    #Number of atoms
    AtomNumberData=np.loadtxt(coords_data_adress,skiprows=6,max_rows=1,dtype=int)
    N_atoms=AtomNumberData[0]

    A_lat=np.loadtxt(coords_data_adress,skiprows=2,max_rows=3)

    coords_data=np.loadtxt(coords_data_adress,skiprows=7,max_rows=N_atoms,dtype=np.ndarray)
    types=np.zeros(N_atoms).astype(int)
    types[np.logical_or(coords_data[:,0]=='B', coords_data[:,0]=='b')]=1
    types[np.logical_or(coords_data[:,0]=='N', coords_data[:,0]=='n')]=2
    
    coords=coords_data[:,1:4].astype(float)  #atomic coordinates
    if N_atoms!=types.size: print('Loading error!')  #the number of atoms in the unit cell

    return Structure(N_atoms,types,coords,A_lat)


#Electronic

def SlaterKoster(rho,orb1,orb2,sk_params):
    """
    INPUTS:
    rho = r2-r1 : distance vector between orbital positions 2 and 1
    orb1, orb2 : orbital name of orbitals 1 and 2 (s,px,py,pz), as strings
    sk_params : the corresponding Slater Koster parameters ACCOUNTING FOR DISTANCE,ETC. as an array/list in the order Vss_sigma,Vsp_sigma,V_pp_sigma,Vpp_pi
    OUTPUTS:
    The hopping between the two orbitals.
    NOTES:
    Based on code by Adel Belayadi on the Kwant mailing list: https://www.mail-archive.com/kwant-discuss@kwant-project.org/msg02169.html
    """

    Vss_sig,Vsp_sig,Vpp_sig,Vpp_pi=sk_params  #read SK parameters

    l,m,n=rho/np.linalg.norm(rho)   #direction cosines

    #Computation of the S-K hopping (res):

    if orb1=='s' and orb2=='s': res  =   Vss_sig  #s_s

    elif orb1=='s' and orb2=='px': res = l*Vsp_sig  #s_px
    elif orb1=='s' and orb2=='py': res = m*Vsp_sig  #s_py
    elif orb1=='s' and orb2=='pz': res = n*Vsp_sig  #s_pz

    elif orb1=='px' and orb2=='s': res = -l*Vsp_sig  #s_px
    elif orb1=='py' and orb2=='s': res = -m*Vsp_sig  #s_py
    elif orb1=='pz' and orb2=='s': res = -n*Vsp_sig  #s_pz

    ## ----------------------------------------------------

    elif orb1=='px' and orb2=='px': res = (l**2)*Vpp_sig + (1-l**2)*Vpp_pi  #px_px
    elif (orb1=='px' and orb2=='py') or (orb1=='py' and orb2=='px'): res = (l*m)*(Vpp_sig - Vpp_pi) #px_py & py_px
    elif (orb1=='px' and orb2=='pz') or (orb1=='pz' and orb2=='px'): res = (l*n)*(Vpp_sig - Vpp_pi) #px_pz & pz_px
    ## ----------------------------------------------------
    elif orb1=='py' and orb2=='py': res = (m**2)*Vpp_sig + (1-m**2)*Vpp_pi  #py_py
    elif (orb1=='py' and orb2=='pz') or (orb1=='pz' and orb2=='py'): res = (m*n)*(Vpp_sig - Vpp_pi)   #py_pz & pz_py
    ## ----------------------------------------------------
    elif orb1=='pz' and orb2=='pz': res = (n**2)*Vpp_sig + (1-n**2)*Vpp_pi  #pz_pz

    return res


def calc_hop_BN(sk: tb.SK, rij: np.ndarray, label_i: str,
                  label_j: str) -> complex:
    
    """
    Computes the hoppings.
    """
    
    at_i=label_i.split(":")[0]   #species name (B or N)
    at_j=label_j.split(":")[0]
    bond=at_i+at_j      #pair name (BN, NB, BB or NN)
    
    lm_i = label_i.split(":")[1]   #orbital name (s, px, py or pz)
    lm_j = label_j.split(":")[1]
    
    r = norm(rij)   #pair distance

    r0 = 1.57  #reference distance for the hopping distance dependence, in AA

    #Slater-Koster hopping parameters and distance dependance

    hop_sss={'BN':-4.0, 'NB':-4.0, 'BB':-4.0,'NN':-4.0}   #hopping value "V_sss" at r=r0
    beta_sss={'BN':4.1, 'NB':4.1, 'BB':4.1,'NN':4.1}   #hopping decay constant "beta"
    v_sss = hop_sss[bond] * exp(-(r/r0-1) * beta_sss[bond])

    hop_sps={'BN':3.8, 'NB':4.3, 'BB':4.05,'NN':4.05}
    beta_sps={'BN':1.7, 'NB':2.3, 'BB':2.0,'NN':2.0}
    v_sps = hop_sps[bond] * exp(-(r/r0-1) * beta_sps[bond])
    
    hop_pps={'BN':5.2, 'NB':5.2, 'BB':5.2,'NN':5.2}
    beta_pps={'BN':1.9, 'NB':1.9, 'BB':1.9,'NN':1.9}
    v_pps = hop_pps[bond] * exp(-(r/r0-1) * beta_pps[bond])
    
    hop_ppp={'BN':-1.8, 'NB':-1.8, 'BB':-1.8,'NN':-1.8}
    beta_ppp={'BN':3, 'NB':3, 'BB':3,'NN':3}
    v_ppp = hop_ppp[bond] * exp(-(r/r0-1) * beta_ppp[bond])

    res=sk.eval(r=rij, label_i=lm_i, label_j=lm_j, v_sss=v_sss, v_sps=v_sps, v_pps=v_pps, v_ppp=v_ppp)


    return res


#The actual TB_PlaS function

def BN_SK(structure, n_per=[1,1,1]):
    
    N_atoms,types,coords,A_lat=structure.unpack()
    N_boron=np.size(np.argwhere(types==1))
    N_nitrogen=N_atoms-N_boron
    #N_el=3*N_boron+5*N_nitrogen  #Total number of electrons in the system  (used to find the Fermi Level)

    # Orbital coordinates
    num_orbitals_per_atom=4
    orbital_coord = np.repeat(structure.coords, num_orbitals_per_atom, axis=0)
    

    # Orbital labels
    B_orbitals = ("B:s", "B:px", "B:py", "B:pz")
    N_orbitals = ("N:s", "N:px", "N:py", "N:pz")
    orbital_label = B_orbitals*N_boron + N_orbitals*N_nitrogen

    # Orbital energies
    # Parameters from Simon Dubois
    Bs, Bp=-1.3, 8.6
    Ns, Np=-10.1,0
    orbital_energy = {"B:s":Bs, "B:px":Bp, "B:py":Bp, "B:pz":Bp, "N:s":Ns, "N:px":Np, "N:py":Np, "N:pz":Np}

    # Create the primitive cell and add orbitals
    cell = tb.PrimitiveCell(lat_vec=A_lat, unit=tb.ANG)
    for i, label in enumerate(orbital_label):
        coord = orbital_coord[i]
        energy = orbital_energy[label]
        cell.add_orbital_cart(coord, energy=energy, label=label)
    
    #add hopping terms
    sk = tb.SK()
    r_cut=1.9
    #Here we will find all neighbors etc. FINDING NEIGHBORS IS NOT NEEDED IF YOU ALREADY HAVE THE H_supercell
    positions=orbital_coord
    A1,A2,A3=A_lat
    for i in range(-n_per[0],1+n_per[0]):
        for j in range(-n_per[1],1+n_per[1]):
            for k in range(-n_per[2],1+n_per[2]):
                translat=i*A1+j*A2+k*A3
                positions2 = positions+translat
                d_min = r_cut * 0.01  #minimum distance, to avoid sites seeing themselves as their own n.n.
                d_max = r_cut
                kdtree1 = cKDTree(positions)
                kdtree2 = cKDTree(positions2)  #the cell, but shifted by i*A1+j*A2+k*A3
                coo = kdtree1.sparse_distance_matrix(kdtree2, d_max, output_type='coo_matrix')
                idx = coo.data > d_min
                row, col = coo.row[idx], coo.col[idx]
                rowcol=np.stack((row,col),axis=1)
                rowcol_unique=rowcol
                for pair in rowcol_unique:
                    rho=positions2[pair[1]]-positions[pair[0]]
                    label_i = cell.get_orbital(pair[0]).label
                    label_j = cell.get_orbital(pair[1]).label
                    hop = calc_hop_BN(sk, rho, label_i, label_j)
                    cell.add_hopping((i, j, k), pair[0], pair[1], energy=hop)


    #Plot of the model
    cell.plot()
    

    #DOS
    k_mesh = tb.gen_kmesh((25, 25, 1))
    energies, dos = cell.calc_dos(k_mesh, enable_mpi=True, e_min=-40, e_max=20)
    tb.Visualizer().plot_dos(energies, dos)

    
    return energies, dos


#SOME TEST CALCULATIONS

if __name__ == "__main__":
    coords_data_adress='monolayer_tri_xsf'
    structure=ReadXSFGeom(coords_data_adress)
    
    energies, dos=BN_SK(structure, n_per=[2,2,0])
