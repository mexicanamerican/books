"""
   Usage: run case6
      Case 6 in Table 4.3, Liska_Wendroff_02.
      solve Euler equations using the HLLC solver and ideal triangular mesh
"""
import numpy as np
from numpy import pi as pi
import matplotlib.pylab as plt
from time import clock

from CREATE_IDEAL_MESH import *
from Meshing_Tools import *
from VISUAL_TRI import *

from IC_conditions import *

from HLLC_FLUX import *
from HLLC_SOLVER_TRI import *

plt.close('all')

# -----------------------------------------------------------
#       case 6: INITIAL CONDITIONS IN EACH QUADRANT
#                        (p,d,u,v)
# -----------------------------------------------------------
upper_left = [1.0, 2.0, 0.75, 0.5]
lower_left = [1.0, 1.0, -0.75, 0.5]
upper_right = [1.0, 1.0, 0.75, -0.5]
lower_right = [1.0, 3.0, -0.75, -0.5]

# simulation time and Courant number for stability
sim_time = 0.3  # simulation time
CFL = 0.1  # Courant number

# -----------------------------------------------------------
#               DOMAIN AND GRID GRANULARITY
# -----------------------------------------------------------
# physical domain
#x1 = 0; x2 = 1; y1 = 0; y2 = 1
x1 = -0.75; x2 = 1.75
y1 = -0.75; y2 = 1.75

h0 = 0.1 # 0.01 

# For case6:
# Using h0 = 0.01, a computational domain [-0.75,1.75]x[-0.75,1.75] 
# a) it generates about 20,000 triangles in [0,1]x[0,1] (and about
#    125,000 triangles in the computational domain)
# b) for sim_time = 0.3, we get CPU time of 7.8 hours, # iterations = 960
# c) use Iplot = 300

# for initial testing use a larger granulariy, for ex, h0 = 0.05

# -----------------------------------------------------------
#               CREATE THE MESH AND THE BOUNDARY
# -----------------------------------------------------------
dx = h0; dy = h0

create_mesh = CREATE_IDEAL_MESH_RECT(x1,x2,y1,y2,dx,dy)
p,tri,bbars,boundary_nodes,boundary_bars = create_mesh()

# ----------------------------------------------------------
#       CREATE GEOMETRY INFO, GHOSTS FOR BC and NEIGHBORS INFO
# ----------------------------------------------------------
geo = GEOMETRY(p,tri)
area, e01,e12,e20,ang01,ang12,ang20 = geo()

gboundary = GHOSTS(p,tri,bbars,e01,e12,e20,ang01,ang12,ang20)
btri,tghost,comp_p,comp_tri = gboundary()

# Notice that the node array 'comp_p' now includes also the additional
# nodes of the ghost triangles added to the boundary

matched_pairs = matching_pairs(bbars,comp_tri)
# Note: len(matched_pairs) = len(bbars) = len(tghost)
#    if matched_pairs[i] = [43,12], this means that the
#    comp_tri[43] is the ghost triangle abutted to the
#    comp_tri[12] triangle inside the physical space

neighbors = vecinos(bbars,comp_tri)
# Note: for each triangle in the physical domain it gives the
#       three neighbors (triangles) abutted to it

plot_ghosts(comp_p,tghost)

# -------------------------------------------------------------
#       APPLY IC ON THE MESH
# -------------------------------------------------------------

# map the initial pressure, density and speeds into the physical mesh
# based on the Initial Conditions specification:
pp,dd,uu,vv = ic_phys(upper_left,lower_left,upper_right,lower_right,p,tri)

start_time = clock()

solver = HLLC_TRI(CFL,pp,dd,uu,vv, \
                 area,e01,e12,e20,ang01,ang12,ang20, \
                 neighbors,matched_pairs,p,tri,x1,x2,y1,y2,Iplot=10)

dd,uu,vv,pp,ee = solver(sim_time)

end_time = clock()
print 'CPU time = %g [sec]' % (end_time - start_time)

# ------------------------------------------------------
#               VISUALIZATION
# ------------------------------------------------------
# visualization domain for final plot
vx1 = 0.0; vx2 = 1.0
vy1 = 0.0; vy2 = 1.0

vis = VISUAL_2D(vx1,vx2,vy1,vy2,sim_time,p,tri,pp,dd,uu,vv)
vis()

