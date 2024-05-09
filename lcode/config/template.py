
lcode_template = '''
# Simulation area:
geometry = c
window-width = {window-width};          r-step = {window-width-step-size}
window-length = {window-length};        xi-step = {xi-step}
time-limit = {time-limit};         time-step = {time-step}
continuation = n # Plasma continuation (no/beam/longplasma, n/y/Y)

# Particle beams: ????
beam-current = 0.00281
rigid-beam = n
beam-substepping-energy = {beam-substepping-energy}
beam-particles-in-layer = 300
beam-profile = """
xishape=b, ampl=1.0, length=1500, rshape=g, radius=1.0, angshape=l, angspread=4.5e-5, energy=7.83e5, m/q=1836, eshape=g, espread=2740.0
"""

# Plasma:
plasma-model = {plasma-model}
plasma-particles-per-cell = {plasma-particles-per-cell} #???????
plasma-profile = {plasma-profile} 
plasma-zshape = {plasma-zshape}                 
plasma-width = {plasma-width} 
plasma-width-2 = {plasma-width-2}
plasma-density-2 = {plasma-density-2}
plasma-temperature = {plasma-temperature}
ion-model = {ion-model}
ion-mass = 157000
substepping-depth = {substepping-depth}
substepping-sensivity = {substepping-sensitivity}

# Every-time-step diagnostics:

# Periodical diagnostics:
output-time-period = {time-step}

#  Colored maps: (Er,Ef,Ez,Phi,Bf,Bz,pr,pf,pz,pri,pfi,pzi
#                 nb,ne,ni,Wf,dW,SEB,Sf,Sf2,Sr,Sr2,dS,dS2):
colormaps-full = ""
colormaps-subwindow = ""
colormaps-type = n
drawn-portion = 1 # Drawn portion of the simulation window
subwindow-xi-from = 0;		subwindow-xi-to = -1500
subwindow-r-from = 0;		subwindow-r-to = 10
output-reference-energy = 1000
output-merging-r = 10;		output-merging-z = 20
palette = d # Colormaps palette (default/greyscale/hue/bluewhitered, d/g/h/b)
                E-step = 0.0005;	               nb-step = 0.0005
            Phi-step = 0.0005;	               ne-step = 0.999
            Bf-step = 0.0005;	               ni-step = 0.01
            Bz-step = 0.05;	             flux-step = 0.02
electron-momenta-step = 0.1;	 r-corrected-flux-step = 0.02
    ion-momenta-step = 0.1;	           energy-step = 10

#  Output of various quantities as functions of xi:
#   (ne,nb,Ez,<Ez>,Bz,Phi,pz,emitt,dW,Wf,ni,pzi)
#   (nb2,Er,Ez2,Bf,Bz2,Fr,pr,pf,<rb>,dS,Sf,SEB,pri,pfi,Ef)
f(xi) = ""
f(xi)-type = Y
axis-radius = 0;		auxillary-radius = 1
            E-scale = 0.02;	              nb-scale = 0.02
            Phi-scale = 0.02;	              ne-scale = 2
            Bz-scale = 0.5;	              ni-scale = 0.1
electron-momenta-scale = 0.5;	            flux-scale = 0.0005
    ion-momenta-scale = 0.5;	          energy-scale = 1
    beam-radius-scale = 10;	       emittance-scale = 1000

#  Beam particle information as pictures (r,pr,pz,M):
output-beam-particles = ""
draw-each = 10
beam-picture-height = 900
beam-pr-scale = 10
beam-a-m-scale = 1000;		beam-pz-scale = 5000

# Output of beam characteristics in histogram form (r,z,M,a):
histogram-output = ""
histogram-output-accel = ""
histogram-type = y
histogram-bins = 300;		beam-angle-scale = 0.02

#  Trajectories of plasma particles:
trajectories-draw = Y
trajectories-each = 1;		trajectories-spacing = 1
trajectories-min-energy = 0;	trajectories-energy-step = 0.5

# Saving run state periodically:
saving-period = {time-step}
save-beam = n
save-plasma = n

'''