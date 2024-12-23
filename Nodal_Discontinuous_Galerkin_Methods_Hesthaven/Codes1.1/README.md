Revision: 1.1 August 10 2009

These MATLAB scripts accompany:

Nodal Discontinuous Methods: Algorithms, Analysis, and Applications 
by Jan S. Hesthaven and Tim Warburton.

To use these scripts you should add this directory and subdirectories 
to your MATLAB path.

The following routines are Driver routines that will solve various 
partial differential equations:

CFD1D/EulerDriver1D.m
CFD2D/CurvedCNSDriver2D.m
CFD2D/CurvedEulerDriver2D.m
CFD2D/CurvedINSDriver2D.m
CFD2D/EulerDriver2D.m
CFD2D/EulerShockDriver2D.m
Codes1D/AdvecDriver1D.m
Codes1D/BurgersDriver1D.m
Codes1D/HeatDriver1D.m
Codes1D/MaxwellDriver1D.m
Codes1D/PoissonCDriver1D.m
Codes2D/CurvedPoissonIPDGDriver2D.m
Codes2D/MaxwellCurvedDriver2D.m
Codes2D/MaxwellDriver2D.m
Codes2D/MaxwellHNonConDriver2D.m
Codes2D/MaxwellPNonConDriver2D.m
Codes2D/PoissonDriver2D.m
Codes3D/AdvecDriver3D.m
Codes3D/MaxwellDriver3D.m
Codes3D/PoissonIPDGDriver3D.m

Permission to use this software for noncommercial research and educational purposes 
is hereby granted without fee. Redistribution, sale, or incorporation of this software
 into a commercial product is prohibited. 

THIS SOFTWARE,INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY
PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHORS OR THE PUBLISHER BE LIABLE FOR ANY
SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS.

Version 1.1

Additions by burakbayramli

To run code from Octave, for example `CFD2D/EulerDriver2D.m` 
use this,

```
addpath('../Codes2D:../Grid/Euler2D:../Codes1D:../ServiceRoutines');
EulerDriver2D;
```


