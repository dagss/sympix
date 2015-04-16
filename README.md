Accompanying code for the SymPix paper
======================================

NOTE: This is (at least not yet) a stand-alone package.
Rather it is a loose collection of code, without any
build system or similar, copied from the Commander 2
project. At the time being this code is supposed to be *read*,
not *run*. There is no build system in this repository that
will build the code for you.

If you are interested in using SymPix, please contact the author and
we can cooperate to create a package around it with the pieces
needed. All code is also part of, and developed and built as part of,
the Commander 2 code, found here:

  https://bitbucket.org/commander/commander2

Explanation:

- sympix.py: This contains the algorithms to compute a grid,
  as well as code for finding neighbouring tiles as a CSC
  matrix, and code for plotting a SymPix map

- sharp.pyx: The SymPixGridPlan class interfaces to libsharp
  for the SymPix grid, and shows up to set up arguments for
  sharp_make_geom_info.

- sympix_mg.f90: This contains code for evaluating block
  matrices, \widehat{B} from the paper.

- legendre_transform.c.in: Fast code for doing basic Legendre
  transforms. Requires being run through preprocessing using the
  Jinja template language.


License: BSD 3-clause
---------------------

Copyright (c) 2015, Dag Sverre Seljebotn, University of Oslo
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
