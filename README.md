lowerhooks.py
=============
Proof of concept code for computing Betti diagrams relative to lower hook
modules. This code accompanies the paper ["Koszul complexes and relative
homological algebra of functors over posets"](https://arxiv.org/abs/2209.05923)
by Wojciech Chachólski, Andrea Guidolin, Isaac Ren, Martina Scolamiero, and
Francesca Tombari.

License
-------
This software is released under
[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html), or (at your option) any
later version as published by the Free Software Foundation.

Details
-------
This code computes the Betti diagrams of a multiparameter persistence module,
given a free presentation. Extra visualization is provided for the 2D case.
All of the code is in `lowerhooks.py`: run
```
python lowerhooks.py --random --draw
```
to generate a random free 2D presentation and draw it. Run
```
python lowerhooks.py --input=example.txt --draw
```
to view the presentation of an example free presentation. Run
`python lowerhooks.py -h` to see all other options.
