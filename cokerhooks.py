#!/usr/bin/env python
# -*-coding:utf8-*-

# lowerhooks.py: a proof-of-concept code for computing relative Betti diagrams
# Copyright (C) 2023  Isaac Ren
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
from collections import namedtuple
import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
import numpy as np

import cProfile

# Program name for the command-line interface
PROG = "lowerhooks"

"""
SECTION: Introduction

We present a finitely presented module over R^2 by generators and relations:

      d
    R -> G -> M -> 0

We fix a set of generators {g_1, ..., g_n} with grades {a_1, ..., a_n} in N^2.
We fix a set of relations {r_1, ..., r_p} with grades {b_1, ..., b_p} and
images {d(r_1), ..., d(r_p)}, expressed as vectors in the basis of generators.

We work modulo 2, but store field values as integers. Thus sums are done by

    (x + y) % 2.

This code then computes the Betti diagrams of M relative to lower hooks using
a "cokernel trick": this has nothing to do with machine learning. Instead, we
use a result that says that we can work with cokernels instead of kernels,
which simplifies things.
"""

"""
SECTION: Classes

We define a class FreePresentation, with general methods and a method to
compute relative Betti diagrams.

We also define a class QuotientSpace, which is a Python named tuple.
"""
class FreePresentation():
    """
    Free presentation of a persistence module M over a finite grid in N^r, .
    Free presentation
    
        R -> G -> M -> 0
    
    of a persistence module M over a finite subgrid of N^r, with
    coefficients in Z/2Z. The presentation is defined over a grid of the form
    [0, n_1] x ... x [0, n_r], and is represented by the generators of the free
    module G, the generators of the free module R, and the matrix of the
    morphism R -> G.
    
    Attributes
    ----------
    dim: int
        Dimension of the grid.
    n_gen, n_rel: int
        Number of generators and relations, respectively.
    shape: list of int
        Maximal coordinate in each dimension of the grid.
    G_grade: np.ndarray
        Array of shape (2, n_gen) of the grades of the generators.
    R_grade: np.ndarray
        Array of shape (2, n_rel) of the grades of the relations.
    R_image: np.ndarray
        Array of shape (n_rel, n_gen) of the matrix representing the
        morphism R -> G.
    matrix: np.ndarray
        Array of shape (n_gen, n_rel + n_gen) used to compute cokernels.
    """
    # __slots__ = "shape", "G_grade", "R_grade", "R_image", "dim"
    
    def __init__(self, shape, G_grade, R_grade, R_image):
        self.shape = shape
        self.G_grade = G_grade
        self.R_grade = R_grade
        self.R_image = R_image
        
        self.dim = len(shape)
        self.n_gen = G_grade.shape[0]
        self.n_rel = R_grade.shape[0]
        
        self.matrix = np.hstack((R_image, np.eye(len(G_grade), dtype=int)))
        
        self.low_of = np.zeros(self.n_gen, dtype=int)
    
    def __str__(self):
        """
        Represent the presentation as a matrix with graded rows and columns:
        
            (  dim   )    relation
            ( labels )     grades
            
               gen      [ relation ]
              grades    [  matrix  ]
        
        Returns
        -------
        output: str
            String representation of the free presentation.
        """
        dim_labels = "xyz." # each label is one character long
        n_special_labels = 3
        max_number = max(
            self.G_grade.max(initial=0),
            self.R_grade.max(initial=0),
            self.R_image.max(initial=0), 0)
        n_char = len(str(max_number))
        
        # Initialize output
        output = ""
        
        # Generate upper part
        for d in range(self.dim):
            # Insert dimension label
            line = "(" + " " * (d * (n_char + 1))
            if d < n_special_labels:
                line += f"{dim_labels[d]:>{n_char}}"
            else:
                line += f"{dim_labels[-1]:>{n_char}}"
            line += " " * ((self.dim - d - 1) * (n_char + 1))
            line += ") " # padding
            
            # Insert relation grades for the given dimension
            line += " ".join(f"{self.R_grade[i, d]:{n_char}d}" for i in range(self.n_rel))
            line += "\n"
            output += line
        
        # Generate lower part
        for i in range(self.n_gen):
            # Insert generator grades for the given generator
            line = " "
            line += " ".join(f"{self.G_grade[i, j]:{n_char}d}" for j in range(self.dim))
            # Insert the row of the relation matrix
            line += " ["
            line += " ".join(f"{self.R_image[i, j]:{n_char}d}" for j in range(self.n_rel))
            line += "]\n"
            output += line
        return output.strip()
    
    def __repr__(self):
        return f"FreePresentation({repr(self.G_grade)}, {repr(self.R_grade)}, {repr(self.R_image)})"
    
    def draw(self, ax, extra=[], aspect=.6, padding=.3, gen_color="#339c9c", rel_color="#e86a58"):
        """
        Draw the free presentation of the grid functor. If self.dim > 2, then
        only the first two dimensions are drawn.
        
        Relations in the top right corner of the grid are not drawn.
        
        Parameters
        ----------
        ax: matplotlib Axis
            Axis where the presentation is drawn.
        extra: list of pairs of float, optional
            Coordinates of extra points to be drawn. Default is the empty list.
        aspect: float, optional
            Desired aspect ratio y/x. Default is .6.
        padding: float, optional
            Relative padding above and to the right of the largest points.
            Default is .3.
        gen_color, rel_color: matplotlib color, optional
            Colors for the generators and relations. Default are #339c9c and
            #e86a58, which correspond to certain shades of turquoise and red.
        """
        # Set up plot
        x_max = max(
            self.G_grade[:, 0].max(initial=1),
            self.R_grade[:, 0].max(initial=1))
        y_max = max(
            self.G_grade[:, 1].max(initial=1),
            self.R_grade[:, 1].max(initial=1))
        
        x_lim = max(x_max, y_max / aspect) * (1 + padding)
        y_lim = max(y_max, x_max * aspect) * (1 + padding * aspect)
        k_rel = np.argmax(self.R_grade[:, 0])
        kill_x, kill_y = self.R_grade[k_rel]
        
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box", share=True)
        ax.set_xlim(-self.shape[0] * 0.1, self.shape[0] * 1.1)
        ax.set_ylim(-self.shape[1] * 0.1, self.shape[1] * 1.1)
        
        # Draw rectangles
        gen_rects = [mpatches.Rectangle(g, kill_x - g[0], kill_y - g[1]) for g in self.G_grade]
        rel_rects = [mpatches.Rectangle(g, kill_x - g[0], kill_y - g[1]) for g in self.R_grade[:k_rel]]
        
        ax.add_collection(PatchCollection(gen_rects, fc = gen_color, ec = "none", alpha=.6))
        # clear before adding relations
        ax.add_collection(PatchCollection(rel_rects, fc = "1", ec = "none", alpha=1))
        ax.add_collection(PatchCollection(rel_rects, fc = rel_color, ec = "none", alpha=.6))
        
        # Draw generators and relations
        radius = x_lim / 100
        gen_points = [mpatches.Circle(gen, radius=radius * 1.2) for gen in self.G_grade]
        rel_points = [mpatches.Circle(rel, radius=radius) for rel in self.R_grade[:k_rel]]
        extra_points = [mpatches.CirclePolygon(p, radius=1.6*radius, resolution=5) for p in extra]
        
        ax.add_collection(PatchCollection(gen_points, fc = "0", ec = "none", zorder = 2.5))
        ax.add_collection(PatchCollection(rel_points, fc = "1", ec = "none", zorder = 2.6))
        ax.add_collection(PatchCollection(extra_points, fc = "red", ec = "none", zorder = 2.4))
        
        # Draw differential
        diff_lines = []
        for i, gen in enumerate(self.G_grade):
            for j, rel in enumerate(self.R_grade[:k_rel]):
                if self.R_image[i, j]:
                    diff_lines.append([gen, rel])
        ax.add_collection(LineCollection(diff_lines, colors = "0"))
    
    
    def all_Betti_diagrams(self):
        """
        Compute all Betti diagrams relative to lower hooks for the free
        presentation.
        
        Returns
        -------
        betti_diagrams: np.ndarray
            Array of shape (n, 2 * self.dim + 2), where n is the sum of the
            sizes of the supports of the relative Betti diagrams. The first
            2 * self.dim values on each row give the coordinates of the
            generator and relation of each lower hook in the resolution, the
            penultimate value is the degree of the lower hook, and the last
            value is the multiplicity of the lower hook in that degree.
        """
        # Get presentation restricted to a sublattice
        sub_presentation, sublattice_index = self._sublattice_presentation()
        
        betti_diagrams = sub_presentation._all_Betti_diagrams_aux()
        
        # Return to real grades
        for d in range(self.dim):
            betti_diagrams[:, d] = sublattice_index[d][betti_diagrams[:, d]]
            betti_diagrams[:, self.dim + d] = sublattice_index[d][betti_diagrams[:, self.dim + d]]
        
        return betti_diagrams
    
    """
    Private methods
    """
    
    def _sublattice_presentation(self):
        """
        Return the presentation that only keeps the subgrid containing generator
        and relation grades.
        
        In order to use the cokernel trick, we also include (-1, -1) in the
        subgrid.
        
        Returns
        -------
        sublattice_presentation: FreePresentation
            Free presentation equal to the original free presentation except
            that the coordinates have been reindexed to make the grid shape as
            small as possible.
        sublattice_index: list of np.ndarray
            The i^th value of the j^th array is the true value of the
            coordinate value i in the j^th dimension.
        """
        # Get indices of sublattice
        sublattice_index = []
        sublattice_shape = []
        G_sub_grade = self.G_grade.copy()
        R_sub_grade = self.R_grade.copy()
        grades = np.concatenate((np.full((1, self.dim), -1, dtype=int), self.G_grade, self.R_grade))
        for d in range(self.dim):
            axis, indices = np.unique(grades[:, d], return_inverse=True)
            sublattice_index.append(axis)
            sublattice_shape.append(len(axis))
            G_sub_grade[:, d] = indices[1:self.n_gen + 1]
            R_sub_grade[:, d] = indices[self.n_gen + 1:]
        sublattice_presentation = FreePresentation(tuple(sublattice_shape), G_sub_grade, R_sub_grade, self.R_image)
        return sublattice_presentation, sublattice_index
    
    def _all_Betti_diagrams_aux(self):
        """
        Auxiliary function computing all of the relative Betti diagrams. This
        is called for the sublattice presentation. With the cokernel trick, we
        compute the homology of Koszul complexes of cokernels everywhere, and
        then discard the homology at any point with a 0 coordinate.
        
        Returns
        -------
        betti_diagrams
            See documentation for all_Betti_diagrams().
        """
        cokernel = np.zeros(self.shape * 2 + (self.n_gen,), dtype=bool)
        quotient = np.zeros(self.shape * 2 + (self.n_gen, self.n_gen), dtype=int)
        relevant = self._relevant_grades()
        
        betti = np.zeros(self.shape * 2 + (2 * self.dim + 1,), dtype=int)
        for index in it.product(*(range(n) for n in self.shape * 2)):
            grade = np.asarray(index)
            if ((grade[:self.dim] <= grade[self.dim:]).all()
                and (grade[:self.dim] != grade[self.dim:]).any()):
                self._coker(grade[:self.dim], grade[self.dim:], outs=(cokernel[index], quotient[index]))
                if self._check_nontrivial(index, relevant):
                    # self._coker(grade[:self.dim], grade[self.dim:], outs=(cokernel[index], quotient[index]))
                    boundary_maps = self._all_boundary_maps(cokernel, quotient, grade)
                    H = self._homology_dims(boundary_maps)
                    betti[index][:len(H)] = H
                # else:
                #     self._coker_without_quotient(grade[:self.dim], grade[self.dim:], cokernel[index])
                    # print(cokernel[index])
        
        temp = np.hstack((np.argwhere(betti), betti[betti.nonzero()].reshape(-1, 1)))
        betti_diagrams = temp[temp[:, 0] > 0]
        betti_diagrams[:, -2] -= 2
        return betti_diagrams
    
    
    def _coker(self, a, b, outs):
        """
        Compute a basis of C := coker(M(a) -> M(b)), which is a subset of the
        presentation generators, as well as the quotient map G(b)/G(a) -> C.
        We use the following snake diagram:
             
             R(a) ---------------> G(a) ->> M(a) -> 0
              |  '->> I(a) >------^ |        |
              v        |            v        v
             R(b) ---- v --------> G(b) ->> M(b) -> 0
                 '->> I(b) >------^ |        |
                       |            |        |
                       v            v        v
                   I(b)/I(a) -> G(b)/G(a) -> C -> 0
        
        Since I(a) is a subspace of G(a), the cokernel C is also the cokernel
        of the map
        
            I(b) -> G(b)/G(a).
        
        We represent G(b)/G(a) and I(b) in the basis of generators (the
        quotient G(b)/G(a) is obtained by choosing the generators with grade
        ≤ b but not ≤ a), and then reduce G(b)/G(a) by I(b) and G(b). The
        nonzero columns of the reduced matrix form a basis of C. When reducing
        by G(b), we make the same transformation of an identity matrix on a
        null matrix. Then, for each element of the basis of C, we replace the
        corresponding column with the column of the identity matrix.
        
        Parameters
        ----------
        a, b: np.ndarray of int
            Arrays of shape (self.dim,) corresponding to the source and target
            grades of the computed cokernel.
        outs: tuple
            Tuple of the form (cokernel, quotient), where:
            - cokernel is an array of shape (self.n_gen,) and type bool, where
              True means that the corresponding generator belongs to the basis
              of C, and False means that it does not;
            - quotient is a matrix of size (self.n_gen, self.n_gen) and type
              int that corresponds to the quotient map.
            The elements of this tuple are modified in place.
        """
        R_quotient = (self.R_grade <= b).all(axis=1) # new array created
        
        
        # G_col = np.any(self.G_grade > a, axis=1)
        # G_b = np.all(self.G_grade <= b, axis=1)
        # np.logical_and(G_col, G_b, out=G_col)
        G_col = np.logical_and((self.G_grade > a).any(axis=1), (self.G_grade <= b).all(axis=1))
        
        A = self.matrix[G_col, :] # copy
        B = outs[1]
        n, p = A.shape
                
        # low_of = np.full(n, -1, dtype=int)
        self.low_of[:] = -1
        for j2 in it.chain(R_quotient.nonzero()[0], G_col.nonzero()[0] + self.n_rel):
            for i in range(n - 1, -1, -1):
                if A[i, j2]: # current low
                    j1 = self.low_of[i]
                    if j1 >= 0: # found this low
                        A[:, j2] = (A[:, j2] + A[:, j1]) % 2
                        i2 = j1 - self.n_rel
                        if i2 >= 0: # update quotient map
                            B[i2, j2 - self.n_rel] = (B[i2, j2 - self.n_rel] + 1) % 2
                    else: # did not find this low
                        self.low_of[i] = j2
                        break
        
        gens = np.any(A[:, self.R_grade.shape[0]:], axis=0, out=outs[0])
        for i, b in enumerate(gens):
            if b:
                B[:, i] = 0
                B[i, i] = 1
    
    
    def _coker_without_quotient(self, a, b, out):
        R_quotient = (self.R_grade <= b).all(axis=1) # new array created
        
        G_col = np.logical_and((self.G_grade > a).any(axis=1), (self.G_grade <= b).all(axis=1))
        
        A = self.matrix[G_col, :] # copy
        n, p = A.shape
                
        low_of = np.full(n, -1, dtype=int)
        for j2 in it.chain(R_quotient.nonzero()[0], G_col.nonzero()[0] + self.n_rel):
            for i in range(n - 1, -1, -1):
                if A[i, j2]: # current low
                    j1 = low_of[i]
                    if j1 >= 0: # found this low
                        A[:, j2] = (A[:, j2] + A[:, j1]) % 2
                    else: # did not find this low
                        low_of[i] = j2
                        break
        
        np.any(A[:, self.R_grade.shape[0]:], axis=0, out=out)
    
    
    def _all_boundary_maps(self, cokernel, quotient, grade):
        """
        Compute all boundary maps for the Koszul complex at a given grade.
        
        Parameters
        ----------
        cokernel: np.ndarray
            Array of shape 2 * self.shape of cokernels, indexed by the
            source and target of the lower hooks.
        quotient: np.ndarray
            Array of shape 2 * self.shape + (self.n_gen, self.n_gen) of
            matrices representing the quotient map G(b)/G(a) -> C.
        grade: np.ndarray
            Grade at which we compute boundary maps.
        """
        shiftable = np.concatenate((grade[:self.dim] > 0, grade[:self.dim] < grade[self.dim:]))
        shift_shape = tuple(2 if b else 1 for b in shiftable)
        shift_gen = tuple((0, 1) if b else (0,) for b in shiftable)
        complex_length = np.count_nonzero(shiftable)
        
        view_slice = tuple(slice(g, None, -1) for g in grade)
        cokernel_view = cokernel[view_slice]
        quotient_view = quotient[view_slice]
        
        # pos keeps track of each cokernel's position in the boundary matrices
        pos = np.zeros(shift_shape, dtype=int)
        sizes = np.zeros(complex_length + 1, dtype=int)
        for shift in it.product(*shift_gen):
            i = sum(shift)
            sh_grade = grade - shift
            pos[shift] = sizes[i]
            sizes[i] += np.count_nonzero(cokernel_view[shift])
        
        bmaps = [np.zeros((sizes[i], sizes[i + 1]), dtype=int) for i in range(complex_length)]
        for s_i in it.product(*shift_gen):
            deg = sum(s_i)
            for d in range(2 * self.dim):
                if s_i[d]: # target exists
                    t_i = s_i[:d] + (0,) + s_i[d + 1:] # target shift
                    M = quotient_view[t_i][cokernel_view[t_i].nonzero()[0].reshape(-1, 1), cokernel_view[s_i].nonzero()[0].reshape(1, -1)]
                    # M = quotient_view[t_i][np.ix_(cokernel_view[t_i], cokernel_view[s_i])] # Why is this slower!!!
                    bmaps[deg - 1][
                        pos[t_i]:pos[t_i] + M.shape[0],
                        pos[s_i]:pos[s_i] + M.shape[1]
                    ] = M
        return bmaps
    
    
    def _homology_dims(self, boundary_maps):
        """
        Compute homology dimensions given boundary maps. Note: this function
        transforms the argument boundary_maps.
        
        Parameters
        ----------
        boundary_maps: list of np.ndarray
            List of matrices corresponding to boundary maps of a chain complex.
        
        Returns
        -------
        h_dims: list of int
            List of dimensions of the homology of the chain complex.
        """
        if len(boundary_maps) == 0:
            return []
        
        h_dims = []
        cycle_dim, boundary_dim = boundary_maps[0].shape[0], 0
        for D in boundary_maps:
            reduced_D, is_low = reduce_standard(D)
            boundary_dim = np.count_nonzero(is_low)
            h_dims.append(cycle_dim - boundary_dim)
            cycle_dim = np.count_nonzero(np.logical_not(reduced_D.any(axis=0)))
        h_dims.append(cycle_dim)
        
        return h_dims
    
    def _relevant_grades(self):
        """
        Compute the grades where the Koszul complex is not guaranteed to be
        zero. For now, we use the following reasoning: poset elements with
        no change in persistence among its ancestors are irrelevant
        
        Returns
        -------
        relevant: np.ndarray of bool
            Array of shape self.shape indicating relevant grades.
        """
        relevant = np.zeros(self.shape, dtype=bool)
        for grade in self.G_grade:
            for d in range(self.dim):
                indices = (tuple(grade[i] for i in range(d))
                    + (slice(grade[d], None),)
                    + tuple(grade[i] for i in range(d + 1, self.dim)))
                relevant[indices] = True
        for grade in self.R_grade:
            for d in range(self.dim):
                indices = (tuple(grade[i] for i in range(d))
                    + (slice(grade[d], None),)
                    + tuple(grade[i] for i in range(d + 1, self.dim)))
                relevant[indices] = True
        return relevant
    
    def _check_nontrivial(self, index, relevant):
        """
        Check if a grade and its ancestors are not all trivial.
        
        Parameters
        ----------
        index: tuple of int
            Grade to be evaluated.
        relevant: np.ndarray of bool
            Array of relevant grades.
        
        Returns
        -------
        bool
            True if any of the ancestors of index are relevant
        """
        s_shift_gen = ((0, 1) if x else (0,) for x in index[:self.dim])
        t_shift_gen = ((0, 1) if index[i] != index[self.dim + i] else (0,) for i in range(self.dim))
        
        s_slice_view = tuple(slice(g, None, -1) for g in index[:self.dim])
        t_slice_view = tuple(slice(g, None, -1) for g in index[self.dim:])
        relevant_s_view = relevant[s_slice_view]
        relevant_t_view = relevant[t_slice_view]
        s_check = False
        t_check = False
        for shift in it.product(*s_shift_gen):
            s_check = s_check or relevant_s_view[shift]
        for shift in it.product(*t_shift_gen):
            t_check = t_check or relevant_t_view[shift]
        return (s_check and t_check)


def random_2D_presentation(grid_size, seed=None, n_gen=5, n_rel=5, image_proba=.5, verbose=False):
    """
    Generate a random presentation on a square 2D grid of given size. First,
    the generators and relations are added randomly, and then coefficients
    between each generator and relation are randomly set to 1. If the relation
    grade is not greater or equal to the generator's, then the coefficient is
    set back to 0. If a relation has no nonzero coefficients, then it is
    removed.
    
    In addition, a relation is added for each generator to cancel it at the
    top right corner of the grid.
    
    Generators and relations are sorted by lexicographical order (which is a
    refinement of the product order).
    
    Parameters
    ----------
    grid_size: int
        Size of grid.
    seed: int, optional
        Seed for generating the random presentation. Default is None, in which
        case a random seed is generated and can be outputted for reuse.
    n_gen: int, optional
        Number of generators to be generated. Default value is 5.
    n_rel: int, optional
        Maximal number of relations to be generated. Default value is 5.
    image_proba: float, optional
        Probability of a nonzero coefficient between each pair of a generator
        and a relation, before checking that the relation's grade is greater or
        equal to the generator's. Default value is .5.
    verbose: bool, optional
        If True, print the parameters of the function call. In particular, the
        seed is printed. Default value is False.
    """
    if seed == None:
        random_seed = np.random.default_rng().integers(2e16)
    else:
        random_seed = seed
    if verbose:
        print(f"[{PROG}] Generating a 2D presentation on a \
{grid_size}x{grid_size} grid with {n_gen} generators, {n_rel} relations, \
{image_proba} image probability and rng seed {random_seed}")
    rng = np.random.default_rng(seed=random_seed)
    G_grade = rng.integers(grid_size, size=(n_gen, 2))
    R_grade = rng.integers(grid_size, size=(n_rel + n_gen, 2))
    R_grade[-n_gen:, :] = grid_size
    
    R_image = rng.binomial(1, image_proba, size=(n_gen, n_rel))
    R_image = np.concatenate((R_image, np.eye(n_gen, dtype=int)), axis=1)
    for i in range(n_gen):
        for j in range(n_rel):
            if (G_grade[i] > R_grade[j]).any():
                R_image[i, j] = 0
    nonzero_cols = R_image.any(axis=0)
    return FreePresentation((grid_size, grid_size), G_grade, R_grade[nonzero_cols], R_image[:, nonzero_cols])


"""
SECTION: Auxiliary functions
"""

def draw_2D_Betti_diagrams(axes, betti_diagrams, max_deg):
    """
    Draw the relative Betti diagrams for a 2D presentation.
    
    Parameters
    ----------
    axes: list of matplotlib Axis
        list of at least max_deg axes where the relative Betti diagrams are
        drawn.
    betti_diagrams: np.ndarray of int
        Array with the data of the relative Betti diagrams.
    max_deg: int
        Maximal degree to be plotted.
    """
    lines = [[] for _ in range(max_deg + 1)]
    for row in betti_diagrams:
        degree = row[4]
        multiplicity = row[5]
        if degree <= max_deg:
            lines[degree].append([row[:2], row[2:4]])
    
    for d in range(max_deg):
        axes[d + 1].add_collection(LineCollection(lines[d], colors = "0.7", linestyle = ":"))
    for d in range(max_deg + 1):
        axes[d].grid(True)
        axes[d].set_title(rf"$\beta_{d}$")
        axes[d].add_collection(LineCollection(lines[d], colors = "0"))


def reduce_standard(A):
    """
    Reduce the matrix A by the "low" algorithm using the standard row order,
    assuming that the coefficients are in Z/2Z.
    Note: A is modified by the function.
    
    Parameters
    ----------
    A: np.ndarary of int
        Array to be reduced. A is modified by the function.
    
    Returns
    -------
    A: np.ndarray of int
        The reduced matrix.
    is_low: np.ndarray of bool
        bool of rows that are low for some column.
    """
    n, p = A.shape
    
    low_of = np.full(n, -1, dtype=int)
    for j2 in range(p):
        for i in range(n - 1, -1, -1):
            if A[i, j2]: # current low
                if low_of[i] > -1: # found this low
                    A[:, j2] = (A[:, j2] + A[:, low_of[i]]) % 2
                else: # did not find this low
                    low_of[i] = j2
                    break
    is_low = (low_of > -1)
    return A, is_low


"""
SECTION: Command-line interface
"""

def run_parser():
    """
    Run the argument parser for the command-line interface.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--draw",
        action="store_true",
        help="draw the relative Betti diagrams.")
    parser.add_argument("-f", "--save-fig",
        action="store",
        help="save the relative Betti diagram to the specified file.",
        metavar="filename")
    parser.add_argument("-i", "--input",
        action="store",
        help="use free presentation from the given file. The file should be a \
text file containing the __repr__ of a FreePresentation object.",
        metavar="filename")
    parser.add_argument("-o", "--output",
        action="store",
        help="store __repr__ of the computed relative Betti diagram array.",
        metavar="filename")
    parser.add_argument("-r", "--random",
        action="store",
        nargs="*",
        default=[],
        help="generate a random free presentation with optional parameters.",
        metavar="param")
    parser.add_argument("-s", "--seed",
        action="store",
        type=int,
        help="seed for randomly generated free presentations.",
        metavar="n")
    parser.add_argument("-v", "--verbose",
        action="store_true",
        help="print more things during the computation")
    args = parser.parse_args()
    
    if args.input is None:
        params = [100, 5, 5, .5] # default grid_size, n_gen, n_rel, image_proba
        param_types = ["int", "int", "int", "float"]
        for i, arg in enumerate(args.random[:4]):
            params[i] = eval(f"{param_types[i]}({arg})")
        presentation = random_2D_presentation(params[0], seed=args.seed, n_gen=params[1], n_rel=params[2], image_proba=params[3], verbose=args.verbose)
    
    if args.input is not None:
        with open(args.input, 'r', encoding="utf-8") as f:
            presentation = eval(f.read())
    
    if args.verbose:
        print(f"[{PROG}] presentation:", presentation, sep="\n")
    
    betti_diagrams = presentation.all_Betti_diagrams()
    
    if args.verbose:
        print(f"[{PROG}] betti diagrams:", betti_diagrams, sep="\n")
    
    if args.output is not None:
        with open(args.output, 'w', encoding="utf-8") as f:
            f.write(repr(betti_diagrams))
    
    if args.draw or (args.save_fig is not None):
        max_deg = presentation.dim * 2 - 2
        matplotlib.rcParams['toolbar'] = 'None'
        fig = plt.figure(figsize=(12, 3), layout="tight")
        ax = fig.subplots(nrows=1, ncols=max_deg + 2, sharex=True, sharey=True, squeeze=True)
        presentation.draw(ax[0])
        draw_2D_Betti_diagrams(ax[1:], betti_diagrams, max_deg)
        
        if args.save_fig is not None:
            plt.savefig(args.save_fig, dpi=300)
        if args.draw:
            plt.show()


if __name__ == "__main__":
    # run_parser()
    with open("example.txt", 'r', encoding="utf-8") as f:
        presentation = eval(f.read())
    cProfile.run("presentation.all_Betti_diagrams()")
    