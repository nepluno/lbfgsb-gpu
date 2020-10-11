/**
 * \copyright 2012 Yun Fei
 * in collaboration with G. Rong, W. Wang and B. Wang
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#ifndef EXAMPLES_DSSCFG_DSSCFG_CUDA_H_
#define EXAMPLES_DSSCFG_DSSCFG_CUDA_H_

//     **********
//
//     Subroutine dsscfg_cuda
//
//     This subroutine computes the function and gradient of the
//     steady state combustion problem.
//
//     The subroutine statement is
//
//       subroutine dsscfg(nx,ny,x,f,fgrad,task,lambda)
//
//     where
//
//       nx is an integer variable.
//         On entry nx is the number of grid points in the first
//            coordinate direction.
//         On exit nx is unchanged.
//
//       ny is an integer variable.
//         On entry ny is the number of grid points in the second
//            coordinate direction.
//         On exit ny is unchanged.
//
//       x is a real precision array of dimension nx*ny.
//         On entry x specifies the vector x if task = 'F', 'G', or 'FG'.
//            Otherwise x need not be specified.
//         On exit x is unchanged if task = 'F', 'G', or 'FG'. Otherwise
//            x is set according to task.
//
//       f is a real precision variable.
//         On entry f need not be specified.
//         On exit f is set to the function evaluated at x if task = 'F'
//            or 'FG'.
//
//       assist_buffer is a real precision array for storing temporary
//       variables.
//         On entry assist_buffer specifies the pointer to buffer if task = 'F',
//         'G', or 'FG'.
//            Otherwise x need not be specified.
//         On exit assist_buffer contains the pointer to buffer if
//            task = 'XS'.
//
//       fgrad is a real precision array of dimension nx*ny.
//         On entry fgrad need not be specified.
//         On exit fgrad contains the gradient evaluated at x if
//            task = 'G' or 'FG'.
//
//       task is a character variable.
//         On entry task specifies the action of the subroutine:
//
//            task               action
//            ----               ------
//            'F'      Evaluate the function at x.
//            'G'      Evaluate the gradient at x.
//            'FG'     Evaluate the function and the gradient at x.
//            'XS'     Set x to the standard starting point xs.
//
//         On exit task is unchanged.
//
//       lambda is a real precision variable.
//         On entry lambda is a nonnegative Frank-Kamenetski parameter.
//         On exit lambda is unchanged.
//
//     MINPACK-2 Project. November 1993.
//     Argonne National Laboratory and University of Minnesota.
//     Brett M. Averick.
//     Adapted by Raymond Y. Fei
//
//     **********

template <typename real>
void dsscfg_cuda(int const& nx, int const& ny, real* x, real& f,
                 real* fgrad, real** assist_buffer, int task,
                 real const& lambda);

#endif
