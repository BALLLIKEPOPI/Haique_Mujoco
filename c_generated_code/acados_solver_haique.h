/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_haique_H_
#define ACADOS_SOLVER_haique_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define HAIQUE_NX     13
#define HAIQUE_NZ     0
#define HAIQUE_NU     10
#define HAIQUE_NP     6
#define HAIQUE_NP_GLOBAL     0
#define HAIQUE_NBX    0
#define HAIQUE_NBX0   13
#define HAIQUE_NBU    10
#define HAIQUE_NSBX   0
#define HAIQUE_NSBU   0
#define HAIQUE_NSH    0
#define HAIQUE_NSH0   0
#define HAIQUE_NSG    0
#define HAIQUE_NSPHI  0
#define HAIQUE_NSHN   0
#define HAIQUE_NSGN   0
#define HAIQUE_NSPHIN 0
#define HAIQUE_NSPHI0 0
#define HAIQUE_NSBXN  0
#define HAIQUE_NS     0
#define HAIQUE_NS0    0
#define HAIQUE_NSN    0
#define HAIQUE_NG     0
#define HAIQUE_NBXN   0
#define HAIQUE_NGN    0
#define HAIQUE_NY0    23
#define HAIQUE_NY     23
#define HAIQUE_NYN    13
#define HAIQUE_N      30
#define HAIQUE_NH     0
#define HAIQUE_NHN    0
#define HAIQUE_NH0    0
#define HAIQUE_NPHI0  0
#define HAIQUE_NPHI   0
#define HAIQUE_NPHIN  0
#define HAIQUE_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct haique_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */

    // dynamics

    external_function_external_param_casadi *expl_vde_forw;
    external_function_external_param_casadi *expl_ode_fun;
    external_function_external_param_casadi *expl_vde_adj;




    // cost






    // constraints







} haique_solver_capsule;

ACADOS_SYMBOL_EXPORT haique_solver_capsule * haique_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int haique_acados_free_capsule(haique_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int haique_acados_create(haique_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int haique_acados_reset(haique_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of haique_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int haique_acados_create_with_discretization(haique_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int haique_acados_update_time_steps(haique_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int haique_acados_update_qp_solver_cond_N(haique_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int haique_acados_update_params(haique_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int haique_acados_update_params_sparse(haique_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int haique_acados_set_p_global_and_precompute_dependencies(haique_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int haique_acados_solve(haique_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int haique_acados_setup_qp_matrices_and_factorize(haique_solver_capsule* capsule);



ACADOS_SYMBOL_EXPORT int haique_acados_free(haique_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void haique_acados_print_stats(haique_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int haique_acados_custom_update(haique_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *haique_acados_get_nlp_in(haique_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *haique_acados_get_nlp_out(haique_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *haique_acados_get_sens_out(haique_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *haique_acados_get_nlp_solver(haique_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *haique_acados_get_nlp_config(haique_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *haique_acados_get_nlp_opts(haique_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *haique_acados_get_nlp_dims(haique_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *haique_acados_get_nlp_plan(haique_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_haique_H_
