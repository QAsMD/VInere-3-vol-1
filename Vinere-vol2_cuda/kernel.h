#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include "flintpp.h"

#define CUDA_CALLABLE_MEMBER __device__ __host__
#define NOTHROW

//Program
cudaError_t mexpWithCuda(unsigned int size, void *m2_nl, void *lc_nl, void *d_nl, void *n_nl);

/********************************************/
//Necessary macros
#define PURGEVARS_L(X) (void)0
#define ISPURGED_L(X) (void)0
#define Assert(a) (void)0

#define CUDA_LT_L(a_l,b_l) \
    (cuda_cmp_l ((a_l), (b_l)) == -1)        /* a_l < b_l        */

#define cuda_lt_l(a_l,b_l) \
    (cuda_cmp_l ((a_l), (b_l)) == -1)        /* a_l < b_l        */


#define CUDA_LE_L(a_l,b_l) \
    (cuda_cmp_l ((a_l), (b_l)) < 1)          /* a_l <= b_l       */

#define cuda_le_l(a_l,b_l) \
    (cuda_cmp_l ((a_l), (b_l)) < 1)          /* a_l <= b_l       */


#define CUDA_GT_L(a_l,b_l) \
    (cuda_cmp_l ((a_l), (b_l)) == 1)         /* a_l > b_l        */

#define cuda_gt_l(a_l,b_l) \
    (cuda_cmp_l ((a_l), (b_l)) == 1)         /* a_l > b_l        */


#define CUDA_GE_L(a_l,b_l) \
    (cuda_cmp_l ((a_l), (b_l)) > -1)         /* a_l >= b_l       */

#define cuda_ge_l(a_l,b_l) \
    (cuda_cmp_l ((a_l), (b_l)) > -1)         /* a_l >= b_l       */


#define CUDA_GTZ_L(a_l) \
    (cuda_cmp_l ((a_l), nul_l) == 1)         /* a_l > 0          */

#define cuda_gtz_l(a_l) \
    (cuda_cmp_l ((a_l), nul_l) == 1)         /* a_l > 0          */

#define CUDA_EQZ_L(a_l) \
    (cuda_equ_l ((a_l), nul_l) == 1)         /* a_l == 0         */

#define cuda_eqz_l(a_l) \
    (cuda_equ_l ((a_l), nul_l) == 1)         /* a_l == 0         */

#define CUDA_EQONE_L(a_l) \
    (cuda_equ_l ((a_l), one_l) == 1)         /* a_l == 1         */

#define cuda_eqone_l(a_l) \
    (cuda_equ_l ((a_l), one_l) == 1)         /* a_l == 1         */

/* Set CLINT-variables to values 0, 1, 2 resp. */
#define CUDA_SETZERO_L(n_l)\
    (*(n_l) = 0)

#define cuda_setzero_l(n_l)\
    (*(n_l) = 0)

#define CUDA_SETONE_L(n_l)\
    (cuda_u2clint_l ((n_l), 1U))

#define cuda_setone_l(n_l)\
    (cuda_u2clint_l ((n_l), 1U))

#define CUDA_SETTWO_L(n_l)\
    (cuda_u2clint_l ((n_l), 2U))

#define cuda_settwo_l(n_l)\
    (cuda_u2clint_l ((n_l), 2U))


/*                                                                            */
/********************************************/

class CUDALINT
{
public:
	CUDA_CALLABLE_MEMBER CUDALINT(void);
	CUDA_CALLABLE_MEMBER CUDALINT(clint*);
	CUDA_CALLABLE_MEMBER ~CUDALINT(void);
	CUDA_CALLABLE_MEMBER const CUDALINT& operator= (const CUDALINT&);

	// LINT::Default-Error-Handler
	CUDA_CALLABLE_MEMBER static void panic(LINT_ERRORS, const char* const, const int, const int);

	// Pointer to type CLINT
	clint* n_l;

	// Status after an operation on a LINT object
	LINT_ERRORS status;
};

//CUDALINT API Declaration
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_mexp_l(CLINT bas_l, CLINT exp_l, CLINT p_l, CLINT m_l);

CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_cpy_l(CLINT dest_l, CLINT src_l);

CUDA_CALLABLE_MEMBER const CUDALINT
cuda_mexp(const CUDALINT& lr, const CUDALINT& ln, const CUDALINT& m);

CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_equ_l(CLINT a_l, CLINT b_l);

CUDA_CALLABLE_MEMBER USHORT __FLINT_API
cuda_invmon_l(CLINT n_l);

CUDA_CALLABLE_MEMBER unsigned int __FLINT_API
cuda_ld_l(CLINT n_l);

CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_mult(CLINT aa_l, CLINT bb_l, CLINT p_l);

CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_mexpkm_l(CLINT bas_l, CLINT exp_l, CLINT p_l, CLINT m_l);

CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_u2clint_l(CLINT num_l, USHORT u);

CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_setbit_l(CLINT a_l, unsigned int pos);

CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_mod_l(CLINT dv_l, CLINT ds_l, CLINT r_l);

CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_sqr(CLINT a_l, CLINT p_l);

CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_sqrmon_l(CLINT a_l, CLINT n_l, USHORT nprime, USHORT logB_r, CLINT p_l);

CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_sub_l(CLINT aa_l, CLINT bb_l, CLINT d_l);

CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_sub(CLINT a_l, CLINT b_l, CLINT d_l);

CUDA_CALLABLE_MEMBER clint * __FLINT_API
cuda_setmax_l(CLINT a_l);

CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_add_l(CLINT a_l, CLINT b_l, CLINT s_l);

CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_mmul_l(CLINT aa_l, CLINT bb_l, CLINT c_l, CLINT m_l);

CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_mulmon_l(CLINT a_l, CLINT b_l, CLINT n_l, USHORT nprime, USHORT logB_r, CLINT p_l);

CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_cmp_l(CLINT a_l, CLINT b_l);

CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_add(CLINT a_l, CLINT b_l, CLINT s_l);

CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_div_l(CLINT d1_l, CLINT d2_l, CLINT quot_l, CLINT rem_l);

CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_inc_l(CLINT a_l);