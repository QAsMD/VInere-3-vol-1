#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include "flintpp.h"

#define CUDA_CALLABLE_MEMBER __device__ __host__
#define NOTHROW

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

/* CLINT-Constant Values */
__device__ clint __FLINT_API_DATA
nul_l[] = { 0, 0, 0, 0, 0 };
__device__ clint __FLINT_API_DATA
one_l[] = { 1, 1, 0, 0, 0 };
__device__ clint __FLINT_API_DATA
two_l[] = { 1, 2, 0, 0, 0 };

/******************************************************************************/

__device__ static int twotab[] =
{ 0, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0,
3, 0, 1, 0, 2, 0, 1, 0, 7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0,
3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0 };


__device__ static USHORT oddtab[] =
{ 0, 1, 1, 3, 1, 5, 3, 7, 1, 9, 5, 11, 3, 13, 7, 15, 1, 17, 9, 19, 5, 21, 11, 23, 3, 25, 13, 27, 7, 29, 15, 31, 1,
33, 17, 35, 9, 37, 19, 39, 5, 41, 21, 43, 11, 45, 23, 47, 3, 49, 25, 51, 13, 53, 27, 55, 7, 57, 29, 59, 15,
61, 31, 63, 1, 65, 33, 67, 17, 69, 35, 71, 9, 73, 37, 75, 19, 77, 39, 79, 5, 81, 41, 83, 21, 85, 43, 87, 11,
89, 45, 91, 23, 93, 47, 95, 3, 97, 49, 99, 25, 101, 51, 103, 13, 105, 53, 107, 27, 109, 55, 111, 7, 113,
57, 115, 29, 117, 59, 119, 15, 121, 61, 123, 31, 125, 63, 127, 1, 129, 65, 131, 33, 133, 67, 135, 17,
137, 69, 139, 35, 141, 71, 143, 9, 145, 73, 147, 37, 149, 75, 151, 19, 153, 77, 155, 39, 157, 79, 159,
5, 161, 81, 163, 41, 165, 83, 167, 21, 169, 85, 171, 43, 173, 87, 175, 11, 177, 89, 179, 45, 181, 91,
183, 23, 185, 93, 187, 47, 189, 95, 191, 3, 193, 97, 195, 49, 197, 99, 199, 25, 201, 101, 203, 51, 205,
103, 207, 13, 209, 105, 211, 53, 213, 107, 215, 27, 217, 109, 219, 55, 221, 111, 223, 7, 225, 113,
227, 57, 229, 115, 231, 29, 233, 117, 235, 59, 237, 119, 239, 15, 241, 121, 243, 61, 245, 123, 247, 31,
249, 125, 251, 63, 253, 127, 255 };


/******************************************************************************/
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