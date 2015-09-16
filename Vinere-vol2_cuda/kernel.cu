#include "kernel.h"


//Program
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, void *m2_nl, void *lc_nl, void *d_nl, void *n_nl);

//LINT M2 = mexp(LC, potential_D[i], N);
__global__ void addKernel(int *c, const int *a, const int *b, short *sM2, short *sLC, short *sD, short *sN)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];

	/**********************************************/

	clint* m2_nl = (clint*)sM2;
	clint* lc_nl = (clint*)sLC;
	clint* d_nl = (clint*)sD;
	clint* n_nl = (clint*)sN;

	CUDALINT M2 = CUDALINT(m2_nl);
	CUDALINT LC = CUDALINT(lc_nl);
	CUDALINT D = CUDALINT(d_nl);
	CUDALINT N = CUDALINT(n_nl);
	M2 = cuda_mexp(LC, D, N);
	/**********************************************/
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

	//******************************************************//
	LINT M2, LC, D, N;
	void* m2_nl;
	void* lc_nl;
	void* d_nl;
	void* n_nl;

	M2 = nextprime(randl(64) + 1, 1);
	LC = nextprime(M2 + 1, 1);
	D = nextprime(LC + 1, 1);
	N = nextprime(D + 1, 1);

	m2_nl = M2.n_l;
	lc_nl = LC.n_l;
	d_nl = D.n_l;
	n_nl = N.n_l;
	//******************************************************//

    // Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize, m2_nl, lc_nl, d_nl, n_nl);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, void *m2_nl, void *lc_nl, void *d_nl, void *n_nl)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

	//******************************************************//
	void *m2_dev_arr = 0;
	void *lc_dev_arr = 0;
	void *d_dev_arr = 0;
	void *n_dev_arr = 0;
	//******************************************************//

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	//******************************************************//
	size_t m2_num_size = 2 * ((short)(*((short*)m2_nl))) + sizeof(short);
	size_t lc_num_size = 2 * ((short)(*((short*)lc_nl))) + sizeof(short);
	size_t d_num_size = 2 * ((short)(*((short*)d_nl))) + sizeof(short);
	size_t n_num_size = 2 * ((short)(*((short*)n_nl))) + sizeof(short);

	cudaStatus = cudaMalloc((void**)(&m2_dev_arr), m2_num_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)(&lc_dev_arr), lc_num_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)(&d_dev_arr), d_num_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)(&n_dev_arr), n_num_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(m2_dev_arr, m2_nl, m2_num_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(lc_dev_arr, lc_nl, lc_num_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(d_dev_arr, d_nl, d_num_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(n_dev_arr, n_nl, n_num_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//******************************************************//

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b, (short*)m2_dev_arr, (short*)lc_dev_arr, (short*)d_dev_arr, (short*)n_dev_arr);

    // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
	cudaFree(m2_dev_arr);
	cudaFree(lc_dev_arr);
	cudaFree(d_dev_arr);
	cudaFree(n_dev_arr);
    
    return cudaStatus;
}

///
/// CUDALINT Operators
///
// Assignment

CUDA_CALLABLE_MEMBER const CUDALINT& CUDALINT::operator= (const CUDALINT& ln)
{
	if (ln.status == E_LINT_INV) panic(E_LINT_INV, "=", 1, __LINE__);

	if (&ln != this)                      // Don't copy object to itself
	{
		cuda_cpy_l(n_l, ln.n_l);
		status = ln.status;
	}
	return *this;
} /*lint !e1539*/

///

///
/// CUDALINT Methods
///
//Default constructor
CUDA_CALLABLE_MEMBER CUDALINT::CUDALINT(void)
{
	n_l = new NOTHROW CLINT;

	if (NULL == n_l)
	{
		panic(E_LINT_NHP, "Default constructor", 0, __LINE__);
	}

	status = E_LINT_INV;
}

//nl Constructor
CUDA_CALLABLE_MEMBER CUDALINT::CUDALINT(clint* nl)
{
	//if (ln.status == E_LINT_INV) panic(E_LINT_INV, "constructor 5", 1, __LINE__);

	n_l = new NOTHROW CLINT;
	if (NULL == n_l)
	{
		panic(E_LINT_NHP, "constructor 5", 0, __LINE__);
	}
	cuda_cpy_l(n_l, nl);
	status = E_LINT_OK;
}

//Default Destructor
CUDA_CALLABLE_MEMBER CUDALINT::~CUDALINT(void)
{
	delete[] n_l;
}

//Panic method overload
CUDA_CALLABLE_MEMBER void CUDALINT::panic(LINT_ERRORS, const char* const, const int, const int)
{
	return;
}
///

///
///	CUDALINT API
///
/******************************************************************************/
/*                                                                            */
/*  Function:  Test whether two CLINT operands are equal                      */
/*  Syntax:    int equ_l (CLINT a_l, CLINT b_l);                              */
/*  Input:     CLINT a_l, b_l                                                 */
/*  Output:    -                                                              */
/*  Returns:   1 : a_l have b_l equal values                                  */
/*             0 : otherwise                                                  */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_equ_l(CLINT a_l, CLINT b_l)
{
	clint *msdptra_l, *msdptrb_l;
	int la = (int)DIGITS_L(a_l);
	int lb = (int)DIGITS_L(b_l);

	if (la == 0 && lb == 0)
	{
		return 1;
	}

	while (a_l[la] == 0 && la > 0)
	{
		--la;
	}

	while (b_l[lb] == 0 && lb > 0)
	{
		--lb;
	}

	if (la == 0 && lb == 0)
	{
		return 1;
	}

	if (la != lb)
	{
		return 0;
	}

	msdptra_l = a_l + la;
	msdptrb_l = b_l + lb;

	while ((*msdptra_l == *msdptrb_l) && (msdptra_l > a_l))
	{
		msdptra_l--;
		msdptrb_l--;
	}

	/* Purging of variables */
	PURGEVARS_L((2, sizeof(la), &la,
		sizeof(lb), &lb));

	ISPURGED_L((2, sizeof(la), &la,
		sizeof(lb), &lb));

	return (msdptra_l > a_l ? 0 : 1);
}
/******************************************************************************/
/*                                                                            */
/*  Function:  Modular exponentiation                                         */
/*             Automatic application of Montgomery exponentiation mexpkm_l    */
/*             if modulus is even, else mexpk_l is used                       */
/*  Syntax:    int mexp_l (CLINT bas_l, CLINT exp_l, CLINT p_l, CLINT m_l);   */
/*  Input:     bas_l (Base), exp_l (Exponent), m_l (Modulus)                  */
/*  Output:    p_l (Remainder of bas_l^exp_l mod m_l)                         */
/*  Returns:   E_CLINT_OK : Everything O.K.                                   */
/*             E_CLINT_DBZ: Division by Zero                                  */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_mexp_l(CLINT bas_l, CLINT exp_l, CLINT p_l, CLINT m_l)
{
	if (ISODD_L(m_l))              /* Montgomery exponentiation possible */
	{
		cuda_mexpkm_l(bas_l, exp_l, p_l, m_l);
	}
	else
	{
		return E_CLINT_DBZ;
		//return mexpk_l(bas_l, exp_l, p_l, m_l);
	}
}
/******************************************************************************/
/*					  nl      nl	                                          */
/*  Function:   Copy CLINT to CLINT                                           */
/*  Syntax:     void cpy_l (CLINT dest_l, CLINT src_l);                       */
/*  Input:      CLINT src_l                                                   */
/*  Output:     CLINT dest_l                                                  */
/*  Returns:    -                                                             */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_cpy_l(CLINT dest_l, CLINT src_l)
{
	clint *lastsrc_l = MSDPTR_L(src_l);
	*dest_l = *src_l;

	while ((*lastsrc_l == 0) && (*dest_l > 0))
	{
		--lastsrc_l;
		--*dest_l;
	}

	while (src_l < lastsrc_l)
	{
		*++dest_l = *++src_l;
	}
}
/*****************************************************************************************/
/*							      														 */
/*  Function:   mexp M2=mexp(LC,D,N)													 */
/*  Syntax:     void cuda_mexp(const CUDALINT& lr, const CUDALINT& ln, const CUDALINT& m)*/
/*  Input:      const CUDALINT& lr, const CUDALINT& ln, const CUDALINT& m                */
/*  Output:     -					                                                     */
/*  Returns:    const CUDALINT															 */
/*																						 */
/*****************************************************************************************/
CUDA_CALLABLE_MEMBER const CUDALINT
cuda_mexp(const CUDALINT& lr, const CUDALINT& ln, const CUDALINT& m)
{
	CUDALINT pot;
	int err;
	if (lr.status == E_LINT_INV) CUDALINT::panic(E_LINT_INV, "mexp", 1, __LINE__);
	if (ln.status == E_LINT_INV) CUDALINT::panic(E_LINT_INV, "mexp", 2, __LINE__);
	if (m.status == E_LINT_INV) CUDALINT::panic(E_LINT_INV, "mexp", 3, __LINE__);

	err = cuda_mexp_l(lr.n_l, ln.n_l, pot.n_l, m.n_l);

	switch (err)
	{
	case E_CLINT_OK:
		pot.status = E_LINT_OK;
		break;
	case E_CLINT_DBZ:
		CUDALINT::panic(E_LINT_DBZ, "mexp", 3, __LINE__);
		break;
	default:
		CUDALINT::panic(E_LINT_ERR, "mexp", err, __LINE__);
	}

	return pot;
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Inverse -n^(-1) mod B for odd n                                */
/*  Syntax:    USHORT invmon_l (CLINT n_l);                                   */
/*  Input:     n_l (Modulus)                                                  */
/*  Output:    -                                                              */
/*  Returns:   -n^(-1) mod B                                                  */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER USHORT __FLINT_API
cuda_invmon_l(CLINT n_l)
{
	unsigned int i;
	ULONG x = 2, y = 1;

	if (ISEVEN_L(n_l))
	{
		return (USHORT)E_CLINT_MOD;
	}

	for (i = 2; i <= BITPERDGT; i++, x <<= 1)
	{
		if (x < (((ULONG)((ULONG)(*LSDPTR_L(n_l)) * (ULONG)y)) & ((x << 1) - 1)))
		{
			y += x;
		}
	}

	return (USHORT)(x - y);
}
/******************************************************************************/
/*                                                                            */
/*  Function:  Calculate number of bits of a CLINT operand                    */
/*             (Integral part of base-2-logarithm + 1)                        */
/*  Syntax:    ld_l (n_l);                                                    */
/*  Input:     n_l (Argument)                                                 */
/*  Output:    -                                                              */
/*  Returns:   Number of relevant binary digits of n_l                        */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER unsigned int __FLINT_API
cuda_ld_l(CLINT n_l)
{
	unsigned int l;
	USHORT test;

	l = (unsigned int)DIGITS_L(n_l);
	while (n_l[l] == 0 && l > 0)
	{
		--l;
	}

	if (l == 0)
	{
		return 0;
	}

	test = n_l[l];
	l <<= LDBITPERDGT;

	while ((test & BASEDIV2) == 0)
	{
		test <<= 1;
		--l;
	}

	/* Purging of variables */
	PURGEVARS_L((1, sizeof(test), &test));
	ISPURGED_L((1, sizeof(test), &test));

	return l;
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Multiplication kernel function                                 */
/*             w/o overflow detection, w/o checking for leading zeros         */
/*             accumulator mode not supported                                 */
/*  Syntax:    void mult (CLINT aa_l, CLINT bb_l, CLINT p_l);                 */
/*  Input:     aa_l, bb_l (Factors)                                           */
/*  Output:    p_l (Product)                                                  */
/*  Returns:   -                                                              */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_mult(CLINT aa_l, CLINT bb_l, CLINT p_l) /* Allow for double length result    */
{
	register clint *cptr_l, *bptr_l;
	clint *a_l, *b_l, *aptr_l, *csptr_l, *msdptra_l, *msdptrb_l;
	USHORT av;
	ULONG carry;

	if (CUDA_EQZ_L(aa_l) || CUDA_EQZ_L(bb_l))
	{
		SETZERO_L(p_l);
		return;
	}

	if (DIGITS_L(aa_l) < DIGITS_L(bb_l))
	{
		a_l = bb_l;
		b_l = aa_l;
	}
	else
	{
		a_l = aa_l;
		b_l = bb_l;
	}

	msdptra_l = MSDPTR_L(a_l);
	msdptrb_l = MSDPTR_L(b_l);

	carry = 0;
	av = *LSDPTR_L(a_l);
	for (bptr_l = LSDPTR_L(b_l), cptr_l = LSDPTR_L(p_l); bptr_l <= msdptrb_l; bptr_l++, cptr_l++)
	{
		*cptr_l = (USHORT)(carry = (ULONG)av * (ULONG)*bptr_l +
			(ULONG)(USHORT)(carry >> BITPERDGT));
	}
	*cptr_l = (USHORT)(carry >> BITPERDGT);

	for (csptr_l = LSDPTR_L(p_l) + 1, aptr_l = LSDPTR_L(a_l) + 1; aptr_l <= msdptra_l; csptr_l++, aptr_l++)
	{
		carry = 0;
		av = *aptr_l;
		for (bptr_l = LSDPTR_L(b_l), cptr_l = csptr_l; bptr_l <= msdptrb_l; bptr_l++, cptr_l++)
		{
			*cptr_l = (USHORT)(carry = (ULONG)av * (ULONG)*bptr_l +
				(ULONG)*cptr_l + (ULONG)(USHORT)(carry >> BITPERDGT));
		}
		*cptr_l = (USHORT)(carry >> BITPERDGT);
	}

	SETDIGITS_L(p_l, DIGITS_L(a_l) + DIGITS_L(b_l));
	RMLDZRS_L(p_l);

	/* Purging of variables */
	PURGEVARS_L((2, sizeof(carry), &carry,
		sizeof(av), &av));

	ISPURGED_L((2, sizeof(carry), &carry,
		sizeof(av), &av));
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Modular Exponentiation for odd moduli (Montgomery reduction)   */
/*  Syntax:    int mexpkm_l (CLINT bas_l, CLINT exp_l, CLINT p_l, CLINT m_l); */
/*  Input:     bas_l (Base), exp_l (Exponent), m_l (Modulus )                 */
/*  Output:    p_l (Remainder of bas_l ^ exp_l mod m_l)                       */
/*  Returns:   E_CLINT_OK : Everything O.K.                                   */
/*             E_CLINT_DBZ: Division by Zero                                  */
/*             E_CLINT_MAL: Error with malloc()                               */
/*             E_CLINT_MOD: Modulus even                                      */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_mexpkm_l(CLINT bas_l, CLINT exp_l, CLINT p_l, CLINT m_l)
{
	CLINT a_l, a2_l, md_l;
	clint e_l[CLINTMAXSHORT + 1];
	clint r_l[CLINTMAXSHORT + 1];
	CLINTD acc_l;
	clint **aptr_l, *ptr_l = NULL;
	int noofdigits, s, t, i;
	unsigned int k, lge, bit, digit, fk, word, pow2k, k_mask;
	USHORT logB_r, mprime;

	if (CUDA_EQZ_L(m_l))
	{
		return E_CLINT_DBZ;          /* Division by Zero */
	}

	if (ISEVEN_L(m_l))
	{
		return E_CLINT_MOD;          /* Modulus even */
	}

	if (CUDA_EQONE_L(m_l))
	{
		CUDA_SETZERO_L(p_l);             /* Modulus = 1 ==> Remainder = 0 */
		return E_CLINT_OK;
	}

	cuda_cpy_l(a_l, bas_l);
	cuda_cpy_l(e_l, exp_l);
	cuda_cpy_l(md_l, m_l);

	if (DIGITS_L(e_l) == 0)
	{
		CUDA_SETONE_L(p_l);
		PURGEVARS_L((2, sizeof(a_l), a_l,
			sizeof(md_l), md_l));
		ISPURGED_L((2, sizeof(a_l), a_l,
			sizeof(md_l), md_l));
		return E_CLINT_OK;
	}

	if (DIGITS_L(a_l) == 0)
	{
		CUDA_SETZERO_L(p_l);

		PURGEVARS_L((2, sizeof(e_l), e_l,
			sizeof(md_l), md_l));
		ISPURGED_L((2, sizeof(e_l), e_l,
			sizeof(md_l), md_l));
		return E_CLINT_OK;
	}

	lge = cuda_ld_l(e_l);

	k = 8;

	while (k > 1 && ((k - 1) * (k << ((k - 1) << 1)) / ((1 << k) - k - 1)) >= lge - 1)
	{
		--k;
	}

	pow2k = 1U << k;                 /*lint !e644 */

	k_mask = pow2k - 1;

	if ((aptr_l = (clint **)malloc(sizeof(clint *) * pow2k)) == NULL)
	{
		PURGEVARS_L((3, sizeof(a_l), a_l,
			sizeof(e_l), e_l,
			sizeof(md_l), md_l));
		ISPURGED_L((3, sizeof(a_l), a_l,
			sizeof(e_l), e_l,
			sizeof(md_l), md_l));
		return E_CLINT_MAL;
	}

	aptr_l[1] = a_l;
	CUDA_SETZERO_L(r_l);
	logB_r = DIGITS_L(md_l);
	cuda_setbit_l(r_l, logB_r << LDBITPERDGT);
	if (DIGITS_L(r_l) > CLINTMAXDIGIT)
	{
		cuda_mod_l(r_l, md_l, r_l);
	}

	mprime = cuda_invmon_l(md_l);

	cuda_mmul_l(a_l, r_l, a_l, md_l);

	if (k > 1)
	{
		if ((ptr_l = (clint *)malloc(sizeof(CLINT) * ((pow2k >> 1) - 1))) == NULL)
		{
			free(aptr_l);
			PURGEVARS_L((3, sizeof(a_l), a_l,
				sizeof(e_l), e_l,
				sizeof(md_l), md_l));
			ISPURGED_L((3, sizeof(a_l), a_l,
				sizeof(e_l), e_l,
				sizeof(md_l), md_l));
			return E_CLINT_MAL;
		}

		aptr_l[2] = a2_l;
		cuda_sqrmon_l(a_l, md_l, mprime, logB_r, aptr_l[2]);

		for (aptr_l[3] = ptr_l, i = 5; i < (int)pow2k; i += 2)
		{
			aptr_l[i] = aptr_l[i - 2] + CLINTMAXSHORT;   /*lint !e661 !e662 */
		}

		for (i = 3; i < (int)pow2k; i += 2)
		{
			cuda_mulmon_l(aptr_l[2], aptr_l[i - 2], md_l, mprime, logB_r, aptr_l[i]);
		}
	}

	*(MSDPTR_L(e_l) + 1) = 0;     /* 0 follows most significant digit of e_l */

	noofdigits = (lge - 1) / k;                                    /*lint !e713 */
	fk = noofdigits * k;                /* >>loss of precision<< not critical */

	word = (unsigned int)(fk >> LDBITPERDGT);        /* fk div 16 */
	bit = (unsigned int)(fk & (BITPERDGT - 1UL));    /* fk mod 16 */

	switch (k)
	{
	case 1:
	case 2:
	case 4:
	case 8:
		digit = ((ULONG)(e_l[word + 1]) >> bit) & k_mask;
		break;
	default:
		digit = ((ULONG)(e_l[word + 1] | ((ULONG)e_l[word + 2]
			<< BITPERDGT)) >> bit) & k_mask;
	}

	if (digit != 0)                  /* k-digit > 0 */
	{
		cuda_cpy_l(acc_l, aptr_l[oddtab[digit]]);

		t = twotab[digit];
		for (; t > 0; t--)
		{
			cuda_sqrmon_l(acc_l, md_l, mprime, logB_r, acc_l);
		}
	}
	else
	{
		cuda_mod_l(r_l, md_l, acc_l);
	}

	for (noofdigits--, fk -= k; noofdigits >= 0; noofdigits--, fk -= k)
	{
		word = (unsigned int)fk >> LDBITPERDGT;       /* fk div 16 */
		bit = (unsigned int)fk & (BITPERDGT - 1UL);   /* fk mod 16 */

		switch (k)
		{
		case 1:
		case 2:
		case 4:
		case 8:
			digit = ((ULONG)(e_l[word + 1]) >> bit) & k_mask;
			break;
		default:
			digit = ((ULONG)(e_l[word + 1] | ((ULONG)e_l[word + 2]
				<< BITPERDGT)) >> bit) & k_mask;
		}

		if (digit != 0)              /* k-digit > 0 */
		{
			t = twotab[digit];

			for (s = (int)(k - t); s > 0; s--)
			{
				cuda_sqrmon_l(acc_l, md_l, mprime, logB_r, acc_l);
			}

			cuda_mulmon_l(acc_l, aptr_l[oddtab[digit]], md_l, mprime, logB_r, acc_l);

			for (; t > 0; t--)
			{
				cuda_sqrmon_l(acc_l, md_l, mprime, logB_r, acc_l);
			}
		}
		else                         /* k-digit == 0 */
		{
			for (s = (int)k; s > 0; s--)
			{
				cuda_sqrmon_l(acc_l, md_l, mprime, logB_r, acc_l);
			}
		}
	}

	cuda_mulmon_l(acc_l, one_l, md_l, mprime, logB_r, p_l);

	free(aptr_l);
	if (ptr_l != NULL)
	{
		free(ptr_l);                /*lint !e644 */
	}

	/* Purging of variables */
	PURGEVARS_L((14, sizeof(i), &i,
		sizeof(noofdigits), &noofdigits,
		sizeof(s), &s,
		sizeof(t), &t,
		sizeof(bit), &bit,
		sizeof(digit), &digit,
		sizeof(k), &k,
		sizeof(lge), &lge,
		sizeof(fk), &fk,
		sizeof(word), &word,
		sizeof(pow2k), &pow2k,
		sizeof(k_mask), &k_mask,
		sizeof(logB_r), &logB_r,
		sizeof(mprime), &mprime));
	PURGEVARS_L((6, sizeof(a_l), a_l,
		sizeof(a2_l), a2_l,
		sizeof(e_l), e_l,
		sizeof(r_l), r_l,
		sizeof(acc_l), acc_l,
		sizeof(md_l), md_l));

	ISPURGED_L((20, sizeof(i), &i,
		sizeof(noofdigits), &noofdigits,
		sizeof(s), &s,
		sizeof(t), &t,
		sizeof(bit), &bit,
		sizeof(digit), &digit,
		sizeof(k), &k,
		sizeof(lge), &lge,
		sizeof(fk), &fk,
		sizeof(word), &word,
		sizeof(pow2k), &pow2k,
		sizeof(k_mask), &k_mask,
		sizeof(logB_r), &logB_r,
		sizeof(mprime), &mprime,
		sizeof(a_l), a_l,
		sizeof(a2_l), a2_l,
		sizeof(e_l), e_l,
		sizeof(r_l), r_l,
		sizeof(acc_l), acc_l,
		sizeof(md_l), md_l));

	return E_CLINT_OK;
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Conversion of an USHORT value to CLINT format                  */
/*  Syntax:    void u2clint_l (CLINT num_l, USHORT u);                        */
/*  Input:     u (Value to be converted)                                      */
/*  Output:    num_l (CLINT variable with value u)                            */
/*  Returns:   -                                                              */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_u2clint_l(CLINT num_l, USHORT u)
{
	*LSDPTR_L(num_l) = u;
	SETDIGITS_L(num_l, 1);
	RMLDZRS_L(num_l);
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Testing and setting of a single bit                            */
/*  Syntax:    int setbit_l (CLINT a_l, unsigned int pos);                    */
/*  Input:     a_l (Argument),                                                */
/*             pos (Position of the bit to be set in a_l, leftmost position   */
/*             is 0)                                                          */
/*  Output:    a_l, bit in position pos set to 1                              */
/*  Returns:   1: bit in position pos had value 1 before it was set           */
/*             0: else                                                        */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_setbit_l(CLINT a_l, unsigned int pos)
{
	int res = 0;
	unsigned int i;
	USHORT shortpos = (USHORT)(pos >> LDBITPERDGT);
	USHORT bitpos = (USHORT)(pos & (BITPERDGT - 1));
	USHORT m = (USHORT)(1U << bitpos);

	if (pos > CLINTMAXBIT)
	{
		return E_CLINT_OFL;
	}

	if (shortpos >= DIGITS_L(a_l))
	{
		/* Fill up with 0 to the requested bitposition */
		for (i = DIGITS_L(a_l) + 1; i <= shortpos + 1U; i++)
		{
			a_l[i] = 0;
		}

		/* Set new length */
		SETDIGITS_L(a_l, shortpos + 1);
	}

	/* Test bit */
	if (a_l[shortpos + 1] & m)
	{
		res = 1;
	}

	/* Set bit */
	a_l[shortpos + 1] |= m;

	/* Purging of variables */
	PURGEVARS_L((4, sizeof(i), &i,
		sizeof(shortpos), &shortpos,
		sizeof(bitpos), &bitpos,
		sizeof(m), &m));

	ISPURGED_L((4, sizeof(i), &i,
		sizeof(shortpos), &shortpos,
		sizeof(bitpos), &bitpos,
		sizeof(m), &m));

	return res;
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Reduction modulo m                                             */
/*  Syntax:    int mod_l (CLINT dv_l, CLINT ds_l, CLINT r_l);                 */
/*  Input:     dv_l (Dividend), ds_l (Divisor)                                */
/*  Output:    r_l (Remainder of dv_l mod ds_l)                               */
/*  Returns:   E_CLINT_OK : Everything O.K.                                   */
/*             E_CLINT_DBZ: Division by Zero                                  */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_mod_l(CLINT dv_l, CLINT ds_l, CLINT r_l)
{
	CLINTD junk_l;
	int err;

	err = cuda_div_l(dv_l, ds_l, junk_l, r_l);

	/* Purging of variables */
	PURGEVARS_L((1, sizeof(junk_l), junk_l));
	ISPURGED_L((1, sizeof(junk_l), junk_l));

	return err;
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Squaring kernel function                                       */
/*             w/o overflow detection, w/o checking for leading zeros         */
/*             accumulator mode not supported                                 */
/*  Syntax:    void sqr (CLINT a_l, CLINT r_l);                               */
/*  Input:     a_l (Factor)                                                   */
/*  Output:    p_l (Square)                                                   */
/*  Returns:   -                                                              */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_sqr(CLINT a_l, CLINT p_l)              /* Allow for double length result     */
{
	register clint *cptr_l, *bptr_l;
	clint *aptr_l, *csptr_l, *msdptra_l, *msdptrb_l, *msdptrc_l;
	USHORT av;
	ULONG carry;

	if (CUDA_EQZ_L(a_l))
	{
		CUDA_SETZERO_L(p_l);
		return;
	}

	msdptrb_l = MSDPTR_L(a_l);
	msdptra_l = msdptrb_l - 1;
	*LSDPTR_L(p_l) = 0;
	carry = 0;
	av = *LSDPTR_L(a_l);
	for (bptr_l = LSDPTR_L(a_l) + 1, cptr_l = LSDPTR_L(p_l) + 1; bptr_l <= msdptrb_l; bptr_l++, cptr_l++)
	{
		*cptr_l = (USHORT)(carry = (ULONG)av * (ULONG)*bptr_l +
			(ULONG)(USHORT)(carry >> BITPERDGT));
	}
	*cptr_l = (USHORT)(carry >> BITPERDGT);

	for (aptr_l = LSDPTR_L(a_l) + 1, csptr_l = LSDPTR_L(p_l) + 3; aptr_l <= msdptra_l; aptr_l++, csptr_l += 2)
	{
		carry = 0;
		av = *aptr_l;
		for (bptr_l = aptr_l + 1, cptr_l = csptr_l; bptr_l <= msdptrb_l; bptr_l++, cptr_l++)
		{
			*cptr_l = (USHORT)(carry = (ULONG)av * (ULONG)*bptr_l +
				(ULONG)*cptr_l + (ULONG)(USHORT)(carry >> BITPERDGT));
		}
		*cptr_l = (USHORT)(carry >> BITPERDGT);
	}

	msdptrc_l = cptr_l;
	carry = 0;
	for (cptr_l = LSDPTR_L(p_l); cptr_l <= msdptrc_l; cptr_l++)
	{
		*cptr_l = (USHORT)(carry = (((ULONG)*cptr_l) << 1) +
			(ULONG)(USHORT)(carry >> BITPERDGT));
	}
	*cptr_l = (USHORT)(carry >> BITPERDGT);

	carry = 0;
	for (bptr_l = LSDPTR_L(a_l), cptr_l = LSDPTR_L(p_l); bptr_l <= msdptrb_l; bptr_l++, cptr_l++)
	{
		*cptr_l = (USHORT)(carry = (ULONG)*bptr_l * (ULONG)*bptr_l +
			(ULONG)*cptr_l + (ULONG)(USHORT)(carry >> BITPERDGT));
		cptr_l++;
		*cptr_l = (USHORT)(carry = (ULONG)*cptr_l + (carry >> BITPERDGT));
	}

	SETDIGITS_L(p_l, DIGITS_L(a_l) << 1);
	RMLDZRS_L(p_l);

	/* Purging of variables */
	PURGEVARS_L((2, sizeof(carry), &carry,
		sizeof(av), &av));

	ISPURGED_L((2, sizeof(carry), &carry,
		sizeof(av), &av));
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Montgomery squaring                                            */
/*  Syntax:    void sqrmon_l (CLINT a_l, CLINT n_l, USHORT nprime,            */
/*                                                 USHORT logB_r, CLINT p_l); */
/*  Input:     a_l (factor),  n_l (Modulus, odd)                              */
/*             nprime (n' mod B),                                             */
/*             logB_r (Integral Part of Logarithm of r to base B)             */
/*             (For an explanation of the operands cf. Chap. 6)               */
/*  Output:    p_l (Remainder a_l * a_l * r^(-1) mod n_l)                     */
/*             with r := B^logB_r, B^(logB_r-1) <= n_l < B^logB_r)            */
/*  Returns:   -                                                              */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_sqrmon_l(CLINT a_l, CLINT n_l, USHORT nprime, USHORT logB_r, CLINT p_l)
{
	clint t_l[2 + (CLINTMAXDIGIT << 1)];
	clint *tptr_l, *nptr_l, *tiptr_l, *lasttnptr, *lastnptr;
	ULONG carry;
	USHORT mi;
	int i;

	cuda_sqr(a_l, t_l);

	lasttnptr = t_l + DIGITS_L(n_l);
	lastnptr = MSDPTR_L(n_l);

	for (i = (int)DIGITS_L(t_l) + 1; i <= (int)(DIGITS_L(n_l) << 1); i++)
	{
		t_l[i] = 0;
	}

	SETDIGITS_L(t_l, MAX(DIGITS_L(t_l), DIGITS_L(n_l) << 1));

	for (tptr_l = LSDPTR_L(t_l); tptr_l <= lasttnptr; tptr_l++)
	{
		carry = 0;
		mi = (USHORT)((ULONG)nprime * (ULONG)*tptr_l);
		for (nptr_l = LSDPTR_L(n_l), tiptr_l = tptr_l; nptr_l <= lastnptr; nptr_l++, tiptr_l++)
		{
			Assert(tiptr_l <= t_l + (CLINTMAXDIGIT << 1));
			*tiptr_l = (USHORT)(carry = (ULONG)mi * (ULONG)*nptr_l +
				(ULONG)*tiptr_l + (ULONG)(USHORT)(carry >> BITPERDGT));
		}

		for (; ((carry >> BITPERDGT) > 0) && tiptr_l <= MSDPTR_L(t_l); tiptr_l++)
		{
			Assert(tiptr_l <= t_l + (CLINTMAXDIGIT << 1));
			*tiptr_l = (USHORT)(carry = (ULONG)*tiptr_l + (ULONG)(USHORT)(carry >> BITPERDGT));
		}

		if (((carry >> BITPERDGT) > 0) && tiptr_l > MSDPTR_L(t_l))
		{
			Assert(tiptr_l <= t_l + 1 + (CLINTMAXDIGIT << 1));
			*tiptr_l = (USHORT)(carry >> BITPERDGT);
			INCDIGITS_L(t_l);
		}
	}

	tptr_l = t_l + logB_r;
	SETDIGITS_L(tptr_l, DIGITS_L(t_l) - logB_r);

	if (CUDA_GE_L(tptr_l, n_l))
	{
		cuda_sub_l(tptr_l, n_l, p_l);
	}
	else
	{
		cuda_cpy_l(p_l, tptr_l);
	}

	Assert(DIGITS_L(p_l) <= CLINTMAXDIGIT);

	/* Purging of variables */
	PURGEVARS_L((3, sizeof(mi), &mi,
		sizeof(carry), &carry,
		sizeof(t_l), t_l));

	ISPURGED_L((3, sizeof(mi), &mi,
		sizeof(carry), &carry,
		sizeof(t_l), t_l));
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Subtraction of one CLINT operand from another                  */
/*  Syntax:    int sub_l (CLINT aa_l, CLINT bb_l, CLINT d_l);                 */
/*  Input:     aa_l, bb_l (Operands)                                          */
/*  Output:    d_l (Value of a_l - b_l)                                       */
/*  Returns:   E_CLINT_OK : Everything O.K.                                   */
/*             E_CLINT_UFL: Underflow                                         */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_sub_l(CLINT aa_l, CLINT bb_l, CLINT d_l)
{
	CLINT b_l;
	clint a_l[CLINTMAXSHORT + 1], t_l[CLINTMAXSHORT + 1], tmp_l[CLINTMAXSHORT + 1];
	int UFL = 0;

	cuda_cpy_l(b_l, bb_l);

	if (CUDA_LT_L(aa_l, b_l))            /* Underflow ? */
	{
		cuda_setmax_l(a_l);              /* We calculate with Nmax             */
		cuda_cpy_l(t_l, aa_l);           /* aa_l will be needed once again, ...*/
		UFL = E_CLINT_UFL;           /*  ... will be corrected at the end  */
	}
	else
	{
		cuda_cpy_l(a_l, aa_l);
	}

	cuda_sub(a_l, b_l, tmp_l);

	if (UFL)
	{                              /* Underflow ? */
		cuda_add_l(tmp_l, t_l, d_l);     /* Correction needed */
		cuda_inc_l(d_l);                 /* One is missing */
	}
	else
	{
		cuda_cpy_l(d_l, tmp_l);
	}

	/* Purging of variables */
	PURGEVARS_L((4, sizeof(a_l), a_l,
		sizeof(b_l), b_l,
		sizeof(t_l), t_l,
		sizeof(tmp_l), tmp_l));

	ISPURGED_L((4, sizeof(a_l), a_l,
		sizeof(b_l), b_l,
		sizeof(t_l), t_l,
		sizeof(tmp_l), tmp_l));

	return UFL;
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Subtraction kernel function                                    */
/*             w/o overflow detection, w/o checking for leading zeros         */
/*  Syntax:    void sub (CLINT a_l, CLINT b_l, CLINT d_l);                    */
/*  Input:     a_l, b_l (Operands)                                            */
/*  Output:    d_l (Difference)                                               */
/*  Returns:   -                                                              */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_sub(CLINT a_l, CLINT b_l, CLINT d_l)
{
	clint *msdptra_l, *msdptrb_l;
	clint *aptr_l = LSDPTR_L(a_l), *bptr_l = LSDPTR_L(b_l), *dptr_l = LSDPTR_L(d_l);
	ULONG carry = 0L;

	msdptra_l = MSDPTR_L(a_l);
	msdptrb_l = MSDPTR_L(b_l);

	SETDIGITS_L(d_l, DIGITS_L(a_l));

	while (bptr_l <= msdptrb_l)
	{
		*dptr_l++ = (USHORT)(carry = (ULONG)*aptr_l++ - (ULONG)*bptr_l++
			- ((carry & BASE) >> BITPERDGT));
	}

	while (aptr_l <= msdptra_l)
	{
		*dptr_l++ = (USHORT)(carry = (ULONG)*aptr_l++
			- ((carry & BASE) >> BITPERDGT));
	}

	RMLDZRS_L(d_l);

	/* Purging of variables */
	PURGEVARS_L((1, sizeof(carry), &carry));
	ISPURGED_L((1, sizeof(carry), &carry));
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Generation of maximum CLINT value 2^CLINTMAXBIT - 1            */
/*  Syntax:    clint * setmax_l (CLINT a_l);                                  */
/*  Input:     a_l CLINT variable                                             */
/*  Output:    a_l set to value of 2^CLINTMAXBIT - 1 = Nmax                   */
/*  Returns:   Address of CLINT variable a_l                                  */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER clint * __FLINT_API
cuda_setmax_l(CLINT a_l)
{
	clint *aptr_l = a_l;
	clint *msdptra_l = a_l + CLINTMAXDIGIT;

	while (++aptr_l <= msdptra_l)
	{
		*aptr_l = BASEMINONE;
	}

	SETDIGITS_L(a_l, CLINTMAXDIGIT);
	return (clint *)a_l;
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Addition of two CLINT operands                                 */
/*  Syntax:    int add_l (CLINT a_l, CLINT b_l, CLINT s_l);                   */
/*  Input:     a_l, b_l (Operands)                                            */
/*  Output:    s_l (Sum)                                                      */
/*  Returns:   E_CLINT_OK : Everything O.K.                                   */
/*             E_CLINT_OFL: Overflow                                          */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_add_l(CLINT a_l, CLINT b_l, CLINT s_l)
{
	clint ss_l[CLINTMAXSHORT + 1];
	int OFL = 0;

	cuda_add(a_l, b_l, ss_l);

	if (DIGITS_L(ss_l) > (USHORT)CLINTMAXDIGIT)       /* Overflow ? */
	{
		ANDMAX_L(ss_l);                  /* Reduction modulo Nmax+1 */
		OFL = E_CLINT_OFL;
	}

	cuda_cpy_l(s_l, ss_l);

	/* Purging of variables */
	PURGEVARS_L((1, sizeof(s_l), ss_l));
	ISPURGED_L((1, sizeof(s_l), ss_l));

	return OFL;
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Modular Multiplication                                         */
/*  Syntax:    int mmul_l (CLINT aa_l, CLINT bb_l, CLINT c_l, CLINT m_l);     */
/*  Input:     aa_l, bb_l, m_l (Operands)                                     */
/*  Output:    c_l (Remainder of aa_l * bb_l mod m_l)                         */
/*  Returns:   E_CLINT_OK : Everything O.K.                                   */
/*             E_CLINT_DBZ: Division by Zero                                  */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_mmul_l(CLINT aa_l, CLINT bb_l, CLINT c_l, CLINT m_l)
{
	CLINT a_l, b_l;
	CLINTD tmp_l;

	if (CUDA_EQZ_L(m_l))
	{
		return E_CLINT_DBZ;          /* Division by Zero */
	}

	cuda_cpy_l(a_l, aa_l);
	cuda_cpy_l(b_l, bb_l);

	cuda_mult(a_l, b_l, tmp_l);
	cuda_mod_l(tmp_l, m_l, c_l);

	/* Purging of variables */
	PURGEVARS_L((3, sizeof(a_l), a_l,
		sizeof(b_l), b_l,
		sizeof(tmp_l), tmp_l));

	ISPURGED_L((3, sizeof(a_l), a_l,
		sizeof(b_l), b_l,
		sizeof(tmp_l), tmp_l));

	return E_CLINT_OK;
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Montgomery multiplication                                      */
/*  Syntax:    void mulmon_l (CLINT a_l, CLINT b_l, CLINT n_l, USHORT nprime, */
/*                                                 USHORT logB_r, CLINT p_l); */
/*  Input:     a_l, b_l (Factors)                                             */
/*             n_l (Modulus, odd, n_l > a_l, b_l)                             */
/*             nprime (-n_l^(-1) mod B)                                       */
/*             logB_r (Integral part of logarithm of r to base B)             */
/*             (For an explanation of the operands cf. Chap. 6)               */
/*  Output:    p_l (Remainder of a_l * b_l * r^(-1) mod n_l)                  */
/*             with r := B^logB_r, B^(logB_r-1) <= n_l < B^logB_r)            */
/*  Returns:   -                                                              */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_mulmon_l(CLINT a_l, CLINT b_l, CLINT n_l, USHORT nprime, USHORT logB_r, CLINT p_l)
{
	clint t_l[2 + (CLINTMAXDIGIT << 1)];
	clint *tptr_l, *nptr_l, *tiptr_l, *lasttnptr, *lastnptr;
	ULONG carry;
	USHORT mi;
	int i;

	cuda_mult(a_l, b_l, t_l);
	Assert(DIGITS_L(t_l) <= (1 + (CLINTMAXDIGIT << 1)));

	lasttnptr = t_l + DIGITS_L(n_l);
	lastnptr = MSDPTR_L(n_l);

	for (i = (int)DIGITS_L(t_l) + 1; i <= (int)(DIGITS_L(n_l) << 1); i++)
	{
		Assert(i < sizeof(t_l));
		t_l[i] = 0;
	}

	SETDIGITS_L(t_l, MAX(DIGITS_L(t_l), DIGITS_L(n_l) << 1));

	Assert(DIGITS_L(t_l) <= (CLINTMAXDIGIT << 1));

	for (tptr_l = LSDPTR_L(t_l); tptr_l <= lasttnptr; tptr_l++)
	{
		carry = 0;
		mi = (USHORT)((ULONG)nprime * (ULONG)*tptr_l);
		for (nptr_l = LSDPTR_L(n_l), tiptr_l = tptr_l; nptr_l <= lastnptr; nptr_l++, tiptr_l++)
		{
			Assert(tiptr_l <= t_l + (CLINTMAXDIGIT << 1));
			*tiptr_l = (USHORT)(carry = (ULONG)mi * (ULONG)*nptr_l +
				(ULONG)*tiptr_l + (ULONG)(USHORT)(carry >> BITPERDGT));
		}

		for (; ((carry >> BITPERDGT) > 0) && tiptr_l <= MSDPTR_L(t_l); tiptr_l++)
		{
			Assert(tiptr_l <= t_l + (CLINTMAXDIGIT << 1));
			*tiptr_l = (USHORT)(carry = (ULONG)*tiptr_l + (ULONG)(USHORT)(carry >> BITPERDGT));
		}

		if (((carry >> BITPERDGT) > 0))
		{
			Assert(tiptr_l <= t_l + 1 + (CLINTMAXDIGIT << 1));
			*tiptr_l = (USHORT)(carry >> BITPERDGT);
			INCDIGITS_L(t_l);
		}
	}

	tptr_l = t_l + logB_r;
	SETDIGITS_L(tptr_l, DIGITS_L(t_l) - logB_r);
	Assert(DIGITS_L(tptr_l) <= (CLINTMAXDIGIT + 1));

	if (CUDA_GE_L(tptr_l, n_l))
	{
		cuda_sub_l(tptr_l, n_l, p_l);
	}
	else
	{
		cuda_cpy_l(p_l, tptr_l);
	}

	Assert(DIGITS_L(p_l) <= CLINTMAXDIGIT);

	/* Purging of variables */
	PURGEVARS_L((3, sizeof(mi), &mi,
		sizeof(carry), &carry,
		sizeof(t_l), t_l));

	ISPURGED_L((3, sizeof(mi), &mi,
		sizeof(carry), &carry,
		sizeof(t_l), t_l));
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Comparison of two CLINT operands                               */
/*  Syntax:    int cmp_l (CLINT a_l, CLINT b_l);                              */
/*  Input:     CLINT a_l, b_l (Values to compare)                             */
/*  Output:    -                                                              */
/*  Returns:   -1: a_l < b_l,                                                 */
/*              0: a_l == b_l,                                                */
/*              1: a_l > b_l                                                  */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_cmp_l(CLINT a_l, CLINT b_l)
{
	clint *msdptra_l, *msdptrb_l;
	int la = (int)DIGITS_L(a_l);
	int lb = (int)DIGITS_L(b_l);

	if (la == 0 && lb == 0)
	{
		return 0;
	}

	while (a_l[la] == 0 && la > 0)
	{
		--la;
	}

	while (b_l[lb] == 0 && lb > 0)
	{
		--lb;
	}

	if (la == 0 && lb == 0)
	{
		return 0;
	}

	if (la > lb)
	{
		PURGEVARS_L((2, sizeof(la), &la,
			sizeof(lb), &lb));
		ISPURGED_L((2, sizeof(la), &la,
			sizeof(lb), &lb));
		return 1;
	}

	if (la < lb)
	{
		PURGEVARS_L((2, sizeof(la), &la,
			sizeof(lb), &lb));
		ISPURGED_L((2, sizeof(la), &la,
			sizeof(lb), &lb));
		return -1;
	}

	msdptra_l = a_l + la;
	msdptrb_l = b_l + lb;

	while ((*msdptra_l == *msdptrb_l) && (msdptra_l > a_l))
	{
		msdptra_l--;
		msdptrb_l--;
	}

	PURGEVARS_L((2, sizeof(la), &la,
		sizeof(lb), &lb));
	ISPURGED_L((2, sizeof(la), &la,
		sizeof(lb), &lb));

	if (msdptra_l == a_l)
	{
		return 0;
	}

	if (*msdptra_l > *msdptrb_l)
	{
		return 1;
	}
	else
	{
		return -1;
	}
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Addition kernel function                                       */
/*             w/o overflow detection, w/o checking for leading zeros         */
/*  Syntax:    void add (CLINT a_l, CLINT b_l, CLINT s_l);                    */
/*  Input:     a_l, b_l (Operands)                                            */
/*  Output:    s_l (Sum)                                                      */
/*  Returns:   -                                                              */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER void __FLINT_API
cuda_add(CLINT a_l, CLINT b_l, CLINT s_l)
{
	clint *msdptra_l, *msdptrb_l;
	clint *aptr_l, *bptr_l, *sptr_l = LSDPTR_L(s_l);
	ULONG carry = 0L;

	if (DIGITS_L(a_l) < DIGITS_L(b_l))
	{
		aptr_l = LSDPTR_L(b_l);
		bptr_l = LSDPTR_L(a_l);
		msdptra_l = MSDPTR_L(b_l);
		msdptrb_l = MSDPTR_L(a_l);
		SETDIGITS_L(s_l, DIGITS_L(b_l));
	}
	else
	{
		aptr_l = LSDPTR_L(a_l);
		bptr_l = LSDPTR_L(b_l);
		msdptra_l = MSDPTR_L(a_l);
		msdptrb_l = MSDPTR_L(b_l);
		SETDIGITS_L(s_l, DIGITS_L(a_l));
	}

	while (bptr_l <= msdptrb_l)
	{
		*sptr_l++ = (USHORT)(carry = (ULONG)*aptr_l++ + (ULONG)*bptr_l++
			+ (ULONG)(USHORT)(carry >> BITPERDGT));
	}
	while (aptr_l <= msdptra_l)
	{
		*sptr_l++ = (USHORT)(carry = (ULONG)*aptr_l++
			+ (ULONG)(USHORT)(carry >> BITPERDGT));
	}
	if (carry & BASE)
	{
		*sptr_l = 1;
		INCDIGITS_L(s_l);
	}

	/* Purging of variables */
	PURGEVARS_L((1, sizeof(carry), &carry));
	ISPURGED_L((1, sizeof(carry), &carry));
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Integer Division                                               */
/*  Syntax:    int div_l (CLINT d1_l, CLINT d2_l, CLINT quot_l, CLINT rem_l); */
/*  Input:     d1_l (Dividend), d2_l (Divisor)                                */
/*  Output:    quot_l (Quotient), rem_l (Remainder)                           */
/*  Returns:   E_CLINT_OK : Everything O.K.                                   */
/*             E_CLINT_DBZ: Division by Zero                                  */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_div_l(CLINT d1_l, CLINT d2_l, CLINT quot_l, CLINT rem_l)
{
	register clint *rptr_l, *bptr_l;
	CLINT b_l;
	clint r_l[2 + (CLINTMAXDIGIT << 1)]; /* Allow double long remainder + 1 digit */
	clint *qptr_l, *msdptrb_l, *msdptrr_l, *lsdptrr_l;
	USHORT bv, rv, qhat, ri, ri_1, ri_2, bn, bn_1;
	ULONG right, left, rhat, borrow, carry, sbitsminusd;
	unsigned int d = 0;
	int i;

	cuda_cpy_l(r_l, d1_l);
	cuda_cpy_l(b_l, d2_l);

	if (CUDA_EQZ_L(b_l))
	{
		PURGEVARS_L((1, sizeof(r_l), r_l));
		ISPURGED_L((1, sizeof(r_l), r_l));

		return E_CLINT_DBZ;          /* Division by Zero */
	}

	if (CUDA_EQZ_L(r_l))
	{
		CUDA_SETZERO_L(quot_l);
		CUDA_SETZERO_L(rem_l);

		PURGEVARS_L((1, sizeof(b_l), b_l));
		ISPURGED_L((1, sizeof(b_l), b_l));

		return E_CLINT_OK;
	}

	i = cuda_cmp_l(r_l, b_l);

	if (i == -1)
	{
		cuda_cpy_l(rem_l, r_l);
		CUDA_SETZERO_L(quot_l);

		PURGEVARS_L((2, sizeof(b_l), b_l,
			sizeof(r_l), r_l));
		ISPURGED_L((2, sizeof(b_l), b_l,
			sizeof(r_l), r_l));
		return E_CLINT_OK;
	}
	else if (i == 0)
	{
		CUDA_SETONE_L(quot_l);
		CUDA_SETZERO_L(rem_l);

		PURGEVARS_L((2, sizeof(b_l), b_l,
			sizeof(r_l), r_l));
		ISPURGED_L((2, sizeof(b_l), b_l,
			sizeof(r_l), r_l));
		return E_CLINT_OK;
	}

	if (DIGITS_L(b_l) == 1)
	{
		goto shortdiv;
	}

	/* Step 1 */
	msdptrb_l = MSDPTR_L(b_l);

	bn = *msdptrb_l;
	while (bn < BASEDIV2)
	{
		d++;
		bn <<= 1;
	}

	sbitsminusd = (int)BITPERDGT - d;

	if (d > 0)
	{
		bn += *(msdptrb_l - 1) >> sbitsminusd;

		if (DIGITS_L(b_l) > 2)
		{
			bn_1 = (USHORT)((*(msdptrb_l - 1) << d) + (*(msdptrb_l - 2) >> sbitsminusd));
		}
		else
		{
			bn_1 = (USHORT)(*(msdptrb_l - 1) << d);
		}
	}
	else
	{
		bn_1 = (USHORT)(*(msdptrb_l - 1));
	}

	/* Steps 2 and 3 */
	msdptrr_l = MSDPTR_L(r_l) + 1;
	lsdptrr_l = MSDPTR_L(r_l) - DIGITS_L(b_l) + 1;
	*msdptrr_l = 0;

	qptr_l = quot_l + DIGITS_L(r_l) - DIGITS_L(b_l) + 1;

	/* Step 4 */
	while (lsdptrr_l >= LSDPTR_L(r_l))
	{
		ri = (USHORT)((*msdptrr_l << d) + (*(msdptrr_l - 1) >> sbitsminusd));

		ri_1 = (USHORT)((*(msdptrr_l - 1) << d) + (*(msdptrr_l - 2) >> sbitsminusd));

		if (msdptrr_l - 3 > r_l)
		{
			ri_2 = (USHORT)((*(msdptrr_l - 2) << d) + (*(msdptrr_l - 3) >> sbitsminusd));
		}
		else
		{
			ri_2 = (USHORT)(*(msdptrr_l - 2) << d);
		}

		if (ri != bn)               /* almost always */
		{
			qhat = (USHORT)((rhat = ((ULONG)ri << BITPERDGT) + (ULONG)ri_1) / bn);
			right = ((rhat = (rhat - (ULONG)bn * qhat)) << BITPERDGT) + ri_2;

			/* test qhat */

			if ((left = (ULONG)bn_1 * qhat) > right)
			{
				qhat--;
				if ((rhat + bn) < BASE)
					/* else bn_1 * qhat < rhat * b_l */
				{
					if ((left - bn_1) > (right + ((ULONG)bn << BITPERDGT)))
					{
						qhat--;
					}
				}
			}
		}
		else                        /* ri == bn, almost never */
		{
			qhat = BASEMINONE;
			right = ((ULONG)(rhat = (ULONG)bn + (ULONG)ri_1) << BITPERDGT) + ri_2;
			if (rhat < BASE)       /* else bn_1 * qhat < rhat * b_l */
			{
				/* test qhat */

				if ((left = (ULONG)bn_1 * qhat) > right)
				{
					qhat--;
					if ((rhat + bn) < BASE)
						/* else bn_1 * qhat < rhat * b_l */
					{
						if ((left - bn_1) > (right + ((ULONG)bn << BITPERDGT)))
						{
							qhat--;
						}
					}
				}
			}
		}

		/* Step 5 */
		borrow = BASE;
		carry = 0;
		for (bptr_l = LSDPTR_L(b_l), rptr_l = lsdptrr_l; bptr_l <= msdptrb_l; bptr_l++, rptr_l++)
		{
			if (borrow >= BASE)
			{
				*rptr_l = (USHORT)(borrow = ((ULONG)(*rptr_l) + BASE -
					(ULONG)(USHORT)(carry = (ULONG)(*bptr_l) *
					qhat + (ULONG)(USHORT)(carry >> BITPERDGT))));
			}
			else
			{
				*rptr_l = (USHORT)(borrow = ((ULONG)(*rptr_l) + BASEMINONEL -
					(ULONG)(USHORT)(carry = (ULONG)(*bptr_l) *
					qhat + (ULONG)(USHORT)(carry >> BITPERDGT))));
			}
		}

		if (borrow >= BASE)
		{
			*rptr_l = (USHORT)(borrow = ((ULONG)(*rptr_l) + BASE -
				(ULONG)(USHORT)(carry >> BITPERDGT)));
		}
		else
		{
			*rptr_l = (USHORT)(borrow = ((ULONG)(*rptr_l) + BASEMINONEL -
				(ULONG)(USHORT)(carry >> BITPERDGT)));
		}

		/* Step 6 */
		*qptr_l = qhat;

		if (borrow < BASE)
		{
			carry = 0;
			for (bptr_l = LSDPTR_L(b_l), rptr_l = lsdptrr_l; bptr_l <= msdptrb_l; bptr_l++, rptr_l++)
			{
				*rptr_l = (USHORT)(carry = ((ULONG)(*rptr_l) + (ULONG)(*bptr_l) +
					(ULONG)(USHORT)(carry >> BITPERDGT)));
			}
			*rptr_l += (USHORT)(carry >> BITPERDGT);
			(*qptr_l)--;
		}

		/* Step 7 */
		msdptrr_l--;
		lsdptrr_l--;
		qptr_l--;
	}

	/* Step 8 */
	SETDIGITS_L(quot_l, DIGITS_L(r_l) - DIGITS_L(b_l) + 1);
	RMLDZRS_L(quot_l);

	SETDIGITS_L(r_l, DIGITS_L(b_l));
	cuda_cpy_l(rem_l, r_l);

	/* Purging of variables */
	PURGEVARS_L((17, sizeof(bv), &bv,
		sizeof(rv), &rv,
		sizeof(qhat), &qhat,
		sizeof(ri), &ri,
		sizeof(ri_1), &ri_1,
		sizeof(ri_2), &ri_2,
		sizeof(bn), &bn,
		sizeof(bn_1), &bn_1,
		sizeof(right), &right,
		sizeof(left), &left,
		sizeof(rhat), &rhat,
		sizeof(borrow), &borrow,
		sizeof(carry), &carry,
		sizeof(sbitsminusd), &sbitsminusd,
		sizeof(d), &d,
		sizeof(b_l), b_l,
		sizeof(r_l), r_l));

	ISPURGED_L((17, sizeof(bv), &bv,
		sizeof(rv), &rv,
		sizeof(qhat), &qhat,
		sizeof(ri), &ri,
		sizeof(ri_1), &ri_1,
		sizeof(ri_2), &ri_2,
		sizeof(bn), &bn,
		sizeof(bn_1), &bn_1,
		sizeof(right), &right,
		sizeof(left), &left,
		sizeof(rhat), &rhat,
		sizeof(borrow), &borrow,
		sizeof(carry), &carry,
		sizeof(sbitsminusd), &sbitsminusd,
		sizeof(d), &d,
		sizeof(b_l), b_l,
		sizeof(r_l), r_l));

	return E_CLINT_OK;

	/* Division by divisor with one-digit */
shortdiv:

	rv = 0;
	bv = *LSDPTR_L(b_l);
	for (rptr_l = MSDPTR_L(r_l), qptr_l = quot_l + DIGITS_L(r_l); rptr_l >= LSDPTR_L(r_l); rptr_l--, qptr_l--)
	{
		*qptr_l = (USHORT)((rhat = ((((ULONG)rv) << BITPERDGT) +
			(ULONG)*rptr_l)) / bv);
		rv = (USHORT)(rhat - (ULONG)bv * (ULONG)*qptr_l);
	}

	SETDIGITS_L(quot_l, DIGITS_L(r_l));

	RMLDZRS_L(quot_l);
	cuda_u2clint_l(rem_l, rv);

	/* Purging of variables */
	PURGEVARS_L((4, sizeof(rv), &rv,
		sizeof(bv), &bv,
		sizeof(b_l), b_l,
		sizeof(r_l), r_l));

	ISPURGED_L((4, sizeof(rv), &rv,
		sizeof(bv), &bv,
		sizeof(b_l), b_l,
		sizeof(r_l), r_l));

	return E_CLINT_OK;
}

/******************************************************************************/
/*                                                                            */
/*  Function:  Increment                                                      */
/*  Syntax:    int inc_l (CLINT a_l);                                         */
/*  Input:     a_l (CLINT value)                                              */
/*  Output:    a_l, incremented by 1                                          */
/*  Returns:   E_CLINT_OK : Everything O.K.                                   */
/*             E_CLINT_OFL: Overflow                                          */
/*                                                                            */
/******************************************************************************/
CUDA_CALLABLE_MEMBER int __FLINT_API
cuda_inc_l(CLINT a_l)
{
	clint *msdptra_l, *aptr_l = LSDPTR_L(a_l);
	ULONG carry = BASE;
	int OFL = 0;

	msdptra_l = MSDPTR_L(a_l);
	while ((aptr_l <= msdptra_l) && (carry & BASE))
	{
		*aptr_l = (USHORT)(carry = 1UL + (ULONG)(*aptr_l));
		aptr_l++;
	}

	if ((aptr_l > msdptra_l) && (carry & BASE))
	{
		*aptr_l = 1;
		INCDIGITS_L(a_l);
		if (DIGITS_L(a_l) > (USHORT)CLINTMAXDIGIT)    /* Overflow ? */
		{
			SETZERO_L(a_l);              /* Reduction modulo Nmax+1 */
			OFL = E_CLINT_OFL;
		}
	}

	/* Purging of variables */
	PURGEVARS_L((1, sizeof(carry), &carry));
	ISPURGED_L((1, sizeof(carry), &carry));

	return OFL;
}