#include "kernel.h"



//Function CUDA analogs
//
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


////////////////////////////////////////////////////////////////////////////////
//           Overloaded operators                                             //
////////////////////////////////////////////////////////////////////////////////

// Assignment

const CUDALINT& CUDALINT::operator= (const CUDALINT& ln)
{
	if (ln.status == E_LINT_INV) panic(E_LINT_INV, "=", 1, __LINE__);

	if (&ln != this)                      // Don't copy object to itself
	{
		cpy_l(n_l, ln.n_l);
		status = ln.status;
	}
	return *this;
} /*lint !e1539*/

//Panic method overload
CUDA_CALLABLE_MEMBER void CUDALINT::panic(LINT_ERRORS, const char* const, const int, const int)
{
	return;
}

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

//Destructor
CUDA_CALLABLE_MEMBER CUDALINT::~CUDALINT(void)
{
	delete[] n_l;
}

//Program
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, void *nl);

//LINT M2 = mexp(LC, potential_D[i], N);
__global__ void addKernel(int *c, const int *a, const int *b, short *LC)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];

	clint* nl_LC = (clint*)LC;
	CUDALINT input1_long = CUDALINT(nl_LC);
	CUDALINT input2;
	input2 = input1_long;
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

	LINT zero;// [arraySize];
	void* nl;// [arraySize];
	zero = nextprime(randl(512) + 1, 1);
	nl = zero.n_l;

	//for (int count = 1; count < arraySize; count++)
	//{
	//	zero[count] = nextprime(zero[count-1], 1);
	//	nl[count] = zero[count].n_l;
	//}

    // Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize, nl);
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
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, void *nl)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

	void *dev_arr = 0;

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

	size_t num_size = 2 * ((short)(*((short*)nl))) + sizeof(short);

	cudaStatus = cudaMalloc((void**)(&dev_arr), num_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_arr, nl, num_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

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
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b, (short*)dev_arr);

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
	cudaFree(dev_arr);
    
    return cudaStatus;
}
