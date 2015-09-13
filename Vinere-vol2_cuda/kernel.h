#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include "flintpp.h"

#define CUDA_CALLABLE_MEMBER __device__ __host__
#define NOTHROW

class CUDALINT
{
public:
	CUDA_CALLABLE_MEMBER CUDALINT(void);
	CUDA_CALLABLE_MEMBER CUDALINT(clint*);
	CUDA_CALLABLE_MEMBER ~CUDALINT(void);
	const CUDALINT& operator= (const CUDALINT&);

	// LINT::Default-Error-Handler
	CUDA_CALLABLE_MEMBER static void panic(LINT_ERRORS, const char* const, const int, const int);

	// Pointer to type CLINT
	clint* n_l;

	// Status after an operation on a LINT object
	LINT_ERRORS status;
};