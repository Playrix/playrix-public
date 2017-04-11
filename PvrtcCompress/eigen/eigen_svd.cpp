
#include "SVD" // Eigen 3.3.3 is required

template<int R, int C>
static inline void pseudoInverse(float A[], float B[])
{
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor | Eigen::AutoAlign, R, C> t_A;

	t_A mA(R, C);

	for (int i = 0; i < R; i++)
	{
		for (int j = 0; j < C; j++)
		{
			mA(i, j) = A[i + j * R];
		}
	}

	Eigen::JacobiSVD<t_A, Eigen::ColPivHouseholderQRPreconditioner> svd(mA, Eigen::ComputeThinU | Eigen::ComputeThinV);

	auto U = svd.matrixU();
	auto S = svd.singularValues();
	auto V = svd.matrixV();

	float Vt[C * C];

	for (int i = 0; i < C; i++)
	{
		float s = S(i);

		if (s != 0.f)
			s = 1.f / s;

		for (int j = 0; j < C; j++)
		{
			Vt[j * C + i] = V(j, i) * s;
		}
	}

	for (int i = 0; i < C; i++)
	{
		for (int j = 0; j < R; j++)
		{
			float v = 0.f;

			for (int k = 0; k < C; k++)
			{
				v += Vt[i * C + k] * U(j, k);
			}

			B[i * R + j] = v;
		}
	}
}

void svd_routine_49_2(float A[], float B[])
{
	pseudoInverse<49, 2>(A, B);
}

void svd_routine_121_8(float A[], float B[])
{
	pseudoInverse<121, 8>(A, B);
}
