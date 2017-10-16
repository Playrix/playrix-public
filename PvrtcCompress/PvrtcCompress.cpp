// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
// --------------------------------------------------------------------------------
//
// Copyright(c) 2017 Playrix LLC
//
// LICENSE: https://mit-license.org

#ifdef WIN32
#include <windows.h>
#pragma warning(push)
#pragma warning(disable : 4458)
#include <gdiplus.h>
#pragma warning(pop)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <smmintrin.h> // SSE4.1
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef WIN32
#pragma comment(lib, "gdiplus.lib")

#define INLINED __forceinline
#define M128I_I16(mm, index) ((mm).m128i_i16[index])
#define M128I_I32(mm, index) ((mm).m128i_i32[index])
#else
#define INLINED __attribute__((always_inline))
#define M128I_I16(mm, index) (reinterpret_cast<int16_t(&)[8]>(mm)[index])
#define M128I_I32(mm, index) (reinterpret_cast<int32_t(&)[4]>(mm)[index])
#endif

// http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html Apple RGB
enum { kAlpha = 1000, kGreen = 672, kRed = 245, kBlue = 83 };

// Linear RGB
//enum { kAlpha = 3, kGreen = 1, kRed = 1, kBlue = 1 };

enum { kColor = kGreen + kRed + kBlue, kCapacity = 2048 >> 2, kStep = 8 };

alignas(16) static const int g_AGRB[4] = { kAlpha, kGreen, kRed, kBlue };
alignas(16) static const int g_GRB[4] = { 0, kGreen, kRed, kBlue };

alignas(16) static const short g_AGRB_I16[8] = { kAlpha, kGreen, kRed, kBlue, kAlpha, kGreen, kRed, kBlue };
alignas(16) static const short g_GRB_I16[8] = { 0, kGreen, kRed, kBlue, 0, kGreen, kRed, kBlue };
alignas(16) static const short g_Transparent2_I16[8] = { 0, -1, -1, -1, 0, -1, -1, -1 };

// (m2,m1,m3,m0)
static const uint32_t g_selection0312[0x10] =
{
	3, 0, 3, 0, 1, 0, 1, 0,
	2, 0, 2, 0, 1, 0, 1, 0
};

static INLINED int Sqr(int x)
{
	return x * x;
}

static INLINED constexpr int Min(int x, int y)
{
	return (x < y) ? x : y;
}

static INLINED constexpr int Max(int x, int y)
{
	return (x > y) ? x : y;
}

static INLINED int MakeColor5(int c)
{
	return c & 0x1F;
}

static INLINED int MakeColor4(int c)
{
	c &= 0x1E;
	return c | (c >> 4);
}

static INLINED int MakeColor3(int c)
{
	c &= 0x1C;
	return c | (c >> 3);
}

static INLINED int MakeAlpha3(int a)
{
	return a & 0xE;
}

static INLINED __m128i HorizontalMinimum4(__m128i me4)
{
	__m128i me2 = _mm_min_epi32(me4, _mm_shuffle_epi32(me4, _MM_SHUFFLE(2, 3, 0, 1)));
	__m128i me1 = _mm_min_epi32(me2, _mm_shuffle_epi32(me2, _MM_SHUFFLE(0, 1, 2, 3)));
	return me1;
}

static int Stride;

static int* _Image;
static ptrdiff_t _ImageToMaskDelta;
static int _Size;
static int _Mask;

static std::atomic_int _Changes;

typedef void(*PBlockKernel)(int y, int x);

static void BlockKernel_Shutdown(int, int)
{
}

static class Worker
{
protected:
	struct Item
	{
		int _Y, _X;

		Item()
		{
		}

		Item(int y, int x)
			: _Y(y)
			, _X(x)
		{
		}
	};

#ifdef WIN32
	CRITICAL_SECTION _Sync;
#else
	std::mutex _Sync;
#endif

	PBlockKernel _BlockKernel;

	int _Index, _Count, _Batch;

	std::atomic_int _Running;
	std::atomic_int _Active;

	Item _Items[kCapacity * kCapacity];

public:
	Worker()
	{
#ifdef WIN32
		if (!InitializeCriticalSectionAndSpinCount(&_Sync, 1000))
			throw std::runtime_error("init");
#endif

		_BlockKernel = &BlockKernel_Shutdown;

		_Index = 0;
		_Count = 0;
		_Batch = 1;

		_Running = 0;
		_Active = 0;
	}

	~Worker()
	{
#ifdef WIN32
		DeleteCriticalSection(&_Sync);
#endif
	}

	void Lock()
	{
#ifdef WIN32
		EnterCriticalSection(&_Sync);
#else
		_Sync.lock();
#endif
	}

	void UnLock()
	{
#ifdef WIN32
		LeaveCriticalSection(&_Sync);
#else
		_Sync.unlock();
#endif
	}

protected:
	static int ThreadProc(Worker* worker)
	{
		for (;; )
		{
			int head, tail;
			{
				worker->Lock();

				head = worker->_Index;
				tail = Min(head + worker->_Batch, worker->_Count);

				if (head < tail)
				{
					worker->_Active++;

					worker->_Index = tail;
				}

				worker->UnLock();
			}

			if (head >= tail)
			{
				std::this_thread::yield();
				continue;
			}

			if (worker->_BlockKernel == &BlockKernel_Shutdown)
				break;

			while (head < tail)
			{
				const Worker::Item& item = worker->_Items[head++];

				worker->_BlockKernel(item._Y, item._X);
			}

			worker->_Active--;
		}

		worker->_Active--;
		worker->_Running--;

		return 0;
	}

public:
	void Start()
	{
		int n = Max(1, (int)std::thread::hardware_concurrency());
		_Running = n;

		for (int i = 0; i < n; i++)
		{
			std::thread thread(ThreadProc, this);
			thread.detach();
		}
	}

protected:
	void WaitRun()
	{
		for (;; )
		{
			std::this_thread::yield();

			int pending, active;
			{
				Lock();

				pending = _Count - _Index;
				active = _Active;

				UnLock();
			}

			if ((pending <= 0) && (active <= 0))
				break;
		}
	}

public:
	void RunKernel(PBlockKernel blockKernel, int step, int batch)
	{
		_BlockKernel = blockKernel;

		int field_size = _Mask + 1;

		step = Min(step, field_size);

		for (int inner_y = 0; inner_y < step; inner_y++)
		{
			for (int inner_x = 0; inner_x < step; inner_x++)
			{
				Lock();

				_Index = 0;
				_Count = 0;

				Worker::Item* item = _Items;

				for (int y = inner_y; y <= _Mask; y += step)
				{
					for (int x = inner_x; x <= _Mask; x += step)
					{
						item->_Y = y;
						item->_X = x;
						item++;

						_Count++;
					}
				}

				_Batch = Max(Min((_Count - 1) / _Running + 1, batch), 1);

				UnLock();

				WaitRun();
			}
		}
	}

	void Finish()
	{
		Lock();

		_BlockKernel = &BlockKernel_Shutdown;
		_Index = 0;
		_Count = _Running;
		_Batch = 1;

		UnLock();

		WaitRun();
	}
} _Worker;

static class Fire
{
protected:
	volatile uint8_t _Flags[kCapacity][kCapacity];

public:
	Fire()
	{
		memset((void*)_Flags, 0, sizeof(_Flags));
	}

	void MarkAll()
	{
		for (int y = 0; y <= _Mask; y++)
		{
			for (int x = 0; x <= _Mask; x++)
			{
				_Flags[y][x] = 0xFF;
			}
		}
	}

	void MarkArea(int y1, int y2, int x1, int x2)
	{
		for (int y = y1; y <= y2; y++)
		{
			for (int x = x1; x <= x2; x++)
			{
				_Flags[y & _Mask][x & _Mask] = 0xFF;
			}
		}
	}

	void Remove(int y, int x, uint8_t f)
	{
		_Flags[y][x] &= ~f;
	}

	bool Has(int y, int x, uint8_t f)
	{
		return (_Flags[y][x] & f) != 0;
	}
} _Fire;

extern void svd_routine_49_2(float A[], float B[]);
extern void svd_routine_121_8(float A[], float B[]);

template<int R, int C>
class SVD
{
protected:
	float t_A[C][R];
	float t_B[C][R];

public:
	SVD()
	{
		static_assert(((R == 49) && (C == 2)) || ((R == 121) && (C == 8)), "Error");
	}

	INLINED void PseudoInverse(const float M[R][C])
	{
		constexpr bool two = (C < 3);
		if (two)
		{
			float T[C][C];

			for (int i = 0; i < C; i++)
			{
				for (int j = i; j < C; j++)
				{
					float v = 0.f;

					for (int k = 0; k < R; k++)
					{
						v += M[k][i] * M[k][j];
					}

					T[i][j] = v;
				}
			}

			float det = T[0][0] * T[1][1] - T[0][1] * T[0][1];
			if (det != 0.f)
			{
				float inv_det = 1.f / det;

				float L[C][C];
				L[0][0] = +T[1][1]; L[0][1] = -T[0][1];
				L[1][0] = -T[0][1]; L[1][1] = +T[0][0];

				for (int i = 0; i < C; i++)
				{
					for (int j = 0; j < R; j++)
					{
						float v = 0.f;

						for (int k = 0; k < C; k++)
						{
							v += L[i][k] * M[j][k];
						}

						t_B[i][j] = v * inv_det;
					}
				}

				return;
			}
		}

		for (int i = 0; i < R; i++)
		{
			for (int j = 0; j < C; j++)
			{
				t_A[j][i] = M[i][j];
			}
		}

		if (two)
		{
			svd_routine_49_2((float*)t_A, (float*)t_B);
		}
		else
		{
			svd_routine_121_8((float*)t_A, (float*)t_B);
		}
	}

	INLINED void Resolve(float AB[C], const float P[R])
	{
		for (int i = 0; i < C; i++)
		{
			float v = 0.f;

			for (int j = 0; j < R; j++)
			{
				v += t_B[i][j] * P[j];
			}

			AB[i] = v;
		}
	}
};

static int GetNextPow2(int k)
{
	if (k == 0)
		return 1;

	k--;

	for (int i = 1; i < 32; i <<= 1)
	{
		k |= k >> i;
	}

	return k + 1;
}

#ifdef WIN32

static bool ReadImage(const char* src_name, uint8_t* &pixels, int &width, int &height, bool flip)
{
	ULONG_PTR gdiplusToken;

	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
	{
		std::wstring wide_src_name;
		wide_src_name.resize(std::mbstowcs(nullptr, src_name, MAX_PATH));
		std::mbstowcs(&wide_src_name.front(), src_name, MAX_PATH);

		Gdiplus::Bitmap bitmap(wide_src_name.c_str(), FALSE);

		width = (int)bitmap.GetWidth();
		height = (int)bitmap.GetHeight();

		Gdiplus::Rect rect(0, 0, width, height);
		Gdiplus::BitmapData data;
		if (bitmap.LockBits(&rect, Gdiplus::ImageLockModeRead, PixelFormat32bppARGB, &data) == 0)
		{
			int stride = width << 2;

			pixels = new uint8_t[height * stride];

			uint8_t* w = pixels;
			for (int y = 0; y < height; y++)
			{
				const uint8_t* r = (const uint8_t*)data.Scan0 + (flip ? height - 1 - y : y) * data.Stride;
				memcpy(w, r, stride);
				w += stride;
			}

			bitmap.UnlockBits(&data);
		}
		else
		{
			pixels = nullptr;
		}
	}
	Gdiplus::GdiplusShutdown(gdiplusToken);

	return pixels != nullptr;
}

static void WriteImage(const char* dst_name, const uint8_t* pixels, int w, int h, bool flip)
{
	ULONG_PTR gdiplusToken;

	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
	{
		Gdiplus::Bitmap bitmap(w, h, PixelFormat32bppARGB);

		Gdiplus::Rect rect(0, 0, w, h);
		Gdiplus::BitmapData data;
		if (bitmap.LockBits(&rect, Gdiplus::ImageLockModeWrite, PixelFormat32bppARGB, &data) == 0)
		{
			for (int y = 0; y < h; y++)
			{
				memcpy((uint8_t*)data.Scan0 + (flip ? h - 1 - y : y) * data.Stride, pixels + y * w * 4, w * 4);
			}

			bitmap.UnlockBits(&data);
		}

		CLSID format;
		bool ok = false;
		{
			UINT num, size;
			Gdiplus::GetImageEncodersSize(&num, &size);
			if (size >= num * sizeof(Gdiplus::ImageCodecInfo))
			{
				Gdiplus::ImageCodecInfo* pArray = (Gdiplus::ImageCodecInfo*)new uint8_t[size];
				Gdiplus::GetImageEncoders(num, size, pArray);

				for (UINT i = 0; i < num; ++i)
				{
					if (pArray[i].FormatID == Gdiplus::ImageFormatPNG)
					{
						format = pArray[i].Clsid;
						ok = true;
						break;
					}
				}

				delete[](uint8_t*)pArray;
			}
		}
		if (ok)
		{
			std::wstring wide_dst_name;
			wide_dst_name.resize(std::mbstowcs(nullptr, dst_name, MAX_PATH));
			std::mbstowcs(&wide_dst_name.front(), dst_name, MAX_PATH);

			ok = (bitmap.Save(wide_dst_name.c_str(), &format) == Gdiplus::Ok);
		}

		printf(ok ? "  Saved %s\n" : "Lost %s\n", dst_name);
	}
	Gdiplus::GdiplusShutdown(gdiplusToken);
}

static void LoadPvrtc(const char* name, uint8_t* buffer, int size)
{
	HANDLE file = CreateFile(name, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, 0, NULL);
	if (file != INVALID_HANDLE_VALUE)
	{
		DWORD transferred;
		BOOL ok = ReadFile(file, buffer, size, &transferred, NULL);

		CloseHandle(file);

		if (ok)
		{
			printf("    Loaded %s\n", name);
		}
	}
}

static void SavePvrtc(const char* name, const uint8_t* buffer, int size)
{
	bool ok = false;

	HANDLE file = CreateFile(name, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (file != INVALID_HANDLE_VALUE)
	{
		DWORD transferred;
		ok = (WriteFile(file, buffer, size, &transferred, NULL) != 0);

		CloseHandle(file);
	}

	printf(ok ? "    Saved %s\n" : "Lost %s\n", name);
}

#endif

static INLINED void ComputeAlphaMaskWithOutline(uint8_t* mask_agrb, const uint8_t* src_bgra, int src_s, int radius)
{
	int full_s = 1 + radius + src_s + radius;

	uint8_t* data = new uint8_t[full_s * full_s];
	memset(data, 0, full_s * full_s);

	int mask = src_s - 1;

	for (int y = 1; y < full_s; y++)
	{
		const uint8_t* r = &src_bgra[((y - radius - 1) & mask) * Stride + 3];
		uint8_t* w = &data[y * full_s];

		for (int x = 1; x < full_s; x++)
		{
			w[x] = (r[((x - radius - 1) & mask) * 4] != 0) ? 1 : 0;
		}
	}

	int* sum = new int[full_s * full_s];
	memset(sum, 0, full_s * full_s * sizeof(int));

	int from_py = 1 * full_s;
	int to_py = full_s * full_s;
	for (int py = from_py; py < to_py; py += full_s)
	{
		int prev_py = py - full_s;

		for (int x = 1; x < full_s; x++)
		{
			sum[py + x] = sum[prev_py + x] - sum[prev_py + x - 1] + data[py + x] + sum[py + x - 1];
		}
	}

	int a = radius + radius + 1;
	for (int y = 0; y < src_s; y++)
	{
		int* w = (int*)&mask_agrb[y * Stride];

		const int* rL = &sum[y * full_s];
		const int* rH = &sum[(y + a) * full_s];

		for (int x = 0; x < src_s; x++, rL++, rH++)
		{
			int v = rH[a] - *rH + *rL - rL[a];

			w[x] = (v != 0) ? -1 : 0xFF;
		}
	}

	delete[] sum, sum = nullptr;

	delete[] data, data = nullptr;
}

static const double g_ssim_16k1L = (0.01 * 255 * 16) * (0.01 * 255 * 16);
static const double g_ssim_16k2L = g_ssim_16k1L * 9;

#define SSIM_INIT() \
	__m128i sa = _mm_setzero_si128(); \
	__m128i sb = _mm_setzero_si128(); \
	__m128i sab = _mm_setzero_si128(); \
	__m128i saa_sbb = _mm_setzero_si128(); \

#define SSIM_UPDATE(a, b) \
	sa = _mm_add_epi32(sa, a); \
	sb = _mm_add_epi32(sb, b); \
	sab = _mm_add_epi32(sab, _mm_mullo_epi16(a, b)); \
	saa_sbb = _mm_add_epi32(saa_sbb, _mm_add_epi32(_mm_mullo_epi16(a, a), _mm_mullo_epi16(b, b))); \

#define SSIM_CLOSE(shift) \
	sab = _mm_slli_epi32(sab, shift + 1); \
	saa_sbb = _mm_slli_epi32(saa_sbb, shift); \
	__m128i sasb = _mm_mullo_epi32(sa, sb); \
	sasb = _mm_add_epi32(sasb, sasb); \
	__m128i sasa_sbsb = _mm_add_epi32(_mm_mullo_epi32(sa, sa), _mm_mullo_epi32(sb, sb)); \
	sab = _mm_sub_epi32(sab, sasb); \
	saa_sbb = _mm_sub_epi32(saa_sbb, sasa_sbsb); \

#define SSIM_OTHER() \
	sab = _mm_unpackhi_epi64(sab, sab); \
	saa_sbb = _mm_unpackhi_epi64(saa_sbb, saa_sbb); \
	sasb = _mm_unpackhi_epi64(sasb, sasb); \
	sasa_sbsb = _mm_unpackhi_epi64(sasa_sbsb, sasa_sbsb); \

#define SSIM_FINAL(dst, p1, p2) \
	__m128d dst; \
	{ \
		__m128d mp1 = _mm_load1_pd(&p1); \
		__m128d mp2 = _mm_load1_pd(&p2); \
		dst = _mm_div_pd( \
			_mm_mul_pd(_mm_add_pd(_mm_cvtepi32_pd(sasb), mp1), _mm_add_pd(_mm_cvtepi32_pd(sab), mp2)), \
			_mm_mul_pd(_mm_add_pd(_mm_cvtepi32_pd(sasa_sbsb), mp1), _mm_add_pd(_mm_cvtepi32_pd(saa_sbb), mp2))); \
	} \


struct Block
{
	static Block Buffer[kCapacity][kCapacity];

	__m128i Colors; // AGRB 4555, A&B

	uint32_t Data[2];

	int Y, X;

	double SSIM;
	int Error;

	bool IsEmpty, IsZero;
	bool IsOpaque, IsDense;

	alignas(16) Block* Matrix[3][3];

	INLINED void Locate(int y, int x)
	{
		Y = y;
		X = x;

		int Yp = (y - 1) & _Mask; int Yn = (y + 1) & _Mask;
		int Xp = (x - 1) & _Mask; int Xn = (x + 1) & _Mask;

		Matrix[0][0] = &Buffer[Yp][Xp];
		Matrix[0][1] = &Buffer[Yp][X];
		Matrix[0][2] = &Buffer[Yp][Xn];

		Matrix[1][0] = &Buffer[Y][Xp];
		Matrix[1][1] = &Buffer[Y][X]; // this
		Matrix[1][2] = &Buffer[Y][Xn];

		Matrix[2][0] = &Buffer[Yn][Xp];
		Matrix[2][1] = &Buffer[Yn][X];
		Matrix[2][2] = &Buffer[Yn][Xn];
	}

	INLINED void Load(const uint8_t input[8])
	{
		memcpy(Data, input, 8);
	}

	INLINED void Save(uint8_t output[8]) const
	{
		memcpy(output, Data, 8);
	}

	INLINED void CheckEmpty()
	{
		IsEmpty = false;

		for (int y = 0; y < 4; y++)
		{
			const uint32_t* source = (const uint32_t*)&_Image[((Y << 2) + y) * _Size + (X << 2)];
			const uint32_t* mask = (const uint32_t*)((const uint8_t*)source + _ImageToMaskDelta);

			for (int x = 0; x < 4; x++)
			{
				if (mask[x] != 0xFFu)
					return;

				if ((source[x] & (0xFFu << 24)) != 0)
					return;
			}
		}

		IsEmpty = true;
	}

	INLINED void CheckOpaque()
	{
		IsOpaque = false;

		for (int y = 0; y < 4; y++)
		{
			const uint32_t* source = (const uint32_t*)&_Image[((Y << 2) + y) * _Size + (X << 2)];
			const uint32_t* mask = (const uint32_t*)((const uint8_t*)source + _ImageToMaskDelta);

			for (int x = 0; x < 4; x++)
			{
				if (mask[x] != (uint32_t)-1)
					return;

				if ((~source[x] & (0xFFu << 24)) != 0)
					return;
			}
		}

		IsOpaque = true;
	}

	INLINED void CheckZero()
	{
		IsZero = false;

		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				if (!Matrix[y][x]->IsEmpty)
					return;
			}
		}

		IsZero = true;
	}

	INLINED void CheckDense()
	{
		IsDense = false;

		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				if (!Matrix[y][x]->IsOpaque)
					return;
			}
		}

		IsDense = true;
	}

	INLINED void UnpackColors()
	{
		__m128i mc = _mm_setzero_si128();

		uint32_t ColorA = Data[1];
		if (ColorA & (1u << 15))
		{
			int r = MakeColor5(ColorA >> 10);
			int g = MakeColor5(ColorA >> 5);
			int b = MakeColor4(ColorA);
			int a = 0xF;

			mc = _mm_insert_epi16(mc, r, 2);
			mc = _mm_insert_epi16(mc, g, 1);
			mc = _mm_insert_epi16(mc, b, 3);
			mc = _mm_insert_epi16(mc, a, 0);
		}
		else
		{
			int r = MakeColor4(ColorA >> (8 - 1));
			int g = MakeColor4(ColorA >> (4 - 1));
			int b = MakeColor3(ColorA + ColorA);
			int a = MakeAlpha3(ColorA >> 11);

			mc = _mm_insert_epi16(mc, r, 2);
			mc = _mm_insert_epi16(mc, g, 1);
			mc = _mm_insert_epi16(mc, b, 3);
			mc = _mm_insert_epi16(mc, a, 0);
		}

		uint32_t ColorB = Data[1] >> 16;
		if (ColorB & (1u << 15))
		{
			int r = MakeColor5(ColorB >> 10);
			int g = MakeColor5(ColorB >> 5);
			int b = MakeColor5(ColorB);
			int a = 0xF;

			mc = _mm_insert_epi16(mc, r, 6);
			mc = _mm_insert_epi16(mc, g, 5);
			mc = _mm_insert_epi16(mc, b, 7);
			mc = _mm_insert_epi16(mc, a, 4);
		}
		else
		{
			int r = MakeColor4(ColorB >> (8 - 1));
			int g = MakeColor4(ColorB >> (4 - 1));
			int b = MakeColor4(ColorB + ColorB);
			int a = MakeAlpha3(ColorB >> 11);

			mc = _mm_insert_epi16(mc, r, 6);
			mc = _mm_insert_epi16(mc, g, 5);
			mc = _mm_insert_epi16(mc, b, 7);
			mc = _mm_insert_epi16(mc, a, 4);
		}

		_mm_store_si128(&Colors, mc);
	}

	INLINED void PackColors()
	{
		uint32_t colorA;
		{
			int a = M128I_I16(Colors, 0) & 0xF;
			int g = MakeColor5(M128I_I16(Colors, 1));
			int r = MakeColor5(M128I_I16(Colors, 2));
			int b = MakeColor5(M128I_I16(Colors, 3));

			int b1 = MakeColor4(b);
			int e1 = Sqr(b - b1) * kBlue + Sqr(a - 0xF) * kAlpha;

			if (e1 <= 0)
			{
				colorA = (1 << 15) | (r << 10) | (g << 5) | (b1 & 0x1E);
			}
			else
			{
				int r0 = MakeColor4(r);
				int g0 = MakeColor4(g);
				int b0 = MakeColor3(b);
				int a0 = MakeAlpha3(a);
				int e0 = Sqr(r - r0) * kRed + Sqr(g - g0) * kGreen + Sqr(b - b0) * kBlue + Sqr(a - a0) * kAlpha;

				if (e1 <= e0)
				{
					colorA = (1 << 15) | (r << 10) | (g << 5) | (b1 & 0x1E);
				}
				else
				{
					colorA = (a0 << 11) | ((r0 & 0x1E) << 7) | ((g0 & 0x1E) << 3) | ((b0 & 0x1C) >> 1);
				}
			}
		}

		uint32_t colorB;
		{
			int a = M128I_I16(Colors, 4) & 0xF;
			int g = MakeColor5(M128I_I16(Colors, 5));
			int r = MakeColor5(M128I_I16(Colors, 6));
			int b = MakeColor5(M128I_I16(Colors, 7));

			int e1 = Sqr(a - 0xF) * kAlpha;

			int r0 = MakeColor4(r);
			int g0 = MakeColor4(g);
			int b0 = MakeColor4(b);
			int a0 = MakeAlpha3(a);
			int e0 = Sqr(r - r0) * kRed + Sqr(g - g0) * kGreen + Sqr(b - b0) * kBlue + Sqr(a - a0) * kAlpha;

			if (e1 <= e0)
			{
				colorB = (1 << 15) | (r << 10) | (g << 5) | b;
			}
			else
			{
				colorB = (a0 << 11) | ((r0 & 0x1E) << 7) | ((g0 & 0x1E) << 3) | ((b0 & 0x1E) >> 1);
			}
		}

		Data[1] = (Data[1] & 1u) | colorA | (colorB << 16);
	}

	void UpdateColors()
	{
		PackColors();
		UnpackColors();
	}

	INLINED static void Macro_Gradient(__m128i& mDL, __m128i& mDR, __m128i& mVL, __m128i& mVR, const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x)
	{
		if ((y ^ 2) < 2)
		{
			if ((x ^ 2) < 2)
			{
				__m128i mTL = _mm_load_si128(&b00.Colors);
				__m128i mBL = _mm_load_si128(&b10.Colors);

				mDL = _mm_sub_epi16(mBL, mTL);
				mVL = _mm_slli_epi16(mTL, 2);

				__m128i mTR = _mm_load_si128(&b01.Colors);
				__m128i mBR = _mm_load_si128(&b11.Colors);

				mDR = _mm_sub_epi16(mBR, mTR);
				mVR = _mm_slli_epi16(mTR, 2);
			}
			else
			{
				__m128i mTL = _mm_load_si128(&b00.Colors);
				__m128i mBL = _mm_load_si128(&b10.Colors);

				mDR = _mm_sub_epi16(mBL, mTL);
				mVR = _mm_slli_epi16(mTL, 2);

				__m128i mTR = _mm_load_si128(&b01.Colors);
				__m128i mBR = _mm_load_si128(&b11.Colors);

				mDL = _mm_sub_epi16(mBR, mTR);
				mVL = _mm_slli_epi16(mTR, 2);
			}
		}
		else
		{
			if ((x ^ 2) < 2)
			{
				__m128i mBL = _mm_load_si128(&b10.Colors);
				__m128i mTL = _mm_load_si128(&b00.Colors);

				mDL = _mm_sub_epi16(mTL, mBL);
				mVL = _mm_slli_epi16(mBL, 2);

				__m128i mBR = _mm_load_si128(&b11.Colors);
				__m128i mTR = _mm_load_si128(&b01.Colors);

				mDR = _mm_sub_epi16(mTR, mBR);
				mVR = _mm_slli_epi16(mBR, 2);
			}
			else
			{
				__m128i mBL = _mm_load_si128(&b10.Colors);
				__m128i mTL = _mm_load_si128(&b00.Colors);

				mDR = _mm_sub_epi16(mTL, mBL);
				mVR = _mm_slli_epi16(mBL, 2);

				__m128i mBR = _mm_load_si128(&b11.Colors);
				__m128i mTR = _mm_load_si128(&b01.Colors);

				mDL = _mm_sub_epi16(mTR, mBR);
				mVL = _mm_slli_epi16(mBR, 2);
			}
		}
	}

	INLINED static void Macro_InterpolateCY(__m128i mDL, __m128i mDR, const __m128i& mVL, const __m128i& mVR, __m128i& mD, __m128i& mV, int y)
	{
		__m128i mL, mR;

		switch (y ^ 2)
		{
		case 3:
			mL = _mm_add_epi16(mDL, mVL);
			mR = _mm_add_epi16(mDR, mVR);
			break;

		case 2:
			mL = _mm_add_epi16(_mm_slli_epi16(mDL, 1), mVL);
			mR = _mm_add_epi16(_mm_slli_epi16(mDR, 1), mVR);
			break;

		case 1:
			mL = _mm_add_epi16(mDL, mVL);
			mR = _mm_add_epi16(mDR, mVR);
			break;

		default:
			mL = mVL;
			mR = mVR;
			break;
		}

		mD = _mm_sub_epi16(mR, mL);
		mV = _mm_slli_epi16(mL, 2);
	}

	INLINED static void Macro_InterpolateCY_Pair(__m128i mDL, __m128i mDR, const __m128i& mVL, const __m128i& mVR, __m128i& mD0, __m128i& mV0, __m128i& mD1, __m128i& mV1, int y)
	{
		__m128i mL1 = _mm_add_epi16(mDL, mVL);
		__m128i mR1 = _mm_add_epi16(mDR, mVR);

		if ((y ^ 2) < 2)
		{
			mD0 = _mm_sub_epi16(mVR, mVL);
			mV0 = _mm_slli_epi16(mVL, 2);

			mD1 = _mm_sub_epi16(mR1, mL1);
			mV1 = _mm_slli_epi16(mL1, 2);
		}
		else
		{
			__m128i mL0 = _mm_add_epi16(mDL, mL1);
			__m128i mR0 = _mm_add_epi16(mDR, mR1);

			mD1 = _mm_sub_epi16(mR1, mL1);
			mV1 = _mm_slli_epi16(mL1, 2);

			mD0 = _mm_sub_epi16(mR0, mL0);
			mV0 = _mm_slli_epi16(mL0, 2);
		}
	}

	INLINED static __m128i Macro_InterpolateCX(__m128i mD, __m128i mV, int x)
	{
		__m128i mC;

		switch (x ^ 2)
		{
		case 3:
			mC = _mm_add_epi16(mD, mV);
			break;

		case 2:
			mC = _mm_add_epi16(_mm_slli_epi16(mD, 1), mV);
			break;

		case 1:
			mC = _mm_add_epi16(mD, mV);
			break;

		default:
			mC = mV;
			break;
		}

		return mC;
	}

	INLINED static void Macro_InterpolateCX_Pair(__m128i mD, __m128i mV, __m128i& mC0, __m128i& mC1, int x)
	{
		mC1 = _mm_add_epi16(mD, mV);

		if ((x ^ 2) < 2)
		{
			mC0 = mV;
		}
		else
		{
			mC0 = _mm_add_epi16(mD, mC1);
		}
	}

	INLINED static __m128i Macro_ExpandC(__m128i mC)
	{
		__m128i ma = _mm_add_epi16(_mm_srai_epi16(mC, 4), mC);

		mC = _mm_add_epi16(_mm_srai_epi16(mC, 1), _mm_srai_epi16(mC, 5 + 1));

		return _mm_blend_epi16(mC, ma, 0x11);
	}

	INLINED static __m128i Macro_ExpandC_GRB(__m128i mC)
	{
		return _mm_add_epi16(_mm_srai_epi16(mC, 1), _mm_srai_epi16(mC, 5 + 1));
	}

	INLINED static void Interpolate2x2(__m128i interpolation[4][4], const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x)
	{
		__m128i mDL, mDR, mVL, mVR;
		Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

		__m128i mD0, mV0, mD1, mV1;
		Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

		__m128i mC00, mC01;
		Macro_InterpolateCX_Pair(mD0, mV0, mC00, mC01, x);

		_mm_store_si128(&interpolation[y + 0][x + 0], Macro_ExpandC(mC00));
		_mm_store_si128(&interpolation[y + 0][x + 1], Macro_ExpandC(mC01));

		__m128i mC10, mC11;
		Macro_InterpolateCX_Pair(mD1, mV1, mC10, mC11, x);

		_mm_store_si128(&interpolation[y + 1][x + 0], Macro_ExpandC(mC10));
		_mm_store_si128(&interpolation[y + 1][x + 1], Macro_ExpandC(mC11));
	}

	INLINED void Interpolate4x4(__m128i interpolation[4][4]) const
	{
		Interpolate2x2(interpolation, *Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 0, 0);
		Interpolate2x2(interpolation, *Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 0, 2);

		Interpolate2x2(interpolation, *Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 2, 0);
		Interpolate2x2(interpolation, *this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 2, 2);
	}

	INLINED static void ModulateTransparentPixel(const __m128i interpolation[4][4], int& p, uint32_t codes, int y, int x, int shift)
	{
		static const int Weights[4] = { 0 + (0 << 16), 1 + (1 << 16), 1 + (1 << 16), 2 + (2 << 16) };
		static const int Masks[4] = { -1, -1, 0xFFFFFF, -1 };

		__m128i mA = _mm_load_si128(&interpolation[y][x]);
		__m128i mB = _mm_unpackhi_epi64(mA, mA);

		size_t code = (codes >> shift) & 3u;

		__m128i mMod = _mm_shuffle_epi32(_mm_cvtsi32_si128(Weights[code]), 0);

		__m128i mC = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_sub_epi16(mB, mA), mMod), 1), mA);

		mC = _mm_shufflelo_epi16(mC, _MM_SHUFFLE(0, 2, 1, 3));

		mC = _mm_packus_epi16(mC, mC);

		p = _mm_cvtsi128_si32(mC) & Masks[code];
	}

	INLINED static void ModulateOpaquePixel(const __m128i interpolation[4][4], int& p, uint32_t codes, int y, int x, int shift)
	{
		static const int Weights[4] = { 0 + (0 << 16), 3 + (3 << 16), 5 + (5 << 16), 8 + (8 << 16) };

		__m128i mA = _mm_load_si128(&interpolation[y][x]);
		__m128i mB = _mm_unpackhi_epi64(mA, mA);

		size_t code = (codes >> shift) & 3u;

		__m128i mMod = _mm_shuffle_epi32(_mm_cvtsi32_si128(Weights[code]), 0);

		__m128i mC = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_sub_epi16(mB, mA), mMod), 3), mA);

		mC = _mm_shufflelo_epi16(mC, _MM_SHUFFLE(0, 2, 1, 3));

		mC = _mm_packus_epi16(mC, mC);

		p = _mm_cvtsi128_si32(mC);
	}

	void Modulate4x4(const __m128i interpolation[4][4], int* output, int stride) const
	{
		uint32_t codes = Data[0];

		if (Data[1] & 1u)
		{
			ModulateTransparentPixel(interpolation, output[0], codes, 0, 0, 0);
			ModulateTransparentPixel(interpolation, output[1], codes, 0, 1, 2);
			ModulateTransparentPixel(interpolation, output[2], codes, 0, 2, 4);
			ModulateTransparentPixel(interpolation, output[3], codes, 0, 3, 6);

			output = (int*)((uint8_t*)output + stride);

			ModulateTransparentPixel(interpolation, output[0], codes, 1, 0, 8);
			ModulateTransparentPixel(interpolation, output[1], codes, 1, 1, 10);
			ModulateTransparentPixel(interpolation, output[2], codes, 1, 2, 12);
			ModulateTransparentPixel(interpolation, output[3], codes, 1, 3, 14);

			output = (int*)((uint8_t*)output + stride);

			ModulateTransparentPixel(interpolation, output[0], codes, 2, 0, 16);
			ModulateTransparentPixel(interpolation, output[1], codes, 2, 1, 18);
			ModulateTransparentPixel(interpolation, output[2], codes, 2, 2, 20);
			ModulateTransparentPixel(interpolation, output[3], codes, 2, 3, 22);

			output = (int*)((uint8_t*)output + stride);

			ModulateTransparentPixel(interpolation, output[0], codes, 3, 0, 24);
			ModulateTransparentPixel(interpolation, output[1], codes, 3, 1, 26);
			ModulateTransparentPixel(interpolation, output[2], codes, 3, 2, 28);
			ModulateTransparentPixel(interpolation, output[3], codes, 3, 3, 30);
		}
		else
		{
			ModulateOpaquePixel(interpolation, output[0], codes, 0, 0, 0);
			ModulateOpaquePixel(interpolation, output[1], codes, 0, 1, 2);
			ModulateOpaquePixel(interpolation, output[2], codes, 0, 2, 4);
			ModulateOpaquePixel(interpolation, output[3], codes, 0, 3, 6);

			output = (int*)((uint8_t*)output + stride);

			ModulateOpaquePixel(interpolation, output[0], codes, 1, 0, 8);
			ModulateOpaquePixel(interpolation, output[1], codes, 1, 1, 10);
			ModulateOpaquePixel(interpolation, output[2], codes, 1, 2, 12);
			ModulateOpaquePixel(interpolation, output[3], codes, 1, 3, 14);

			output = (int*)((uint8_t*)output + stride);

			ModulateOpaquePixel(interpolation, output[0], codes, 2, 0, 16);
			ModulateOpaquePixel(interpolation, output[1], codes, 2, 1, 18);
			ModulateOpaquePixel(interpolation, output[2], codes, 2, 2, 20);
			ModulateOpaquePixel(interpolation, output[3], codes, 2, 3, 22);

			output = (int*)((uint8_t*)output + stride);

			ModulateOpaquePixel(interpolation, output[0], codes, 3, 0, 24);
			ModulateOpaquePixel(interpolation, output[1], codes, 3, 1, 26);
			ModulateOpaquePixel(interpolation, output[2], codes, 3, 2, 28);
			ModulateOpaquePixel(interpolation, output[3], codes, 3, 3, 30);
		}
	}

	INLINED static void FastPackOpaqueTransparentPixel_Pair(const __m128i interpolation[4][4], const int& p, uint32_t& codesO, uint32_t& codesT, int y, int x, int shift, int& sumO, int& sumT)
	{
		__m128i mCx = _mm_load_si128(&interpolation[y][x + 0]);
		__m128i mCy = _mm_load_si128(&interpolation[y][x + 1]);

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		__m128i mc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&p));

		__m128i mmask = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)((const uint8_t*)&p + _ImageToMaskDelta)));

		//

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);
		__m128i mBT = _mm_and_si128(mAT, _mm_load_si128((const __m128i*)g_Transparent2_I16));

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);
		mAT = _mm_sub_epi16(mAT, mc);
		mBT = _mm_sub_epi16(mBT, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);
		mAT = _mm_mullo_epi16(mAT, mAT);
		mBT = _mm_mullo_epi16(mBT, mBT);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_AGRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAC = _mm_min_epu16(mAC, mlimit);
		mBC = _mm_min_epu16(mBC, mlimit);
		mAT = _mm_min_epu16(mAT, mlimit);
		mBT = _mm_min_epu16(mBT, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAC = _mm_madd_epi16(mAC, magrb);
		mBC = _mm_madd_epi16(mBC, magrb);
		mAT = _mm_madd_epi16(mAT, magrb);
		mBT = _mm_madd_epi16(mBT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);
		__m128i m2T = _mm_hadd_epi32(mAT, mBT);

		m2 = _mm_shuffle_epi32(m2, _MM_SHUFFLE(3, 1, 2, 0));
		m2C = _mm_shuffle_epi32(m2C, _MM_SHUFFLE(3, 1, 2, 0));
		m2T = _mm_shuffle_epi32(m2T, _MM_SHUFFLE(3, 1, 2, 0));

		{
			__m128i me4 = _mm_unpacklo_epi64(m2, m2C);
			__m128i me1 = HorizontalMinimum4(me4);

			codesO |= g_selection0312[(uint32_t)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4, me1)))] << shift;

			sumO += _mm_cvtsi128_si32(me1);
		}
		{
			__m128i me4 = _mm_unpacklo_epi64(m2, m2T);
			__m128i me1 = HorizontalMinimum4(me4);

			codesT |= g_selection0312[(uint32_t)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4, me1)))] << shift;

			sumT += _mm_cvtsi128_si32(me1);
		}

		{
			__m128i me4 = _mm_unpackhi_epi64(m2, m2C);
			__m128i me1 = HorizontalMinimum4(me4);

			codesO |= g_selection0312[(uint32_t)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4, me1)))] << (shift + 2);

			sumO += _mm_cvtsi128_si32(me1);
		}
		{
			__m128i me4 = _mm_unpackhi_epi64(m2, m2T);
			__m128i me1 = HorizontalMinimum4(me4);

			codesT |= g_selection0312[(uint32_t)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4, me1)))] << (shift + 2);

			sumT += _mm_cvtsi128_si32(me1);
		}
	}

	INLINED static void PackOpaqueTransparentPixel(const __m128i interpolation[4][4], const int& p, uint32_t& codesO, uint32_t& codesT, int y, int x, int shift, int& sumO, int& sumT)
	{
		__m128i mc = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(p));
		mc = _mm_shuffle_epi32(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mA = _mm_load_si128(&interpolation[y][x]);
		__m128i mB = _mm_unpackhi_epi64(mA, mA);

		mA = _mm_cvtepi16_epi32(mA);
		mB = _mm_cvtepi16_epi32(mB);

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);
		__m128i mBT = _mm_insert_epi16(mAT, 0, 0);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);
		mAT = _mm_sub_epi16(mAT, mc);
		mBT = _mm_sub_epi16(mBT, mc);

		__m128i magrb = _mm_load_si128((const __m128i*)g_AGRB);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);
		mAT = _mm_mullo_epi16(mAT, mAT);
		mBT = _mm_mullo_epi16(mBT, mBT);

		magrb = _mm_and_si128(magrb, _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)&p + _ImageToMaskDelta))));

		mA = _mm_mullo_epi32(mA, magrb);
		mB = _mm_mullo_epi32(mB, magrb);
		mAC = _mm_mullo_epi32(mAC, magrb);
		mBC = _mm_mullo_epi32(mBC, magrb);
		mAT = _mm_mullo_epi32(mAT, magrb);
		mBT = _mm_mullo_epi32(mBT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);
		__m128i m2T = _mm_hadd_epi32(mAT, mBT);

		{
			__m128i me4 = _mm_hadd_epi32(m2, m2C);
			__m128i me1 = HorizontalMinimum4(me4);

			codesO |= g_selection0312[(uint32_t)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4, me1)))] << shift;

			sumO += _mm_cvtsi128_si32(me1);
		}
		{
			__m128i me4 = _mm_hadd_epi32(m2, m2T);
			__m128i me1 = HorizontalMinimum4(me4);

			codesT |= g_selection0312[(uint32_t)_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4, me1)))] << shift;

			sumT += _mm_cvtsi128_si32(me1);
		}
	}


	INLINED void PackOpaqueTransparent4x4(const __m128i interpolation[4][4], int& errorO, uint32_t& answerO, int& errorT, uint32_t& answerT) const
	{
		if (Error < 0x7FFF * Min(Min(kGreen, kRed), kBlue))
		{
			const int* source = &_Image[(Y << 2) * _Size + (X << 2)];

			uint32_t codesO = 0, codesT = 0;
			int sumO = 0, sumT = 0;

			FastPackOpaqueTransparentPixel_Pair(interpolation, source[0], codesO, codesT, 0, 0, 0, sumO, sumT);
			FastPackOpaqueTransparentPixel_Pair(interpolation, source[2], codesO, codesT, 0, 2, 4, sumO, sumT);

			source += _Size;

			FastPackOpaqueTransparentPixel_Pair(interpolation, source[0], codesO, codesT, 1, 0, 8, sumO, sumT);
			FastPackOpaqueTransparentPixel_Pair(interpolation, source[2], codesO, codesT, 1, 2, 12, sumO, sumT);

			source += _Size;

			FastPackOpaqueTransparentPixel_Pair(interpolation, source[0], codesO, codesT, 2, 0, 16, sumO, sumT);
			FastPackOpaqueTransparentPixel_Pair(interpolation, source[2], codesO, codesT, 2, 2, 20, sumO, sumT);

			source += _Size;

			FastPackOpaqueTransparentPixel_Pair(interpolation, source[0], codesO, codesT, 3, 0, 24, sumO, sumT);
			FastPackOpaqueTransparentPixel_Pair(interpolation, source[2], codesO, codesT, 3, 2, 28, sumO, sumT);

			if (Min(sumO, sumT) < 0x7FFF * Min(Min(kGreen, kRed), kBlue))
			{
				errorO = sumO;
				answerO = codesO;
				errorT = sumT;
				answerT = codesT;
				return;
			}
		}

		const int* source = &_Image[(Y << 2) * _Size + (X << 2)];

		uint32_t codesO = 0, codesT = 0;
		int sumO = 0, sumT = 0;

		PackOpaqueTransparentPixel(interpolation, source[0], codesO, codesT, 0, 0, 0, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[1], codesO, codesT, 0, 1, 2, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[2], codesO, codesT, 0, 2, 4, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[3], codesO, codesT, 0, 3, 6, sumO, sumT);

		source += _Size;

		PackOpaqueTransparentPixel(interpolation, source[0], codesO, codesT, 1, 0, 8, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[1], codesO, codesT, 1, 1, 10, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[2], codesO, codesT, 1, 2, 12, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[3], codesO, codesT, 1, 3, 14, sumO, sumT);

		source += _Size;

		PackOpaqueTransparentPixel(interpolation, source[0], codesO, codesT, 2, 0, 16, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[1], codesO, codesT, 2, 1, 18, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[2], codesO, codesT, 2, 2, 20, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[3], codesO, codesT, 2, 3, 22, sumO, sumT);

		source += _Size;

		PackOpaqueTransparentPixel(interpolation, source[0], codesO, codesT, 3, 0, 24, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[1], codesO, codesT, 3, 1, 26, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[2], codesO, codesT, 3, 2, 28, sumO, sumT);
		PackOpaqueTransparentPixel(interpolation, source[3], codesO, codesT, 3, 3, 30, sumO, sumT);

		errorO = sumO;
		answerO = codesO;
		errorT = sumT;
		answerT = codesT;
	}

	void PackModulation()
	{
		if (IsEmpty)
		{
			Error = 0;
			Data[0] = 0xAAAAAAAAu;
			Data[1] |= 1u;
			return;
		}

		__m128i interpolation[4][4];
		Interpolate4x4(interpolation);

		int errorO, errorT;
		uint32_t answerO, answerT;
		PackOpaqueTransparent4x4(interpolation, errorO, answerO, errorT, answerT);

		if (errorO == errorT)
		{
			if (Data[1] & 1u)
			{
				Error = errorT;
				Data[0] = answerT;
			}
			else
			{
				Error = errorO;
				Data[0] = answerO;
			}
		}
		else if (errorO < errorT)
		{
			Error = errorO;
			Data[0] = answerO;
			Data[1] &= ~1u;
		}
		else
		{
			Error = errorT;
			Data[0] = answerT;
			Data[1] |= 1u;
		}
	}

	INLINED void Egoist()
	{
		__m128i m0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(-1), 0);
		__m128i m1 = _mm_setzero_si128();

		for (int y = 0; y < 4; y++)
		{
			const int* source = &_Image[((Y << 2) + y) * _Size + (X << 2)];

			__m128i mc = _mm_loadu_si128((const __m128i*)source);

			m0 = _mm_min_epu8(m0, mc);
			m1 = _mm_max_epu8(m1, mc);
		}

		m0 = _mm_min_epu8(m0, _mm_shuffle_epi32(m0, _MM_SHUFFLE(2, 3, 0, 1)));
		m1 = _mm_max_epu8(m1, _mm_shuffle_epi32(m1, _MM_SHUFFLE(2, 3, 0, 1)));

		m0 = _mm_min_epu8(m0, _mm_shuffle_epi32(m0, _MM_SHUFFLE(1, 0, 3, 2)));
		m1 = _mm_max_epu8(m1, _mm_shuffle_epi32(m1, _MM_SHUFFLE(1, 0, 3, 2)));

		__m128i mC = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(m0, m1));

		mC = _mm_shufflelo_epi16(mC, _MM_SHUFFLE(0, 2, 1, 3));
		mC = _mm_shufflehi_epi16(mC, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i ma = _mm_srai_epi16(mC, 4);
		mC = _mm_srai_epi16(mC, 3);

		__m128i mBA = _mm_load_si128(&Colors);

		mBA = _mm_blend_epi16(mBA, ma, 0x11);
		mBA = _mm_blend_epi16(mBA, mC, 0xEE);

		_mm_store_si128(&Colors, mBA);

		UpdateColors();
	}

	///////////////////
	// Fast Estimate //
	///////////////////

	INLINED static void FastEstimateOpaqueTransparent_Pair(__m128i mD, __m128i mV, int x, int& sumO, int& sumT, const int* source)
	{
		__m128i mCx, mCy;
		Macro_InterpolateCX_Pair(mD, mV, mCx, mCy, x);

		__m128i mc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)source));

		mCx = Macro_ExpandC(mCx);
		mCy = Macro_ExpandC(mCy);

		__m128i mmask = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)((const uint8_t*)source + _ImageToMaskDelta)));

		//

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);
		__m128i mBT = _mm_and_si128(mAT, _mm_load_si128((const __m128i*)g_Transparent2_I16));

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);
		mAT = _mm_sub_epi16(mAT, mc);
		mBT = _mm_sub_epi16(mBT, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);
		mAT = _mm_mullo_epi16(mAT, mAT);
		mBT = _mm_mullo_epi16(mBT, mBT);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_AGRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAC = _mm_min_epu16(mAC, mlimit);
		mBC = _mm_min_epu16(mBC, mlimit);
		mAT = _mm_min_epu16(mAT, mlimit);
		mBT = _mm_min_epu16(mBT, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAC = _mm_madd_epi16(mAC, magrb);
		mBC = _mm_madd_epi16(mBC, magrb);
		mAT = _mm_madd_epi16(mAT, magrb);
		mBT = _mm_madd_epi16(mBT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);
		__m128i m2T = _mm_hadd_epi32(mAT, mBT);

		{
			m2C = _mm_min_epi32(m2C, m2);
			m2C = _mm_min_epi32(m2C, _mm_shuffle_epi32(m2C, _MM_SHUFFLE(1, 0, 3, 2)));

			sumO += _mm_cvtsi128_si32(m2C);
			sumO += _mm_cvtsi128_si32(_mm_shuffle_epi32(m2C, _MM_SHUFFLE(2, 3, 0, 1)));
		}
		{
			m2T = _mm_min_epi32(m2T, m2);
			m2T = _mm_min_epi32(m2T, _mm_shuffle_epi32(m2T, _MM_SHUFFLE(1, 0, 3, 2)));

			sumT += _mm_cvtsi128_si32(m2T);
			sumT += _mm_cvtsi128_si32(_mm_shuffle_epi32(m2T, _MM_SHUFFLE(2, 3, 0, 1)));
		}
	}


	INLINED static int FastEstimateTransparent_Pair(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mCx, mCy;
		Macro_InterpolateCX_Pair(mD, mV, mCx, mCy, x);

		__m128i mc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)source));

		mCx = Macro_ExpandC(mCx);
		mCy = Macro_ExpandC(mCy);

		__m128i mmask = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)((const uint8_t*)source + _ImageToMaskDelta)));

		//

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);
		__m128i mBT = _mm_and_si128(mAT, _mm_load_si128((const __m128i*)g_Transparent2_I16));

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAT = _mm_sub_epi16(mAT, mc);
		mBT = _mm_sub_epi16(mBT, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAT = _mm_mullo_epi16(mAT, mAT);
		mBT = _mm_mullo_epi16(mBT, mBT);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_AGRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAT = _mm_min_epu16(mAT, mlimit);
		mBT = _mm_min_epu16(mBT, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAT = _mm_madd_epi16(mAT, magrb);
		mBT = _mm_madd_epi16(mBT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2T = _mm_hadd_epi32(mAT, mBT);

		{
			m2T = _mm_min_epi32(m2T, m2);
			m2T = _mm_min_epi32(m2T, _mm_shuffle_epi32(m2T, _MM_SHUFFLE(1, 0, 3, 2)));

			return _mm_cvtsi128_si32(m2T) + _mm_cvtsi128_si32(_mm_shuffle_epi32(m2T, _MM_SHUFFLE(2, 3, 0, 1)));
		}
	}

	INLINED static int FastEstimateOpaque_Pair(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mCx, mCy;
		Macro_InterpolateCX_Pair(mD, mV, mCx, mCy, x);

		__m128i mc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)source));

		mCx = Macro_ExpandC(mCx);
		mCy = Macro_ExpandC(mCy);

		__m128i mmask = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)((const uint8_t*)source + _ImageToMaskDelta)));

		//

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_AGRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAC = _mm_min_epu16(mAC, mlimit);
		mBC = _mm_min_epu16(mBC, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAC = _mm_madd_epi16(mAC, magrb);
		mBC = _mm_madd_epi16(mBC, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);

		{
			m2C = _mm_min_epi32(m2C, m2);
			m2C = _mm_min_epi32(m2C, _mm_shuffle_epi32(m2C, _MM_SHUFFLE(1, 0, 3, 2)));

			return _mm_cvtsi128_si32(m2C) + _mm_cvtsi128_si32(_mm_shuffle_epi32(m2C, _MM_SHUFFLE(2, 3, 0, 1)));
		}
	}

	INLINED static int FastEstimateTransparent_Duo(__m128i mD, __m128i mV, int x, const int* source0, const int* source1)
	{
		__m128i mCx, mCy;
		Macro_InterpolateCX_Pair(mD, mV, mCx, mCy, x);

		__m128i mcx = _mm_cvtsi32_si128(*source0);
		__m128i mcy = _mm_cvtsi32_si128(*source1);
		__m128i mc = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(mcx, mcy));

		mCx = Macro_ExpandC(mCx);
		mCy = Macro_ExpandC(mCy);

		__m128i mmaskx = _mm_cvtsi32_si128(*(const int*)((const uint8_t*)source0 + _ImageToMaskDelta));
		__m128i mmasky = _mm_cvtsi32_si128(*(const int*)((const uint8_t*)source1 + _ImageToMaskDelta));
		__m128i mmask = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(mmaskx, mmasky));

		//

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);
		__m128i mBT = _mm_and_si128(mAT, _mm_load_si128((const __m128i*)g_Transparent2_I16));

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAT = _mm_sub_epi16(mAT, mc);
		mBT = _mm_sub_epi16(mBT, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAT = _mm_mullo_epi16(mAT, mAT);
		mBT = _mm_mullo_epi16(mBT, mBT);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_AGRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAT = _mm_min_epu16(mAT, mlimit);
		mBT = _mm_min_epu16(mBT, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAT = _mm_madd_epi16(mAT, magrb);
		mBT = _mm_madd_epi16(mBT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2T = _mm_hadd_epi32(mAT, mBT);

		{
			m2T = _mm_min_epi32(m2T, m2);
			m2T = _mm_min_epi32(m2T, _mm_shuffle_epi32(m2T, _MM_SHUFFLE(1, 0, 3, 2)));

			return _mm_cvtsi128_si32(m2T) + _mm_cvtsi128_si32(_mm_shuffle_epi32(m2T, _MM_SHUFFLE(2, 3, 0, 1)));
		}
	}

	INLINED static int FastEstimateOpaque_Duo(__m128i mD, __m128i mV, int x, const int* source0, const int* source1)
	{
		__m128i mCx, mCy;
		Macro_InterpolateCX_Pair(mD, mV, mCx, mCy, x);

		__m128i mcx = _mm_cvtsi32_si128(*source0);
		__m128i mcy = _mm_cvtsi32_si128(*source1);
		__m128i mc = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(mcx, mcy));

		mCx = Macro_ExpandC(mCx);
		mCy = Macro_ExpandC(mCy);

		__m128i mmaskx = _mm_cvtsi32_si128(*(const int*)((const uint8_t*)source0 + _ImageToMaskDelta));
		__m128i mmasky = _mm_cvtsi32_si128(*(const int*)((const uint8_t*)source1 + _ImageToMaskDelta));
		__m128i mmask = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(mmaskx, mmasky));

		//

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_AGRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAC = _mm_min_epu16(mAC, mlimit);
		mBC = _mm_min_epu16(mBC, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAC = _mm_madd_epi16(mAC, magrb);
		mBC = _mm_madd_epi16(mBC, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);

		{
			m2C = _mm_min_epi32(m2C, m2);
			m2C = _mm_min_epi32(m2C, _mm_shuffle_epi32(m2C, _MM_SHUFFLE(1, 0, 3, 2)));

			return _mm_cvtsi128_si32(m2C) + _mm_cvtsi128_si32(_mm_shuffle_epi32(m2C, _MM_SHUFFLE(2, 3, 0, 1)));
		}
	}

	INLINED static int FastEstimateTransparent(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mC = Macro_ExpandC(Macro_InterpolateCX(mD, mV, x));

		__m128i mc = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*source));
		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_unpacklo_epi64(mc, mc);

		__m128i mA = mC;
		__m128i mB = _mm_unpackhi_epi64(mC, mC);

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);
		__m128i mBT = _mm_insert_epi16(mAT, 0, 0);

		__m128i m2 = _mm_unpacklo_epi64(mA, mB);
		__m128i m2T = _mm_unpacklo_epi64(mAT, mBT);

		__m128i mmask = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)source + _ImageToMaskDelta)));
		mmask = _mm_unpacklo_epi64(mmask, mmask);

		m2 = _mm_sub_epi16(m2, mc);
		m2T = _mm_sub_epi16(m2T, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		m2 = _mm_mullo_epi16(m2, m2);
		m2T = _mm_mullo_epi16(m2T, m2T);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_AGRB_I16));

		m2 = _mm_min_epu16(m2, mlimit);
		m2T = _mm_min_epu16(m2T, mlimit);

		m2 = _mm_madd_epi16(m2, magrb);
		m2T = _mm_madd_epi16(m2T, magrb);

		__m128i me4 = _mm_hadd_epi32(m2, m2T);
		__m128i me1 = HorizontalMinimum4(me4);

		return _mm_cvtsi128_si32(me1);
	}

	INLINED static int FastEstimateOpaque(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mC = Macro_ExpandC(Macro_InterpolateCX(mD, mV, x));

		__m128i mc = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*source));
		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_unpacklo_epi64(mc, mc);

		__m128i mA = mC;
		__m128i mB = _mm_unpackhi_epi64(mC, mC);

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		__m128i m2 = _mm_unpacklo_epi64(mA, mB);
		__m128i m2C = _mm_unpacklo_epi64(mAC, mBC);

		__m128i mmask = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)source + _ImageToMaskDelta)));
		mmask = _mm_unpacklo_epi64(mmask, mmask);

		m2 = _mm_sub_epi16(m2, mc);
		m2C = _mm_sub_epi16(m2C, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		m2 = _mm_mullo_epi16(m2, m2);
		m2C = _mm_mullo_epi16(m2C, m2C);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_AGRB_I16));

		m2 = _mm_min_epu16(m2, mlimit);
		m2C = _mm_min_epu16(m2C, mlimit);

		m2 = _mm_madd_epi16(m2, magrb);
		m2C = _mm_madd_epi16(m2C, magrb);

		__m128i me4 = _mm_hadd_epi32(m2, m2C);
		__m128i me1 = HorizontalMinimum4(me4);

		return _mm_cvtsi128_si32(me1);
	}

	INLINED void FastEstimateOpaqueTransparent2x2(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x, int& sumO, int& sumT) const
	{
		__m128i mDL, mDR, mVL, mVR;
		Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

		__m128i mD0, mV0, mD1, mV1;
		Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

		const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

		FastEstimateOpaqueTransparent_Pair(mD0, mV0, x, sumO, sumT, source);
		FastEstimateOpaqueTransparent_Pair(mD1, mV1, x, sumO, sumT, source + _Size);
	}

	INLINED int FastEstimate2x2(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (Data[1] & 1u)
		{
			if (IsEmpty)
				return 0;

			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD0, mV0, mD1, mV1;
			Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateTransparent_Pair(mD0, mV0, x, source);
			sum += FastEstimateTransparent_Pair(mD1, mV1, x, source + _Size);

			return sum;
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD0, mV0, mD1, mV1;
			Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateOpaque_Pair(mD0, mV0, x, source);
			sum += FastEstimateOpaque_Pair(mD1, mV1, x, source + _Size);

			return sum;
		}
	}

	INLINED int FastEstimate2x1(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (Data[1] & 1u)
		{
			if (IsEmpty)
				return 0;

			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateTransparent_Pair(mD, mV, x, source);

			return sum;
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateOpaque_Pair(mD, mV, x, source);

			return sum;
		}
	}

	INLINED int FastEstimate1x2(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (Data[1] & 1u)
		{
			if (IsEmpty)
				return 0;

			__m128i mDT, mDB, mVT, mVB;
			Macro_Gradient(mDT, mDB, mVT, mVB, b00, b10, b01, b11, x, y);

			__m128i mD, mV;
			Macro_InterpolateCY(mDT, mDB, mVT, mVB, mD, mV, x);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateTransparent_Duo(mD, mV, y, source, source + _Size);

			return sum;
		}
		else
		{
			__m128i mDT, mDB, mVT, mVB;
			Macro_Gradient(mDT, mDB, mVT, mVB, b00, b10, b01, b11, x, y);

			__m128i mD, mV;
			Macro_InterpolateCY(mDT, mDB, mVT, mVB, mD, mV, x);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateOpaque_Duo(mD, mV, y, source, source + _Size);

			return sum;
		}
	}

	INLINED int FastEstimate1x1(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (Data[1] & 1u)
		{
			if (IsEmpty)
				return 0;

			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			return FastEstimateTransparent(mD, mV, x, source);
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			return FastEstimateOpaque(mD, mV, x, source);
		}
	}

	//////////////////////
	// Precise Estimate //
	//////////////////////

	INLINED static void PreciseEstimateOpaqueTransparent(__m128i mD, __m128i mV, int x, int& sumO, int& sumT, const int* source)
	{
		__m128i mC = Macro_ExpandC(Macro_InterpolateCX(mD, mV, x));

		__m128i mc = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*source));
		mc = _mm_shuffle_epi32(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mA = mC;
		__m128i mB = _mm_unpackhi_epi64(mA, mA);

		mA = _mm_cvtepi16_epi32(mA);
		mB = _mm_cvtepi16_epi32(mB);

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);
		__m128i mBT = _mm_insert_epi16(mAT, 0, 0);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);
		mAT = _mm_sub_epi16(mAT, mc);
		mBT = _mm_sub_epi16(mBT, mc);

		__m128i magrb = _mm_load_si128((const __m128i*)g_AGRB);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);
		mAT = _mm_mullo_epi16(mAT, mAT);
		mBT = _mm_mullo_epi16(mBT, mBT);

		magrb = _mm_and_si128(magrb, _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)source + _ImageToMaskDelta))));

		mA = _mm_mullo_epi32(mA, magrb);
		mB = _mm_mullo_epi32(mB, magrb);
		mAC = _mm_mullo_epi32(mAC, magrb);
		mBC = _mm_mullo_epi32(mBC, magrb);
		mAT = _mm_mullo_epi32(mAT, magrb);
		mBT = _mm_mullo_epi32(mBT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);
		__m128i m2T = _mm_hadd_epi32(mAT, mBT);

		{
			__m128i me4 = _mm_hadd_epi32(m2, m2C);
			__m128i me1 = HorizontalMinimum4(me4);

			sumO += _mm_cvtsi128_si32(me1);
		}
		{
			__m128i me4 = _mm_hadd_epi32(m2, m2T);
			__m128i me1 = HorizontalMinimum4(me4);

			sumT += _mm_cvtsi128_si32(me1);
		}
	}


	INLINED static int PreciseEstimateTransparent(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mC = Macro_ExpandC(Macro_InterpolateCX(mD, mV, x));

		__m128i mc = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*source));
		mc = _mm_shuffle_epi32(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mA = mC;
		__m128i mB = _mm_unpackhi_epi64(mA, mA);

		mA = _mm_cvtepi16_epi32(mA);
		mB = _mm_cvtepi16_epi32(mB);

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);
		__m128i mBT = _mm_insert_epi16(mAT, 0, 0);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAT = _mm_sub_epi16(mAT, mc);
		mBT = _mm_sub_epi16(mBT, mc);

		__m128i magrb = _mm_load_si128((const __m128i*)g_AGRB);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAT = _mm_mullo_epi16(mAT, mAT);
		mBT = _mm_mullo_epi16(mBT, mBT);

		magrb = _mm_and_si128(magrb, _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)source + _ImageToMaskDelta))));

		mA = _mm_mullo_epi32(mA, magrb);
		mB = _mm_mullo_epi32(mB, magrb);
		mAT = _mm_mullo_epi32(mAT, magrb);
		mBT = _mm_mullo_epi32(mBT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2T = _mm_hadd_epi32(mAT, mBT);

		__m128i me4 = _mm_hadd_epi32(m2, m2T);
		__m128i me1 = HorizontalMinimum4(me4);

		return _mm_cvtsi128_si32(me1);
	}

	INLINED static int PreciseEstimateOpaque(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mC = Macro_ExpandC(Macro_InterpolateCX(mD, mV, x));

		__m128i mc = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*source));
		mc = _mm_shuffle_epi32(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mA = mC;
		__m128i mB = _mm_unpackhi_epi64(mA, mA);

		mA = _mm_cvtepi16_epi32(mA);
		mB = _mm_cvtepi16_epi32(mB);

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);

		__m128i magrb = _mm_load_si128((const __m128i*)g_AGRB);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);

		magrb = _mm_and_si128(magrb, _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)source + _ImageToMaskDelta))));

		mA = _mm_mullo_epi32(mA, magrb);
		mB = _mm_mullo_epi32(mB, magrb);
		mAC = _mm_mullo_epi32(mAC, magrb);
		mBC = _mm_mullo_epi32(mBC, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);

		__m128i me4 = _mm_hadd_epi32(m2, m2C);
		__m128i me1 = HorizontalMinimum4(me4);

		return _mm_cvtsi128_si32(me1);
	}

	INLINED void PreciseEstimateOpaqueTransparent2x2(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x, int& sumO, int& sumT) const
	{
		__m128i mDL, mDR, mVL, mVR;
		Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

		__m128i mD0, mV0, mD1, mV1;
		Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

		const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

		PreciseEstimateOpaqueTransparent(mD0, mV0, x + 0, sumO, sumT, source + 0);
		PreciseEstimateOpaqueTransparent(mD0, mV0, x + 1, sumO, sumT, source + 1);

		source += _Size;

		PreciseEstimateOpaqueTransparent(mD1, mV1, x + 0, sumO, sumT, source + 0);
		PreciseEstimateOpaqueTransparent(mD1, mV1, x + 1, sumO, sumT, source + 1);
	}

	INLINED int PreciseEstimate2x2(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (Data[1] & 1u)
		{
			if (IsEmpty)
				return 0;

			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD0, mV0, mD1, mV1;
			Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateTransparent(mD0, mV0, x + 0, source + 0);
			sum += PreciseEstimateTransparent(mD0, mV0, x + 1, source + 1);

			source += _Size;

			sum += PreciseEstimateTransparent(mD1, mV1, x + 0, source + 0);
			sum += PreciseEstimateTransparent(mD1, mV1, x + 1, source + 1);

			return sum;
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD0, mV0, mD1, mV1;
			Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateOpaque(mD0, mV0, x + 0, source + 0);
			sum += PreciseEstimateOpaque(mD0, mV0, x + 1, source + 1);

			source += _Size;

			sum += PreciseEstimateOpaque(mD1, mV1, x + 0, source + 0);
			sum += PreciseEstimateOpaque(mD1, mV1, x + 1, source + 1);

			return sum;
		}
	}

	INLINED int PreciseEstimate2x1(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (Data[1] & 1u)
		{
			if (IsEmpty)
				return 0;

			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateTransparent(mD, mV, x + 0, source + 0);
			sum += PreciseEstimateTransparent(mD, mV, x + 1, source + 1);

			return sum;
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateOpaque(mD, mV, x + 0, source + 0);
			sum += PreciseEstimateOpaque(mD, mV, x + 1, source + 1);

			return sum;
		}
	}

	INLINED int PreciseEstimate1x2(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (Data[1] & 1u)
		{
			if (IsEmpty)
				return 0;

			__m128i mDT, mDB, mVT, mVB;
			Macro_Gradient(mDT, mDB, mVT, mVB, b00, b10, b01, b11, x, y);

			__m128i mD, mV;
			Macro_InterpolateCY(mDT, mDB, mVT, mVB, mD, mV, x);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateTransparent(mD, mV, y + 0, source);
			sum += PreciseEstimateTransparent(mD, mV, y + 1, source + _Size);

			return sum;
		}
		else
		{
			__m128i mDT, mDB, mVT, mVB;
			Macro_Gradient(mDT, mDB, mVT, mVB, b00, b10, b01, b11, x, y);

			__m128i mD, mV;
			Macro_InterpolateCY(mDT, mDB, mVT, mVB, mD, mV, x);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateOpaque(mD, mV, y + 0, source);
			sum += PreciseEstimateOpaque(mD, mV, y + 1, source + _Size);

			return sum;
		}
	}

	INLINED int PreciseEstimate1x1(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (Data[1] & 1u)
		{
			if (IsEmpty)
				return 0;

			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			return PreciseEstimateTransparent(mD, mV, x, source);
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			return PreciseEstimateOpaque(mD, mV, x, source);
		}
	}

	int64_t Estimate7x7(int64_t water) const
	{
		int64_t sum4, sum9;

		int64_t sum = 0;

		if (!IsEmpty)
		{
			int sumO = 0, sumT = 0;

			FastEstimateOpaqueTransparent2x2(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 0, 0, sumO, sumT);
			sum = (sumO <= sumT) ? sumO : sumT;
			if (sum >= water)
				return sum;

			FastEstimateOpaqueTransparent2x2(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 0, 2, sumO, sumT);
			sum = (sumO <= sumT) ? sumO : sumT;
			if (sum >= water)
				return sum;

			FastEstimateOpaqueTransparent2x2(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 2, 0, sumO, sumT);
			sum = (sumO <= sumT) ? sumO : sumT;
			if (sum >= water)
				return sum;

			FastEstimateOpaqueTransparent2x2(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 2, 2, sumO, sumT);
			sum = (sumO <= sumT) ? sumO : sumT;
			if (sum >= water)
				return sum;
		}

		sum4 = sum;

		sum += Matrix[1][2]->FastEstimate2x2(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 0, 0);
		if (sum >= water)
			return sum;

		sum += Matrix[1][2]->FastEstimate2x2(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 2, 0);
		if (sum >= water)
			return sum;

		sum += Matrix[2][1]->FastEstimate2x2(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 0, 0);
		if (sum >= water)
			return sum;

		sum += Matrix[2][1]->FastEstimate2x2(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 0, 2);
		if (sum >= water)
			return sum;

		sum += Matrix[2][2]->FastEstimate2x2(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 0, 0);
		if (sum >= water)
			return sum;

		sum9 = sum;

		//

		sum += Matrix[0][1]->FastEstimate2x1(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 3, 0);
		if (sum >= water)
			return sum;

		sum += Matrix[0][1]->FastEstimate2x1(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 3, 2);
		if (sum >= water)
			return sum;

		sum += Matrix[0][2]->FastEstimate2x1(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 3, 0);
		if (sum >= water)
			return sum;

		//

		sum += Matrix[1][0]->FastEstimate1x2(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 0, 3);
		if (sum >= water)
			return sum;

		sum += Matrix[1][0]->FastEstimate1x2(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 2, 3);
		if (sum >= water)
			return sum;

		sum += Matrix[2][0]->FastEstimate1x2(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 0, 3);
		if (sum >= water)
			return sum;

		//

		sum += Matrix[0][0]->FastEstimate1x1(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 3, 3);
		if (sum >= water)
			return sum;

		if (sum < 0x7FFF * Min(Min(kGreen, kRed), kBlue))
		{
			return sum;
		}

		sum = 0;

		if (sum9 < 0x7FFF * Min(Min(kGreen, kRed), kBlue))
		{
			sum = sum9;
		}
		else
		{
			if (sum4 < 0x7FFF * Min(Min(kGreen, kRed), kBlue))
			{
				sum = sum4;
			}
			else
			{
				if (!IsEmpty)
				{
					int sumO = 0, sumT = 0;

					PreciseEstimateOpaqueTransparent2x2(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 0, 0, sumO, sumT);
					PreciseEstimateOpaqueTransparent2x2(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 0, 2, sumO, sumT);
					PreciseEstimateOpaqueTransparent2x2(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 2, 0, sumO, sumT);
					PreciseEstimateOpaqueTransparent2x2(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 2, 2, sumO, sumT);

					sum = (sumO <= sumT) ? sumO : sumT;
					if (sum >= water)
						return sum;
				}
			}

			sum += Matrix[1][2]->PreciseEstimate2x2(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 0, 0);
			sum += Matrix[1][2]->PreciseEstimate2x2(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 2, 0);
			sum += Matrix[2][1]->PreciseEstimate2x2(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 0, 0);
			sum += Matrix[2][1]->PreciseEstimate2x2(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 0, 2);
			sum += Matrix[2][2]->PreciseEstimate2x2(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 0, 0);
			if (sum >= water)
				return sum;
		}

		//

		sum += Matrix[0][1]->PreciseEstimate2x1(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 3, 0);
		if (sum >= water)
			return sum;

		sum += Matrix[0][1]->PreciseEstimate2x1(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 3, 2);
		if (sum >= water)
			return sum;

		sum += Matrix[0][2]->PreciseEstimate2x1(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 3, 0);
		if (sum >= water)
			return sum;

		//

		sum += Matrix[1][0]->PreciseEstimate1x2(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 0, 3);
		if (sum >= water)
			return sum;

		sum += Matrix[1][0]->PreciseEstimate1x2(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 2, 3);
		if (sum >= water)
			return sum;

		sum += Matrix[2][0]->PreciseEstimate1x2(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 0, 3);
		if (sum >= water)
			return sum;

		//

		sum += Matrix[0][0]->PreciseEstimate1x1(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 3, 3);
		if (sum >= water)
			return sum;

		return sum;
	}

	//////////////////
	// Climber Test //
	//////////////////

	INLINED bool ClimberTest(__m128i& backup, int64_t& water)
	{
		int64_t error = Estimate7x7(water);
		if (water > error)
		{
			water = error;
			backup = Colors;
			return true;
		}
		else
		{
			Colors = backup;
			return false;
		}
	}

	INLINED void ChangeAlpha(__m128i& backup, int64_t& water, int& changes, int mark, int group, int mask)
	{
		if ((changes != 0) || (mask < mark))
		{
			for (;; )
			{
				int a = M128I_I16(Colors, group);
				if (a <= 0) break;
				if (a >= 0xF)
				{
					if (IsOpaque)
						break;

					M128I_I16(Colors, group) = 0xE;
					M128I_I16(Colors, group + 1) = (short)MakeColor4(M128I_I16(Colors, group + 1));
					M128I_I16(Colors, group + 2) = (short)MakeColor4(M128I_I16(Colors, group + 2));
					if (group == 0)
						M128I_I16(Colors, group + 3) = (short)MakeColor3(M128I_I16(Colors, group + 3));
					else
						M128I_I16(Colors, group + 3) = (short)MakeColor4(M128I_I16(Colors, group + 3));
				}
				else
				{
					a = MakeAlpha3(a - 2);
					M128I_I16(Colors, group) = (short)a;
				}

				if (ClimberTest(backup, water)) changes = mask; else break;
			}
		}

		// because of alpha loop side effects
		if (((changes != 0) || (mask <= mark)) && (changes != mask))
		{
			for (;; )
			{
				int a = M128I_I16(Colors, group);
				if (a >= 0xF) break;
				if (a == 0xE)
				{
					M128I_I16(Colors, group) = 0xF;
					//M128I_I16(Colors, group + 1) = (short)MakeColor5(M128I_I16(Colors, group + 1));
					//M128I_I16(Colors, group + 2) = (short)MakeColor5(M128I_I16(Colors, group + 2));
					if (group == 0)
						M128I_I16(Colors, group + 3) = (short)MakeColor4(M128I_I16(Colors, group + 3));
					//else
					//	M128I_I16(Colors, group + 3) = (short)MakeColor5(M128I_I16(Colors, group + 3));
				}
				else
				{
					a = MakeAlpha3(a + 2);
					M128I_I16(Colors, group) = (short)a;
				}

				if (ClimberTest(backup, water)) changes = mask + mask; else break;
			}
		}
	}

	INLINED void ChangeColors(__m128i& backup, int64_t& water, int& changes, int mark, int mode, bool hasG, bool hasR, bool hasB, int group, int mask)
	{
		if ((changes != 0) || (mask < mark))
		{
			for (;; )
			{
				bool other = false;

				if (hasG)
				{
					int c = M128I_I16(Colors, group + 1);
					if (c > 0)
					{
						c = mode ? c - 1 : MakeColor4(c - 2);
						M128I_I16(Colors, group + 1) = (short)c;
						other = true;
					}
				}

				if (hasR)
				{
					int c = M128I_I16(Colors, group + 2);
					if (c > 0)
					{
						c = mode ? c - 1 : MakeColor4(c - 2);
						M128I_I16(Colors, group + 2) = (short)c;
						other = true;
					}
				}

				if (hasB)
				{
					int c = M128I_I16(Colors, group + 3);
					if (c > 0)
					{
						if (group == 0)
							c = mode ? MakeColor4(c - 2) : MakeColor3(c - 4);
						else
							c = mode ? c - 1 : MakeColor4(c - 2);
						M128I_I16(Colors, group + 3) = (short)c;
						other = true;
					}
				}

				if (!other)
					break;

				if (ClimberTest(backup, water)) changes = mask; else break;
			}
		}

		if (((changes != 0) || (mask + mask < mark)) && (changes != mask))
		{
			for (;; )
			{
				bool other = false;

				if (hasG)
				{
					int c = M128I_I16(Colors, group + 1);
					if (c < 0x1F)
					{
						c = mode ? c + 1 : MakeColor4(c + 2);
						M128I_I16(Colors, group + 1) = (short)c;
						other = true;
					}
				}

				if (hasR)
				{
					int c = M128I_I16(Colors, group + 2);
					if (c < 0x1F)
					{
						c = mode ? c + 1 : MakeColor4(c + 2);
						M128I_I16(Colors, group + 2) = (short)c;
						other = true;
					}
				}

				if (hasB)
				{
					int c = M128I_I16(Colors, group + 3);
					if (c < 0x1F)
					{
						if (group == 0)
							c = mode ? MakeColor4(c + 2) : MakeColor3(c + 4);
						else
							c = mode ? c + 1 : MakeColor4(c + 2);
						M128I_I16(Colors, group + 3) = (short)c;
						other = true;
					}
				}

				if (!other)
					break;

				if (ClimberTest(backup, water)) changes = mask + mask; else break;
			}
		}
	}

	///////////////////////
	// Fast Estimate GRB //
	///////////////////////

	INLINED static void FastEstimateOpaqueTransparent_GRB_Pair(__m128i mD, __m128i mV, int x, int& sumO, int& sumT, const int* source)
	{
		__m128i mCx, mCy;
		Macro_InterpolateCX_Pair(mD, mV, mCx, mCy, x);

		__m128i mc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)source));

		mCx = Macro_ExpandC_GRB(mCx);
		mCy = Macro_ExpandC_GRB(mCy);

		__m128i mmask = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)((const uint8_t*)source + _ImageToMaskDelta)));

		//

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);
		mAT = _mm_sub_epi16(mAT, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);
		mAT = _mm_mullo_epi16(mAT, mAT);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_GRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAC = _mm_min_epu16(mAC, mlimit);
		mBC = _mm_min_epu16(mBC, mlimit);
		mAT = _mm_min_epu16(mAT, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAC = _mm_madd_epi16(mAC, magrb);
		mBC = _mm_madd_epi16(mBC, magrb);
		mAT = _mm_madd_epi16(mAT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);
		__m128i m2T = _mm_hadd_epi32(mAT, mAT);

		{
			m2C = _mm_min_epi32(m2C, m2);
			m2C = _mm_min_epi32(m2C, _mm_shuffle_epi32(m2C, _MM_SHUFFLE(1, 0, 3, 2)));

			sumO += _mm_cvtsi128_si32(m2C);
			sumO += _mm_cvtsi128_si32(_mm_shuffle_epi32(m2C, _MM_SHUFFLE(2, 3, 0, 1)));
		}
		{
			m2T = _mm_min_epi32(m2T, m2);
			m2T = _mm_min_epi32(m2T, _mm_shuffle_epi32(m2T, _MM_SHUFFLE(1, 0, 3, 2)));

			sumT += _mm_cvtsi128_si32(m2T);
			sumT += _mm_cvtsi128_si32(_mm_shuffle_epi32(m2T, _MM_SHUFFLE(2, 3, 0, 1)));
		}
	}


	INLINED static int FastEstimateTransparent_GRB_Pair(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mCx, mCy;
		Macro_InterpolateCX_Pair(mD, mV, mCx, mCy, x);

		__m128i mc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)source));

		mCx = Macro_ExpandC_GRB(mCx);
		mCy = Macro_ExpandC_GRB(mCy);

		__m128i mmask = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)((const uint8_t*)source + _ImageToMaskDelta)));

		//

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAT = _mm_sub_epi16(mAT, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAT = _mm_mullo_epi16(mAT, mAT);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_GRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAT = _mm_min_epu16(mAT, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAT = _mm_madd_epi16(mAT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2T = _mm_hadd_epi32(mAT, mAT);

		{
			m2T = _mm_min_epi32(m2T, m2);
			m2T = _mm_min_epi32(m2T, _mm_shuffle_epi32(m2T, _MM_SHUFFLE(1, 0, 3, 2)));

			return _mm_cvtsi128_si32(m2T) + _mm_cvtsi128_si32(_mm_shuffle_epi32(m2T, _MM_SHUFFLE(2, 3, 0, 1)));
		}
	}

	INLINED static int FastEstimateOpaque_GRB_Pair(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mCx, mCy;
		Macro_InterpolateCX_Pair(mD, mV, mCx, mCy, x);

		__m128i mc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)source));

		mCx = Macro_ExpandC_GRB(mCx);
		mCy = Macro_ExpandC_GRB(mCy);

		__m128i mmask = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)((const uint8_t*)source + _ImageToMaskDelta)));

		//

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_GRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAC = _mm_min_epu16(mAC, mlimit);
		mBC = _mm_min_epu16(mBC, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAC = _mm_madd_epi16(mAC, magrb);
		mBC = _mm_madd_epi16(mBC, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);

		{
			m2C = _mm_min_epi32(m2C, m2);
			m2C = _mm_min_epi32(m2C, _mm_shuffle_epi32(m2C, _MM_SHUFFLE(1, 0, 3, 2)));

			return _mm_cvtsi128_si32(m2C) + _mm_cvtsi128_si32(_mm_shuffle_epi32(m2C, _MM_SHUFFLE(2, 3, 0, 1)));
		}
	}

	INLINED static int FastEstimateTransparent_GRB_Duo(__m128i mD, __m128i mV, int x, const int* source0, const int* source1)
	{
		__m128i mCx, mCy;
		Macro_InterpolateCX_Pair(mD, mV, mCx, mCy, x);

		__m128i mcx = _mm_cvtsi32_si128(*source0);
		__m128i mcy = _mm_cvtsi32_si128(*source1);
		__m128i mc = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(mcx, mcy));

		mCx = Macro_ExpandC_GRB(mCx);
		mCy = Macro_ExpandC_GRB(mCy);

		__m128i mmaskx = _mm_cvtsi32_si128(*(const int*)((const uint8_t*)source0 + _ImageToMaskDelta));
		__m128i mmasky = _mm_cvtsi32_si128(*(const int*)((const uint8_t*)source1 + _ImageToMaskDelta));
		__m128i mmask = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(mmaskx, mmasky));

		//

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAT = _mm_sub_epi16(mAT, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAT = _mm_mullo_epi16(mAT, mAT);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_GRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAT = _mm_min_epu16(mAT, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAT = _mm_madd_epi16(mAT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2T = _mm_hadd_epi32(mAT, mAT);

		{
			m2T = _mm_min_epi32(m2T, m2);
			m2T = _mm_min_epi32(m2T, _mm_shuffle_epi32(m2T, _MM_SHUFFLE(1, 0, 3, 2)));

			return _mm_cvtsi128_si32(m2T) + _mm_cvtsi128_si32(_mm_shuffle_epi32(m2T, _MM_SHUFFLE(2, 3, 0, 1)));
		}
	}

	INLINED static int FastEstimateOpaque_GRB_Duo(__m128i mD, __m128i mV, int x, const int* source0, const int* source1)
	{
		__m128i mCx, mCy;
		Macro_InterpolateCX_Pair(mD, mV, mCx, mCy, x);

		__m128i mcx = _mm_cvtsi32_si128(*source0);
		__m128i mcy = _mm_cvtsi32_si128(*source1);
		__m128i mc = _mm_cvtepu8_epi16(_mm_unpacklo_epi32(mcx, mcy));

		mCx = Macro_ExpandC_GRB(mCx);
		mCy = Macro_ExpandC_GRB(mCy);

		__m128i mmaskx = _mm_cvtsi32_si128(*(const int*)((const uint8_t*)source0 + _ImageToMaskDelta));
		__m128i mmasky = _mm_cvtsi32_si128(*(const int*)((const uint8_t*)source1 + _ImageToMaskDelta));
		__m128i mmask = _mm_cvtepi8_epi16(_mm_unpacklo_epi32(mmaskx, mmasky));

		//

		__m128i mA = _mm_unpacklo_epi64(mCx, mCy);
		__m128i mB = _mm_unpackhi_epi64(mCx, mCy);

		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_shufflehi_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_GRB_I16));

		mA = _mm_min_epu16(mA, mlimit);
		mB = _mm_min_epu16(mB, mlimit);
		mAC = _mm_min_epu16(mAC, mlimit);
		mBC = _mm_min_epu16(mBC, mlimit);

		mA = _mm_madd_epi16(mA, magrb);
		mB = _mm_madd_epi16(mB, magrb);
		mAC = _mm_madd_epi16(mAC, magrb);
		mBC = _mm_madd_epi16(mBC, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);

		{
			m2C = _mm_min_epi32(m2C, m2);
			m2C = _mm_min_epi32(m2C, _mm_shuffle_epi32(m2C, _MM_SHUFFLE(1, 0, 3, 2)));

			return _mm_cvtsi128_si32(m2C) + _mm_cvtsi128_si32(_mm_shuffle_epi32(m2C, _MM_SHUFFLE(2, 3, 0, 1)));
		}
	}

	INLINED static int FastEstimateTransparent_GRB(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mC = Macro_ExpandC_GRB(Macro_InterpolateCX(mD, mV, x));

		__m128i mc = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*source));
		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_unpacklo_epi64(mc, mc);

		__m128i mA = mC;
		__m128i mB = _mm_unpackhi_epi64(mC, mC);

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);

		__m128i m2 = _mm_unpacklo_epi64(mA, mB);
		__m128i m2T = _mm_unpacklo_epi64(mAT, mAT);

		__m128i mmask = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)source + _ImageToMaskDelta)));
		mmask = _mm_unpacklo_epi64(mmask, mmask);

		m2 = _mm_sub_epi16(m2, mc);
		m2T = _mm_sub_epi16(m2T, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		m2 = _mm_mullo_epi16(m2, m2);
		m2T = _mm_mullo_epi16(m2T, m2T);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_GRB_I16));

		m2 = _mm_min_epu16(m2, mlimit);
		m2T = _mm_min_epu16(m2T, mlimit);

		m2 = _mm_madd_epi16(m2, magrb);
		m2T = _mm_madd_epi16(m2T, magrb);

		__m128i me4 = _mm_hadd_epi32(m2, m2T);
		__m128i me1 = HorizontalMinimum4(me4);

		return _mm_cvtsi128_si32(me1);
	}

	INLINED static int FastEstimateOpaque_GRB(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mC = Macro_ExpandC_GRB(Macro_InterpolateCX(mD, mV, x));

		__m128i mc = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*source));
		mc = _mm_shufflelo_epi16(mc, _MM_SHUFFLE(0, 2, 1, 3));
		mc = _mm_unpacklo_epi64(mc, mc);

		__m128i mA = mC;
		__m128i mB = _mm_unpackhi_epi64(mC, mC);

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		__m128i m2 = _mm_unpacklo_epi64(mA, mB);
		__m128i m2C = _mm_unpacklo_epi64(mAC, mBC);

		__m128i mmask = _mm_cvtepi8_epi16(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)source + _ImageToMaskDelta)));
		mmask = _mm_unpacklo_epi64(mmask, mmask);

		m2 = _mm_sub_epi16(m2, mc);
		m2C = _mm_sub_epi16(m2C, mc);

		__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

		m2 = _mm_mullo_epi16(m2, m2);
		m2C = _mm_mullo_epi16(m2C, m2C);

		__m128i magrb = _mm_and_si128(mmask, _mm_load_si128((const __m128i*)g_GRB_I16));

		m2 = _mm_min_epu16(m2, mlimit);
		m2C = _mm_min_epu16(m2C, mlimit);

		m2 = _mm_madd_epi16(m2, magrb);
		m2C = _mm_madd_epi16(m2C, magrb);

		__m128i me4 = _mm_hadd_epi32(m2, m2C);
		__m128i me1 = HorizontalMinimum4(me4);

		return _mm_cvtsi128_si32(me1);
	}

	INLINED void FastEstimateOpaqueTransparent2x2_GRB(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x, int& sumO, int& sumT) const
	{
		__m128i mDL, mDR, mVL, mVR;
		Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

		__m128i mD0, mV0, mD1, mV1;
		Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

		const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

		FastEstimateOpaqueTransparent_GRB_Pair(mD0, mV0, x, sumO, sumT, source);
		FastEstimateOpaqueTransparent_GRB_Pair(mD1, mV1, x, sumO, sumT, source + _Size);
	}

	INLINED int FastEstimate2x2_GRB(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (IsEmpty)
			return 0;

		if (Data[1] & 1u)
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD0, mV0, mD1, mV1;
			Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateTransparent_GRB_Pair(mD0, mV0, x, source);
			sum += FastEstimateTransparent_GRB_Pair(mD1, mV1, x, source + _Size);

			return sum;
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD0, mV0, mD1, mV1;
			Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateOpaque_GRB_Pair(mD0, mV0, x, source);
			sum += FastEstimateOpaque_GRB_Pair(mD1, mV1, x, source + _Size);

			return sum;
		}
	}

	INLINED int FastEstimate2x1_GRB(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (IsEmpty)
			return 0;

		if (Data[1] & 1u)
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateTransparent_GRB_Pair(mD, mV, x, source);

			return sum;
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateOpaque_GRB_Pair(mD, mV, x, source);

			return sum;
		}
	}

	INLINED int FastEstimate1x2_GRB(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (IsEmpty)
			return 0;

		if (Data[1] & 1u)
		{
			__m128i mDT, mDB, mVT, mVB;
			Macro_Gradient(mDT, mDB, mVT, mVB, b00, b10, b01, b11, x, y);

			__m128i mD, mV;
			Macro_InterpolateCY(mDT, mDB, mVT, mVB, mD, mV, x);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateTransparent_GRB_Duo(mD, mV, y, source, source + _Size);

			return sum;
		}
		else
		{
			__m128i mDT, mDB, mVT, mVB;
			Macro_Gradient(mDT, mDB, mVT, mVB, b00, b10, b01, b11, x, y);

			__m128i mD, mV;
			Macro_InterpolateCY(mDT, mDB, mVT, mVB, mD, mV, x);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = FastEstimateOpaque_GRB_Duo(mD, mV, y, source, source + _Size);

			return sum;
		}
	}

	INLINED int FastEstimate1x1_GRB(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (IsEmpty)
			return 0;

		if (Data[1] & 1u)
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			return FastEstimateTransparent_GRB(mD, mV, x, source);
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			return FastEstimateOpaque_GRB(mD, mV, x, source);
		}
	}

	//////////////////////////
	// Precise Estimate GRB //
	//////////////////////////

	INLINED static void PreciseEstimateOpaqueTransparent_GRB(__m128i mD, __m128i mV, int x, int& sumO, int& sumT, const int* source)
	{
		__m128i mC = Macro_ExpandC_GRB(Macro_InterpolateCX(mD, mV, x));

		__m128i mc = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*source));
		mc = _mm_shuffle_epi32(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mA = mC;
		__m128i mB = _mm_unpackhi_epi64(mA, mA);

		mA = _mm_cvtepi16_epi32(mA);
		mB = _mm_cvtepi16_epi32(mB);

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);
		mAT = _mm_sub_epi16(mAT, mc);

		__m128i magrb = _mm_load_si128((const __m128i*)g_GRB);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);
		mAT = _mm_mullo_epi16(mAT, mAT);

		magrb = _mm_and_si128(magrb, _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)source + _ImageToMaskDelta))));

		mA = _mm_mullo_epi32(mA, magrb);
		mB = _mm_mullo_epi32(mB, magrb);
		mAC = _mm_mullo_epi32(mAC, magrb);
		mBC = _mm_mullo_epi32(mBC, magrb);
		mAT = _mm_mullo_epi32(mAT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);
		__m128i m2T = _mm_hadd_epi32(mAT, mAT);

		{
			__m128i me4 = _mm_hadd_epi32(m2, m2C);
			__m128i me1 = HorizontalMinimum4(me4);

			sumO += _mm_cvtsi128_si32(me1);
		}
		{
			__m128i me4 = _mm_hadd_epi32(m2, m2T);
			__m128i me1 = HorizontalMinimum4(me4);

			sumT += _mm_cvtsi128_si32(me1);
		}
	}


	INLINED static int PreciseEstimateTransparent_GRB(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mC = Macro_ExpandC_GRB(Macro_InterpolateCX(mD, mV, x));

		__m128i mc = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*source));
		mc = _mm_shuffle_epi32(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mA = mC;
		__m128i mB = _mm_unpackhi_epi64(mA, mA);

		mA = _mm_cvtepi16_epi32(mA);
		mB = _mm_cvtepi16_epi32(mB);

		__m128i mAT = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAT = _mm_sub_epi16(mAT, mc);

		__m128i magrb = _mm_load_si128((const __m128i*)g_GRB);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAT = _mm_mullo_epi16(mAT, mAT);

		magrb = _mm_and_si128(magrb, _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)source + _ImageToMaskDelta))));

		mA = _mm_mullo_epi32(mA, magrb);
		mB = _mm_mullo_epi32(mB, magrb);
		mAT = _mm_mullo_epi32(mAT, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2T = _mm_hadd_epi32(mAT, mAT);

		__m128i me4 = _mm_hadd_epi32(m2, m2T);
		__m128i me1 = HorizontalMinimum4(me4);

		return _mm_cvtsi128_si32(me1);
	}

	INLINED static int PreciseEstimateOpaque_GRB(__m128i mD, __m128i mV, int x, const int* source)
	{
		__m128i mC = Macro_ExpandC_GRB(Macro_InterpolateCX(mD, mV, x));

		__m128i mc = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*source));
		mc = _mm_shuffle_epi32(mc, _MM_SHUFFLE(0, 2, 1, 3));

		__m128i mA = mC;
		__m128i mB = _mm_unpackhi_epi64(mA, mA);

		mA = _mm_cvtepi16_epi32(mA);
		mB = _mm_cvtepi16_epi32(mB);

		__m128i mBA = _mm_sub_epi16(mB, mA);
		__m128i mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
		__m128i mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);

		mA = _mm_sub_epi16(mA, mc);
		mB = _mm_sub_epi16(mB, mc);
		mAC = _mm_sub_epi16(mAC, mc);
		mBC = _mm_sub_epi16(mBC, mc);

		__m128i magrb = _mm_load_si128((const __m128i*)g_GRB);

		mA = _mm_mullo_epi16(mA, mA);
		mB = _mm_mullo_epi16(mB, mB);
		mAC = _mm_mullo_epi16(mAC, mAC);
		mBC = _mm_mullo_epi16(mBC, mBC);

		magrb = _mm_and_si128(magrb, _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(const int*)((const uint8_t*)source + _ImageToMaskDelta))));

		mA = _mm_mullo_epi32(mA, magrb);
		mB = _mm_mullo_epi32(mB, magrb);
		mAC = _mm_mullo_epi32(mAC, magrb);
		mBC = _mm_mullo_epi32(mBC, magrb);

		__m128i m2 = _mm_hadd_epi32(mA, mB);
		__m128i m2C = _mm_hadd_epi32(mAC, mBC);

		__m128i me4 = _mm_hadd_epi32(m2, m2C);
		__m128i me1 = HorizontalMinimum4(me4);

		return _mm_cvtsi128_si32(me1);
	}

	INLINED void PreciseEstimateOpaqueTransparent2x2_GRB(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x, int& sumO, int& sumT) const
	{
		__m128i mDL, mDR, mVL, mVR;
		Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

		__m128i mD0, mV0, mD1, mV1;
		Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

		const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

		PreciseEstimateOpaqueTransparent_GRB(mD0, mV0, x + 0, sumO, sumT, source + 0);
		PreciseEstimateOpaqueTransparent_GRB(mD0, mV0, x + 1, sumO, sumT, source + 1);

		source += _Size;

		PreciseEstimateOpaqueTransparent_GRB(mD1, mV1, x + 0, sumO, sumT, source + 0);
		PreciseEstimateOpaqueTransparent_GRB(mD1, mV1, x + 1, sumO, sumT, source + 1);
	}

	INLINED int PreciseEstimate2x2_GRB(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (IsEmpty)
			return 0;

		if (Data[1] & 1u)
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD0, mV0, mD1, mV1;
			Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateTransparent_GRB(mD0, mV0, x + 0, source + 0);
			sum += PreciseEstimateTransparent_GRB(mD0, mV0, x + 1, source + 1);

			source += _Size;

			sum += PreciseEstimateTransparent_GRB(mD1, mV1, x + 0, source + 0);
			sum += PreciseEstimateTransparent_GRB(mD1, mV1, x + 1, source + 1);

			return sum;
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD0, mV0, mD1, mV1;
			Macro_InterpolateCY_Pair(mDL, mDR, mVL, mVR, mD0, mV0, mD1, mV1, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateOpaque_GRB(mD0, mV0, x + 0, source + 0);
			sum += PreciseEstimateOpaque_GRB(mD0, mV0, x + 1, source + 1);

			source += _Size;

			sum += PreciseEstimateOpaque_GRB(mD1, mV1, x + 0, source + 0);
			sum += PreciseEstimateOpaque_GRB(mD1, mV1, x + 1, source + 1);

			return sum;
		}
	}

	INLINED int PreciseEstimate2x1_GRB(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (IsEmpty)
			return 0;

		if (Data[1] & 1u)
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateTransparent_GRB(mD, mV, x + 0, source + 0);
			sum += PreciseEstimateTransparent_GRB(mD, mV, x + 1, source + 1);

			return sum;
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateOpaque_GRB(mD, mV, x + 0, source + 0);
			sum += PreciseEstimateOpaque_GRB(mD, mV, x + 1, source + 1);

			return sum;
		}
	}

	INLINED int PreciseEstimate1x2_GRB(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (IsEmpty)
			return 0;

		if (Data[1] & 1u)
		{
			__m128i mDT, mDB, mVT, mVB;
			Macro_Gradient(mDT, mDB, mVT, mVB, b00, b10, b01, b11, x, y);

			__m128i mD, mV;
			Macro_InterpolateCY(mDT, mDB, mVT, mVB, mD, mV, x);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateTransparent_GRB(mD, mV, y + 0, source);
			sum += PreciseEstimateTransparent_GRB(mD, mV, y + 1, source + _Size);

			return sum;
		}
		else
		{
			__m128i mDT, mDB, mVT, mVB;
			Macro_Gradient(mDT, mDB, mVT, mVB, b00, b10, b01, b11, x, y);

			__m128i mD, mV;
			Macro_InterpolateCY(mDT, mDB, mVT, mVB, mD, mV, x);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			int sum = PreciseEstimateOpaque_GRB(mD, mV, y + 0, source);
			sum += PreciseEstimateOpaque_GRB(mD, mV, y + 1, source + _Size);

			return sum;
		}
	}

	INLINED int PreciseEstimate1x1_GRB(const Block& b00, const Block& b01, const Block& b10, const Block& b11, int y, int x) const
	{
		if (IsEmpty)
			return 0;

		if (Data[1] & 1u)
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			return PreciseEstimateTransparent_GRB(mD, mV, x, source);
		}
		else
		{
			__m128i mDL, mDR, mVL, mVR;
			Macro_Gradient(mDL, mDR, mVL, mVR, b00, b01, b10, b11, y, x);

			__m128i mD, mV;
			Macro_InterpolateCY(mDL, mDR, mVL, mVR, mD, mV, y);

			const int* source = &_Image[((Y << 2) + y) * _Size + ((X << 2) + x)];

			return PreciseEstimateOpaque_GRB(mD, mV, x, source);
		}
	}

	int64_t Estimate7x7_GRB(int64_t water) const
	{
		int64_t sum4, sum9;

		int64_t sum = 0;

		if (!IsEmpty)
		{
			int sumO = 0, sumT = 0;

			FastEstimateOpaqueTransparent2x2_GRB(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 0, 0, sumO, sumT);
			sum = (sumO <= sumT) ? sumO : sumT;
			if (sum >= water)
				return sum;

			FastEstimateOpaqueTransparent2x2_GRB(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 0, 2, sumO, sumT);
			sum = (sumO <= sumT) ? sumO : sumT;
			if (sum >= water)
				return sum;

			FastEstimateOpaqueTransparent2x2_GRB(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 2, 0, sumO, sumT);
			sum = (sumO <= sumT) ? sumO : sumT;
			if (sum >= water)
				return sum;

			FastEstimateOpaqueTransparent2x2_GRB(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 2, 2, sumO, sumT);
			sum = (sumO <= sumT) ? sumO : sumT;
			if (sum >= water)
				return sum;
		}

		sum4 = sum;

		sum += Matrix[1][2]->FastEstimate2x2_GRB(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 0, 0);
		if (sum >= water)
			return sum;

		sum += Matrix[1][2]->FastEstimate2x2_GRB(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 2, 0);
		if (sum >= water)
			return sum;

		sum += Matrix[2][1]->FastEstimate2x2_GRB(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 0, 0);
		if (sum >= water)
			return sum;

		sum += Matrix[2][1]->FastEstimate2x2_GRB(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 0, 2);
		if (sum >= water)
			return sum;

		sum += Matrix[2][2]->FastEstimate2x2_GRB(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 0, 0);
		if (sum >= water)
			return sum;

		sum9 = sum;

		//

		sum += Matrix[0][1]->FastEstimate2x1_GRB(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 3, 0);
		if (sum >= water)
			return sum;

		sum += Matrix[0][1]->FastEstimate2x1_GRB(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 3, 2);
		if (sum >= water)
			return sum;

		sum += Matrix[0][2]->FastEstimate2x1_GRB(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 3, 0);
		if (sum >= water)
			return sum;

		//

		sum += Matrix[1][0]->FastEstimate1x2_GRB(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 0, 3);
		if (sum >= water)
			return sum;

		sum += Matrix[1][0]->FastEstimate1x2_GRB(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 2, 3);
		if (sum >= water)
			return sum;

		sum += Matrix[2][0]->FastEstimate1x2_GRB(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 0, 3);
		if (sum >= water)
			return sum;

		//

		sum += Matrix[0][0]->FastEstimate1x1_GRB(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 3, 3);
		if (sum >= water)
			return sum;

		if (sum < 0x7FFF * Min(Min(kGreen, kRed), kBlue))
		{
			return sum;
		}

		sum = 0;

		if (sum9 < 0x7FFF * Min(Min(kGreen, kRed), kBlue))
		{
			sum = sum9;
		}
		else
		{
			if (sum4 < 0x7FFF * Min(Min(kGreen, kRed), kBlue))
			{
				sum = sum4;
			}
			else
			{
				if (!IsEmpty)
				{
					int sumO = 0, sumT = 0;

					PreciseEstimateOpaqueTransparent2x2_GRB(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 0, 0, sumO, sumT);
					PreciseEstimateOpaqueTransparent2x2_GRB(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 0, 2, sumO, sumT);
					PreciseEstimateOpaqueTransparent2x2_GRB(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 2, 0, sumO, sumT);
					PreciseEstimateOpaqueTransparent2x2_GRB(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 2, 2, sumO, sumT);

					sum = (sumO <= sumT) ? sumO : sumT;
					if (sum >= water)
						return sum;
				}
			}

			sum += Matrix[1][2]->PreciseEstimate2x2_GRB(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 0, 0);
			sum += Matrix[1][2]->PreciseEstimate2x2_GRB(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 2, 0);
			sum += Matrix[2][1]->PreciseEstimate2x2_GRB(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 0, 0);
			sum += Matrix[2][1]->PreciseEstimate2x2_GRB(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 0, 2);
			sum += Matrix[2][2]->PreciseEstimate2x2_GRB(*this, *Matrix[1][2], *Matrix[2][1], *Matrix[2][2], 0, 0);
			if (sum >= water)
				return sum;
		}

		//

		sum += Matrix[0][1]->PreciseEstimate2x1_GRB(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 3, 0);
		if (sum >= water)
			return sum;

		sum += Matrix[0][1]->PreciseEstimate2x1_GRB(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 3, 2);
		if (sum >= water)
			return sum;

		sum += Matrix[0][2]->PreciseEstimate2x1_GRB(*Matrix[0][1], *Matrix[0][2], *this, *Matrix[1][2], 3, 0);
		if (sum >= water)
			return sum;

		//

		sum += Matrix[1][0]->PreciseEstimate1x2_GRB(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 0, 3);
		if (sum >= water)
			return sum;

		sum += Matrix[1][0]->PreciseEstimate1x2_GRB(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 2, 3);
		if (sum >= water)
			return sum;

		sum += Matrix[2][0]->PreciseEstimate1x2_GRB(*Matrix[1][0], *this, *Matrix[2][0], *Matrix[2][1], 0, 3);
		if (sum >= water)
			return sum;

		//

		sum += Matrix[0][0]->PreciseEstimate1x1_GRB(*Matrix[0][0], *Matrix[0][1], *Matrix[1][0], *this, 3, 3);
		if (sum >= water)
			return sum;

		return sum;
	}

	//////////////////////
	// Climber Test GRB //
	//////////////////////

	INLINED bool ClimberTest_GRB(__m128i& backup, int64_t& water)
	{
		int64_t error = Estimate7x7_GRB(water);
		if (water > error)
		{
			water = error;
			backup = Colors;
			return true;
		}
		else
		{
			Colors = backup;
			return false;
		}
	}

	INLINED void ChangeColors_GRB(__m128i& backup, int64_t& water, int& changes, int mark, bool hasG, bool hasR, bool hasB, int group, int mask)
	{
		if ((changes != 0) || (mask < mark))
		{
			for (;; )
			{
				bool other = false;

				if (hasG)
				{
					int c = M128I_I16(Colors, group + 1);
					if (c > 0)
					{
						c = c - 1;
						M128I_I16(Colors, group + 1) = (short)c;
						other = true;
					}
				}

				if (hasR)
				{
					int c = M128I_I16(Colors, group + 2);
					if (c > 0)
					{
						c = c - 1;
						M128I_I16(Colors, group + 2) = (short)c;
						other = true;
					}
				}

				if (hasB)
				{
					int c = M128I_I16(Colors, group + 3);
					if (c > 0)
					{
						if (group == 0)
							c = MakeColor4(c - 2);
						else
							c = c - 1;
						M128I_I16(Colors, group + 3) = (short)c;
						other = true;
					}
				}

				if (!other)
					break;

				if (ClimberTest_GRB(backup, water)) changes = mask; else break;
			}
		}

		if (((changes != 0) || (mask + mask < mark)) && (changes != mask))
		{
			for (;; )
			{
				bool other = false;

				if (hasG)
				{
					int c = M128I_I16(Colors, group + 1);
					if (c < 0x1F)
					{
						c = c + 1;
						M128I_I16(Colors, group + 1) = (short)c;
						other = true;
					}
				}

				if (hasR)
				{
					int c = M128I_I16(Colors, group + 2);
					if (c < 0x1F)
					{
						c = c + 1;
						M128I_I16(Colors, group + 2) = (short)c;
						other = true;
					}
				}

				if (hasB)
				{
					int c = M128I_I16(Colors, group + 3);
					if (c < 0x1F)
					{
						if (group == 0)
							c = MakeColor4(c + 2);
						else
							c = c + 1;
						M128I_I16(Colors, group + 3) = (short)c;
						other = true;
					}
				}

				if (!other)
					break;

				if (ClimberTest_GRB(backup, water)) changes = mask + mask; else break;
			}
		}
	}

	/////////////
	// Climber //
	/////////////

	void Climber(bool& more)
	{
		if (IsZero)
			return;

		int64_t water = 0;
		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				water += Matrix[y][x]->Error;
			}
		}
		if (water <= 0)
			return;

		bool check_alpha = true;
		if (IsDense & (M128I_I16(Colors, 0) >= 0xF) & (M128I_I16(Colors, 4) >= 0xF))
		{
			if ((M128I_I16(Matrix[0][0]->Colors, 0) >= 0xF) & (M128I_I16(Matrix[0][0]->Colors, 4) >= 0xF) &
				(M128I_I16(Matrix[0][1]->Colors, 0) >= 0xF) & (M128I_I16(Matrix[0][1]->Colors, 4) >= 0xF) &
				(M128I_I16(Matrix[0][2]->Colors, 0) >= 0xF) & (M128I_I16(Matrix[0][2]->Colors, 4) >= 0xF) &
				(M128I_I16(Matrix[1][0]->Colors, 0) >= 0xF) & (M128I_I16(Matrix[1][0]->Colors, 4) >= 0xF) &
				(M128I_I16(Matrix[1][2]->Colors, 0) >= 0xF) & (M128I_I16(Matrix[1][2]->Colors, 4) >= 0xF) &
				(M128I_I16(Matrix[2][0]->Colors, 0) >= 0xF) & (M128I_I16(Matrix[2][0]->Colors, 4) >= 0xF) &
				(M128I_I16(Matrix[2][1]->Colors, 0) >= 0xF) & (M128I_I16(Matrix[2][1]->Colors, 4) >= 0xF) &
				(M128I_I16(Matrix[2][2]->Colors, 0) >= 0xF) & (M128I_I16(Matrix[2][2]->Colors, 4) >= 0xF))
			{
				check_alpha = false;
			}
		}

		if (check_alpha)
		{
			water = Estimate7x7(water);
		}
		else
		{
			water = Estimate7x7_GRB(water);
		}
		if (water <= 0)
			return;

		__m128i backup = Colors; uint32_t backup1 = Data[1];

		int mark = 0x100000;

		if (check_alpha)
		{
			for (;; )
			{
				int changes = 0;

				ChangeAlpha(backup, water, changes, mark, 0, 1);

				int modeA = (M128I_I16(Colors, 0) >= 0xF);

				ChangeColors(backup, water, changes, mark, modeA, true, true, true, 0, 4);
				ChangeColors(backup, water, changes, mark, modeA, true, false, false, 0, 0x10);
				ChangeColors(backup, water, changes, mark, modeA, false, true, false, 0, 0x40);
				ChangeColors(backup, water, changes, mark, modeA, false, false, true, 0, 0x100);

				//

				ChangeAlpha(backup, water, changes, mark, 4, 0x400);

				int modeB = (M128I_I16(Colors, 4) >= 0xF);

				ChangeColors(backup, water, changes, mark, modeB, true, true, true, 4, 0x1000);
				ChangeColors(backup, water, changes, mark, modeB, true, false, false, 4, 0x4000);
				ChangeColors(backup, water, changes, mark, modeB, false, true, false, 4, 0x10000);
				ChangeColors(backup, water, changes, mark, modeB, false, false, true, 4, 0x40000);

				if (!changes)
					break;

				mark = changes;
			}
		}
		else
		{
			for (;; )
			{
				int changes = 0;

				ChangeColors_GRB(backup, water, changes, mark, true, true, true, 0, 4);
				ChangeColors_GRB(backup, water, changes, mark, true, false, false, 0, 0x10);
				ChangeColors_GRB(backup, water, changes, mark, false, true, false, 0, 0x40);
				ChangeColors_GRB(backup, water, changes, mark, false, false, true, 0, 0x100);

				//

				ChangeColors_GRB(backup, water, changes, mark, true, true, true, 4, 0x1000);
				ChangeColors_GRB(backup, water, changes, mark, true, false, false, 4, 0x4000);
				ChangeColors_GRB(backup, water, changes, mark, false, true, false, 4, 0x10000);
				ChangeColors_GRB(backup, water, changes, mark, false, false, true, 4, 0x40000);

				if (!changes)
					break;

				mark = changes;
			}
		}

		if (mark == 0x100000)
		{
			Data[1] = backup1;
			return;
		}

		PackColors();

		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				Matrix[y][x]->PackModulation();
			}
		}

		more = true;
	}

	INLINED void Delta4x4(int* output, int stride) const
	{
		for (int y = 0; y < 4; y++)
		{
			const int* source = &_Image[((Y << 2) + y) * _Size + (X << 2)];

			__m128i mc = _mm_loadu_si128((const  __m128i*)source);
			__m128i mx = _mm_load_si128((const  __m128i*)output);

			__m128i m = _mm_subs_epu8(mc, mx);

			_mm_store_si128((__m128i*)output, m);

			output = (int*)((uint8_t*)output + stride);
		}
	}

	double FinalPack4x4(const __m128i interpolation[4][4], int error, uint32_t& answer, bool transparent)
	{
		if (error <= 0)
			return 1.0;

		alignas(16) int src[16], msk[16];
		{
			const int* source = &_Image[(Y << 2) * _Size + (X << 2)];
			const int* mask = (const int*)((const uint8_t*)source + _ImageToMaskDelta);

			_mm_store_si128((__m128i*)&src[0], _mm_loadu_si128((const __m128i*)source));
			_mm_store_si128((__m128i*)&msk[0], _mm_loadu_si128((const __m128i*)mask));

			source += _Size; mask += _Size;

			_mm_store_si128((__m128i*)&src[4], _mm_loadu_si128((const __m128i*)source));
			_mm_store_si128((__m128i*)&msk[4], _mm_loadu_si128((const __m128i*)mask));

			source += _Size; mask += _Size;

			_mm_store_si128((__m128i*)&src[8], _mm_loadu_si128((const __m128i*)source));
			_mm_store_si128((__m128i*)&msk[8], _mm_loadu_si128((const __m128i*)mask));

			source += _Size; mask += _Size;

			_mm_store_si128((__m128i*)&src[12], _mm_loadu_si128((const __m128i*)source));
			_mm_store_si128((__m128i*)&msk[12], _mm_loadu_si128((const __m128i*)mask));
		}

		int ways[16];
		__m128i vals[16];

		for (int i = 0; i < 16; i++)
		{
			__m128i mc = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(src[i]));
			mc = _mm_shuffle_epi32(mc, _MM_SHUFFLE(0, 2, 1, 3));

			__m128i mA = _mm_load_si128(&((const __m128i*)&interpolation[0][0])[i]);
			__m128i mB = _mm_unpackhi_epi64(mA, mA);

			mA = _mm_cvtepi16_epi32(mA);
			mB = _mm_cvtepi16_epi32(mB);

			__m128i mmask = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(msk[i]));

			mc = _mm_and_si128(mc, mmask);
			mA = _mm_and_si128(mA, mmask);
			mB = _mm_and_si128(mB, mmask);

			src[i] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi16(mc, mc), mc));

			__m128i mAC, mBC;
			if (transparent)
			{
				mAC = _mm_srai_epi16(_mm_add_epi16(mA, mB), 1);
				mBC = _mm_insert_epi16(mAC, 0, 0);
			}
			else
			{
				__m128i mBA = _mm_sub_epi16(mB, mA);
				mAC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 1), mBA), 3), mA);
				mBC = _mm_add_epi16(_mm_srai_epi16(_mm_add_epi16(_mm_slli_epi16(mBA, 2), mBA), 3), mA);
			}

			_mm_store_si128(&vals[i], _mm_packus_epi16(_mm_packus_epi16(mA, mAC), _mm_packus_epi16(mBC, mB)));

			int good = 0xF;
			if (_mm_movemask_epi8(_mm_cmpeq_epi32(mA, mAC)) == 0xFFFF) good &= ~2;
			if (_mm_movemask_epi8(_mm_cmpeq_epi32(mAC, mBC)) == 0xFFFF) good &= ~4;
			if (_mm_movemask_epi8(_mm_cmpeq_epi32(mBC, mB)) == 0xFFFF) good &= ~8;
			if (_mm_movemask_epi8(_mm_cmpeq_epi32(mAC, mB)) == 0xFFFF) good &= ~8; // punch-through

			mA = _mm_sub_epi16(mA, mc);
			mB = _mm_sub_epi16(mB, mc);
			mAC = _mm_sub_epi16(mAC, mc);
			mBC = _mm_sub_epi16(mBC, mc);

			__m128i magrb = _mm_load_si128((const __m128i*)g_AGRB);

			mA = _mm_mullo_epi16(mA, mA);
			mB = _mm_mullo_epi16(mB, mB);
			mAC = _mm_mullo_epi16(mAC, mAC);
			mBC = _mm_mullo_epi16(mBC, mBC);

			mA = _mm_mullo_epi32(mA, magrb);
			mB = _mm_mullo_epi32(mB, magrb);
			mAC = _mm_mullo_epi32(mAC, magrb);
			mBC = _mm_mullo_epi32(mBC, magrb);

			__m128i m10 = _mm_hadd_epi32(mA, mAC);
			__m128i m32 = _mm_hadd_epi32(mBC, mB);
			__m128i me4 = _mm_hadd_epi32(m10, m32);
			__m128i me1 = HorizontalMinimum4(me4);

			int way = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4, me1)));
			ways[i] = (way & good) | (1 << 4);
		}

		int loops[16];

		for (int i = 0; i < 16; i++)
		{
			int k = 0;
			while ((ways[i] & (1 << k)) == 0) k++;
			loops[i] = k;
		}

		double best = -(kAlpha + kColor + 0.1);

		for (;; )
		{
			SSIM_INIT();

			for (int i = 0; i < 16; i++)
			{
				__m128i mt = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(M128I_I32(vals[i], loops[i])));

				__m128i mb = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(src[i]));

				SSIM_UPDATE(mt, mb);
			}

			SSIM_CLOSE(4);

			SSIM_FINAL(mssim_ga, g_ssim_16k1L, g_ssim_16k2L);
			SSIM_OTHER();
			SSIM_FINAL(mssim_br, g_ssim_16k1L, g_ssim_16k2L);

			double ssim = _mm_cvtsd_f64(mssim_ga) * kAlpha;
			ssim += _mm_cvtsd_f64(_mm_unpackhi_pd(mssim_ga, mssim_ga)) * kGreen;
			ssim += _mm_cvtsd_f64(mssim_br) * kRed;
			ssim += _mm_cvtsd_f64(_mm_unpackhi_pd(mssim_br, mssim_br)) * kBlue;

			if (best < ssim)
			{
				best = ssim;

				uint32_t v = 0;
				for (int j = 0; j < 16; j++)
				{
					v |= ((uint32_t)loops[j]) << (j + j);
				}
				answer = v;
			}

			int i = 0;
			for (;; )
			{
				int k = loops[i];
				if (ways[i] != (1 << k))
				{
					do { k++; } while ((ways[i] & (1 << k)) == 0);
					if (k < 4)
					{
						loops[i] = k;
						break;
					}

					k = 0;
					while ((ways[i] & (1 << k)) == 0) k++;
					loops[i] = k;
				}

				if (++i >= 16)
					return best * (1.0 / (kAlpha + kColor));
			}
		}
	}

	void FinalPackModulation()
	{
		if (IsEmpty)
		{
			SSIM = 1.0;
			Error = 0;
			Data[0] = 0xAAAAAAAAu;
			Data[1] |= 1u;
			return;
		}

		__m128i interpolation[4][4];
		Interpolate4x4(interpolation);

		int errorO, errorT;
		uint32_t answerO, answerT;
		PackOpaqueTransparent4x4(interpolation, errorO, answerO, errorT, answerT);

		if (errorO == errorT)
		{
			double ssimO = FinalPack4x4(interpolation, errorO, answerO, false);
			double ssimT = FinalPack4x4(interpolation, errorT, answerT, true);

			if (ssimO == ssimT)
			{
				if (Data[1] & 1u)
				{
					SSIM = ssimT;
					Error = errorT;
					Data[0] = answerT;
				}
				else
				{
					SSIM = ssimO;
					Error = errorO;
					Data[0] = answerO;
				}
			}
			else if (ssimO > ssimT)
			{
				SSIM = ssimO;
				Error = errorO;
				Data[0] = answerO;
				Data[1] &= ~1u;
			}
			else
			{
				SSIM = ssimT;
				Error = errorT;
				Data[0] = answerT;
				Data[1] |= 1u;
			}
		}
		else if (errorO < errorT)
		{
			SSIM = FinalPack4x4(interpolation, errorO, answerO, false);
			Error = errorO;
			Data[0] = answerO;
			Data[1] &= ~1u;
		}
		else
		{
			SSIM = FinalPack4x4(interpolation, errorT, answerT, true);
			Error = errorT;
			Data[0] = answerT;
			Data[1] |= 1u;
		}
	}
};

Block Block::Buffer[kCapacity][kCapacity];

static int _Twiddle[0x100];

static void InitTwiddle()
{
	for (int i = 0; i < 0x100; i++)
	{
		int v = 0;

		if (i & 1) v |= 1;
		if (i & 2) v |= 4;
		if (i & 4) v |= 0x10;
		if (i & 8) v |= 0x40;
		if (i & 0x10) v |= 0x100;
		if (i & 0x20) v |= 0x400;
		if (i & 0x40) v |= 0x1000;
		if (i & 0x80) v |= 0x4000;

		_Twiddle[i] = v;
	}
}

static __m128i _Window1Inner[2][3][3][4][4];

static void Window1Inner_Inner(__m128i zond, int yy, int xx, int zz)
{
	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			Block::Buffer[y][x].Colors = _mm_setzero_si128();
		}
	}

	Block::Buffer[yy][xx].Colors = zond;

	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			Block::Buffer[y][x].Interpolate4x4(_Window1Inner[zz][y][x]);
		}
	}
}

static void InitWindow1Inner()
{
	_Mask = 3;

	for (int y = 0; y < 4; y++)
	{
		for (int x = 0; x < 4; x++)
		{
			Block::Buffer[y][x].Locate(y, x);
		}
	}

	__m128i zondA = _mm_shufflelo_epi16(_mm_cvtsi32_si128(1), 0);
	__m128i zondB = _mm_unpacklo_epi64(_mm_setzero_si128(), zondA);

	Window1Inner_Inner(zondA, 1, 1, 0);
	Window1Inner_Inner(zondB, 1, 1, 1);
}

struct Window1
{
	alignas(16) Block* Matrix[3][3];

	int64_t unused;

	alignas(16) int Field[12][12];

	alignas(8) float Wgrb[7 * 7][2];
	alignas(8) float AB[2];
	alignas(8) float P[7 * 7];

	SVD<49, 2> _svd;

	Window1()
	{
	}

	void Inner(int zz)
	{
		int stride = sizeof(Field[0]);

		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				Matrix[y][x]->Modulate4x4(_Window1Inner[zz][y][x], &Field[y << 2][x << 2], stride);
			}
		}

		for (int y = 0; y < 7; y++)
		{
			for (int x = 0; x < 7; x++)
			{
				uint32_t v = (uint32_t)Field[y + 3][x + 3];
				Wgrb[y * 7 + x][zz] = (float)(int)((v >> 8) & 0xFF);
			}
		}
	}

	void Outer()
	{
		for (int y = 0 + 1; y < 3 - 1; y++)
		{
			for (int x = 0 + 1; x < 3 - 1; x++)
			{
				Matrix[y][x]->Colors = _mm_setzero_si128();
			}
		}

		int stride = sizeof(Field[0]);

		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				__m128i interpolation[4][4];
				Matrix[y][x]->Interpolate4x4(interpolation);

				Matrix[y][x]->Modulate4x4(interpolation, &Field[y << 2][x << 2], stride);

				Matrix[y][x]->Delta4x4(&Field[y << 2][x << 2], stride);
			}
		}
	}

	void Reverse(int shift)
	{
		for (int y = 0; y < 7; y++)
		{
			for (int x = 0; x < 7; x++)
			{
				uint32_t v = (uint32_t)Field[y + 3][x + 3];
				P[y * 7 + x] = (float)(int)((v >> shift) & 0xFF);
			}
		}

		_svd.Resolve(AB, P);
	}

	void ReverseY()
	{
		for (int y = 0; y < 7; y++)
		{
			for (int x = 0; x < 7; x++)
			{
				uint32_t v = (uint32_t)Field[y + 3][x + 3];
				P[y * 7 + x] = float(((v) & 0xFF) * kBlue + ((v >> 8) & 0xFF) * kGreen + ((v >> 16) & 0xFF) * kRed) * (1.f / kColor);
			}
		}

		_svd.Resolve(AB, P);
	}

	INLINED int64_t Error() const
	{
		int64_t error = 0;

		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				error += Matrix[y][x]->Error;
			}
		}

		return error;
	}

	INLINED void Pick(__m128i& colors, int color_index, int cell_index, int high_value) const
	{
		float fa = AB[cell_index];
		float fb = AB[cell_index + 1];

		const float r = high_value + 0.1f;

		if ((fa >= -r) && (fa <= r) &&
			(fb >= -r) && (fb <= r))
		{
			int a = (int)(fa + 0.5f);
			int b = (int)(fb + 0.5f);

			if (a < 0)
				a = 0;
			if (b < 0)
				b = 0;
			if (a > high_value)
				a = high_value;
			if (b > high_value)
				b = high_value;

			M128I_I16(colors, color_index) = (short)a;
			M128I_I16(colors, color_index + 4) = (short)b;
		}
	}

	void Optimize(int Y, int X, bool& more)
	{
		{
			Block& b = Block::Buffer[Y][X];
			if (b.Matrix[1][1]->IsZero)
				return;

			memcpy(Matrix, b.Matrix, sizeof(Matrix));
		}

		int64_t water = Error();
		if (water <= 0)
			return;

		__m128i backup = Matrix[1][1]->Colors; uint32_t backup1 = Matrix[1][1]->Data[1];

		Inner(0);
		Inner(1);

		Outer();

		__m128i current = backup;

		_svd.PseudoInverse(Wgrb);

		Reverse(0);
		Pick(current, 3, 0, 31);

		Reverse(8);
		Pick(current, 1, 0, 31);

		Reverse(16);
		Pick(current, 2, 0, 31);

		if (!Matrix[1][1]->IsDense)
		{
			ReverseY();
			Pick(current, 0, 0, 31); M128I_I16(current, 0) >>= 1;
		}

		Matrix[1][1]->Colors = current;
		if (_mm_movemask_epi8(_mm_cmpeq_epi16(current, backup)) != 0xFFFF)
		{
			Matrix[1][1]->UpdateColors();
		}

		int diff = _mm_movemask_epi8(_mm_cmpeq_epi16(Matrix[1][1]->Colors, backup));
		if (diff == 0xFFFF)
			return;

		for (int y = 0; y < 3; y++)
		{
			for (int x = 0; x < 3; x++)
			{
				Matrix[y][x]->PackModulation();
			}
		}

		{
			bool better = false;

			Matrix[1][1]->Climber(better);
		}

		int64_t error = Error();
		if (water > error)
		{
			more = true;
		}
		else if (water < error)
		{
			Matrix[1][1]->Colors = backup; Matrix[1][1]->Data[1] = backup1;

			for (int y = 0; y < 3; y++)
			{
				for (int x = 0; x < 3; x++)
				{
					Matrix[y][x]->PackModulation();
				}
			}
		}
	}
};

static __m128i _Window4Inner[8][4][4][4][4];

static void Window4Inner_Inner(__m128i zond, int yy, int xx, int zz)
{
	for (int y = 0; y < 4; y++)
	{
		for (int x = 0; x < 4; x++)
		{
			Block::Buffer[y][x].Colors = _mm_setzero_si128();
		}
	}

	Block::Buffer[yy][xx].Colors = zond;

	for (int y = 0; y < 4; y++)
	{
		for (int x = 0; x < 4; x++)
		{
			Block::Buffer[y][x].Interpolate4x4(_Window4Inner[zz][y][x]);
		}
	}
}

static void InitWindow4Inner()
{
	_Mask = 3;

	for (int y = 0; y < 4; y++)
	{
		for (int x = 0; x < 4; x++)
		{
			Block::Buffer[y][x].Locate(y, x);
		}
	}

	__m128i zondA = _mm_shufflelo_epi16(_mm_cvtsi32_si128(1), 0);
	__m128i zondB = _mm_unpacklo_epi64(_mm_setzero_si128(), zondA);

	Window4Inner_Inner(zondA, 1, 1, 0);
	Window4Inner_Inner(zondB, 1, 1, 1);

	Window4Inner_Inner(zondA, 1, 2, 2);
	Window4Inner_Inner(zondB, 1, 2, 3);

	Window4Inner_Inner(zondA, 2, 1, 4);
	Window4Inner_Inner(zondB, 2, 1, 5);

	Window4Inner_Inner(zondA, 2, 2, 6);
	Window4Inner_Inner(zondB, 2, 2, 7);
}

struct Window4
{
	alignas(16) Block* Matrix[4][4];

	alignas(16) int Field[16][16];

	alignas(16) float Wgrb[11 * 11][8];
	alignas(16) float AB[8];
	alignas(16) float P[11 * 11];

	SVD<121, 8> _svd;

	Window4()
	{
	}

	void Inner(int zz)
	{
		int stride = sizeof(Field[0]);

		for (int y = 0; y < 4; y++)
		{
			for (int x = 0; x < 4; x++)
			{
				Matrix[y][x]->Modulate4x4(_Window4Inner[zz][y][x], &Field[y << 2][x << 2], stride);
			}
		}

		for (int y = 0; y < 11; y++)
		{
			for (int x = 0; x < 11; x++)
			{
				uint32_t v = (uint32_t)Field[y + 3][x + 3];
				Wgrb[y * 11 + x][zz] = (float)(int)((v >> 8) & 0xFF);
			}
		}
	}

	void Outer()
	{
		for (int y = 0 + 1; y < 4 - 1; y++)
		{
			for (int x = 0 + 1; x < 4 - 1; x++)
			{
				Matrix[y][x]->Colors = _mm_setzero_si128();
			}
		}

		int stride = sizeof(Field[0]);

		for (int y = 0; y < 4; y++)
		{
			for (int x = 0; x < 4; x++)
			{
				__m128i interpolation[4][4];
				Matrix[y][x]->Interpolate4x4(interpolation);

				Matrix[y][x]->Modulate4x4(interpolation, &Field[y << 2][x << 2], stride);

				Matrix[y][x]->Delta4x4(&Field[y << 2][x << 2], stride);
			}
		}
	}

	void Reverse(int shift)
	{
		for (int y = 0; y < 11; y++)
		{
			for (int x = 0; x < 11; x++)
			{
				uint32_t v = (uint32_t)Field[y + 3][x + 3];
				P[y * 11 + x] = (float)(int)((v >> shift) & 0xFF);
			}
		}

		_svd.Resolve(AB, P);
	}

	void ReverseY()
	{
		for (int y = 0; y < 11; y++)
		{
			for (int x = 0; x < 11; x++)
			{
				uint32_t v = (uint32_t)Field[y + 3][x + 3];
				P[y * 11 + x] = float(((v) & 0xFF) * kBlue + ((v >> 8) & 0xFF) * kGreen + ((v >> 16) & 0xFF) * kRed) * (1.f / kColor);
			}
		}

		_svd.Resolve(AB, P);
	}

	INLINED int64_t Error() const
	{
		int64_t error = 0;

		for (int y = 0; y < 4; y++)
		{
			for (int x = 0; x < 4; x++)
			{
				error += Matrix[y][x]->Error;
			}
		}

		return error;
	}

	INLINED void Pick(__m128i& colors, int color_index, int cell_index, int high_value) const
	{
		float fa = AB[cell_index];
		float fb = AB[cell_index + 1];

		const float r = high_value + 0.1f;

		if ((fa >= -r) && (fa <= r) &&
			(fb >= -r) && (fb <= r))
		{
			int a = (int)(fa + 0.5f);
			int b = (int)(fb + 0.5f);

			if (a < 0)
				a = 0;
			if (b < 0)
				b = 0;
			if (a > high_value)
				a = high_value;
			if (b > high_value)
				b = high_value;

			M128I_I16(colors, color_index) = (short)a;
			M128I_I16(colors, color_index + 4) = (short)b;
		}
	}

	void Optimize(int Y, int X, bool& more)
	{
		{
			Block& b = Block::Buffer[Y][X];
			if (b.Matrix[1][1]->IsZero & b.Matrix[1][2]->IsZero & b.Matrix[2][1]->IsZero & b.Matrix[2][2]->IsZero)
				return;

			Matrix[0][0] = b.Matrix[0][0];
			Matrix[0][1] = b.Matrix[0][1];
			Matrix[0][2] = b.Matrix[0][2];
			Matrix[0][3] = b.Matrix[0][2]->Matrix[1][2];

			Matrix[1][0] = b.Matrix[1][0];
			Matrix[1][1] = b.Matrix[1][1];
			Matrix[1][2] = b.Matrix[1][2];
			Matrix[1][3] = b.Matrix[1][2]->Matrix[1][2];

			Matrix[2][0] = b.Matrix[2][0];
			Matrix[2][1] = b.Matrix[2][1];
			Matrix[2][2] = b.Matrix[2][2];
			Matrix[2][3] = b.Matrix[2][2]->Matrix[1][2];

			Matrix[3][0] = b.Matrix[2][0]->Matrix[2][1];
			Matrix[3][1] = b.Matrix[2][1]->Matrix[2][1];
			Matrix[3][2] = b.Matrix[2][2]->Matrix[2][1];
			Matrix[3][3] = b.Matrix[2][2]->Matrix[2][2];
		}

		int64_t water = Error();
		if (water <= 0)
			return;

		__m128i backup11 = Matrix[1][1]->Colors; uint32_t backup1_11 = Matrix[1][1]->Data[1];
		__m128i backup12 = Matrix[1][2]->Colors; uint32_t backup1_12 = Matrix[1][2]->Data[1];
		__m128i backup21 = Matrix[2][1]->Colors; uint32_t backup1_21 = Matrix[2][1]->Data[1];
		__m128i backup22 = Matrix[2][2]->Colors; uint32_t backup1_22 = Matrix[2][2]->Data[1];

		Inner(0);
		Inner(1);

		Inner(2);
		Inner(3);

		Inner(4);
		Inner(5);

		Inner(6);
		Inner(7);

		Outer();

		__m128i current11 = backup11;
		__m128i current12 = backup12;
		__m128i current21 = backup21;
		__m128i current22 = backup22;

		_svd.PseudoInverse(Wgrb);

		Reverse(0);
		Pick(current11, 3, 0, 31);
		Pick(current12, 3, 2, 31);
		Pick(current21, 3, 4, 31);
		Pick(current22, 3, 6, 31);

		Reverse(8);
		Pick(current11, 1, 0, 31);
		Pick(current12, 1, 2, 31);
		Pick(current21, 1, 4, 31);
		Pick(current22, 1, 6, 31);

		Reverse(16);
		Pick(current11, 2, 0, 31);
		Pick(current12, 2, 2, 31);
		Pick(current21, 2, 4, 31);
		Pick(current22, 2, 6, 31);

		if (!(Matrix[1][1]->IsDense & Matrix[1][2]->IsDense & Matrix[2][1]->IsDense & Matrix[2][2]->IsDense))
		{
			ReverseY();
			Pick(current11, 0, 0, 31); M128I_I16(current11, 0) >>= 1; M128I_I16(current11, 4) >>= 1;
			Pick(current12, 0, 2, 31); M128I_I16(current12, 0) >>= 1; M128I_I16(current12, 4) >>= 1;
			Pick(current21, 0, 4, 31); M128I_I16(current21, 0) >>= 1; M128I_I16(current21, 4) >>= 1;
			Pick(current22, 0, 6, 31); M128I_I16(current22, 0) >>= 1; M128I_I16(current22, 4) >>= 1;
		}

		Matrix[1][1]->Colors = current11;
		if (_mm_movemask_epi8(_mm_cmpeq_epi16(current11, backup11)) != 0xFFFF)
		{
			Matrix[1][1]->UpdateColors();
		}

		Matrix[1][2]->Colors = current12;
		if (_mm_movemask_epi8(_mm_cmpeq_epi16(current12, backup12)) != 0xFFFF)
		{
			Matrix[1][2]->UpdateColors();
		}

		Matrix[2][1]->Colors = current21;
		if (_mm_movemask_epi8(_mm_cmpeq_epi16(current21, backup21)) != 0xFFFF)
		{
			Matrix[2][1]->UpdateColors();
		}

		Matrix[2][2]->Colors = current22;
		if (_mm_movemask_epi8(_mm_cmpeq_epi16(current22, backup22)) != 0xFFFF)
		{
			Matrix[2][2]->UpdateColors();
		}

		int diff =
			_mm_movemask_epi8(_mm_cmpeq_epi16(Matrix[1][1]->Colors, backup11)) +
			_mm_movemask_epi8(_mm_cmpeq_epi16(Matrix[1][2]->Colors, backup12)) +
			_mm_movemask_epi8(_mm_cmpeq_epi16(Matrix[2][1]->Colors, backup21)) +
			_mm_movemask_epi8(_mm_cmpeq_epi16(Matrix[2][2]->Colors, backup22));
		if (diff == (0xFFFF * 4))
			return;

		for (int y = 0; y < 4; y++)
		{
			for (int x = 0; x < 4; x++)
			{
				Matrix[y][x]->PackModulation();
			}
		}

		{
			int mark = 33;

			for (;;)
			{
				int changes = 0;

				if (/*(changes != 0) ||*/ (11 < mark))
				{
					bool better = false;
					Matrix[1][1]->Climber(better);
					if (better) changes = 11;
				}

				if ((changes != 0) || (12 < mark))
				{
					bool better = false;
					Matrix[1][2]->Climber(better);
					if (better) changes = 12;
				}

				if ((changes != 0) || (21 < mark))
				{
					bool better = false;
					Matrix[2][1]->Climber(better);
					if (better) changes = 21;
				}

				if ((changes != 0) || (22 < mark))
				{
					bool better = false;
					Matrix[2][2]->Climber(better);
					if (better) changes = 22;
				}

				if (!changes)
					break;

				mark = changes;
			}
		}

		int64_t error = Error();
		if (water > error)
		{
			more = true;
		}
		else if (water < error)
		{
			Matrix[1][1]->Colors = backup11; Matrix[1][1]->Data[1] = backup1_11;
			Matrix[1][2]->Colors = backup12; Matrix[1][2]->Data[1] = backup1_12;
			Matrix[2][1]->Colors = backup21; Matrix[2][1]->Data[1] = backup1_21;
			Matrix[2][2]->Colors = backup22; Matrix[2][2]->Data[1] = backup1_22;

			for (int y = 0; y < 4; y++)
			{
				for (int x = 0; x < 4; x++)
				{
					Matrix[y][x]->PackModulation();
				}
			}
		}
	}
};

static void BlockKernel_Prepare(int y, int x)
{
	Block& block = Block::Buffer[y][x];

	block.UnpackColors();

	block.CheckEmpty();

	block.CheckOpaque();
}

static void BlockKernel_ZeroDense(int y, int x)
{
	Block& block = Block::Buffer[y][x];

	block.CheckZero();

	block.CheckDense();
}

static void BlockKernel_Egoist(int y, int x)
{
	Block& block = Block::Buffer[y][x];

	block.Egoist();
}

static void BlockKernel_PackModulation(int y, int x)
{
	Block& block = Block::Buffer[y][x];

	block.PackModulation();
}

static void BlockKernel_Climber(int y, int x)
{
	bool hot = _Fire.Has(y, x, 1);
	if (!hot)
		return;

	Block& block = Block::Buffer[y][x];

	bool better = false;

	block.Climber(better);

	if (better)
	{
		_Changes.store(1);
		_Fire.MarkArea(y - 1, y + 1, x - 1, x + 1);
	}

	_Fire.Remove(y, x, 1);
}

static void BlockKernel_Window1(int y, int x)
{
	bool hot = _Fire.Has(y, x, 2);
	if (!hot)
		return;

	bool better = false;

	{
		Window1 wnd;
		wnd.Optimize(y, x, better);
	}

	if (better)
	{
		_Changes.store(1);
		_Fire.MarkArea(y - 1, y + 1, x - 1, x + 1);
	}

	_Fire.Remove(y, x, 2);
}

static void BlockKernel_Window4(int y, int x)
{
	bool hot = _Fire.Has(y, x, 4);
	if (!hot)
	{
		int yn = (y + 1) & _Mask;
		int xn = (x + 1) & _Mask;

		bool warm = _Fire.Has(y, xn, 4);
		warm |= _Fire.Has(yn, x, 4);
		warm |= _Fire.Has(yn, xn, 4);
		if (!warm)
			return;
	}

	bool better = false;

	{
		Window4 wnd;
		wnd.Optimize(y, x, better);
	}

	if (better)
	{
		_Changes.store(1);
		_Fire.MarkArea(y - 1, y + 2, x - 1, x + 2);
	}

	_Fire.Remove(y, x, 4);
}

static void BlockKernel_Vanish(int y, int x)
{
	Block& block = Block::Buffer[y][x];

	if (block.IsZero)
	{
		block.Colors = _mm_setzero_si128();
		block.UpdateColors();
	}
}

static void BlockKernel_FinalPackModulation(int y, int x)
{
	Block& block = Block::Buffer[y][x];

	block.FinalPackModulation();
}

static int64_t Compress(uint8_t* dst_pvrtc, const uint8_t* mask_agrb, const uint8_t* src_bgra, int src_s, int passes, bool incremental)
{
	_Image = (int*)src_bgra;
	_ImageToMaskDelta = mask_agrb - src_bgra;
	_Size = src_s;
	_Mask = (src_s >> 2) - 1;

	auto t0 = std::chrono::high_resolution_clock::now();

	for (int y = 0; y <= _Mask; y++)
	{
		for (int x = 0; x <= _Mask; x++)
		{
			Block::Buffer[y][x].Locate(y, x);
		}
	}

	uint32_t any_data = 0;

	for (int y = 0; y <= _Mask; y++)
	{
		int twiddleY = (_Twiddle[y >> 8] << (16 + 3)) + (_Twiddle[y & 0xFF] << 3);

		for (int x = 0; x <= _Mask; x++)
		{
			int twiddleX = (_Twiddle[x >> 8] << (16 + 4)) + (_Twiddle[x & 0xFF] << 4);

			Block& block = Block::Buffer[y][x];

			block.Load(&dst_pvrtc[twiddleY + twiddleX]);

			any_data |= block.Data[0];
			any_data |= block.Data[1];
		}
	}

	_Worker.Start();

	_Worker.RunKernel(&BlockKernel_Prepare, 1, 0x100);

	_Worker.RunKernel(&BlockKernel_ZeroDense, 1, 0x100);

	if ((any_data == 0) || !incremental)
	{
		printf("    Step = 0\tEgoist\t\t\t");

		_Worker.RunKernel(&BlockKernel_Egoist, 1, 0x20);
	}

	_Worker.RunKernel(&BlockKernel_PackModulation, 1, 0x20);

	_Fire.MarkAll();

	static const int CLIMBER = 8;
	static const int WINDOW1 = 4;

	int quota_climber = CLIMBER;
	int quota_window1 = WINDOW1;
	int quota_window4 = passes;

	bool last_climber = false;
	bool last_window1 = false;
	bool last_window4 = false;

	double water_psnr = 0;
	int64_t water = 1LL << 62;

	for (int step = 1; step <= 1000; step++)
	{
		int64_t error = 0;

		for (int y = 0; y <= _Mask; y++)
		{
			for (int x = 0; x <= _Mask; x++)
			{
				error += Block::Buffer[y][x].Error;
			}
		}

		if (error > 0)
		{
			double psnr = 10.0 * log((255.0 * 255.0) * kColor * (_Size * _Size) / error) / log(10.0);

			printf("Texture wPSNR = %f  \r", psnr);

			// Специальный случай /draft /incremental когда первый Climber не дал заметных улучшений
			if (incremental && (passes <= 0) && (quota_climber == CLIMBER - 1) && (quota_window1 == WINDOW1) && (psnr - water_psnr < 0.05))
			{
				printf("Already compressed\t\t\t");
				break;
			}

			water_psnr = psnr;
		}
		else
		{
			printf("Exactly\n");
			break;
		}

		if (water <= error)
		{
			if (last_climber)
				quota_climber = 0;
			else if (last_window1)
				quota_window1 = 0;
			else if (last_window4)
				quota_window4 = 0;
			else
				break;
		}
		water = error;

		last_climber = false;
		last_window1 = false;
		last_window4 = false;

		printf("    Step = %i\t\t\t\t", step);

		if (--quota_climber >= 0)
		{
			printf("\b\b\b\b\b\b\b\b");
			printf("\b\b\b\b\b\b\b\b");
			printf("\b\b\b\b\b\b\b\b");
			printf("Climber\t\t\t");

			_Changes.store(0);
			_Worker.RunKernel(&BlockKernel_Climber, kStep, 0x20);
			if (_Changes.load())
			{
				last_climber = true;
				continue;
			}

			// Специальный случай /draft /incremental когда первый Climber не дал улучшений
			if (incremental && (passes <= 0) && (quota_climber == CLIMBER - 1) && (quota_window1 == WINDOW1))
			{
				printf("\b\b\b\b\b\b\b\b");
				printf("\b\b\b\b\b\b\b\b");
				printf("Already\t\t");
				break;
			}
		}

		if (--quota_window1 >= 0)
		{
			printf("\b\b\b\b\b\b\b\b");
			printf("\b\b\b\b\b\b\b\b");
			printf("Window1\t\t");

			_Changes.store(0);
			_Worker.RunKernel(&BlockKernel_Window1, kStep, 8);
			if (_Changes.load())
			{
				quota_climber = CLIMBER;
				last_window1 = true;
				continue;
			}
		}

		if (--quota_window4 >= 0)
		{
			printf("\b\b\b\b\b\b\b\b");
			printf("Window4\t");

			_Changes.store(0);
			_Worker.RunKernel(&BlockKernel_Window4, kStep, 2);
			if (_Changes.load())
			{
				quota_climber = CLIMBER;
				quota_window1 = WINDOW1;
				last_window4 = true;
				continue;
			}
		}

		break;
	}

	_Worker.RunKernel(&BlockKernel_Vanish, 1, 0x20);

	_Worker.RunKernel(&BlockKernel_FinalPackModulation, 1, 0x20);

	_Worker.Finish();

	int64_t mse = 0;
	double mssim = 0;

	for (int y = 0; y <= _Mask; y++)
	{
		int twiddleY = (_Twiddle[y >> 8] << (16 + 3)) + (_Twiddle[y & 0xFF] << 3);

		for (int x = 0; x <= _Mask; x++)
		{
			int twiddleX = (_Twiddle[x >> 8] << (16 + 4)) + (_Twiddle[x & 0xFF] << 4);

			Block& block = Block::Buffer[y][x];

			block.Save(&dst_pvrtc[twiddleY + twiddleX]);

			mse += Block::Buffer[y][x].Error;
			mssim += Block::Buffer[y][x].SSIM;
		}
	}

	if (mse > 0)
	{
		printf("Texture wPSNR = %f\n", 10.0 * log((255.0 * 255.0) * kColor * (_Size * _Size) / mse) / log(10.0));

		printf("\t\t\t\t\tTexture wSSIM_4x4 = %.8f\n", mssim * 16.0 / (_Size * _Size));
	}
	else
	{
		printf("      Exactly\n");
	}

	return mse;
}

static void DecompressState(uint8_t* dst_bgra, int dst_s)
{
	_Image = (int*)dst_bgra;
	_ImageToMaskDelta = 0;
	_Size = dst_s;
	_Mask = (dst_s >> 2) - 1;

	for (int y = 0; y <= _Mask; y++)
	{
		for (int x = 0; x <= _Mask; x++)
		{
			Block::Buffer[y][x].UnpackColors();
		}
	}

	for (int y = 0; y <= _Mask; y++)
	{
		for (int x = 0; x <= _Mask; x++)
		{
			Block& block = Block::Buffer[y][x];

			__m128i interpolation[4][4];
			block.Interpolate4x4(interpolation);

			int* source = &_Image[(y << 2) * _Size + (x << 2)];
			block.Modulate4x4(interpolation, source, _Size << 2);
		}
	}
}

static void PackTexture(const char* name, int Size, uint8_t* dst_bgra, const char* psnr, uint8_t* mask_agrb, const uint8_t* src_bgra, int src_s, int passes, bool incremental)
{
	uint8_t* dst_pvrtc = new uint8_t[Size];
	memset(dst_pvrtc, 0, Size);

	if (incremental)
	{
		LoadPvrtc(name, dst_pvrtc, Size);
	}

	auto start = std::chrono::high_resolution_clock::now();

	int64_t mse = Compress(dst_pvrtc, mask_agrb, src_bgra, src_s, passes, incremental);

	auto finish = std::chrono::high_resolution_clock::now();

	SavePvrtc(name, dst_pvrtc, Size);

	int span = Max((int)std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count(), 1);

	if (mse > 0)
	{
		int n = (src_s * src_s) >> 4;

		printf("    Texture %s = %f, elapsed %i ms, %i bps\n",
			psnr,
			10.0 * log((255.0 * 255.0) * kColor * (src_s * src_s) / mse) / log(10.0),
			span,
			(int)(n * 1000LL / span));
	}
	else
	{
		printf("    Exactly, elapsed %i ms\n", span);
	}

	DecompressState(dst_bgra, src_s);

	delete[] dst_pvrtc;
}

int PvrtcMainWithArgs(const std::vector<std::string>& args)
{
	bool flip = true;
	bool incremental = false;
	int passes = 2;
	int border = 1;

	const char* src_name = nullptr;
	const char* dst_rgba_name = nullptr;
	const char* result_name = nullptr;

	for (int i = 0, n = (int)args.size(); i < n; i++)
	{
		const char* arg = args[i].c_str();

		if (arg[0] == '/')
		{
			if (strcmp(arg, "/draft") == 0)
			{
				passes = 0;
				continue;
			}
			else if (strcmp(arg, "/incremental") == 0)
			{
				incremental = true;
				continue;
			}
			else if (strcmp(arg, "/retina") == 0)
			{
				border = 2;
				continue;
			}
			else if (strcmp(arg, "/debug") == 0)
			{
				if (++i < n)
				{
					result_name = args[i].c_str();
				}
				continue;
			}
#ifdef WIN32
			else
			{
				printf("Unknown %s\n", arg);
				continue;
			}
#endif
		}

		if (src_name == nullptr)
		{
			src_name = arg;
		}
		else if (dst_rgba_name == nullptr)
		{
			dst_rgba_name = arg;
		}
		else
		{
			printf("Error: %s\n", arg);
			return 1;
		}
	}

	if (!src_name)
	{
		printf("No input\n");
		return 1;
	}

	uint8_t* src_image_bgra;
	int src_image_w, src_image_h;

	if (!ReadImage(src_name, src_image_bgra, src_image_w, src_image_h, flip))
	{
		printf("Problem with image %s\n", src_name);
		return 1;
	}

	printf("Loaded %s\n", src_name);

	int src_texture_s = GetNextPow2(Max(src_image_w, src_image_h));

	if (src_texture_s < 8)
		src_texture_s = 8;

	if (src_texture_s > (kCapacity << 2))
	{
		printf("Huge image %s\n", src_name);
		return 1;
	}

	int c = 4;
	int src_image_stride = src_image_w * c;
	int src_texture_stride = src_texture_s * c;

	uint8_t* src_texture_bgra = new uint8_t[src_texture_s * src_texture_stride];

	for (int i = 0; i < src_image_h; i++)
	{
		memcpy(&src_texture_bgra[i * src_texture_stride], &src_image_bgra[i * src_image_stride], src_image_stride);

		for (int j = src_image_stride; j < src_texture_stride; j += c)
		{
			memcpy(&src_texture_bgra[i * src_texture_stride + j], &src_image_bgra[i * src_image_stride + src_image_stride - c], c);
		}
	}

	for (int i = src_image_h; i < src_texture_s; i++)
	{
		memcpy(&src_texture_bgra[i * src_texture_stride], &src_texture_bgra[(src_image_h - 1) * src_texture_stride], src_texture_stride);
	}

	printf("  Image %dx%d, Texture %dx%d\n", src_image_w, src_image_h, src_texture_s, src_texture_s);

	Stride = src_texture_stride;

	uint8_t* dst_texture_bgra = new uint8_t[src_texture_s * src_texture_stride];

	int Size = (src_texture_s * src_texture_s) >> 1;

	InitTwiddle();
	InitWindow1Inner();
	InitWindow4Inner();

	if ((dst_rgba_name != nullptr) && dst_rgba_name[0])
	{
		uint8_t* mask_agrb = new uint8_t[src_texture_s * src_texture_stride];

		ComputeAlphaMaskWithOutline(mask_agrb, src_texture_bgra, src_texture_s, border);

		PackTexture(dst_rgba_name, Size, dst_texture_bgra, "wPSNR", mask_agrb, src_texture_bgra, src_texture_s, passes, incremental);

		delete[] mask_agrb;
	}

	if ((result_name != nullptr) && result_name[0])
	{
		WriteImage(result_name, dst_texture_bgra, src_texture_s, src_texture_s, flip);
	}

	delete[] dst_texture_bgra;
	delete[] src_texture_bgra;
	delete[] src_image_bgra;

	return 0;
}

int __cdecl main(int argc, char* argv[])
{
	if (argc < 2)
	{
		printf("Usage: PvrtcCompress [/draft] [/incremental] [/retina] src [dst_rgba] [/debug result.png]\n");
		return 1;
	}

	std::vector<std::string> args;
	args.reserve(argc);

	for (int i = 1; i < argc; i++)
	{
		args.emplace_back(argv[i]);
	}

	return PvrtcMainWithArgs(args);
}
