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

#ifndef WIN32
#include <boost/thread.hpp>
#ifdef __APPLE__
#include <libkern/OSByteOrder.h>
#else
#include <byteswap.h>
#endif
#endif

#ifdef WIN32
#pragma comment(lib, "gdiplus.lib")

#define INLINED __forceinline
#define NOTINLINED __declspec(noinline)
#define M128I_I32(mm, index) ((mm).m128i_i32[index])
#else
#define INLINED __attribute__((always_inline))
#define NOTINLINED __attribute__((noinline))
#define M128I_I32(mm, index) (reinterpret_cast<int32_t(&)[4]>(mm)[index])
#endif

typedef struct alignas(16) { int Data[8 * 4]; int Count, unused; uint8_t Shift[8]; } Half;
typedef struct alignas(16) { Half A, B; } Elem;

typedef struct alignas(16) { int Data[16 * 4]; uint8_t Shift[16]; int Count, unused0, unused1, unused2; } Area;
typedef struct alignas(16) { short Mask[16], U[16], V[16], Data[16]; } Surface;

typedef struct alignas(8) { int Error, Color; } Node;

// http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html sRGB
enum { kGreen = 715, kRed = 213, kBlue = 72 };

// Linear RGB
//enum { kGreen = 1, kRed = 1, kBlue = 1 };

enum { kColor = kGreen + kRed + kBlue, kUnknownError = (255 * 255) * kColor * (4 * 4) + 1 };

alignas(16) static const int g_table[8][2] = { { 2, 8 },{ 5, 17 },{ 9, 29 },{ 13, 42 },{ 18, 60 },{ 24, 80 },{ 33, 106 },{ 47, 183 } };

alignas(16) static const int g_tableHT[8] = { 3, 6, 11, 16, 23, 32, 41, 64 };

alignas(16) static const short g_tableA[16][8] = {
	{ -3, -6, -9, -15, 2, 5, 8, 14 },
	{ -3, -7, -10, -13, 2, 6, 9, 12 },
	{ -2, -5, -8, -13, 1, 4, 7, 12 },
	{ -2, -4, -6, -13, 1, 3, 5, 12 },
	{ -3, -6, -8, -12, 2, 5, 7, 11 },
	{ -3, -7, -9, -11, 2, 6, 8, 10 },
	{ -4, -7, -8, -11, 3, 6, 7, 10 },
	{ -3, -5, -8, -11, 2, 4, 7, 10 },
	{ -2, -6, -8, -10, 1, 5, 7, 9 },
	{ -2, -5, -8, -10, 1, 4, 7, 9 },
	{ -2, -4, -8, -10, 1, 3, 7, 9 },
	{ -2, -5, -7, -10, 1, 4, 6, 9 },
	{ -3, -4, -7, -10, 2, 3, 6, 9 },
	{ -1, -2, -3, -10, 0, 1, 2, 9 },
	{ -4, -6, -8, -9, 3, 5, 7, 8 },
	{ -3, -5, -7, -9, 2, 4, 6, 8 }
};

alignas(16) static const short g_GRB_I16[8] = { kGreen, kRed, kBlue, 0, kGreen, kRed, kBlue, 0 };
alignas(16) static const short g_GR_I16[8] = { kGreen, kRed, kGreen, kRed, kGreen, kRed, kGreen, kRed };
alignas(16) static const short g_GB_I16[8] = { kGreen, kBlue, kGreen, kBlue, kGreen, kBlue, kGreen, kBlue };

alignas(16) static const int g_colors4[0x10] =
{
	0x00, 0x11, 0x22, 0x33,
	0x44, 0x55, 0x66, 0x77,
	0x88, 0x99, 0xAA, 0xBB,
	0xCC, 0xDD, 0xEE, 0xFF
};

alignas(16) static const int g_colors5[0x20] =
{
	0x00, 0x08, 0x10, 0x18,
	0x21, 0x29, 0x31, 0x39,
	0x42, 0x4A, 0x52, 0x5A,
	0x63, 0x6B, 0x73, 0x7B,
	0x84, 0x8C, 0x94, 0x9C,
	0xA5, 0xAD, 0xB5, 0xBD,
	0xC6, 0xCE, 0xD6, 0xDE,
	0xE7, 0xEF, 0xF7, 0xFF
};

static __m128i g_errors4[8][0x100][0x10 >> 2];
static __m128i g_errors5[8][0x100][0x20 >> 2];
static __m128i g_errorsH[8][0x100][0x100 >> 2];
static __m128i g_errorsT[8][0x100][0x100 >> 2];
static __m128i g_errorsA[0x100][0x100][0x100 >> 2];

static __m128i g_stripesH[8][0x100][0x10 >> 2];
static __m128i g_stripesT[8][0x100][0x10 >> 2];
static __m128i g_stripesA[0x100][0x100][0x10 >> 2];

static const double g_ssim_8k1L = (0.01 * 255 * 8) * (0.01 * 255 * 8);
static const double g_ssim_8k2L = g_ssim_8k1L * 9;

static const double g_ssim_16k1L = (0.01 * 255 * 16) * (0.01 * 255 * 16);
static const double g_ssim_16k2L = g_ssim_16k1L * 9;

static const int WorkerThreadStackSize = 3 * 1024 * 1024;

static int Stride;

#ifdef WIN32

static INLINED uint32_t BSWAP(uint32_t x)
{
	return _byteswap_ulong(x);
}

static INLINED uint64_t BSWAP64(uint64_t x)
{
	return _byteswap_uint64(x);
}

static INLINED uint32_t BROR(uint32_t x)
{
	return _rotr(x, 8);
}

#else

static INLINED uint32_t BSWAP(uint32_t x)
{
#ifdef __APPLE__
	return OSSwapInt32(x);
#else
	return bswap_32(x);
#endif
}

static INLINED uint64_t BSWAP64(uint64_t x)
{
#ifdef __APPLE__
	return OSSwapInt64(x);
#else
	return bswap_64(x);
#endif
}

static INLINED uint32_t BROR(uint32_t x)
{
	return (x >> 8) | (x << (32 - 8));
}

static INLINED void __debugbreak()
{
}

#endif

static INLINED int Min(int x, int y)
{
	return (x < y) ? x : y;
}

static INLINED int Max(int x, int y)
{
	return (x > y) ? x : y;
}

static INLINED int ExpandColor4(int c)
{
	return (c << 4) ^ c;
}

static INLINED __m128i HorizontalMinimum4(__m128i me4)
{
	__m128i me2 = _mm_min_epi32(me4, _mm_shuffle_epi32(me4, _MM_SHUFFLE(2, 3, 0, 1)));
	__m128i me1 = _mm_min_epi32(me2, _mm_shuffle_epi32(me2, _MM_SHUFFLE(0, 1, 2, 3)));
	return me1;
}

static INLINED __m128i HorizontalSum4(__m128i me4)
{
	__m128i me2 = _mm_add_epi32(me4, _mm_shuffle_epi32(me4, _MM_SHUFFLE(2, 3, 0, 1)));
	__m128i me1 = _mm_add_epi32(me2, _mm_shuffle_epi32(me2, _MM_SHUFFLE(0, 1, 2, 3)));
	return me1;
}

static void InitLevelErrors()
{
	for (int q = 0; q < 8; q++)
	{
		int q0 = g_table[q][0];
		int q1 = g_table[q][1];

		for (int i = 0; i < 0x10; i++)
		{
			int c = g_colors4[i];

			int t0 = Min(c + q0, 255);
			int t1 = Min(c + q1, 255);
			int t2 = Max(c - q0, 0);
			int t3 = Max(c - q1, 0);

			for (int x = 0, n = 0x100; x < n; x++)
			{
				int v = Min(Min(abs(x - t0), abs(x - t1)), Min(abs(x - t2), abs(x - t3)));

				((int*)g_errors4[q][x])[i] = v * v;
			}
		}

		for (int i = 0; i < 0x20; i++)
		{
			int c = g_colors5[i];

			int t0 = Min(c + q0, 255);
			int t1 = Min(c + q1, 255);
			int t2 = Max(c - q0, 0);
			int t3 = Max(c - q1, 0);

			for (int x = 0, n = 0x100; x < n; x++)
			{
				int v = Min(Min(abs(x - t0), abs(x - t1)), Min(abs(x - t2), abs(x - t3)));

				((int*)g_errors5[q][x])[i] = v * v;
			}
		}
	}

	for (size_t q = 0x10; q < 0x100; q++)
	{
		__m128i mscale = _mm_shufflelo_epi16(_mm_cvtsi64_si128(q >> 4), 0);
		mscale = _mm_unpacklo_epi64(mscale, mscale);

		__m128i mtable = _mm_load_si128((const __m128i*)&((const uint8_t*)g_tableA)[(q << 4) & 0xF0]);
		mtable = _mm_mullo_epi16(mtable, mscale);

		for (size_t alpha = 0; alpha < 0x100; alpha++)
		{
			__m128i ma = _mm_shufflelo_epi16(_mm_cvtsi64_si128(alpha), 0);
			ma = _mm_unpacklo_epi64(ma, ma);

			__m128i mt = _mm_add_epi16(ma, mtable);
			mt = _mm_packus_epi16(mt, mt);

			__m128i mzero = _mm_setzero_si128();
			mt = _mm_unpacklo_epi8(mt, mzero);

			for (size_t x = 0; x < 0x100; x++)
			{
				__m128i mx = _mm_shufflelo_epi16(_mm_cvtsi64_si128(x), 0);
				mx = _mm_unpacklo_epi64(mx, mx);

				__m128i m = _mm_sub_epi16(mx, mt);
				m = _mm_mullo_epi16(m, m);

				__m128i mmin = _mm_minpos_epu16(m);

				((int*)g_errorsA[alpha][x])[q] = (int)_mm_cvtsi128_si64(mmin) & 0xFFFF;
			}
		}
	}

	for (size_t alpha = 0; alpha < 0x100; alpha++)
	{
		for (size_t x = 0; x < 0x100; x++)
		{
			auto errors = (const int*)g_errorsA[alpha][x] + 0x10;

			for (size_t scale = 1; scale < 0x10; scale++)
			{
				int v = *errors++;

				for (int i = 1; i < 0x10; i++)
				{
					v = Min(v, *errors++);
				}

				((int*)g_stripesA[alpha][x])[scale] = v;
			}
		}
	}

	for (int q = 0; q < 8; q++)
	{
		int d = g_tableHT[q];

		for (int i = 0; i < 0x100; i++)
		{
			int a = ((i >> 4) & 0xC) + ((i >> 2) & 3);
			int b = ((i >> 2) & 0xC) + (i & 3);

			int ca = ExpandColor4(a);
			int cb = ExpandColor4(b);

			int t0 = Min(ca + d, 255);
			int t1 = Max(ca - d, 0);
			int t2 = Min(cb + d, 255);
			int t3 = Max(cb - d, 0);

			for (int x = 0; x < 0x100; x++)
			{
				int v = Min(Min(abs(x - t0), abs(x - t1)), Min(abs(x - t2), abs(x - t3)));

				((int*)g_errorsH[q][x])[i] = v * v;
			}
		}

		for (int i = 0; i < 0x100; i++)
		{
			int a = ((i >> 4) & 0xC) + ((i >> 2) & 3);
			int b = ((i >> 2) & 0xC) + (i & 3);

			int ca = ExpandColor4(a);
			int cb = ExpandColor4(b);

			int t0 = ca;
			int t1 = Min(cb + d, 255);
			int t2 = cb;
			int t3 = Max(cb - d, 0);

			for (int x = 0; x < 0x100; x++)
			{
				int v = Min(Min(abs(x - t0), abs(x - t1)), Min(abs(x - t2), abs(x - t3)));

				((int*)g_errorsT[q][x])[i] = v * v;
			}
		}
	}

	for (int q = 0; q < 8; q++)
	{
		for (int x = 0; x < 0x100; x++)
		{
			auto errors = (const int*)g_errorsH[q][x];

			for (size_t z = 0; z < 0x10; z++)
			{
				int v = *errors++;

				for (int i = 1; i < 0x10; i++)
				{
					v = Min(v, *errors++);
				}

				((int*)g_stripesH[q][x])[z] = v;
			}
		}

		for (int x = 0; x < 0x100; x++)
		{
			auto errors = (const int*)g_errorsT[q][x];

			for (size_t z = 0; z < 0x10; z++)
			{
				int v = *errors++;

				for (int i = 1; i < 0x10; i++)
				{
					v = Min(v, *errors++);
				}

				((int*)g_stripesT[q][x])[z] = v;
			}
		}
	}
}

static INLINED void GuessLevels(const Half& half, size_t offset, Node nodes[0x10 + 1], int weight, int water, int q)
{
#define PROCESS_PIXEL(index) \
	{ \
		const __m128i* p = errors[(size_t)(uint32_t)half.Data[j + index]]; \
		sum0 = _mm_add_epi32(sum0, _mm_load_si128(p + 0)); \
		sum1 = _mm_add_epi32(sum1, _mm_load_si128(p + 1)); \
		sum2 = _mm_add_epi32(sum2, _mm_load_si128(p + 2)); \
		sum3 = _mm_add_epi32(sum3, _mm_load_si128(p + 3)); \
	} \

#define STORE_QUAD(index) \
	if (_mm_movemask_epi8(_mm_cmpgt_epi32(mtop, sum##index)) != 0) \
	{ \
		__m128i sum = _mm_mullo_epi32(_mm_min_epi32(sum##index, mtop), mweight); \
		__m128i mc = _mm_load_si128((__m128i*)&g_colors4[index * 4]); \
	 	level = _mm_min_epi32(level, sum); \
		__m128i mL = _mm_unpacklo_epi32(sum, mc); \
		__m128i mH = _mm_unpackhi_epi32(sum, mc); \
		_mm_store_si128((__m128i*)&nodes[w + 0], mL); \
		_mm_store_si128((__m128i*)&nodes[w + 2], mH); \
		w += 4; \
	} \

	__m128i sum0 = _mm_setzero_si128();
	__m128i sum1 = _mm_setzero_si128();
	__m128i sum2 = _mm_setzero_si128();
	__m128i sum3 = _mm_setzero_si128();

	auto errors = g_errors4[q];

	int k = half.Count; size_t j = offset;
	if (k & 8)
	{
		PROCESS_PIXEL(0);
		PROCESS_PIXEL(4);
		PROCESS_PIXEL(8);
		PROCESS_PIXEL(12);
		PROCESS_PIXEL(16);
		PROCESS_PIXEL(20);
		PROCESS_PIXEL(24);
		PROCESS_PIXEL(28);
	}
	else
	{
		if (k & 4)
		{
			PROCESS_PIXEL(0);
			PROCESS_PIXEL(4);
			PROCESS_PIXEL(8);
			PROCESS_PIXEL(12);

			j += 16;
		}

		if (k & 2)
		{
			PROCESS_PIXEL(0);
			PROCESS_PIXEL(4);

			j += 8;
		}

		if (k & 1)
		{
			PROCESS_PIXEL(0);
		}
	}

	int top = (water + weight - 1) / weight;
	__m128i mweight = _mm_shuffle_epi32(_mm_cvtsi32_si128(weight), 0);
	__m128i mtop = _mm_shuffle_epi32(_mm_cvtsi32_si128(top), 0);
	__m128i level = _mm_mullo_epi32(mtop, mweight);

	size_t w = 0;

	STORE_QUAD(0);
	STORE_QUAD(1);
	STORE_QUAD(2);
	STORE_QUAD(3);

	nodes[0x10].Error = _mm_cvtsi128_si32(HorizontalMinimum4(level));
	nodes[0x10].Color = (int)w;

#undef PROCESS_PIXEL
#undef STORE_QUAD
}

static INLINED void AdjustLevels(const Half& half, size_t offset, Node nodes[0x20 + 1], int weight, int water, int q)
{
#define PROCESS_PIXEL(index) \
	{ \
		const __m128i* p = errors[(size_t)(uint32_t)half.Data[j + index]]; \
		sum0 = _mm_add_epi32(sum0, _mm_load_si128(p + 0)); \
		sum1 = _mm_add_epi32(sum1, _mm_load_si128(p + 1)); \
		sum2 = _mm_add_epi32(sum2, _mm_load_si128(p + 2)); \
		sum3 = _mm_add_epi32(sum3, _mm_load_si128(p + 3)); \
		sum4 = _mm_add_epi32(sum4, _mm_load_si128(p + 4)); \
		sum5 = _mm_add_epi32(sum5, _mm_load_si128(p + 5)); \
		sum6 = _mm_add_epi32(sum6, _mm_load_si128(p + 6)); \
		sum7 = _mm_add_epi32(sum7, _mm_load_si128(p + 7)); \
	} \

#define STORE_QUAD(index) \
	if (_mm_movemask_epi8(_mm_cmpgt_epi32(mtop, sum##index)) != 0) \
	{ \
		__m128i sum = _mm_mullo_epi32(_mm_min_epi32(sum##index, mtop), mweight); \
		__m128i mc = _mm_load_si128((__m128i*)&g_colors5[index * 4]); \
	 	level = _mm_min_epi32(level, sum); \
		__m128i mL = _mm_unpacklo_epi32(sum, mc); \
		__m128i mH = _mm_unpackhi_epi32(sum, mc); \
		_mm_store_si128((__m128i*)&nodes[w + 0], mL); \
		_mm_store_si128((__m128i*)&nodes[w + 2], mH); \
		w += 4; \
	} \

	__m128i sum0 = _mm_setzero_si128();
	__m128i sum1 = _mm_setzero_si128();
	__m128i sum2 = _mm_setzero_si128();
	__m128i sum3 = _mm_setzero_si128();
	__m128i sum4 = _mm_setzero_si128();
	__m128i sum5 = _mm_setzero_si128();
	__m128i sum6 = _mm_setzero_si128();
	__m128i sum7 = _mm_setzero_si128();

	auto errors = g_errors5[q];

	int k = half.Count; size_t j = offset;
	if (k & 8)
	{
		PROCESS_PIXEL(0);
		PROCESS_PIXEL(4);
		PROCESS_PIXEL(8);
		PROCESS_PIXEL(12);
		PROCESS_PIXEL(16);
		PROCESS_PIXEL(20);
		PROCESS_PIXEL(24);
		PROCESS_PIXEL(28);
	}
	else
	{
		if (k & 4)
		{
			PROCESS_PIXEL(0);
			PROCESS_PIXEL(4);
			PROCESS_PIXEL(8);
			PROCESS_PIXEL(12);

			j += 16;
		}

		if (k & 2)
		{
			PROCESS_PIXEL(0);
			PROCESS_PIXEL(4);

			j += 8;
		}

		if (k & 1)
		{
			PROCESS_PIXEL(0);
		}
	}

	int top = (water + weight - 1) / weight;
	__m128i mweight = _mm_shuffle_epi32(_mm_cvtsi32_si128(weight), 0);
	__m128i mtop = _mm_shuffle_epi32(_mm_cvtsi32_si128(top), 0);
	__m128i level = _mm_mullo_epi32(mtop, mweight);

	size_t w = 0;

	STORE_QUAD(0);
	STORE_QUAD(1);
	STORE_QUAD(2);
	STORE_QUAD(3);
	STORE_QUAD(4);
	STORE_QUAD(5);
	STORE_QUAD(6);
	STORE_QUAD(7);

	nodes[0x20].Error = _mm_cvtsi128_si32(HorizontalMinimum4(level));
	nodes[0x20].Color = (int)w;

#undef PROCESS_PIXEL
#undef STORE_QUAD
}

static INLINED void CombineStripes(const Area& area, size_t offset, int chunks[0x10], const __m128i(*stripes)[4], int weight)
{
#define PROCESS_PIXEL(index) \
	{ \
		const __m128i* p = stripes[(size_t)(uint32_t)area.Data[j + index]]; \
		sum0 = _mm_add_epi32(sum0, _mm_load_si128(p + 0)); \
		sum1 = _mm_add_epi32(sum1, _mm_load_si128(p + 1)); \
		sum2 = _mm_add_epi32(sum2, _mm_load_si128(p + 2)); \
		sum3 = _mm_add_epi32(sum3, _mm_load_si128(p + 3)); \
	} \

#define STORE_QUAD(index) \
	_mm_store_si128((__m128i*)&chunks[index << 2], _mm_mullo_epi32(sum##index, mweight)); \

	__m128i mweight = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)weight), 0);

	__m128i sum0 = _mm_setzero_si128();
	__m128i sum1 = _mm_setzero_si128();
	__m128i sum2 = _mm_setzero_si128();
	__m128i sum3 = _mm_setzero_si128();

	int k = area.Count; size_t j = offset;
	if (k & 16)
	{
		PROCESS_PIXEL(0);
		PROCESS_PIXEL(4);
		PROCESS_PIXEL(8);
		PROCESS_PIXEL(12);
		PROCESS_PIXEL(16);
		PROCESS_PIXEL(20);
		PROCESS_PIXEL(24);
		PROCESS_PIXEL(28);

		PROCESS_PIXEL(32);
		PROCESS_PIXEL(36);
		PROCESS_PIXEL(40);
		PROCESS_PIXEL(44);
		PROCESS_PIXEL(48);
		PROCESS_PIXEL(52);
		PROCESS_PIXEL(56);
		PROCESS_PIXEL(60);
	}
	else
	{
		if (k & 8)
		{
			PROCESS_PIXEL(0);
			PROCESS_PIXEL(4);
			PROCESS_PIXEL(8);
			PROCESS_PIXEL(12);
			PROCESS_PIXEL(16);
			PROCESS_PIXEL(20);
			PROCESS_PIXEL(24);
			PROCESS_PIXEL(28);

			j += 32;
		}

		if (k & 4)
		{
			PROCESS_PIXEL(0);
			PROCESS_PIXEL(4);
			PROCESS_PIXEL(8);
			PROCESS_PIXEL(12);

			j += 16;
		}

		if (k & 2)
		{
			PROCESS_PIXEL(0);
			PROCESS_PIXEL(4);

			j += 8;
		}

		if (k & 1)
		{
			PROCESS_PIXEL(0);
		}
	}

	STORE_QUAD(0);
	STORE_QUAD(1);
	STORE_QUAD(2);
	STORE_QUAD(3);

#undef PROCESS_PIXEL
#undef STORE_QUAD
}

static INLINED void CombineLevels(const Area& area, size_t offset, Node nodes[0x100 + 1], const __m128i(*errors)[64], const int chunks[0x10], int weight, int water)
{
#define PROCESS_PIXEL(index) \
	{ \
		const __m128i* p = errors[(size_t)(uint32_t)area.Data[j + index]]; \
		sum0 = _mm_add_epi32(sum0, _mm_load_si128(p + 0)); \
		sum1 = _mm_add_epi32(sum1, _mm_load_si128(p + 1)); \
		sum2 = _mm_add_epi32(sum2, _mm_load_si128(p + 2)); \
		sum3 = _mm_add_epi32(sum3, _mm_load_si128(p + 3)); \
	} \

#define STORE_QUAD(index) \
	if (_mm_movemask_epi8(_mm_cmpgt_epi32(mtop, sum##index)) != 0) \
	{ \
		__m128i sum = _mm_mullo_epi32(_mm_min_epi32(sum##index, mtop), mweight); \
		__m128i mc = _mm_cvtepu8_epi32(_mm_cvtsi64_si128(colors)); \
	 	level = _mm_min_epi32(level, sum); \
		__m128i mL = _mm_unpacklo_epi32(sum, mc); \
		__m128i mH = _mm_unpackhi_epi32(sum, mc); \
		_mm_store_si128((__m128i*)&nodes[w + 0], mL); \
		_mm_store_si128((__m128i*)&nodes[w + 2], mH); \
		w += 4; \
	} \
	colors += 0x10101010; \

	int top = (water + weight - 1) / weight;
	__m128i mweight = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)weight), 0);
	__m128i mtop = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)top), 0);
	__m128i level = _mm_mullo_epi32(mtop, mweight);

	size_t w = 0;

	for (int z = 0; z < 0x10; z++)
	{
		if (chunks[z] >= water)
		{
			errors = (const __m128i(*)[64])((const __m128i*)errors + 4);
			continue;
		}

		__m128i sum0 = _mm_setzero_si128();
		__m128i sum1 = _mm_setzero_si128();
		__m128i sum2 = _mm_setzero_si128();
		__m128i sum3 = _mm_setzero_si128();

		int k = area.Count; size_t j = offset;
		if (k & 16)
		{
			PROCESS_PIXEL(0);
			PROCESS_PIXEL(4);
			PROCESS_PIXEL(8);
			PROCESS_PIXEL(12);
			PROCESS_PIXEL(16);
			PROCESS_PIXEL(20);
			PROCESS_PIXEL(24);
			PROCESS_PIXEL(28);

			if (_mm_movemask_epi8(_mm_cmpgt_epi32(mtop, _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3)))) == 0)
			{
				errors = (const __m128i(*)[64])((const __m128i*)errors + 4);
				continue;
			}

			PROCESS_PIXEL(32);
			PROCESS_PIXEL(36);
			PROCESS_PIXEL(40);
			PROCESS_PIXEL(44);
			PROCESS_PIXEL(48);
			PROCESS_PIXEL(52);
			PROCESS_PIXEL(56);
			PROCESS_PIXEL(60);
		}
		else
		{
			if (k & 8)
			{
				PROCESS_PIXEL(0);
				PROCESS_PIXEL(4);
				PROCESS_PIXEL(8);
				PROCESS_PIXEL(12);
				PROCESS_PIXEL(16);
				PROCESS_PIXEL(20);
				PROCESS_PIXEL(24);
				PROCESS_PIXEL(28);

				j += 32;
			}

			if (k & 4)
			{
				PROCESS_PIXEL(0);
				PROCESS_PIXEL(4);
				PROCESS_PIXEL(8);
				PROCESS_PIXEL(12);

				j += 16;
			}

			if (k & 2)
			{
				PROCESS_PIXEL(0);
				PROCESS_PIXEL(4);

				j += 8;
			}

			if (k & 1)
			{
				PROCESS_PIXEL(0);
			}
		}

		size_t colors = (uint32_t)((((z << 4) & 0xC0) + ((z << 2) & 0xC)) * 0x01010101 + 0x03020100);

		STORE_QUAD(0);
		STORE_QUAD(1);
		STORE_QUAD(2);
		STORE_QUAD(3);

		errors = (const __m128i(*)[64])((const __m128i*)errors + 4);
	}

	nodes[0x100].Error = (int)_mm_cvtsi128_si64(HorizontalMinimum4(level));
	nodes[0x100].Color = (int)w;

#undef PROCESS_PIXEL
#undef STORE_QUAD
}

static INLINED void AlphaStripes(const Area& area, int chunks[0x10], const __m128i(*stripes)[4])
{
#define PROCESS_PIXEL(index) \
	{ \
		const __m128i* p = stripes[(size_t)(uint32_t)area.Data[index]]; \
		sum0 = _mm_add_epi32(sum0, _mm_load_si128(p + 0)); \
		sum1 = _mm_add_epi32(sum1, _mm_load_si128(p + 1)); \
		sum2 = _mm_add_epi32(sum2, _mm_load_si128(p + 2)); \
		sum3 = _mm_add_epi32(sum3, _mm_load_si128(p + 3)); \
	} \

#define STORE_QUAD(index) \
	_mm_store_si128((__m128i*)&chunks[index << 2], sum##index); \

	__m128i sum0 = _mm_setzero_si128();
	__m128i sum1 = _mm_setzero_si128();
	__m128i sum2 = _mm_setzero_si128();
	__m128i sum3 = _mm_setzero_si128();

	PROCESS_PIXEL(0);
	PROCESS_PIXEL(4);
	PROCESS_PIXEL(8);
	PROCESS_PIXEL(12);
	PROCESS_PIXEL(16);
	PROCESS_PIXEL(20);
	PROCESS_PIXEL(24);
	PROCESS_PIXEL(28);

	PROCESS_PIXEL(32);
	PROCESS_PIXEL(36);
	PROCESS_PIXEL(40);
	PROCESS_PIXEL(44);
	PROCESS_PIXEL(48);
	PROCESS_PIXEL(52);
	PROCESS_PIXEL(56);
	PROCESS_PIXEL(60);

	STORE_QUAD(0);
	STORE_QUAD(1);
	STORE_QUAD(2);
	STORE_QUAD(3);

#undef PROCESS_PIXEL
#undef STORE_QUAD
}

static INLINED int AlphaLevels(const Area& area, const __m128i(*errors)[64], const int chunks[0x10], int water, int& last_q_way)
{
#define PROCESS_PIXEL(index) \
	{ \
		const __m128i* p = errors[(size_t)(uint32_t)area.Data[index]]; \
		sum0 = _mm_add_epi32(sum0, _mm_load_si128(p + 0)); \
		sum1 = _mm_add_epi32(sum1, _mm_load_si128(p + 1)); \
		sum2 = _mm_add_epi32(sum2, _mm_load_si128(p + 2)); \
		sum3 = _mm_add_epi32(sum3, _mm_load_si128(p + 3)); \
	} \

	__m128i best = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)water), 0);

	for (int q = 0x10 << 16; !(q & (0x100 << 16)); q += 0x10 << 16)
	{
		errors = (const __m128i(*)[64])((const __m128i*)errors + 4);

		if (chunks[q >> (16 + 4)] >= water)
			continue;

		__m128i sum0 = _mm_setzero_si128();
		__m128i sum1 = _mm_setzero_si128();
		__m128i sum2 = _mm_setzero_si128();
		__m128i sum3 = _mm_setzero_si128();

		PROCESS_PIXEL(0);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3)))) == 0)
			continue;
		PROCESS_PIXEL(4);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3)))) == 0)
			continue;
		PROCESS_PIXEL(8);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3)))) == 0)
			continue;
		PROCESS_PIXEL(12);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3)))) == 0)
			continue;

		PROCESS_PIXEL(16);
		PROCESS_PIXEL(20);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3)))) == 0)
			continue;
		PROCESS_PIXEL(24);
		PROCESS_PIXEL(28);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3)))) == 0)
			continue;

		PROCESS_PIXEL(32);
		PROCESS_PIXEL(36);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3)))) == 0)
			continue;
		PROCESS_PIXEL(40);
		PROCESS_PIXEL(44);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3)))) == 0)
			continue;

		PROCESS_PIXEL(48);
		PROCESS_PIXEL(52);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3)))) == 0)
			continue;
		PROCESS_PIXEL(56);
		PROCESS_PIXEL(60);
		__m128i cur = _mm_min_epi32(_mm_min_epi32(sum0, sum1), _mm_min_epi32(sum2, sum3));
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, cur)) == 0)
			continue;

		best = HorizontalMinimum4(cur);

		__m128i m10 = _mm_packs_epi16(_mm_cmpeq_epi32(sum0, best), _mm_cmpeq_epi32(sum1, best));
		__m128i m32 = _mm_packs_epi16(_mm_cmpeq_epi32(sum2, best), _mm_cmpeq_epi32(sum3, best));
		last_q_way = _mm_movemask_epi8(_mm_packs_epi16(m10, m32)) + q;

		water = _mm_cvtsi128_si32(best);
		if (water <= 0)
			break;
	}

	return water;

#undef PROCESS_PIXEL
}


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


static INLINED void DecompressBlockAlphaEnhanced(const uint8_t input[8], uint8_t* __restrict cell, size_t stride)
{
	uint64_t codes = *(uint64_t*)input;

	__m128i mscale = _mm_shufflelo_epi16(_mm_cvtsi64_si128((codes >> 12) & 0xF), 0);
	mscale = _mm_unpacklo_epi64(mscale, mscale);

	__m128i mtable = _mm_load_si128((const __m128i*)&((const uint8_t*)g_tableA)[(codes >> 4) & 0xF0]);
	mtable = _mm_mullo_epi16(mtable, mscale);

	__m128i ma = _mm_shufflelo_epi16(_mm_cvtsi64_si128(codes & 0xFF), 0);
	ma = _mm_unpacklo_epi64(ma, ma);

	__m128i mt = _mm_add_epi16(ma, mtable);
	mt = _mm_packus_epi16(mt, mt);

	alignas(8) uint8_t alphas[8];
	_mm_storel_epi64((__m128i*)alphas, mt);

	codes = BSWAP64(codes);

	cell[3] = alphas[(codes >> (15 * 3)) & 7];
	cell[7] = alphas[(codes >> (11 * 3)) & 7];
	cell[11] = alphas[(codes >> (7 * 3)) & 7];
	cell[15] = alphas[(codes >> (3 * 3)) & 7];

	cell += stride;

	cell[3] = alphas[(codes >> (14 * 3)) & 7];
	cell[7] = alphas[(codes >> (10 * 3)) & 7];
	cell[11] = alphas[(codes >> (6 * 3)) & 7];
	cell[15] = alphas[(codes >> (2 * 3)) & 7];

	cell += stride;

	cell[3] = alphas[(codes >> (13 * 3)) & 7];
	cell[7] = alphas[(codes >> (9 * 3)) & 7];
	cell[11] = alphas[(codes >> (5 * 3)) & 7];
	cell[15] = alphas[(codes >> (1 * 3)) & 7];

	cell += stride;

	cell[3] = alphas[(codes >> (12 * 3)) & 7];
	cell[7] = alphas[(codes >> (8 * 3)) & 7];
	cell[11] = alphas[(codes >> (4 * 3)) & 7];
	cell[15] = alphas[(codes >> (0 * 3)) & 7];
}

static INLINED int CompareBlocksAlpha(const uint8_t* __restrict cell1, size_t stride1, const uint8_t* __restrict cell2, size_t stride2)
{
	int err = 0;

	for (int y = 0; y < 4; y++)
	{
		int da = cell1[3] - cell2[3];
		err += da * da;

		da = cell1[7] - cell2[7];
		err += da * da;

		da = cell1[11] - cell2[11];
		err += da * da;

		da = cell1[15] - cell2[15];
		err += da * da;

		cell1 += stride1;
		cell2 += stride2;
	}

	return err;
}

static INLINED double CompareBlocksAlphaSSIM(const uint8_t* __restrict cell1, size_t stride1, const uint8_t* __restrict cell2, size_t stride2)
{
	SSIM_INIT();

	for (int y = 0; y < 4; y++)
	{
		for (int x = 0; x < 4; x++)
		{
			__m128i mt = _mm_cvtsi32_si128(cell2[x * 4 + 3]);

			__m128i mb = _mm_cvtsi32_si128(cell1[x * 4 + 3]);

			SSIM_UPDATE(mt, mb);
		}

		cell1 += stride1;
		cell2 += stride2;
	}

	SSIM_CLOSE(4);

	SSIM_FINAL(mssim, g_ssim_16k1L, g_ssim_16k2L);

	return _mm_cvtsd_f64(mssim);
}


static INLINED void ComputeTableAlphaEnhanced(uint8_t output[8], const Area& area, int alpha, int q)
{
	uint64_t answer = (alpha << 8) | q;

	__m128i mscale = _mm_shufflelo_epi16(_mm_cvtsi64_si128((size_t)(uint32_t)(q >> 4)), 0);
	mscale = _mm_unpacklo_epi64(mscale, mscale);

	__m128i mtable = _mm_load_si128((const __m128i*)&((const uint8_t*)g_tableA)[(size_t)(uint32_t)((q << 4) & 0xF0)]);
	mtable = _mm_mullo_epi16(mtable, mscale);

	__m128i ma = _mm_shufflelo_epi16(_mm_cvtsi64_si128((size_t)(uint32_t)alpha), 0);
	ma = _mm_unpacklo_epi64(ma, ma);

	__m128i mt = _mm_add_epi16(ma, mtable);
	mt = _mm_packus_epi16(mt, mt);

	alignas(8) uint8_t vals[8];
	_mm_storel_epi64((__m128i*)vals, mt);

	int good = 0xFF;

	if (vals[0] == vals[1]) good &= ~2;
	if (vals[1] == vals[2]) good &= ~4;
	if (vals[2] == vals[3]) good &= ~8;

	if (vals[0] == vals[4]) good &= ~0x10;

	if (vals[4] == vals[5]) good &= ~0x20;
	if (vals[5] == vals[6]) good &= ~0x40;
	if (vals[6] == vals[7]) good &= ~0x80;

	int ways[16];

	__m128i mzero = _mm_setzero_si128();
	mt = _mm_unpacklo_epi8(mt, mzero);

	__m128i mtH = _mm_unpackhi_epi16(mt, mzero);
	__m128i mtL = _mm_unpacklo_epi16(mt, mzero);

	for (size_t i = 0; i < 16; i++)
	{
		__m128i mb = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)area.Data[i << 2]), 0);

		__m128i me4H = _mm_abs_epi16(_mm_sub_epi16(mb, mtH));
		__m128i me4L = _mm_abs_epi16(_mm_sub_epi16(mb, mtL));

		__m128i me4 = _mm_min_epi16(me4H, me4L);
		__m128i me2 = _mm_min_epi16(me4, _mm_shuffle_epi32(me4, _MM_SHUFFLE(2, 3, 0, 1)));
		__m128i me1 = _mm_min_epi16(me2, _mm_shuffle_epi32(me2, _MM_SHUFFLE(0, 1, 2, 3)));

		int wayH = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4H, me1)));
		int wayL = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4L, me1)));
		ways[i] = (((wayH << 4) | wayL) & good) | (1 << 8);
	}

	int loops[16];

	for (size_t i = 0; i < 16; i++)
	{
		int k = 0;
		while ((ways[i] & (1 << k)) == 0) k++;
		loops[i] = k;
	}

	double best = -1.1;
	uint64_t codes = 0;

	for (;; )
	{
		SSIM_INIT();

		for (size_t i = 0; i < 16; i++)
		{
			__m128i mx = _mm_cvtsi64_si128((size_t)(uint32_t)vals[(size_t)(uint32_t)loops[i]]);

			__m128i mb = _mm_cvtsi64_si128((size_t)(uint32_t)area.Data[i << 2]);

			SSIM_UPDATE(mx, mb);
		}

		SSIM_CLOSE(4);

		SSIM_FINAL(mssim, g_ssim_16k1L, g_ssim_16k2L);

		double ssim = _mm_cvtsd_f64(mssim);

		if (best < ssim)
		{
			best = ssim;

			uint64_t v = 0;
			for (size_t j = 0; j < 16; j++)
			{
				v |= ((uint64_t)(uint32_t)loops[j]) << (j + j + j);
			}
			codes = v;

			if (best >= 1.0)
				break;
		}

		size_t i = 0;
		for (;; )
		{
			int k = loops[i];
			if (ways[i] != (1 << k))
			{
				do { k++; } while ((ways[i] & (1 << k)) == 0);
				if (k < 8)
				{
					loops[i] = k;
					break;
				}

				k = 0;
				while ((ways[i] & (1 << k)) == 0) k++;
				loops[i] = k;
			}

			i++;
			if (i & 16)
				break;
		}
		if (i & 16)
			break;
	}

	for (uint64_t order = 0xFB73EA62D951C840uLL; order != 0; order >>= 4)
	{
		int shift = order & 0xF;

		uint64_t code = (codes >> (shift + shift + shift)) & 7u;

		answer = (answer << 3) + code;
	}

	*(uint64_t*)output = BSWAP64(answer);
}

static int CompressBlockAlphaEnhanced(uint8_t output[8], const uint8_t* __restrict cell, size_t stride, int input_error)
{
	Area area;

	{
		const uint8_t* src = cell;

		area.Data[0] = src[3];
		area.Data[4] = src[7];
		area.Data[8] = src[11];
		area.Data[12] = src[15];

		src += stride;

		area.Data[16] = src[3];
		area.Data[20] = src[7];
		area.Data[24] = src[11];
		area.Data[28] = src[15];

		src += stride;

		area.Data[32] = src[3];
		area.Data[36] = src[7];
		area.Data[40] = src[11];
		area.Data[44] = src[15];

		src += stride;

		area.Data[48] = src[3];
		area.Data[52] = src[7];
		area.Data[56] = src[11];
		area.Data[60] = src[15];
	}

	int sum_alpha = 0;
	for (int i = 0; i < 16 * 4; i += 4)
	{
		sum_alpha += area.Data[i];
	}
	int avg_alpha = (sum_alpha + 8) >> 4;

	alignas(16) int chunks[0x10];

	int water = input_error;

	int best_a = output[0];
	int best_q = output[1];

	int delta = 0;
	for (;;)
	{
		int last_a = avg_alpha + delta;
		int last_q_way = 0;

		AlphaStripes(area, chunks, g_stripesA[last_a]);
		int err = AlphaLevels(area, g_errorsA[last_a], chunks, water, last_q_way);

		if (water > err)
		{
			water = err;

			best_a = last_a;
			best_q = last_q_way >> 16;

			for (int i = 0; i < 0x10; i++)
			{
				if (last_q_way & (1 << i))
				{
					best_q += i;
					break;
				}
			}

			if (water <= 0)
				break;
		}

		delta = (int)(~(uint32_t)delta) + ((delta < 0) ? 1 : 0);
		if (((avg_alpha + delta) & ~0xFF) != 0)
		{
			delta = (int)(~(uint32_t)delta) + ((delta < 0) ? 1 : 0);
			if (((avg_alpha + delta) & ~0xFF) != 0)
				break;
		}
	}

	ComputeTableAlphaEnhanced(output, area, best_a, best_q);

	return water;
}


static INLINED void DecompressHalfColor(int pL[4], int pH[4], int color, int q, uint32_t data, int shift)
{
	alignas(16) static const int g_delta[8][2] =
	{
		{ 2 * 0x010101, (8 ^ 2) * 0x010101 },
		{ 5 * 0x010101, (17 ^ 5) * 0x010101 },
		{ 9 * 0x010101, (29 ^ 9) * 0x010101 },
		{ 13 * 0x010101, (42 ^ 13) * 0x010101 },
		{ 18 * 0x010101, (60 ^ 18) * 0x010101 },
		{ 24 * 0x010101, (80 ^ 24) * 0x010101 },
		{ 33 * 0x010101, (106 ^ 33) * 0x010101 },
		{ 47 * 0x010101, (183 ^ 47) * 0x010101 }
	};

	alignas(16) static const int g_mask[16][4] =
	{
		{ 0, 0, 0, 0 },{ -1, 0, 0, 0 },{ 0, -1, 0, 0 },{ -1, -1, 0, 0 },
		{ 0, 0, -1, 0 },{ -1, 0, -1, 0 },{ 0, -1, -1, 0 },{ -1, -1, -1, 0 },
		{ 0, 0, 0, -1 },{ -1, 0, 0, -1 },{ 0, -1, 0, -1 },{ -1, -1, 0, -1 },
		{ 0, 0, -1, -1 },{ -1, 0, -1, -1 },{ 0, -1, -1, -1 },{ -1, -1, -1, -1 }
	};

	__m128i mt0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_delta[q][0]), 0);
	__m128i mt10 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_delta[q][1]), 0);

	__m128i mMaskL = _mm_load_si128((const __m128i*)&((const uint8_t*)g_mask)[(shift < 4 ? data << (4 - shift) : data >> (shift - 4)) & 0xF0]);
	__m128i mMaskH = _mm_load_si128((const __m128i*)&((const uint8_t*)g_mask)[(data >> (shift + 4 - 4)) & 0xF0]);

	__m128i mtL = _mm_xor_si128(_mm_and_si128(mMaskL, mt10), mt0);
	__m128i mtH = _mm_xor_si128(_mm_and_si128(mMaskH, mt10), mt0);

	mMaskL = _mm_load_si128((const __m128i*)&((const uint8_t*)g_mask)[(data >> (shift + 16 + 0 - 4)) & 0xF0]);
	mMaskH = _mm_load_si128((const __m128i*)&((const uint8_t*)g_mask)[(data >> (shift + 16 + 4 - 4)) & 0xF0]);

	__m128i mc = _mm_shuffle_epi32(_mm_cvtsi32_si128(color), 0);

	__m128i mcL = _mm_or_si128(_mm_and_si128(mMaskL, _mm_subs_epu8(mc, mtL)), _mm_andnot_si128(mMaskL, _mm_adds_epu8(mc, mtL)));
	__m128i mcH = _mm_or_si128(_mm_and_si128(mMaskH, _mm_subs_epu8(mc, mtH)), _mm_andnot_si128(mMaskH, _mm_adds_epu8(mc, mtH)));

	_mm_storeu_si128((__m128i*)pL, mcL);
	_mm_storeu_si128((__m128i*)pH, mcH);
}

static INLINED void DecompressBlockColorT(const uint8_t input[8], uint8_t* __restrict cell, size_t stride)
{
	uint32_t c = BSWAP(*(const uint32_t*)input);

	uint32_t d = (c >> 1) & 6;
	uint32_t a = (c >> 9) & (0xC << 16);
	uint32_t b = (c << 4) & (0xF << 16);

	d += c & 1;
	a += (c >> 8) & (3 << 16);
	b += c & (0xF << 8);

	__m128i md = _mm_shufflelo_epi16(_mm_cvtsi64_si128((size_t)(uint32_t)g_tableHT[(size_t)d]), _MM_SHUFFLE(3, 0, 0, 0));

	a += (c >> 12) & (0xF << 8);
	b += (c >> 4) & 0xF;
	a += (c >> 16) & 0xF;

	b += b << 4;
	a += a << 4;

	b |= 0xFFu << 24;
	a |= 0xFFu << 24;

	__m128i mb = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((size_t)b));
	__m128i ma = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((size_t)a));

	__m128i mt10 = _mm_unpacklo_epi64(ma, _mm_add_epi16(mb, md));
	__m128i mt32 = _mm_unpacklo_epi64(mb, _mm_sub_epi16(mb, md));

	__m128i mt = _mm_packus_epi16(mt10, mt32);

	alignas(16) uint32_t colors[4];
	_mm_store_si128((__m128i*)colors, mt);

	size_t codes = (size_t)BSWAP(*(const uint32_t*)(input + 4));

	const size_t mask = 0x1111;

	size_t row = (((codes >> 16) & mask) << 1) + (codes & mask);
	*(uint32_t*)cell = colors[row & 3];
	*(uint32_t*)(cell + 4) = colors[(row >> 4) & 3];
	*(uint32_t*)(cell + 8) = colors[(row >> 8) & 3];
	*(uint32_t*)(cell + 12) = colors[(row >> 12) & 3];

	cell += stride;

	row = (((codes >> 17) & mask) << 1) + ((codes >> 1) & mask);
	*(uint32_t*)cell = colors[row & 3];
	*(uint32_t*)(cell + 4) = colors[(row >> 4) & 3];
	*(uint32_t*)(cell + 8) = colors[(row >> 8) & 3];
	*(uint32_t*)(cell + 12) = colors[(row >> 12) & 3];

	cell += stride;

	row = (((codes >> 18) & mask) << 1) + ((codes >> 2) & mask);
	*(uint32_t*)cell = colors[row & 3];
	*(uint32_t*)(cell + 4) = colors[(row >> 4) & 3];
	*(uint32_t*)(cell + 8) = colors[(row >> 8) & 3];
	*(uint32_t*)(cell + 12) = colors[(row >> 12) & 3];

	cell += stride;

	row = (((codes >> 19) & mask) << 1) + ((codes >> 3) & mask);
	*(uint32_t*)cell = colors[row & 3];
	*(uint32_t*)(cell + 4) = colors[(row >> 4) & 3];
	*(uint32_t*)(cell + 8) = colors[(row >> 8) & 3];
	*(uint32_t*)(cell + 12) = colors[(row >> 12) & 3];
}

static INLINED void DecompressBlockColorH(const uint8_t input[8], uint8_t* __restrict cell, size_t stride)
{
	uint32_t c = BSWAP(*(const uint32_t*)input);

	uint32_t d = c & 4;
	uint32_t a = (c >> 11) & (0xF << 16);
	uint32_t b = (c << 5) & (0xF << 16);

	d += (c << 1) & 2;
	a += (c >> 15) & ((0xE << 8) + 7);
	b += (c << 1) & (0xF << 8);

	a += (c >> 12) & (1 << 8);
	b += (c >> 3) & 0xF;
	a += (c >> 16) & 8;

	d |= (a >= b) ? 1 : 0;

	b += b << 4;
	a += a << 4;

	__m128i md = _mm_shufflelo_epi16(_mm_cvtsi64_si128((size_t)(uint32_t)g_tableHT[(size_t)d]), _MM_SHUFFLE(3, 0, 0, 0));

	b |= 0xFFu << 24;
	a |= 0xFFu << 24;

	__m128i mb = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((size_t)b));
	__m128i ma = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((size_t)a));

	__m128i mt32 = _mm_unpacklo_epi64(_mm_add_epi16(mb, md), _mm_sub_epi16(mb, md));
	__m128i mt10 = _mm_unpacklo_epi64(_mm_add_epi16(ma, md), _mm_sub_epi16(ma, md));

	__m128i mt = _mm_packus_epi16(mt10, mt32);

	alignas(16) uint32_t colors[4];
	_mm_store_si128((__m128i*)colors, mt);

	size_t codes = (size_t)BSWAP(*(const uint32_t*)(input + 4));

	const size_t mask = 0x1111;

	size_t row = (((codes >> 16) & mask) << 1) + (codes & mask);
	*(uint32_t*)cell = colors[row & 3];
	*(uint32_t*)(cell + 4) = colors[(row >> 4) & 3];
	*(uint32_t*)(cell + 8) = colors[(row >> 8) & 3];
	*(uint32_t*)(cell + 12) = colors[(row >> 12) & 3];

	cell += stride;

	row = (((codes >> 17) & mask) << 1) + ((codes >> 1) & mask);
	*(uint32_t*)cell = colors[row & 3];
	*(uint32_t*)(cell + 4) = colors[(row >> 4) & 3];
	*(uint32_t*)(cell + 8) = colors[(row >> 8) & 3];
	*(uint32_t*)(cell + 12) = colors[(row >> 12) & 3];

	cell += stride;

	row = (((codes >> 18) & mask) << 1) + ((codes >> 2) & mask);
	*(uint32_t*)cell = colors[row & 3];
	*(uint32_t*)(cell + 4) = colors[(row >> 4) & 3];
	*(uint32_t*)(cell + 8) = colors[(row >> 8) & 3];
	*(uint32_t*)(cell + 12) = colors[(row >> 12) & 3];

	cell += stride;

	row = (((codes >> 19) & mask) << 1) + ((codes >> 3) & mask);
	*(uint32_t*)cell = colors[row & 3];
	*(uint32_t*)(cell + 4) = colors[(row >> 4) & 3];
	*(uint32_t*)(cell + 8) = colors[(row >> 8) & 3];
	*(uint32_t*)(cell + 12) = colors[(row >> 12) & 3];
}

static INLINED void DecompressBlockColorP(const uint8_t input[8], uint8_t* __restrict cell, size_t stride)
{
	uint32_t d0 = BSWAP(*(const uint32_t*)input);
	uint32_t d1 = BSWAP(*(const uint32_t*)(input + 4));

	uint32_t co = (d0 >> 7) & (0xFC << 16);
	uint32_t ch = (d0 << 17) & (0xF8 << 16);
	uint32_t cv = (d1 << 5) & (0xFC << 16);

	co += (d0 >> 9) & ((0x80 << 8) + 0x80);
	ch += (d0 << 18) & (4 << 16);
	cv += (d1 << 3) & (0xFE << 8);

	co += (d0 >> 8) & (0x7E << 8);
	ch += (d1 >> 16) & (0xFE << 8);
	cv += (d1 << 2) & 0xFC;

	co += (d0 >> 6) & 0x60;
	ch += (d1 >> 17) & 0xFC;
	co += (d0 >> 5) & 0x1C;

	cv += (cv >> 6) & ((3 << 16) + 3);
	ch += (ch >> 6) & ((3 << 16) + 3);
	co += (co >> 6) & ((3 << 16) + 3);

	cv += (cv >> 7) & (1 << 8);
	ch += (ch >> 7) & (1 << 8);
	co += (co >> 7) & (1 << 8);

	cv |= 0xFFu << 24;
	ch |= 0xFFu << 24;
	co |= 0xFFu << 24;

	__m128i mv = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((size_t)cv));
	__m128i mh = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((size_t)ch));
	__m128i mo = _mm_cvtepu8_epi16(_mm_cvtsi64_si128((size_t)co));
	__m128i m2 = _mm_cvtepu8_epi16(_mm_cvtsi64_si128(0x02020202));

	mv = _mm_sub_epi16(mv, mo);
	mh = _mm_sub_epi16(mh, mo);

	mo = _mm_slli_epi16(mo, 2);
	mo = _mm_add_epi16(mo, m2);

	__m128i mvv = _mm_unpacklo_epi64(mv, mv);
	__m128i mhh = _mm_unpacklo_epi64(mh, mh);
	__m128i moo = _mm_unpacklo_epi64(mo, mo);

	mh = _mm_shuffle_epi32(mh, _MM_SHUFFLE(1, 0, 3, 2));
	mhh = _mm_add_epi16(mhh, mhh);

	__m128i mtL = _mm_add_epi16(moo, mh);
	__m128i mtH = _mm_add_epi16(mhh, mtL);

	_mm_storeu_si128((__m128i*)cell, _mm_packus_epi16(_mm_srai_epi16(mtL, 2), _mm_srai_epi16(mtH, 2)));

	cell += stride;

	mtL = _mm_add_epi16(mtL, mvv);
	mtH = _mm_add_epi16(mtH, mvv);

	_mm_storeu_si128((__m128i*)cell, _mm_packus_epi16(_mm_srai_epi16(mtL, 2), _mm_srai_epi16(mtH, 2)));

	cell += stride;

	mtL = _mm_add_epi16(mtL, mvv);
	mtH = _mm_add_epi16(mtH, mvv);

	_mm_storeu_si128((__m128i*)cell, _mm_packus_epi16(_mm_srai_epi16(mtL, 2), _mm_srai_epi16(mtH, 2)));

	cell += stride;

	mtL = _mm_add_epi16(mtL, mvv);
	mtH = _mm_add_epi16(mtH, mvv);

	_mm_storeu_si128((__m128i*)cell, _mm_packus_epi16(_mm_srai_epi16(mtL, 2), _mm_srai_epi16(mtH, 2)));
}

static INLINED void DecompressBlockColor(const uint8_t input[8], uint8_t* __restrict cell, size_t stride)
{
	int a, b;

	uint32_t c = *(const uint32_t*)input;

	c = BROR(BSWAP(c));

	if ((c & (2 << 24)) == 0)
	{
		a = c & 0xF0F0F0; a |= a >> 4;
		b = c & 0x0F0F0F; b |= b << 4;
	}
	else
	{
		a = c & 0xF8F8F8;
		b = ((((c & 0x070707) ^ 0x444444) - 0x040404) << 3) + a;
		if (b & 0x01010100)
		{
			if (b & 0x01000000)
			{
				DecompressBlockColorT(input, cell, stride);
				return;
			}

			if (b & 0x00010000)
			{
				DecompressBlockColorH(input, cell, stride);
				return;
			}

			DecompressBlockColorP(input, cell, stride);
			return;
		}
		b &= 0xF8F8F8;

		a |= (a >> 5) & 0x070707;
		b |= (b >> 5) & 0x070707;
	}

	a |= 0xFFu << 24;
	b |= 0xFFu << 24;

	uint32_t way = BSWAP(*(const uint32_t*)&input[4]);

	if ((c & (1 << 24)) == 0)
	{
		alignas(16) int buffer[4][4];

		int qa = (c >> (5 + 24)) & 7;
		DecompressHalfColor(buffer[0], buffer[1], a, qa, way, 0);

		int qb = (c >> (2 + 24)) & 7;
		DecompressHalfColor(buffer[2], buffer[3], b, qb, way, 8);

		__m128i row0 = _mm_load_si128((const __m128i*)buffer[0]);
		__m128i row1 = _mm_load_si128((const __m128i*)buffer[1]);
		__m128i row2 = _mm_load_si128((const __m128i*)buffer[2]);
		__m128i row3 = _mm_load_si128((const __m128i*)buffer[3]);

		__m128i tmp0 = _mm_unpacklo_epi32(row0, row1);
		__m128i tmp2 = _mm_unpacklo_epi32(row2, row3);
		__m128i tmp1 = _mm_unpackhi_epi32(row0, row1);
		__m128i tmp3 = _mm_unpackhi_epi32(row2, row3);

		_mm_storeu_si128((__m128i*)cell, _mm_unpacklo_epi64(tmp0, tmp2));
		_mm_storeu_si128((__m128i*)(cell + stride), _mm_unpackhi_epi64(tmp0, tmp2));

		cell += stride + stride;

		_mm_storeu_si128((__m128i*)cell, _mm_unpacklo_epi64(tmp1, tmp3));
		_mm_storeu_si128((__m128i*)(cell + stride), _mm_unpackhi_epi64(tmp1, tmp3));
	}
	else
	{
		way =
			((way & 0x00080008u) << 9) ^
			((way & 0x00840084u) << 6) ^
			((way & 0x08420842u) << 3) ^
			((way & 0x84218421u)) ^
			((way & 0x42104210u) >> 3) ^
			((way & 0x21002100u) >> 6) ^
			((way & 0x10001000u) >> 9);

		int qa = (c >> (5 + 24)) & 7;
		DecompressHalfColor((int*)cell, (int*)(cell + stride), a, qa, way, 0);

		cell += stride + stride;

		int qb = (c >> (2 + 24)) & 7;
		DecompressHalfColor((int*)cell, (int*)(cell + stride), b, qb, way, 8);
	}
}

static INLINED int CompareBlocksColor(const uint8_t* __restrict cell1, size_t stride1, const uint8_t* __restrict cell2, size_t stride2)
{
	int err_g = 0;
	int err_r = 0;
	int err_b = 0;

	for (int y = 0; y < 4; y++)
	{
		if (cell1[3])
		{
			int db = cell1[0] - cell2[0];
			err_b += db * db;

			int dg = cell1[1] - cell2[1];
			err_g += dg * dg;

			int dr = cell1[2] - cell2[2];
			err_r += dr * dr;
		}

		if (cell1[7])
		{
			int db = cell1[4] - cell2[4];
			err_b += db * db;

			int dg = cell1[5] - cell2[5];
			err_g += dg * dg;

			int dr = cell1[6] - cell2[6];
			err_r += dr * dr;
		}

		if (cell1[11])
		{
			int db = cell1[8] - cell2[8];
			err_b += db * db;

			int dg = cell1[9] - cell2[9];
			err_g += dg * dg;

			int dr = cell1[10] - cell2[10];
			err_r += dr * dr;
		}

		if (cell1[15])
		{
			int db = cell1[12] - cell2[12];
			err_b += db * db;

			int dg = cell1[13] - cell2[13];
			err_g += dg * dg;

			int dr = cell1[14] - cell2[14];
			err_r += dr * dr;
		}

		cell1 += stride1;
		cell2 += stride2;
	}

	return err_g * kGreen + err_r * kRed + err_b * kBlue;
}

static INLINED double CompareBlocksColorSSIM(const uint8_t* __restrict cell1, size_t stride1, const uint8_t* __restrict cell2, size_t stride2)
{
	SSIM_INIT();

	for (int y = 0; y < 4; y++)
	{
		for (int x = 0; x < 4; x++)
		{
			if (cell1[x * 4 + 3])
			{
				__m128i mt = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int*)&cell2[x * 4]));

				__m128i mb = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int*)&cell1[x * 4]));

				SSIM_UPDATE(mt, mb);
			}
		}

		cell1 += stride1;
		cell2 += stride2;
	}

	SSIM_CLOSE(4);

	SSIM_FINAL(mssim_gb, g_ssim_16k1L, g_ssim_16k2L);
	SSIM_OTHER();
	SSIM_FINAL(mssim_r, g_ssim_16k1L, g_ssim_16k2L);

	double ssim =
		_mm_cvtsd_f64(mssim_gb) * kBlue +
		_mm_cvtsd_f64(_mm_unpackhi_pd(mssim_gb, mssim_gb)) * kGreen +
		_mm_cvtsd_f64(mssim_r) * kRed;

	return ssim * (1.0 / kColor);
}


#define SWAP_PAIR(a, b) { Node va = nodep[a], vb = nodep[b]; if( va.Error > vb.Error ) { nodep[a] = vb; nodep[b] = va; } }

static void SortNodes2Shifted(Node* __restrict nodep)
{
#define SWAP(a, b) SWAP_PAIR(a - 1, b - 1)

	// http://jgamble.ripco.net/cgi-bin/nw.cgi?inputs=2&algorithm=best&output=macro
	SWAP(0, 1);

#undef SWAP
}

static void SortNodes4Shifted(Node* __restrict nodep)
{
#define SWAP(a, b) SWAP_PAIR(a - 2, b - 2)

	// http://jgamble.ripco.net/cgi-bin/nw.cgi?inputs=4&algorithm=best&output=macro
	SWAP(0, 1);
	SWAP(2, 3);
	SWAP(0, 2);
	SWAP(1, 3);
	SWAP(1, 2);

#undef SWAP
}

static void SortNodes6Shifted(Node* __restrict nodep)
{
#define SWAP(a, b) SWAP_PAIR(a - 3, b - 3)

	// http://jgamble.ripco.net/cgi-bin/nw.cgi?inputs=6&algorithm=best&output=macro
	SWAP(1, 2);
	SWAP(0, 2);
	SWAP(0, 1);
	SWAP(4, 5);
	SWAP(3, 5);
	SWAP(3, 4);
	SWAP(0, 3);
	SWAP(1, 4);
	SWAP(2, 5);
	SWAP(2, 4);
	SWAP(1, 3);
	SWAP(2, 3);

#undef SWAP
}

static void SortNodes8Shifted(Node* __restrict nodep)
{
#define SWAP(a, b) SWAP_PAIR(a - 4, b - 4)

	// http://jgamble.ripco.net/cgi-bin/nw.cgi?inputs=8&algorithm=best&output=macro
	SWAP(0, 1);
	SWAP(2, 3);
	SWAP(0, 2);
	SWAP(1, 3);
	SWAP(1, 2);
	SWAP(4, 5);
	SWAP(6, 7);
	SWAP(4, 6);
	SWAP(5, 7);
	SWAP(5, 6);
	SWAP(0, 4);
	SWAP(1, 5);
	SWAP(1, 4);
	SWAP(2, 6);
	SWAP(3, 7);
	SWAP(3, 6);
	SWAP(2, 4);
	SWAP(3, 5);
	SWAP(3, 4);

#undef SWAP
}

static void SortNodesCShifted(Node* __restrict nodep)
{
#define SWAP(a, b) SWAP_PAIR(a - 6, b - 6)

	// http://jgamble.ripco.net/cgi-bin/nw.cgi?inputs=12&algorithm=best&output=macro
	SWAP(0, 1);
	SWAP(2, 3);
	SWAP(4, 5);
	SWAP(6, 7);
	SWAP(8, 9);
	SWAP(10, 11);
	SWAP(1, 3);
	SWAP(5, 7);
	SWAP(9, 11);
	SWAP(0, 2);
	SWAP(4, 6);
	SWAP(8, 10);
	SWAP(1, 2);
	SWAP(5, 6);
	SWAP(9, 10);
	SWAP(1, 5);
	SWAP(6, 10);
	SWAP(5, 9);
	SWAP(2, 6);
	SWAP(1, 5);
	SWAP(6, 10);
	SWAP(0, 4);
	SWAP(7, 11);
	SWAP(3, 7);
	SWAP(4, 8);
	SWAP(0, 4);
	SWAP(7, 11);
	SWAP(1, 4);
	SWAP(7, 10);
	SWAP(3, 8);
	SWAP(2, 3);
	SWAP(8, 9);
	SWAP(2, 4);
	SWAP(7, 9);
	SWAP(3, 5);
	SWAP(6, 8);
	SWAP(3, 4);
	SWAP(5, 6);
	SWAP(7, 8);

#undef SWAP
}

static void SortNodes10Shifted(Node* __restrict nodep)
{
#define SWAP(a, b) SWAP_PAIR(a - 8, b - 8)

	// http://jgamble.ripco.net/cgi-bin/nw.cgi?inputs=16&algorithm=best&output=macro
	SWAP(0, 1);
	SWAP(2, 3);
	SWAP(4, 5);
	SWAP(6, 7);
	SWAP(8, 9);
	SWAP(10, 11);
	SWAP(12, 13);
	SWAP(14, 15);
	SWAP(0, 2);
	SWAP(4, 6);
	SWAP(8, 10);
	SWAP(12, 14);
	SWAP(1, 3);
	SWAP(5, 7);
	SWAP(9, 11);
	SWAP(13, 15);
	SWAP(0, 4);
	SWAP(8, 12);
	SWAP(1, 5);
	SWAP(9, 13);
	SWAP(2, 6);
	SWAP(10, 14);
	SWAP(3, 7);
	SWAP(11, 15);
	SWAP(0, 8);
	SWAP(1, 9);
	SWAP(2, 10);
	SWAP(3, 11);
	SWAP(4, 12);
	SWAP(5, 13);
	SWAP(6, 14);
	SWAP(7, 15);
	SWAP(5, 10);
	SWAP(6, 9);
	SWAP(3, 12);
	SWAP(13, 14);
	SWAP(7, 11);
	SWAP(1, 2);
	SWAP(4, 8);
	SWAP(1, 4);
	SWAP(7, 13);
	SWAP(2, 8);
	SWAP(11, 14);
	SWAP(2, 4);
	SWAP(5, 6);
	SWAP(9, 10);
	SWAP(11, 13);
	SWAP(3, 8);
	SWAP(7, 12);
	SWAP(6, 8);
	SWAP(10, 12);
	SWAP(3, 5);
	SWAP(7, 9);
	SWAP(3, 4);
	SWAP(5, 6);
	SWAP(7, 8);
	SWAP(9, 10);
	SWAP(11, 12);
	SWAP(6, 7);
	SWAP(8, 9);

#undef SWAP
}

static void SortNodes18Shifted(Node* __restrict nodep)
{
#define SWAP(a, b) SWAP_PAIR(a - 0xC, b - 0xC)

	// http://jgamble.ripco.net/cgi-bin/nw.cgi?inputs=24&algorithm=best&output=macro
	SWAP(1, 2);
	SWAP(0, 2);
	SWAP(0, 1);
	SWAP(4, 5);
	SWAP(3, 5);
	SWAP(3, 4);
	SWAP(0, 3);
	SWAP(1, 4);
	SWAP(2, 5);
	SWAP(2, 4);
	SWAP(1, 3);
	SWAP(2, 3);
	SWAP(7, 8);
	SWAP(6, 8);
	SWAP(6, 7);
	SWAP(10, 11);
	SWAP(9, 11);
	SWAP(9, 10);
	SWAP(6, 9);
	SWAP(7, 10);
	SWAP(8, 11);
	SWAP(8, 10);
	SWAP(7, 9);
	SWAP(8, 9);
	SWAP(0, 6);
	SWAP(1, 7);
	SWAP(2, 8);
	SWAP(2, 7);
	SWAP(1, 6);
	SWAP(2, 6);
	SWAP(3, 9);
	SWAP(4, 10);
	SWAP(5, 11);
	SWAP(5, 10);
	SWAP(4, 9);
	SWAP(5, 9);
	SWAP(3, 6);
	SWAP(4, 7);
	SWAP(5, 8);
	SWAP(5, 7);
	SWAP(4, 6);
	SWAP(5, 6);
	SWAP(13, 14);
	SWAP(12, 14);
	SWAP(12, 13);
	SWAP(16, 17);
	SWAP(15, 17);
	SWAP(15, 16);
	SWAP(12, 15);
	SWAP(13, 16);
	SWAP(14, 17);
	SWAP(14, 16);
	SWAP(13, 15);
	SWAP(14, 15);
	SWAP(19, 20);
	SWAP(18, 20);
	SWAP(18, 19);
	SWAP(22, 23);
	SWAP(21, 23);
	SWAP(21, 22);
	SWAP(18, 21);
	SWAP(19, 22);
	SWAP(20, 23);
	SWAP(20, 22);
	SWAP(19, 21);
	SWAP(20, 21);
	SWAP(12, 18);
	SWAP(13, 19);
	SWAP(14, 20);
	SWAP(14, 19);
	SWAP(13, 18);
	SWAP(14, 18);
	SWAP(15, 21);
	SWAP(16, 22);
	SWAP(17, 23);
	SWAP(17, 22);
	SWAP(16, 21);
	SWAP(17, 21);
	SWAP(15, 18);
	SWAP(16, 19);
	SWAP(17, 20);
	SWAP(17, 19);
	SWAP(16, 18);
	SWAP(17, 18);
	SWAP(0, 12);
	SWAP(1, 13);
	SWAP(2, 14);
	SWAP(2, 13);
	SWAP(1, 12);
	SWAP(2, 12);
	SWAP(3, 15);
	SWAP(4, 16);
	SWAP(5, 17);
	SWAP(5, 16);
	SWAP(4, 15);
	SWAP(5, 15);
	SWAP(3, 12);
	SWAP(4, 13);
	SWAP(5, 14);
	SWAP(5, 13);
	SWAP(4, 12);
	SWAP(5, 12);
	SWAP(6, 18);
	SWAP(7, 19);
	SWAP(8, 20);
	SWAP(8, 19);
	SWAP(7, 18);
	SWAP(8, 18);
	SWAP(9, 21);
	SWAP(10, 22);
	SWAP(11, 23);
	SWAP(11, 22);
	SWAP(10, 21);
	SWAP(11, 21);
	SWAP(9, 18);
	SWAP(10, 19);
	SWAP(11, 20);
	SWAP(11, 19);
	SWAP(10, 18);
	SWAP(11, 18);
	SWAP(6, 12);
	SWAP(7, 13);
	SWAP(8, 14);
	SWAP(8, 13);
	SWAP(7, 12);
	SWAP(8, 12);
	SWAP(9, 15);
	SWAP(10, 16);
	SWAP(11, 17);
	SWAP(11, 16);
	SWAP(10, 15);
	SWAP(11, 15);
	SWAP(9, 12);
	SWAP(10, 13);
	SWAP(11, 14);
	SWAP(11, 13);
	SWAP(10, 12);
	SWAP(11, 12);

#undef SWAP
}

static void SortNodes20Shifted(Node* __restrict nodep)
{
#define SWAP(a, b) SWAP_PAIR(a - 0x10, b - 0x10)

	// http://jgamble.ripco.net/cgi-bin/nw.cgi?inputs=32&algorithm=best&output=macro
	SWAP(0, 1);
	SWAP(2, 3);
	SWAP(0, 2);
	SWAP(1, 3);
	SWAP(1, 2);
	SWAP(4, 5);
	SWAP(6, 7);
	SWAP(4, 6);
	SWAP(5, 7);
	SWAP(5, 6);
	SWAP(0, 4);
	SWAP(1, 5);
	SWAP(1, 4);
	SWAP(2, 6);
	SWAP(3, 7);
	SWAP(3, 6);
	SWAP(2, 4);
	SWAP(3, 5);
	SWAP(3, 4);
	SWAP(8, 9);
	SWAP(10, 11);
	SWAP(8, 10);
	SWAP(9, 11);
	SWAP(9, 10);
	SWAP(12, 13);
	SWAP(14, 15);
	SWAP(12, 14);
	SWAP(13, 15);
	SWAP(13, 14);
	SWAP(8, 12);
	SWAP(9, 13);
	SWAP(9, 12);
	SWAP(10, 14);
	SWAP(11, 15);
	SWAP(11, 14);
	SWAP(10, 12);
	SWAP(11, 13);
	SWAP(11, 12);
	SWAP(0, 8);
	SWAP(1, 9);
	SWAP(1, 8);
	SWAP(2, 10);
	SWAP(3, 11);
	SWAP(3, 10);
	SWAP(2, 8);
	SWAP(3, 9);
	SWAP(3, 8);
	SWAP(4, 12);
	SWAP(5, 13);
	SWAP(5, 12);
	SWAP(6, 14);
	SWAP(7, 15);
	SWAP(7, 14);
	SWAP(6, 12);
	SWAP(7, 13);
	SWAP(7, 12);
	SWAP(4, 8);
	SWAP(5, 9);
	SWAP(5, 8);
	SWAP(6, 10);
	SWAP(7, 11);
	SWAP(7, 10);
	SWAP(6, 8);
	SWAP(7, 9);
	SWAP(7, 8);
	SWAP(16, 17);
	SWAP(18, 19);
	SWAP(16, 18);
	SWAP(17, 19);
	SWAP(17, 18);
	SWAP(20, 21);
	SWAP(22, 23);
	SWAP(20, 22);
	SWAP(21, 23);
	SWAP(21, 22);
	SWAP(16, 20);
	SWAP(17, 21);
	SWAP(17, 20);
	SWAP(18, 22);
	SWAP(19, 23);
	SWAP(19, 22);
	SWAP(18, 20);
	SWAP(19, 21);
	SWAP(19, 20);
	SWAP(24, 25);
	SWAP(26, 27);
	SWAP(24, 26);
	SWAP(25, 27);
	SWAP(25, 26);
	SWAP(28, 29);
	SWAP(30, 31);
	SWAP(28, 30);
	SWAP(29, 31);
	SWAP(29, 30);
	SWAP(24, 28);
	SWAP(25, 29);
	SWAP(25, 28);
	SWAP(26, 30);
	SWAP(27, 31);
	SWAP(27, 30);
	SWAP(26, 28);
	SWAP(27, 29);
	SWAP(27, 28);
	SWAP(16, 24);
	SWAP(17, 25);
	SWAP(17, 24);
	SWAP(18, 26);
	SWAP(19, 27);
	SWAP(19, 26);
	SWAP(18, 24);
	SWAP(19, 25);
	SWAP(19, 24);
	SWAP(20, 28);
	SWAP(21, 29);
	SWAP(21, 28);
	SWAP(22, 30);
	SWAP(23, 31);
	SWAP(23, 30);
	SWAP(22, 28);
	SWAP(23, 29);
	SWAP(23, 28);
	SWAP(20, 24);
	SWAP(21, 25);
	SWAP(21, 24);
	SWAP(22, 26);
	SWAP(23, 27);
	SWAP(23, 26);
	SWAP(22, 24);
	SWAP(23, 25);
	SWAP(23, 24);
	SWAP(0, 16);
	SWAP(1, 17);
	SWAP(1, 16);
	SWAP(2, 18);
	SWAP(3, 19);
	SWAP(3, 18);
	SWAP(2, 16);
	SWAP(3, 17);
	SWAP(3, 16);
	SWAP(4, 20);
	SWAP(5, 21);
	SWAP(5, 20);
	SWAP(6, 22);
	SWAP(7, 23);
	SWAP(7, 22);
	SWAP(6, 20);
	SWAP(7, 21);
	SWAP(7, 20);
	SWAP(4, 16);
	SWAP(5, 17);
	SWAP(5, 16);
	SWAP(6, 18);
	SWAP(7, 19);
	SWAP(7, 18);
	SWAP(6, 16);
	SWAP(7, 17);
	SWAP(7, 16);
	SWAP(8, 24);
	SWAP(9, 25);
	SWAP(9, 24);
	SWAP(10, 26);
	SWAP(11, 27);
	SWAP(11, 26);
	SWAP(10, 24);
	SWAP(11, 25);
	SWAP(11, 24);
	SWAP(12, 28);
	SWAP(13, 29);
	SWAP(13, 28);
	SWAP(14, 30);
	SWAP(15, 31);
	SWAP(15, 30);
	SWAP(14, 28);
	SWAP(15, 29);
	SWAP(15, 28);
	SWAP(12, 24);
	SWAP(13, 25);
	SWAP(13, 24);
	SWAP(14, 26);
	SWAP(15, 27);
	SWAP(15, 26);
	SWAP(14, 24);
	SWAP(15, 25);
	SWAP(15, 24);
	SWAP(8, 16);
	SWAP(9, 17);
	SWAP(9, 16);
	SWAP(10, 18);
	SWAP(11, 19);
	SWAP(11, 18);
	SWAP(10, 16);
	SWAP(11, 17);
	SWAP(11, 16);
	SWAP(12, 20);
	SWAP(13, 21);
	SWAP(13, 20);
	SWAP(14, 22);
	SWAP(15, 23);
	SWAP(15, 22);
	SWAP(14, 20);
	SWAP(15, 21);
	SWAP(15, 20);
	SWAP(12, 16);
	SWAP(13, 17);
	SWAP(13, 16);
	SWAP(14, 18);
	SWAP(15, 19);
	SWAP(15, 18);
	SWAP(14, 16);
	SWAP(15, 17);
	SWAP(15, 16);

#undef SWAP
}

#undef SWAP_PAIR

static INLINED void SortNodes10(Node nodes[0x10 + 1], int water)
{
	int n = nodes[0x10].Color;
	int w = 0;

	for (int i = 0; i < n; i++)
	{
		Node temp = nodes[i];
		if (temp.Error < water)
		{
			nodes[w++] = temp;
		}
	}

	nodes[0x10].Color = w;

	if (w <= 2)
	{
		for (int i = w; i < 2; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes2Shifted(&nodes[1]);
	}
	else if (w <= 4)
	{
		for (int i = w; i < 4; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes4Shifted(&nodes[2]);
	}
	else if (w <= 6)
	{
		for (int i = w; i < 6; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes6Shifted(&nodes[3]);
	}
	else if (w <= 8)
	{
		for (int i = w; i < 8; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes8Shifted(&nodes[4]);
	}
	else if (w <= 0xC)
	{
		for (int i = w; i < 0xC; i++)
		{
			nodes[i].Error = water;
		}

		SortNodesCShifted(&nodes[6]);
	}
	else
	{
		for (int i = w; i < 0x10; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes10Shifted(&nodes[8]);
	}
}

static INLINED void SortNodes20(Node nodes[0x20 + 1], int water)
{
	int n = nodes[0x20].Color;
	int w = 0;

	for (int i = 0; i < n; i++)
	{
		Node temp = nodes[i];
		if (temp.Error < water)
		{
			nodes[w++] = temp;
		}
	}

	nodes[0x20].Color = w;

	if (w <= 2)
	{
		for (int i = w; i < 2; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes2Shifted(&nodes[1]);
	}
	else if (w <= 4)
	{
		for (int i = w; i < 4; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes4Shifted(&nodes[2]);
	}
	else if (w <= 6)
	{
		for (int i = w; i < 6; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes6Shifted(&nodes[3]);
	}
	else if (w <= 8)
	{
		for (int i = w; i < 8; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes8Shifted(&nodes[4]);
	}
	else if (w <= 0xC)
	{
		for (int i = w; i < 0xC; i++)
		{
			nodes[i].Error = water;
		}

		SortNodesCShifted(&nodes[6]);
	}
	else if (w <= 0x10)
	{
		for (int i = w; i < 0x10; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes10Shifted(&nodes[8]);
	}
	else if (w <= 0x18)
	{
		for (int i = w; i < 0x18; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes18Shifted(&nodes[0xC]);
	}
	else
	{
		for (int i = w; i < 0x20; i++)
		{
			nodes[i].Error = water;
		}

		SortNodes20Shifted(&nodes[0x10]);
	}
}

static int __cdecl StableCompareNodes(const void* lhs, const void* rhs)
{
	const Node* a = (const Node*)lhs;
	const Node* b = (const Node*)rhs;

	int v = a->Error - b->Error;
	if (v == 0)
		v = a->Color - b->Color;

	return v;
}

static INLINED int Sort100(Node A[0x101], int water)
{
	int n = A[0x100].Color;
	int w = 0;

	for (int i = 0; i < n; i++)
	{
		Node temp = A[i];
		if (temp.Error < water)
		{
			A[w++] = temp;
		}
	}

	qsort(A, w, sizeof(Node), &StableCompareNodes);

	return w;
}


static INLINED __m128i load_color_BGRA(const uint8_t color[4])
{
	__m128i margb = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int*)color));

	return _mm_shuffle_epi32(margb, _MM_SHUFFLE(3, 0, 2, 1));
}

static INLINED __m128i load_color_GRB(const uint8_t color[4])
{
	return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int*)color));
}

static INLINED __m128i load_color_GR(const uint8_t color[2])
{
	__m128i mrg = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(uint16_t*)color));

	return _mm_unpacklo_epi64(mrg, mrg);
}

static INLINED int ComputeErrorGRB(const Half& half, const uint8_t color[4], int water, int q)
{
	__m128i mc = load_color_GRB(color);

	__m128i best = _mm_cvtsi32_si128(water);

	__m128i mt0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_table[q][0]), 0);
	__m128i mt1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_table[q][1]), 0);

	__m128i mt2 = _mm_subs_epu8(mc, mt0);
	__m128i mt3 = _mm_subs_epu8(mc, mt1);

	mt0 = _mm_adds_epu8(mc, mt0);
	mt1 = _mm_adds_epu8(mc, mt1);

	__m128i mt10 = _mm_packus_epi32(mt0, mt1);
	__m128i mt32 = _mm_packus_epi32(mt2, mt3);

	__m128i mgrb = _mm_load_si128((const __m128i*)g_GRB_I16);

	__m128i sum = _mm_setzero_si128();

	__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

	int k = half.Count, i = 0;

	while ((k -= 2) >= 0)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&half.Data[i]);
		__m128i my = _mm_load_si128((const __m128i*)&half.Data[i + 4]);

		mx = _mm_packus_epi32(mx, mx);
		my = _mm_packus_epi32(my, my);

		__m128i m10x = _mm_sub_epi16(mt10, mx);
		__m128i m10y = _mm_sub_epi16(mt10, my);
		__m128i m32x = _mm_sub_epi16(mt32, mx);
		__m128i m32y = _mm_sub_epi16(mt32, my);

		m10x = _mm_mullo_epi16(m10x, m10x);
		m10y = _mm_mullo_epi16(m10y, m10y);
		m32x = _mm_mullo_epi16(m32x, m32x);
		m32y = _mm_mullo_epi16(m32y, m32y);

		m10x = _mm_min_epu16(m10x, mlimit);
		m10y = _mm_min_epu16(m10y, mlimit);
		m32x = _mm_min_epu16(m32x, mlimit);
		m32y = _mm_min_epu16(m32y, mlimit);

		m10x = _mm_madd_epi16(m10x, mgrb);
		m10y = _mm_madd_epi16(m10y, mgrb);
		m32x = _mm_madd_epi16(m32x, mgrb);
		m32y = _mm_madd_epi16(m32y, mgrb);

		__m128i me4x = _mm_hadd_epi32(m10x, m32x);
		__m128i me4y = _mm_hadd_epi32(m10y, m32y);

		__m128i me1x = HorizontalMinimum4(me4x);
		__m128i me1y = HorizontalMinimum4(me4y);

		sum = _mm_add_epi32(sum, me1x);
		sum = _mm_add_epi32(sum, me1y);

		i += 8;

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	if (k & 1)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&half.Data[i]);
		mx = _mm_packus_epi32(mx, mx);

		__m128i m10 = _mm_sub_epi16(mt10, mx);
		__m128i m32 = _mm_sub_epi16(mt32, mx);

		m10 = _mm_mullo_epi16(m10, m10);
		m32 = _mm_mullo_epi16(m32, m32);

		m10 = _mm_min_epu16(m10, mlimit);
		m32 = _mm_min_epu16(m32, mlimit);

		m10 = _mm_madd_epi16(m10, mgrb);
		m32 = _mm_madd_epi16(m32, mgrb);

		__m128i me4 = _mm_hadd_epi32(m10, m32);
		__m128i me1 = HorizontalMinimum4(me4);

		sum = _mm_add_epi32(sum, me1);

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	static_assert((kGreen >= kBlue) && (kRed >= kBlue), "Error");
	int int_sum = _mm_cvtsi128_si32(sum);
	if (int_sum < 0x7FFF * kBlue)
	{
		return int_sum;
	}

	sum = _mm_setzero_si128();

	mgrb = _mm_cvtepi16_epi32(mgrb);

	mt0 = _mm_cvtepi16_epi32(mt10);
	mt1 = _mm_unpackhi_epi16(mt10, sum);
	mt2 = _mm_cvtepi16_epi32(mt32);
	mt3 = _mm_unpackhi_epi16(mt32, sum);

	for (k = half.Count, i = 0; --k >= 0; i += 4)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&half.Data[i]);

		__m128i m0 = _mm_sub_epi16(mt0, mx);
		__m128i m1 = _mm_sub_epi16(mt1, mx);
		__m128i m2 = _mm_sub_epi16(mt2, mx);
		__m128i m3 = _mm_sub_epi16(mt3, mx);

		m0 = _mm_mullo_epi16(m0, m0);
		m1 = _mm_mullo_epi16(m1, m1);
		m2 = _mm_mullo_epi16(m2, m2);
		m3 = _mm_mullo_epi16(m3, m3);

		m0 = _mm_mullo_epi32(m0, mgrb);
		m1 = _mm_mullo_epi32(m1, mgrb);
		m2 = _mm_mullo_epi32(m2, mgrb);
		m3 = _mm_mullo_epi32(m3, mgrb);

		__m128i m10 = _mm_hadd_epi32(m0, m1);
		__m128i m32 = _mm_hadd_epi32(m2, m3);
		__m128i me4 = _mm_hadd_epi32(m10, m32);
		__m128i me1 = HorizontalMinimum4(me4);

		sum = _mm_add_epi32(sum, me1);
	}

	return _mm_cvtsi128_si32(sum);
}

static INLINED int ComputeErrorGR(const Half& half, const uint8_t color[2], int water, int q)
{
	__m128i mc = load_color_GR(color);

	__m128i best = _mm_cvtsi32_si128(water);

	__m128i mtt = _mm_loadl_epi64((const __m128i*)&g_table[q][0]);
	__m128i mt10 = _mm_unpacklo_epi32(mtt, mtt);

	__m128i mt32 = _mm_subs_epu8(mc, mt10);

	mt10 = _mm_adds_epu8(mc, mt10);

	__m128i mt3210 = _mm_packus_epi32(mt10, mt32);

	__m128i mgr = _mm_load_si128((const __m128i*)g_GR_I16);

	__m128i sum = _mm_setzero_si128();

	__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

	int k = half.Count, i = 0;

	while ((k -= 4) >= 0)
	{
		__m128i mx = _mm_loadl_epi64((const __m128i*)&half.Data[i]);
		__m128i my = _mm_loadl_epi64((const __m128i*)&half.Data[i + 4]);
		__m128i mz = _mm_loadl_epi64((const __m128i*)&half.Data[i + 8]);
		__m128i mw = _mm_loadl_epi64((const __m128i*)&half.Data[i + 12]);

		mx = _mm_unpacklo_epi64(mx, mx);
		my = _mm_unpacklo_epi64(my, my);
		mz = _mm_unpacklo_epi64(mz, mz);
		mw = _mm_unpacklo_epi64(mw, mw);

		mx = _mm_packus_epi32(mx, mx);
		my = _mm_packus_epi32(my, my);
		mz = _mm_packus_epi32(mz, mz);
		mw = _mm_packus_epi32(mw, mw);

		__m128i m3210x = _mm_sub_epi16(mt3210, mx);
		__m128i m3210y = _mm_sub_epi16(mt3210, my);
		__m128i m3210z = _mm_sub_epi16(mt3210, mz);
		__m128i m3210w = _mm_sub_epi16(mt3210, mw);

		m3210x = _mm_mullo_epi16(m3210x, m3210x);
		m3210y = _mm_mullo_epi16(m3210y, m3210y);
		m3210z = _mm_mullo_epi16(m3210z, m3210z);
		m3210w = _mm_mullo_epi16(m3210w, m3210w);

		m3210x = _mm_min_epu16(m3210x, mlimit);
		m3210y = _mm_min_epu16(m3210y, mlimit);
		m3210z = _mm_min_epu16(m3210z, mlimit);
		m3210w = _mm_min_epu16(m3210w, mlimit);

		m3210x = _mm_madd_epi16(m3210x, mgr);
		m3210y = _mm_madd_epi16(m3210y, mgr);
		m3210z = _mm_madd_epi16(m3210z, mgr);
		m3210w = _mm_madd_epi16(m3210w, mgr);

		__m128i me1x = HorizontalMinimum4(m3210x);
		__m128i me1y = HorizontalMinimum4(m3210y);
		__m128i me1z = HorizontalMinimum4(m3210z);
		__m128i me1w = HorizontalMinimum4(m3210w);

		sum = _mm_add_epi32(sum, me1x);
		sum = _mm_add_epi32(sum, me1y);
		sum = _mm_add_epi32(sum, me1z);
		sum = _mm_add_epi32(sum, me1w);

		i += 16;

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	if (k & 2)
	{
		__m128i mx = _mm_loadl_epi64((const __m128i*)&half.Data[i]);
		__m128i my = _mm_loadl_epi64((const __m128i*)&half.Data[i + 4]);

		mx = _mm_unpacklo_epi64(mx, mx);
		my = _mm_unpacklo_epi64(my, my);

		mx = _mm_packus_epi32(mx, mx);
		my = _mm_packus_epi32(my, my);

		__m128i m3210x = _mm_sub_epi16(mt3210, mx);
		__m128i m3210y = _mm_sub_epi16(mt3210, my);

		m3210x = _mm_mullo_epi16(m3210x, m3210x);
		m3210y = _mm_mullo_epi16(m3210y, m3210y);

		m3210x = _mm_min_epu16(m3210x, mlimit);
		m3210y = _mm_min_epu16(m3210y, mlimit);

		m3210x = _mm_madd_epi16(m3210x, mgr);
		m3210y = _mm_madd_epi16(m3210y, mgr);

		__m128i me1x = HorizontalMinimum4(m3210x);
		__m128i me1y = HorizontalMinimum4(m3210y);

		sum = _mm_add_epi32(sum, me1x);
		sum = _mm_add_epi32(sum, me1y);

		i += 8;

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	if (k & 1)
	{
		__m128i mx = _mm_loadl_epi64((const __m128i*)&half.Data[i]);
		mx = _mm_unpacklo_epi64(mx, mx);
		mx = _mm_packus_epi32(mx, mx);

		__m128i m3210 = _mm_sub_epi16(mt3210, mx);
		m3210 = _mm_mullo_epi16(m3210, m3210);
		m3210 = _mm_min_epu16(m3210, mlimit);
		m3210 = _mm_madd_epi16(m3210, mgr);

		__m128i me1 = HorizontalMinimum4(m3210);

		sum = _mm_add_epi32(sum, me1);

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	static_assert(kGreen >= kRed, "Error");
	int int_sum = _mm_cvtsi128_si32(sum);
	if (int_sum < 0x7FFF * kRed)
	{
		return int_sum;
	}

	sum = _mm_setzero_si128();

	mgr = _mm_cvtepi16_epi32(mgr);

	mt10 = _mm_cvtepi16_epi32(mt3210);
	mt32 = _mm_unpackhi_epi16(mt3210, sum);

	for (k = half.Count, i = 0; --k >= 0; i += 4)
	{
		__m128i mx = _mm_loadl_epi64((const __m128i*)&half.Data[i]);
		mx = _mm_unpacklo_epi64(mx, mx);

		__m128i m10 = _mm_sub_epi16(mt10, mx);
		__m128i m32 = _mm_sub_epi16(mt32, mx);

		m10 = _mm_mullo_epi16(m10, m10);
		m32 = _mm_mullo_epi16(m32, m32);

		m10 = _mm_mullo_epi32(m10, mgr);
		m32 = _mm_mullo_epi32(m32, mgr);

		__m128i me4 = _mm_hadd_epi32(m10, m32);
		__m128i me1 = HorizontalMinimum4(me4);

		sum = _mm_add_epi32(sum, me1);
	}

	return _mm_cvtsi128_si32(sum);
}

struct BlockStateColor
{
	uint8_t a[4], b[4];
	int qa, qb, mode;
};

struct GuessStateColor
{
	alignas(16) Node node0[0x12], node1[0x12], node2[0x12];

	int stop;

	INLINED GuessStateColor()
	{
	}

	INLINED void Init(const Half& half, int water, int q)
	{
		GuessLevels(half, 0, node0, kGreen, water, q);
		stop = node0[0x10].Error;

		if (stop >= water)
			return;

		GuessLevels(half, 1, node1, kRed, water - stop, q);
		stop += node1[0x10].Error;

		if (stop >= water)
			return;

		GuessLevels(half, 2, node2, kBlue, water - stop, q);
		stop += node2[0x10].Error;
	}

	INLINED void Sort(int water)
	{
		SortNodes10(node0, water - node1[0x10].Error - node2[0x10].Error);
		SortNodes10(node1, water - node0[0x10].Error - node2[0x10].Error);
		SortNodes10(node2, water - node0[0x10].Error - node1[0x10].Error);
	}
};

struct AdjustStateColor
{
	bool flag_sort;
	bool flag_error;
	bool flag_minimum;
	bool flag_swap;

	int stop;
	uint32_t flag0, flag1, flag2;
	int unused0, unused1, unused2;

	alignas(16) Node node0[0x22], node1[0x22], node2[0x22];

	int ErrorsG[0x20];
	int ErrorsGR[0x20 * 0x20];
	bool LazyGR[0x20 * 0x20];
	int ErrorsGRB[0x20 * 0x20 * 0x20];

	int swap0[0x20], swap1[0x20], swap2[0x20];

	alignas(16) Node diff0[0x20][10], diff1[0x20][10], diff2[0x20][10];

	INLINED AdjustStateColor()
	{
	}

	INLINED void Init(const Half& half, int water, int q)
	{
		bool f = false;
		flag_sort = f;
		flag_error = f;
		flag_minimum = f;
		flag_swap = f;

		AdjustLevels(half, 0, node0, kGreen, water, q);
		stop = node0[0x20].Error;

		if (stop >= water)
			return;

		AdjustLevels(half, 1, node1, kRed, water - stop, q);
		stop += node1[0x20].Error;

		if (stop >= water)
			return;

		AdjustLevels(half, 2, node2, kBlue, water - stop, q);
		stop += node2[0x20].Error;
	}

	INLINED void Sort(int water)
	{
		if (flag_sort)
			return;
		flag_sort = true;

		SortNodes20(node0, water - node1[0x20].Error - node2[0x20].Error);
		SortNodes20(node1, water - node0[0x20].Error - node2[0x20].Error);
		SortNodes20(node2, water - node0[0x20].Error - node1[0x20].Error);
	}

	NOTINLINED void DoWalk(const Half& half, int water, int q)
	{
		uint8_t c[4];

		int min01 = water - node2[0x20].Error;

		for (int c0 = 0; c0 < node0[0x20].Color; c0++)
		{
			int e0 = node0[c0].Error;
			if (e0 + node1[0x20].Error + node2[0x20].Error >= water)
				break;

			c[0] = (uint8_t)node0[c0].Color;

			int min1 = water - node2[0x20].Error;

			for (int c1 = 0; c1 < node1[0x20].Color; c1++)
			{
				int e1 = node1[c1].Error + e0;
				if (e1 + node2[0x20].Error >= water)
					break;

				c[1] = (uint8_t)node1[c1].Color;

				e1 = ComputeErrorGR(half, c, water - node2[0x20].Error, q);

				if (min1 > e1)
					min1 = e1;

				int originGR = (c0 << 5) + c1;
				ErrorsGR[originGR] = e1;
				LazyGR[originGR] = true;
			}

			ErrorsG[c0] = min1;

			if (min01 > min1)
				min01 = min1;
		}

		stop = min01 + node2[0x20].Error;
	}

	INLINED void Walk(const Half& half, int water, int q)
	{
		if (flag_error)
			return;
		flag_error = true;

		DoWalk(half, water, q);
	}

	NOTINLINED void DoBottom(const Half& half, int water, int q)
	{
		uint8_t c[4];

		int blue_max = 0xFF;
		int minimum = Min(water, stop - node2[0x20].Error + (blue_max * blue_max * kBlue) * half.Count);

		for (int c0 = 0; c0 < node0[0x20].Color; c0++)
		{
			int e0 = node0[c0].Error;
			if (e0 + node1[0x20].Error + node2[0x20].Error >= minimum)
				break;

			if (ErrorsG[c0] + node2[0x20].Error >= minimum)
				continue;

			c[0] = (uint8_t)node0[c0].Color;

			for (int c1 = 0; c1 < node1[0x20].Color; c1++)
			{
				int e1 = node1[c1].Error + e0;
				if (e1 + node2[0x20].Error >= minimum)
					break;

				int originGR = (c0 << 5) + c1;
				e1 = ErrorsGR[originGR];
				if (e1 + node2[0x20].Error >= minimum)
					continue;

				c[1] = (uint8_t)node1[c1].Color;

				LazyGR[originGR] = false;

				int origin = originGR << 5;

				for (int c2 = 0; c2 < node2[0x20].Color; c2++)
				{
					int e2 = node2[c2].Error + e1;
					if (e2 >= minimum)
					{
						ErrorsGRB[c2 + origin] = -1;
						continue;
					}

					c[2] = (uint8_t)node2[c2].Color;

					e2 = ComputeErrorGRB(half, c, water, q);
					ErrorsGRB[c2 + origin] = e2;

					if (minimum > e2)
						minimum = e2;
				}
			}
		}

		stop = minimum;
	}

	INLINED void Bottom(const Half& half, int water, int q)
	{
		if (flag_minimum)
			return;
		flag_minimum = true;

		DoBottom(half, water, q);
	}

	INLINED void Index()
	{
		if (flag_swap)
			return;
		flag_swap = true;

		memset(swap0, -1, sizeof(swap0));
		memset(swap1, -1, sizeof(swap1));
		memset(swap2, -1, sizeof(swap2));

		for (int i = 0, n = node0[0x20].Color; i < n; i++)
		{
			int c0 = node0[i].Color >> 3;
			swap0[c0] = i;
		}

		for (int i = 0, n = node1[0x20].Color; i < n; i++)
		{
			int c1 = node1[i].Color >> 3;
			swap1[c1] = i;
		}

		for (int i = 0, n = node2[0x20].Color; i < n; i++)
		{
			int c2 = node2[i].Color >> 3;
			swap2[c2] = i;
		}

		flag0 = 0;
		flag1 = 0;
		flag2 = 0;
	}
};

struct AdjustStateColorGroup
{
	AdjustStateColor S[8];

	INLINED AdjustStateColorGroup()
	{
	}

	INLINED void Init(const Half& half, int water)
	{
		for (int q = 0; q < 8; q++)
		{
			S[q].Init(half, water, q);
		}
	}

	INLINED int Best(int water) const
	{
		int best = water;

		for (int q = 0; q < 8; q++)
		{
			best = Min(best, S[q].stop);
		}

		return best;
	}
};

static INLINED void ComputeTableColor(const Half& half, const uint8_t color[4], int q, uint32_t& index)
{
	size_t halfSize = (size_t)(uint32_t)half.Count;
	if (halfSize == 0)
		return;

	__m128i mc = load_color_GRB(color);

	__m128i mt0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_table[q][0]), 0);
	__m128i mt1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_table[q][1]), 0);

	__m128i mt2 = _mm_subs_epu8(mc, mt0);
	__m128i mt3 = _mm_subs_epu8(mc, mt1);

	mt0 = _mm_adds_epu8(mc, mt0);
	mt1 = _mm_adds_epu8(mc, mt1);

	int good = 0xF;
	if ((_mm_movemask_epi8(_mm_cmpeq_epi32(mt0, mt1)) | 0xF000) == 0xFFFF) good &= ~2;
	if ((_mm_movemask_epi8(_mm_cmpeq_epi32(mt2, mt3)) | 0xF000) == 0xFFFF) good &= ~8;

	__m128i mgrb = _mm_loadl_epi64((const __m128i*)g_GRB_I16);
	mgrb = _mm_cvtepi16_epi32(mgrb);

	int ways[8];

	for (size_t k = 0, i = 0; k < halfSize; k++, i += 4)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&half.Data[i]);

		__m128i m0 = _mm_sub_epi16(mt0, mx);
		__m128i m1 = _mm_sub_epi16(mt1, mx);
		__m128i m2 = _mm_sub_epi16(mt2, mx);
		__m128i m3 = _mm_sub_epi16(mt3, mx);

		m0 = _mm_mullo_epi16(m0, m0);
		m1 = _mm_mullo_epi16(m1, m1);
		m2 = _mm_mullo_epi16(m2, m2);
		m3 = _mm_mullo_epi16(m3, m3);

		m0 = _mm_mullo_epi32(m0, mgrb);
		m1 = _mm_mullo_epi32(m1, mgrb);
		m2 = _mm_mullo_epi32(m2, mgrb);
		m3 = _mm_mullo_epi32(m3, mgrb);

		__m128i m10 = _mm_hadd_epi32(m0, m1);
		__m128i m32 = _mm_hadd_epi32(m2, m3);
		__m128i me4 = _mm_hadd_epi32(m10, m32);
		__m128i me1 = HorizontalMinimum4(me4);

		int way = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4, me1)));
		ways[k] = (way & good) | (1 << 4);
	}

	__m128i vals[4];
	_mm_store_si128(&vals[0], mt0);
	_mm_store_si128(&vals[1], mt1);
	_mm_store_si128(&vals[2], mt2);
	_mm_store_si128(&vals[3], mt3);

	int loops[8];

	for (size_t i = 0; i < halfSize; i++)
	{
		int k = 0;
		while ((ways[i] & (1 << k)) == 0) k++;
		loops[i] = k;
	}

	double best = -(kColor + 0.1);
	uint32_t codes = 0;

	for (;; )
	{
		SSIM_INIT();

		for (size_t i = 0; i < halfSize; i++)
		{
			__m128i mt = _mm_load_si128(&vals[(size_t)(uint32_t)loops[i]]);

			__m128i mb = _mm_load_si128((const __m128i*)&half.Data[i << 2]);

			SSIM_UPDATE(mt, mb);
		}

		SSIM_CLOSE(3);

		SSIM_FINAL(mssim_rg, g_ssim_8k1L, g_ssim_8k2L);
		SSIM_OTHER();
		SSIM_FINAL(mssim_b, g_ssim_8k1L, g_ssim_8k2L);

		double ssim =
			_mm_cvtsd_f64(mssim_rg) * kGreen +
			_mm_cvtsd_f64(_mm_unpackhi_pd(mssim_rg, mssim_rg)) * kRed +
			_mm_cvtsd_f64(mssim_b) * kBlue;

		if (best < ssim)
		{
			best = ssim;

			uint32_t v = 0;
			for (size_t j = 0; j < halfSize; j++)
			{
				v |= ((uint32_t)loops[j]) << (j + j);
			}
			codes = v;

			if (best >= kColor)
				break;
		}

		size_t i = 0;
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

			i++;
			if (i >= halfSize)
				break;
		}
		if (i >= halfSize)
			break;
	}

	for (size_t j = 0; j < halfSize; j++)
	{
		int shift = half.Shift[j];

		uint32_t code = ((codes & 2u) << (16 - 1)) | (codes & 1u);

		codes >>= 2;

		index |= code << shift;
	}
}

static INLINED int GuessColor4(const Half& half, uint8_t color[4], int water, int& table)
{
	GuessStateColor S;

	for (int q = 0; q < 8; q++)
	{
		S.Init(half, water, q);

		if (S.stop >= water)
			continue;

		S.Sort(water);

		uint8_t c[4];

		for (int c0 = 0; c0 < S.node0[0x10].Color; c0++)
		{
			int e0 = S.node0[c0].Error;
			if (e0 + S.node1[0x10].Error + S.node2[0x10].Error >= water)
				break;

			c[0] = (uint8_t)S.node0[c0].Color;

			for (int c1 = 0; c1 < S.node1[0x10].Color; c1++)
			{
				int e1 = S.node1[c1].Error + e0;
				if (e1 + S.node2[0x10].Error >= water)
					break;

				c[1] = (uint8_t)S.node1[c1].Color;

				e1 = ComputeErrorGR(half, c, water - S.node2[0x10].Error, q);
				if (e1 + S.node2[0x10].Error >= water)
					continue;

				for (int c2 = 0; c2 < S.node2[0x10].Color; c2++)
				{
					int e2 = S.node2[c2].Error + e1;
					if (e2 >= water)
						break;

					c[2] = (uint8_t)S.node2[c2].Color;

					e2 = ComputeErrorGRB(half, c, water, q);

					if (water > e2)
					{
						water = e2;

						memcpy(color, c, 4);

						table = q;
					}
				}
			}
		}
	}

	return water;
}

static int CompressBlockColor44(BlockStateColor &s, const Elem& elem, int water, int mode)
{
	uint8_t a[4], b[4];
	int qa, qb;

	int err = GuessColor4(elem.A, a, water, qa);
	if (err >= water)
		return water;

	err += GuessColor4(elem.B, b, water - err, qb);
	if (err >= water)
		return water;

	memcpy(s.a, a, 4);
	memcpy(s.b, b, 4);
	s.qa = qa;
	s.qb = qb;
	s.mode = mode;

	return err;
}

static INLINED int DifferentialColors3(int Id, uint32_t& flag, const Node node[0x20], const int swap[0x20], Node diff[8 + 1], int water)
{
	if ((flag & (1u << Id)) == 0)
	{
		flag |= (1u << Id);

		int Ld = Max(Id - 4, 0);
		int Hd = Min(Id + 3, 31);

		int w = 0;

		for (int d = Ld; d <= Hd; d++)
		{
			int b = swap[d];
			if (b < 0)
				continue;

			int error = node[b].Error;
			if (error < water)
			{
				diff[w].Error = error;
				diff[w].Color = b;
				w++;
			}
		}

		diff[8].Color = w;

		if (w <= 2)
		{
			for (int i = w; i < 2; i++)
			{
				diff[i].Error = water;
			}

			SortNodes2Shifted(&diff[1]);
		}
		else if (w <= 4)
		{
			for (int i = w; i < 4; i++)
			{
				diff[i].Error = water;
			}

			SortNodes4Shifted(&diff[2]);
		}
		else if (w <= 6)
		{
			for (int i = w; i < 6; i++)
			{
				diff[i].Error = water;
			}

			SortNodes6Shifted(&diff[3]);
		}
		else
		{
			for (int i = w; i < 8; i++)
			{
				diff[i].Error = water;
			}

			SortNodes8Shifted(&diff[4]);
		}
	}

	return diff[0].Error;
}

static INLINED int AdjustColors53(const Elem& elem, uint8_t color[4], uint8_t other[4], int water, int qa, int qb, AdjustStateColor& SA, AdjustStateColor& SB, int bestA, int bestB)
{
	uint8_t a[4], b[4];

	SB.Sort(water - bestA);
	SB.Walk(elem.B, water - bestA, qb);
	if (Max(bestA, SA.stop) + SB.stop >= water)
		return water;

	SB.Bottom(elem.B, water - bestA, qb);
	if (Max(bestA, SA.stop) + SB.stop >= water)
		return water;

	SA.Sort(water - bestB);
	SA.Walk(elem.A, water - bestB, qa);
	if (SA.stop + SB.stop >= water)
		return water;

	SA.Bottom(elem.A, water - bestB, qa);
	if (SA.stop + SB.stop >= water)
		return water;

	SB.Index();

	int most = water - SB.stop;

	for (int a0 = 0; a0 < SA.node0[0x20].Color; a0++)
	{
		int e0 = SA.node0[a0].Error;
		if (e0 + SA.node1[0x20].Error + SA.node2[0x20].Error >= most)
			break;

		if (SA.ErrorsG[a0] + SA.node2[0x20].Error >= most)
			continue;

		a[0] = (uint8_t)SA.node0[a0].Color;

		int Id0 = a[0] >> 3;
		Node* diff0 = SB.diff0[Id0];
		int min0 = DifferentialColors3(Id0, SB.flag0, SB.node0, SB.swap0, diff0, water - bestA);

		if (SA.ErrorsG[a0] + SA.node2[0x20].Error + min0 + SB.node1[0x20].Error + SB.node2[0x20].Error >= water)
			continue;

		for (int a1 = 0; a1 < SA.node1[0x20].Color; a1++)
		{
			int e1 = SA.node1[a1].Error + e0;
			if (e1 + SA.node2[0x20].Error >= most)
				break;

			int a_originGR = (a0 << 5) + a1;
			e1 = SA.ErrorsGR[a_originGR];
			if (e1 + SA.node2[0x20].Error >= most)
				continue;

			int a_origin = a_originGR << 5;

			a[1] = (uint8_t)SA.node1[a1].Color;

			int Id1 = a[1] >> 3;
			Node* diff1 = SB.diff1[Id1];
			int min1 = DifferentialColors3(Id1, SB.flag1, SB.node1, SB.swap1, diff1, water - bestA);

			if (e1 + SA.node2[0x20].Error + min0 + min1 + SB.node2[0x20].Error >= water)
				continue;

			if (SA.LazyGR[a_originGR])
			{
				SA.LazyGR[a_originGR] = false;

				for (int c2 = 0; c2 < SA.node2[0x20].Color; c2++)
				{
					SA.ErrorsGRB[c2 + a_origin] = -1;
				}
			}

			for (int a2 = 0; a2 < SA.node2[0x20].Color; a2++)
			{
				int e2 = SA.node2[a2].Error + e1;
				if (e2 >= most)
					break;

				a[2] = (uint8_t)SA.node2[a2].Color;

				int ea = SA.ErrorsGRB[a2 + a_origin];
				if (ea < 0)
				{
					ea = ComputeErrorGRB(elem.A, a, water - bestB, qa);
					SA.ErrorsGRB[a2 + a_origin] = ea;
				}

				e2 = ea;
				if (e2 >= most)
					continue;

				int Id2 = a[2] >> 3;
				Node* diff2 = SB.diff2[Id2];
				int min2 = DifferentialColors3(Id2, SB.flag2, SB.node2, SB.swap2, diff2, water - bestA);

				if (e2 + min0 + min1 + min2 >= water)
					continue;

				for (int d0 = 0; d0 < diff0[8].Color; d0++)
				{
					int e3 = diff0[d0].Error + e2;
					if (e3 + min1 + min2 >= water)
						break;

					int b0 = diff0[d0].Color;
					if (e2 + SB.ErrorsG[b0] + min2 >= water)
						continue;

					for (int d1 = 0; d1 < diff1[8].Color; d1++)
					{
						int e4 = diff1[d1].Error + e3;
						if (e4 + min2 >= water)
							break;

						int b1 = diff1[d1].Color;
						int b_originGR = (b0 << 5) + b1;
						e4 = SB.ErrorsGR[b_originGR] + e2;
						if (e4 + min2 >= water)
							continue;

						int b_origin = b_originGR << 5;

						if (SB.LazyGR[b_originGR])
						{
							SB.LazyGR[b_originGR] = false;

							for (int c2 = 0; c2 < SB.node2[0x20].Color; c2++)
							{
								SB.ErrorsGRB[c2 + b_origin] = -1;
							}
						}

						for (int d2 = 0; d2 < diff2[8].Color; d2++)
						{
							int e5 = diff2[d2].Error + e4;
							if (e5 >= water)
								break;

							int b2 = diff2[d2].Color;
							int eb = SB.ErrorsGRB[b2 + b_origin];
							if (eb < 0)
							{
								b[0] = (uint8_t)SB.node0[b0].Color;
								b[1] = (uint8_t)SB.node1[b1].Color;
								b[2] = (uint8_t)SB.node2[b2].Color;

								eb = ComputeErrorGRB(elem.B, b, water - bestA, qb);
								SB.ErrorsGRB[b2 + b_origin] = eb;
							}

							e5 = eb + e2;

							if (water > e5)
							{
								water = e5;
								most = water - SB.stop;

								memcpy(color, a, 4);

								other[0] = (uint8_t)SB.node0[b0].Color;
								other[1] = (uint8_t)SB.node1[b1].Color;
								other[2] = (uint8_t)SB.node2[b2].Color;
							}
						}
					}
				}
			}
		}
	}

	return water;
}

static int CompressBlockColor53(BlockStateColor &s, const Elem& elem, int water, int mode)
{
	AdjustStateColorGroup GB;

	GB.Init(elem.B, water);

	int stopB = GB.Best(water);
	if (stopB >= water)
		return water;

	AdjustStateColorGroup GA;

	GA.Init(elem.A, water - stopB);

	for (int qa = 0; qa < 8; qa++)
	{
		AdjustStateColor& SA = GA.S[qa];

		for (int qb = 0; qb < 8; qb++)
		{
			AdjustStateColor& SB = GB.S[qb];

			int bestB = GB.Best(water);
			int bestA = GA.Best(water - bestB);
			if (bestA + bestB >= water)
				return water;

			if (Max(bestA, SA.stop) + bestB >= water)
				break;

			if (Max(bestA, SA.stop) + SB.stop >= water)
				continue;

			int error = AdjustColors53(elem, s.a, s.b, water, qa, qb, SA, SB, bestA, bestB);

			if (water > error)
			{
				water = error;

				s.qa = qa;
				s.qb = qb;
				s.mode = mode;
			}
		}

		SA.stop = water;
	}

	return water;
}

static INLINED void FilterPixelsColor(Half& half, uint32_t order)
{
	size_t w = 0;

	for (size_t i = 0; i < 8 * 4; i += 4)
	{
		__m128i m = _mm_load_si128((const __m128i*)&half.Data[i]);

		int a = _mm_extract_epi16(m, 6);

		_mm_store_si128((__m128i*)&half.Data[w * 4], m);

		half.Shift[w] = order & 0xF;

		order >>= 4;

		w += (a != 0) ? 1 : 0;
	}

	half.Count = (int)w;
}

static double EstimateChromaDispersion(const Half& half)
{
	static const double g_r_nn[7] =
	{
		kRed / (2.0 * 2.0), kRed / (3.0 * 3.0), kRed / (4.0 * 4.0), kRed / (5.0 * 5.0), kRed / (6.0 * 6.0), kRed / (7.0 * 7.0), kRed / (8.0 * 8.0)
	};

	static const double g_b_nn[7] =
	{
		kBlue / (2.0 * 2.0), kBlue / (3.0 * 3.0), kBlue / (4.0 * 4.0), kBlue / (5.0 * 5.0), kBlue / (6.0 * 6.0), kBlue / (7.0 * 7.0), kBlue / (8.0 * 8.0)
	};

	int n = half.Count;
	if (n < 2)
		return 0;

	int sr = 0, sb = 0, srr = 0, sbb = 0;

	for (int i = 0; i < n; i++)
	{
		int g = half.Data[i * 4 + 0];
		int r = half.Data[i * 4 + 1];
		int b = half.Data[i * 4 + 2];

		r -= g;
		b -= g;

		sr += r;
		sb += b;

		srr += r * r;
		sbb += b * b;
	}

	return (srr * n - sr * sr) * g_r_nn[n - 2] + (sbb * n - sb * sb) * g_b_nn[n - 2];
}

static int CompressBlockColor(uint8_t output[8], const uint8_t* __restrict cell, size_t stride, int input_error)
{
	Elem norm, flip;

	{
		const uint8_t* src = cell;

		__m128i c00 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(flip.A.Data + 0), c00); _mm_store_si128((__m128i*)(norm.A.Data + 0), c00);
		__m128i c01 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(flip.A.Data + 4), c01); _mm_store_si128((__m128i*)(norm.A.Data + 4), c01);
		__m128i c02 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(flip.A.Data + 8), c02); _mm_store_si128((__m128i*)(norm.B.Data + 0), c02);
		__m128i c03 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(flip.A.Data + 12), c03); _mm_store_si128((__m128i*)(norm.B.Data + 4), c03);

		src += stride;

		__m128i c10 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(flip.A.Data + 16), c10); _mm_store_si128((__m128i*)(norm.A.Data + 8), c10);
		__m128i c11 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(flip.A.Data + 20), c11); _mm_store_si128((__m128i*)(norm.A.Data + 12), c11);
		__m128i c12 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(flip.A.Data + 24), c12); _mm_store_si128((__m128i*)(norm.B.Data + 8), c12);
		__m128i c13 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(flip.A.Data + 28), c13); _mm_store_si128((__m128i*)(norm.B.Data + 12), c13);

		src += stride;

		__m128i c20 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(flip.B.Data + 0), c20); _mm_store_si128((__m128i*)(norm.A.Data + 16), c20);
		__m128i c21 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(flip.B.Data + 4), c21); _mm_store_si128((__m128i*)(norm.A.Data + 20), c21);
		__m128i c22 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(flip.B.Data + 8), c22); _mm_store_si128((__m128i*)(norm.B.Data + 16), c22);
		__m128i c23 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(flip.B.Data + 12), c23); _mm_store_si128((__m128i*)(norm.B.Data + 20), c23);

		src += stride;

		__m128i c30 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(flip.B.Data + 16), c30); _mm_store_si128((__m128i*)(norm.A.Data + 24), c30);
		__m128i c31 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(flip.B.Data + 20), c31); _mm_store_si128((__m128i*)(norm.A.Data + 28), c31);
		__m128i c32 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(flip.B.Data + 24), c32); _mm_store_si128((__m128i*)(norm.B.Data + 24), c32);
		__m128i c33 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(flip.B.Data + 28), c33); _mm_store_si128((__m128i*)(norm.B.Data + 28), c33);
	}

	FilterPixelsColor(norm.A, 0x73625140u);
	FilterPixelsColor(norm.B, 0x73625140u + 0x88888888u);

	FilterPixelsColor(flip.A, 0xD951C840u);
	FilterPixelsColor(flip.B, 0xD951C840u + 0x22222222u);

	BlockStateColor s = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, 0, 0, 0 };

	int err = input_error;
	if (err > 0)
	{
		double Dnorm = EstimateChromaDispersion(norm.A) + EstimateChromaDispersion(norm.B);
		double Dflip = EstimateChromaDispersion(flip.A) + EstimateChromaDispersion(flip.B);

		if (Dnorm <= Dflip)
		{
			int err_norm44 = CompressBlockColor44(s, norm, err, 0);
			if (err > err_norm44)
				err = err_norm44;

			if (err > 0)
			{
				int err_flip44 = CompressBlockColor44(s, flip, err, 1);
				if (err > err_flip44)
					err = err_flip44;

				if (err > 0)
				{
					int err_norm53 = CompressBlockColor53(s, norm, err, 2);
					if (err > err_norm53)
						err = err_norm53;

					if (err > 0)
					{
						int err_flip53 = CompressBlockColor53(s, flip, err, 3);
						if (err > err_flip53)
							err = err_flip53;
					}
				}
			}
		}
		else
		{
			int err_flip44 = CompressBlockColor44(s, flip, err, 1);
			if (err > err_flip44)
				err = err_flip44;

			if (err > 0)
			{
				int err_norm44 = CompressBlockColor44(s, norm, err, 0);
				if (err > err_norm44)
					err = err_norm44;

				if (err > 0)
				{
					int err_flip53 = CompressBlockColor53(s, flip, err, 3);
					if (err > err_flip53)
						err = err_flip53;

					if (err > 0)
					{
						int err_norm53 = CompressBlockColor53(s, norm, err, 2);
						if (err > err_norm53)
							err = err_norm53;
					}
				}
			}
		}
	}

	if (input_error > err)
	{
		uint32_t index = 0;

		bool f = (s.mode & 1) != 0;
		ComputeTableColor(f ? flip.A : norm.A, s.a, s.qa, index);
		ComputeTableColor(f ? flip.B : norm.B, s.b, s.qb, index);

		if (s.mode & 2)
		{
			output[0] = (uint8_t)((s.a[1] & 0xF8) ^ (((s.b[1] >> 3) - (s.a[1] >> 3)) & 7));
			output[1] = (uint8_t)((s.a[0] & 0xF8) ^ (((s.b[0] >> 3) - (s.a[0] >> 3)) & 7));
			output[2] = (uint8_t)((s.a[2] & 0xF8) ^ (((s.b[2] >> 3) - (s.a[2] >> 3)) & 7));
		}
		else
		{
			output[0] = (s.a[1] & 0xF0) ^ (s.b[1] & 0x0F);
			output[1] = (s.a[0] & 0xF0) ^ (s.b[0] & 0x0F);
			output[2] = (s.a[2] & 0xF0) ^ (s.b[2] & 0x0F);
		}

		output[3] = (uint8_t)((s.qa << 5) ^ (s.qb << 2) ^ s.mode);

		*(uint32_t*)&output[4] = BSWAP(index);
	}

	return err;
}


static INLINED int ComputeErrorFourGRB(const Area& area, __m128i mc0, __m128i mc1, const __m128i& mc2, const __m128i& mc3, int water)
{
	__m128i best = _mm_cvtsi64_si128((size_t)(uint32_t)water);

	__m128i mt10 = _mm_packus_epi32(mc0, mc1);
	__m128i mt32 = _mm_packus_epi32(mc2, mc3);

	__m128i mgrb = _mm_load_si128((const __m128i*)g_GRB_I16);

	__m128i sum = _mm_setzero_si128();

	__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

	int k = area.Count, i = 0;

	while ((k -= 2) >= 0)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&area.Data[i]);
		__m128i my = _mm_load_si128((const __m128i*)&area.Data[i + 4]);

		mx = _mm_packus_epi32(mx, mx);
		my = _mm_packus_epi32(my, my);

		__m128i m10x = _mm_sub_epi16(mt10, mx);
		__m128i m10y = _mm_sub_epi16(mt10, my);
		__m128i m32x = _mm_sub_epi16(mt32, mx);
		__m128i m32y = _mm_sub_epi16(mt32, my);

		m10x = _mm_mullo_epi16(m10x, m10x);
		m10y = _mm_mullo_epi16(m10y, m10y);
		m32x = _mm_mullo_epi16(m32x, m32x);
		m32y = _mm_mullo_epi16(m32y, m32y);

		m10x = _mm_min_epu16(m10x, mlimit);
		m10y = _mm_min_epu16(m10y, mlimit);
		m32x = _mm_min_epu16(m32x, mlimit);
		m32y = _mm_min_epu16(m32y, mlimit);

		m10x = _mm_madd_epi16(m10x, mgrb);
		m10y = _mm_madd_epi16(m10y, mgrb);
		m32x = _mm_madd_epi16(m32x, mgrb);
		m32y = _mm_madd_epi16(m32y, mgrb);

		__m128i me4x = _mm_hadd_epi32(m10x, m32x);
		__m128i me4y = _mm_hadd_epi32(m10y, m32y);

		__m128i me1x = HorizontalMinimum4(me4x);
		__m128i me1y = HorizontalMinimum4(me4y);

		sum = _mm_add_epi32(sum, me1x);
		sum = _mm_add_epi32(sum, me1y);

		i += 8;

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	if (k & 1)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&area.Data[i]);
		mx = _mm_packus_epi32(mx, mx);

		__m128i m10 = _mm_sub_epi16(mt10, mx);
		__m128i m32 = _mm_sub_epi16(mt32, mx);

		m10 = _mm_mullo_epi16(m10, m10);
		m32 = _mm_mullo_epi16(m32, m32);

		m10 = _mm_min_epu16(m10, mlimit);
		m32 = _mm_min_epu16(m32, mlimit);

		m10 = _mm_madd_epi16(m10, mgrb);
		m32 = _mm_madd_epi16(m32, mgrb);

		__m128i me4 = _mm_hadd_epi32(m10, m32);
		__m128i me1 = HorizontalMinimum4(me4);

		sum = _mm_add_epi32(sum, me1);

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	static_assert((kGreen >= kBlue) && (kRed >= kBlue), "Error");
	int int_sum = _mm_cvtsi128_si32(sum);
	if (int_sum < 0x7FFF * kBlue)
	{
		return int_sum;
	}

	sum = _mm_setzero_si128();

	mgrb = _mm_cvtepi16_epi32(mgrb);

	__m128i mt0 = _mm_cvtepi16_epi32(mt10);
	__m128i mt1 = _mm_unpackhi_epi16(mt10, sum);
	__m128i mt2 = _mm_cvtepi16_epi32(mt32);
	__m128i mt3 = _mm_unpackhi_epi16(mt32, sum);

	for (k = area.Count, i = 0; --k >= 0; i += 4)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&area.Data[i]);

		__m128i m0 = _mm_sub_epi16(mt0, mx);
		__m128i m1 = _mm_sub_epi16(mt1, mx);
		__m128i m2 = _mm_sub_epi16(mt2, mx);
		__m128i m3 = _mm_sub_epi16(mt3, mx);

		m0 = _mm_mullo_epi16(m0, m0);
		m1 = _mm_mullo_epi16(m1, m1);
		m2 = _mm_mullo_epi16(m2, m2);
		m3 = _mm_mullo_epi16(m3, m3);

		m0 = _mm_mullo_epi32(m0, mgrb);
		m1 = _mm_mullo_epi32(m1, mgrb);
		m2 = _mm_mullo_epi32(m2, mgrb);
		m3 = _mm_mullo_epi32(m3, mgrb);

		__m128i m10 = _mm_hadd_epi32(m0, m1);
		__m128i m32 = _mm_hadd_epi32(m2, m3);
		__m128i me4 = _mm_hadd_epi32(m10, m32);
		__m128i me1 = HorizontalMinimum4(me4);

		sum = _mm_add_epi32(sum, me1);
	}

	return (int)_mm_cvtsi128_si64(sum);
}

static INLINED int ComputeErrorFourGR(const Area& area, __m128i mc0, __m128i mc1, const __m128i& mc2, const __m128i& mc3, int water)
{
	__m128i best = _mm_cvtsi64_si128((size_t)(uint32_t)water);

	__m128i mt10 = _mm_unpacklo_epi64(mc0, mc1);
	__m128i mt32 = _mm_unpacklo_epi64(mc2, mc3);

	__m128i mt3210 = _mm_packus_epi32(mt10, mt32);

	__m128i mgr = _mm_load_si128((const __m128i*)g_GR_I16);

	__m128i sum = _mm_setzero_si128();

	__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

	int k = area.Count, i = 0;

	while ((k -= 4) >= 0)
	{
		__m128i mx = _mm_loadl_epi64((const __m128i*)&area.Data[i]);
		__m128i my = _mm_loadl_epi64((const __m128i*)&area.Data[i + 4]);
		__m128i mz = _mm_loadl_epi64((const __m128i*)&area.Data[i + 8]);
		__m128i mw = _mm_loadl_epi64((const __m128i*)&area.Data[i + 12]);

		mx = _mm_unpacklo_epi64(mx, mx);
		my = _mm_unpacklo_epi64(my, my);
		mz = _mm_unpacklo_epi64(mz, mz);
		mw = _mm_unpacklo_epi64(mw, mw);

		mx = _mm_packus_epi32(mx, mx);
		my = _mm_packus_epi32(my, my);
		mz = _mm_packus_epi32(mz, mz);
		mw = _mm_packus_epi32(mw, mw);

		__m128i m3210x = _mm_sub_epi16(mt3210, mx);
		__m128i m3210y = _mm_sub_epi16(mt3210, my);
		__m128i m3210z = _mm_sub_epi16(mt3210, mz);
		__m128i m3210w = _mm_sub_epi16(mt3210, mw);

		m3210x = _mm_mullo_epi16(m3210x, m3210x);
		m3210y = _mm_mullo_epi16(m3210y, m3210y);
		m3210z = _mm_mullo_epi16(m3210z, m3210z);
		m3210w = _mm_mullo_epi16(m3210w, m3210w);

		m3210x = _mm_min_epu16(m3210x, mlimit);
		m3210y = _mm_min_epu16(m3210y, mlimit);
		m3210z = _mm_min_epu16(m3210z, mlimit);
		m3210w = _mm_min_epu16(m3210w, mlimit);

		m3210x = _mm_madd_epi16(m3210x, mgr);
		m3210y = _mm_madd_epi16(m3210y, mgr);
		m3210z = _mm_madd_epi16(m3210z, mgr);
		m3210w = _mm_madd_epi16(m3210w, mgr);

		__m128i me1x = HorizontalMinimum4(m3210x);
		__m128i me1y = HorizontalMinimum4(m3210y);
		__m128i me1z = HorizontalMinimum4(m3210z);
		__m128i me1w = HorizontalMinimum4(m3210w);

		sum = _mm_add_epi32(sum, me1x);
		sum = _mm_add_epi32(sum, me1y);
		sum = _mm_add_epi32(sum, me1z);
		sum = _mm_add_epi32(sum, me1w);

		i += 16;

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	if (k & 2)
	{
		__m128i mx = _mm_loadl_epi64((const __m128i*)&area.Data[i]);
		__m128i my = _mm_loadl_epi64((const __m128i*)&area.Data[i + 4]);

		mx = _mm_unpacklo_epi64(mx, mx);
		my = _mm_unpacklo_epi64(my, my);

		mx = _mm_packus_epi32(mx, mx);
		my = _mm_packus_epi32(my, my);

		__m128i m3210x = _mm_sub_epi16(mt3210, mx);
		__m128i m3210y = _mm_sub_epi16(mt3210, my);

		m3210x = _mm_mullo_epi16(m3210x, m3210x);
		m3210y = _mm_mullo_epi16(m3210y, m3210y);

		m3210x = _mm_min_epu16(m3210x, mlimit);
		m3210y = _mm_min_epu16(m3210y, mlimit);

		m3210x = _mm_madd_epi16(m3210x, mgr);
		m3210y = _mm_madd_epi16(m3210y, mgr);

		__m128i me1x = HorizontalMinimum4(m3210x);
		__m128i me1y = HorizontalMinimum4(m3210y);

		sum = _mm_add_epi32(sum, me1x);
		sum = _mm_add_epi32(sum, me1y);

		i += 8;

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	if (k & 1)
	{
		__m128i mx = _mm_loadl_epi64((const __m128i*)&area.Data[i]);
		mx = _mm_unpacklo_epi64(mx, mx);
		mx = _mm_packus_epi32(mx, mx);

		__m128i m3210 = _mm_sub_epi16(mt3210, mx);
		m3210 = _mm_mullo_epi16(m3210, m3210);
		m3210 = _mm_min_epu16(m3210, mlimit);
		m3210 = _mm_madd_epi16(m3210, mgr);

		__m128i me1 = HorizontalMinimum4(m3210);

		sum = _mm_add_epi32(sum, me1);

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	static_assert(kGreen >= kRed, "Error");
	int int_sum = _mm_cvtsi128_si32(sum);
	if (int_sum < 0x7FFF * kRed)
	{
		return int_sum;
	}

	sum = _mm_setzero_si128();

	mgr = _mm_cvtepi16_epi32(mgr);

	mt10 = _mm_cvtepi16_epi32(mt3210);
	mt32 = _mm_unpackhi_epi16(mt3210, sum);

	for (k = area.Count, i = 0; --k >= 0; i += 4)
	{
		__m128i mx = _mm_loadl_epi64((const __m128i*)&area.Data[i]);
		mx = _mm_unpacklo_epi64(mx, mx);

		__m128i m10 = _mm_sub_epi16(mt10, mx);
		__m128i m32 = _mm_sub_epi16(mt32, mx);

		m10 = _mm_mullo_epi16(m10, m10);
		m32 = _mm_mullo_epi16(m32, m32);

		m10 = _mm_mullo_epi32(m10, mgr);
		m32 = _mm_mullo_epi32(m32, mgr);

		__m128i me4 = _mm_hadd_epi32(m10, m32);
		__m128i me1 = HorizontalMinimum4(me4);

		sum = _mm_add_epi32(sum, me1);
	}

	return (int)_mm_cvtsi128_si64(sum);
}

static INLINED int ComputeErrorFourGB(const Area& area, __m128i mc0, __m128i mc1, const __m128i& mc2, const __m128i& mc3, int water)
{
	__m128i best = _mm_cvtsi64_si128((size_t)(uint32_t)water);

	__m128i mt10 = _mm_unpacklo_epi64(_mm_shuffle_epi32(mc0, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_epi32(mc1, _MM_SHUFFLE(2, 0, 2, 0)));
	__m128i mt32 = _mm_unpacklo_epi64(_mm_shuffle_epi32(mc2, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_epi32(mc3, _MM_SHUFFLE(2, 0, 2, 0)));

	__m128i mt3210 = _mm_packus_epi32(mt10, mt32);

	__m128i mgb = _mm_load_si128((const __m128i*)g_GB_I16);

	__m128i sum = _mm_setzero_si128();

	__m128i mlimit = _mm_shuffle_epi32(_mm_cvtsi32_si128(0x7FFF7FFF), 0);

	int k = area.Count, i = 0;

	while ((k -= 4) >= 0)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&area.Data[i]);
		__m128i my = _mm_load_si128((const __m128i*)&area.Data[i + 4]);
		__m128i mz = _mm_load_si128((const __m128i*)&area.Data[i + 8]);
		__m128i mw = _mm_load_si128((const __m128i*)&area.Data[i + 12]);

		mx = _mm_shuffle_epi32(mx, _MM_SHUFFLE(2, 0, 2, 0));
		my = _mm_shuffle_epi32(my, _MM_SHUFFLE(2, 0, 2, 0));
		mz = _mm_shuffle_epi32(mz, _MM_SHUFFLE(2, 0, 2, 0));
		mw = _mm_shuffle_epi32(mw, _MM_SHUFFLE(2, 0, 2, 0));

		mx = _mm_packus_epi32(mx, mx);
		my = _mm_packus_epi32(my, my);
		mz = _mm_packus_epi32(mz, mz);
		mw = _mm_packus_epi32(mw, mw);

		__m128i m3210x = _mm_sub_epi16(mt3210, mx);
		__m128i m3210y = _mm_sub_epi16(mt3210, my);
		__m128i m3210z = _mm_sub_epi16(mt3210, mz);
		__m128i m3210w = _mm_sub_epi16(mt3210, mw);

		m3210x = _mm_mullo_epi16(m3210x, m3210x);
		m3210y = _mm_mullo_epi16(m3210y, m3210y);
		m3210z = _mm_mullo_epi16(m3210z, m3210z);
		m3210w = _mm_mullo_epi16(m3210w, m3210w);

		m3210x = _mm_min_epu16(m3210x, mlimit);
		m3210y = _mm_min_epu16(m3210y, mlimit);
		m3210z = _mm_min_epu16(m3210z, mlimit);
		m3210w = _mm_min_epu16(m3210w, mlimit);

		m3210x = _mm_madd_epi16(m3210x, mgb);
		m3210y = _mm_madd_epi16(m3210y, mgb);
		m3210z = _mm_madd_epi16(m3210z, mgb);
		m3210w = _mm_madd_epi16(m3210w, mgb);

		__m128i me1x = HorizontalMinimum4(m3210x);
		__m128i me1y = HorizontalMinimum4(m3210y);
		__m128i me1z = HorizontalMinimum4(m3210z);
		__m128i me1w = HorizontalMinimum4(m3210w);

		sum = _mm_add_epi32(sum, me1x);
		sum = _mm_add_epi32(sum, me1y);
		sum = _mm_add_epi32(sum, me1z);
		sum = _mm_add_epi32(sum, me1w);

		i += 16;

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	if (k & 2)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&area.Data[i]);
		__m128i my = _mm_load_si128((const __m128i*)&area.Data[i + 4]);

		mx = _mm_shuffle_epi32(mx, _MM_SHUFFLE(2, 0, 2, 0));
		my = _mm_shuffle_epi32(my, _MM_SHUFFLE(2, 0, 2, 0));

		mx = _mm_packus_epi32(mx, mx);
		my = _mm_packus_epi32(my, my);

		__m128i m3210x = _mm_sub_epi16(mt3210, mx);
		__m128i m3210y = _mm_sub_epi16(mt3210, my);

		m3210x = _mm_mullo_epi16(m3210x, m3210x);
		m3210y = _mm_mullo_epi16(m3210y, m3210y);

		m3210x = _mm_min_epu16(m3210x, mlimit);
		m3210y = _mm_min_epu16(m3210y, mlimit);

		m3210x = _mm_madd_epi16(m3210x, mgb);
		m3210y = _mm_madd_epi16(m3210y, mgb);

		__m128i me1x = HorizontalMinimum4(m3210x);
		__m128i me1y = HorizontalMinimum4(m3210y);

		sum = _mm_add_epi32(sum, me1x);
		sum = _mm_add_epi32(sum, me1y);

		i += 8;

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	if (k & 1)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&area.Data[i]);
		mx = _mm_shuffle_epi32(mx, _MM_SHUFFLE(2, 0, 2, 0));
		mx = _mm_packus_epi32(mx, mx);

		__m128i m3210 = _mm_sub_epi16(mt3210, mx);
		m3210 = _mm_mullo_epi16(m3210, m3210);
		m3210 = _mm_min_epu16(m3210, mlimit);
		m3210 = _mm_madd_epi16(m3210, mgb);

		__m128i me1 = HorizontalMinimum4(m3210);

		sum = _mm_add_epi32(sum, me1);

		if (_mm_movemask_epi8(_mm_cmpgt_epi32(best, sum)) == 0)
			return water;
	}

	static_assert(kGreen >= kBlue, "Error");
	int int_sum = _mm_cvtsi128_si32(sum);
	if (int_sum < 0x7FFF * kBlue)
	{
		return int_sum;
	}

	sum = _mm_setzero_si128();

	mgb = _mm_cvtepi16_epi32(mgb);

	mt10 = _mm_cvtepi16_epi32(mt3210);
	mt32 = _mm_unpackhi_epi16(mt3210, sum);

	for (k = area.Count, i = 0; --k >= 0; i += 4)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&area.Data[i]);
		mx = _mm_shuffle_epi32(mx, _MM_SHUFFLE(2, 0, 2, 0));

		__m128i m10 = _mm_sub_epi16(mt10, mx);
		__m128i m32 = _mm_sub_epi16(mt32, mx);

		m10 = _mm_mullo_epi16(m10, m10);
		m32 = _mm_mullo_epi16(m32, m32);

		m10 = _mm_mullo_epi32(m10, mgb);
		m32 = _mm_mullo_epi32(m32, mgb);

		__m128i me4 = _mm_hadd_epi32(m10, m32);
		__m128i me1 = HorizontalMinimum4(me4);

		sum = _mm_add_epi32(sum, me1);
	}

	return (int)_mm_cvtsi128_si64(sum);
}

static INLINED void ComputeTableColorFour(const Area& area, __m128i mc0, __m128i mc1, const __m128i& mc2, const __m128i& mc3, uint32_t& index)
{
	size_t areaSize = (size_t)(uint32_t)area.Count;
	if (areaSize == 0)
		return;

	int good = 0xF;
	if ((_mm_movemask_epi8(_mm_cmpeq_epi32(mc0, mc1)) | 0xF000) == 0xFFFF) good &= ~2;
	if ((_mm_movemask_epi8(_mm_cmpeq_epi32(mc0, mc2)) | 0xF000) == 0xFFFF) good &= ~4;
	if ((_mm_movemask_epi8(_mm_cmpeq_epi32(mc0, mc3)) | 0xF000) == 0xFFFF) good &= ~8;
	if ((_mm_movemask_epi8(_mm_cmpeq_epi32(mc1, mc2)) | 0xF000) == 0xFFFF) good &= ~4;
	if ((_mm_movemask_epi8(_mm_cmpeq_epi32(mc1, mc3)) | 0xF000) == 0xFFFF) good &= ~8;
	if ((_mm_movemask_epi8(_mm_cmpeq_epi32(mc2, mc3)) | 0xF000) == 0xFFFF) good &= ~8;

	__m128i mgrb = _mm_loadl_epi64((const __m128i*)g_GRB_I16);
	mgrb = _mm_cvtepi16_epi32(mgrb);

	int ways[16];

	for (size_t k = 0, i = 0; k < areaSize; k++, i += 4)
	{
		__m128i mx = _mm_load_si128((const __m128i*)&area.Data[i]);

		__m128i m0 = _mm_sub_epi16(mc0, mx);
		__m128i m1 = _mm_sub_epi16(mc1, mx);
		__m128i m2 = _mm_sub_epi16(mc2, mx);
		__m128i m3 = _mm_sub_epi16(mc3, mx);

		m0 = _mm_mullo_epi16(m0, m0);
		m1 = _mm_mullo_epi16(m1, m1);
		m2 = _mm_mullo_epi16(m2, m2);
		m3 = _mm_mullo_epi16(m3, m3);

		m0 = _mm_mullo_epi32(m0, mgrb);
		m1 = _mm_mullo_epi32(m1, mgrb);
		m2 = _mm_mullo_epi32(m2, mgrb);
		m3 = _mm_mullo_epi32(m3, mgrb);

		__m128i m10 = _mm_hadd_epi32(m0, m1);
		__m128i m32 = _mm_hadd_epi32(m2, m3);
		__m128i me4 = _mm_hadd_epi32(m10, m32);
		__m128i me1 = HorizontalMinimum4(me4);

		int way = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4, me1)));
		ways[k] = (way & good) | (1 << 4);
	}

	__m128i vals[4];
	_mm_store_si128(&vals[0], mc0);
	_mm_store_si128(&vals[1], mc1);
	_mm_store_si128(&vals[2], mc2);
	_mm_store_si128(&vals[3], mc3);

	int loops[16];

	for (size_t i = 0; i < areaSize; i++)
	{
		int k = 0;
		while ((ways[i] & (1 << k)) == 0) k++;
		loops[i] = k;
	}

	double best = -(kColor + 0.1);
	uint32_t codes = 0;

	for (;; )
	{
		SSIM_INIT();

		for (size_t i = 0; i < areaSize; i++)
		{
			__m128i mx = _mm_load_si128(&vals[(size_t)(uint32_t)loops[i]]);

			__m128i mb = _mm_load_si128((const __m128i*)&area.Data[i << 2]);

			SSIM_UPDATE(mx, mb);
		}

		SSIM_CLOSE(4);

		SSIM_FINAL(mssim_rg, g_ssim_16k1L, g_ssim_16k2L);
		SSIM_OTHER();
		SSIM_FINAL(mssim_b, g_ssim_16k1L, g_ssim_16k2L);

		double ssim =
			_mm_cvtsd_f64(mssim_rg) * kGreen +
			_mm_cvtsd_f64(_mm_unpackhi_pd(mssim_rg, mssim_rg)) * kRed +
			_mm_cvtsd_f64(mssim_b) * kBlue;

		if (best < ssim)
		{
			best = ssim;

			uint32_t v = 0;
			for (size_t j = 0; j < areaSize; j++)
			{
				v |= ((uint32_t)loops[j]) << (j + j);
			}
			codes = v;

			if (best >= kColor)
				break;
		}

		size_t i = 0;
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

			i++;
			if (i >= areaSize)
				break;
		}
		if (i >= areaSize)
			break;
	}

	for (size_t j = 0; j < areaSize; j++)
	{
		int shift = area.Shift[j];

		uint32_t code = ((codes & 2u) << (16 - 1)) | (codes & 1u);

		codes >>= 2;

		index |= code << shift;
	}
}

static int CompressBlockColorH(uint8_t output[8], const Area& area, int input_error)
{
	int water = input_error;

	uint8_t best_a[4], best_b[4];
	int best_q = 0;

	alignas(16) int chunks0[0x10], chunks1[0x10], chunks2[0x10];
	alignas(16) Node err0[0x101], err1[0x101], err2[0x101];

	int memGB[0x100];

	for (int q = 0; q < 8; q++)
	{
		const auto stripes = g_stripesH[q];
		const auto errors = g_errorsH[q];

		CombineStripes(area, 0, chunks0, stripes, kGreen);
		CombineLevels(area, 0, err0, errors, chunks0, kGreen, water);
		int min0 = err0[0x100].Error;
		if (min0 >= water)
			continue;

		CombineStripes(area, 1, chunks1, stripes, kRed);
		CombineLevels(area, 1, err1, errors, chunks1, kRed, water - min0);
		int min1 = err1[0x100].Error;
		if (min0 + min1 >= water)
			continue;

		CombineStripes(area, 2, chunks2, stripes, kBlue);
		CombineLevels(area, 2, err2, errors, chunks2, kBlue, water - min0 - min1);
		int min2 = err2[0x100].Error;
		if (min0 + min1 + min2 >= water)
			continue;

		for (int i1 = 0, n1 = err1[0x100].Color; i1 < n1; i1++)
		{
			int c1 = err1[i1].Color;
			if ((c1 >> 4) > (c1 & 0xF)) // if ((a[1] << 16) > (b[1] << 16))
			{
				err1[i1].Error = water;
			}
		}

		int n0 = Sort100(err0, water - min1 - min2);
		int n1 = Sort100(err1, water - min0 - min2);
		int n2 = Sort100(err2, water - min0 - min1);

		int d = g_tableHT[q];

		uint8_t a[4], b[4];

		for (int i0 = 0; i0 < n0; i0++)
		{
			int e0 = err0[i0].Error;
			if (e0 + min1 + min2 >= water)
				break;

			int c0 = err0[i0].Color;
			a[0] = (uint8_t)ExpandColor4(c0 >> 4);
			b[0] = (uint8_t)ExpandColor4(c0 & 0xF);

			memset(memGB, -1, n2 * sizeof(int));

			for (int i1 = 0; i1 < n1; i1++)
			{
				int e1 = err1[i1].Error + e0;
				if (e1 + min2 >= water)
					break;

				int c1 = err1[i1].Color;
				a[1] = (uint8_t)ExpandColor4(c1 >> 4);
				b[1] = (uint8_t)ExpandColor4(c1 & 0xF);

				int compare_a1a0_b1b0 = ((a[1] - b[1]) << 16) + ((a[0] - b[0]) << 8);
				if (compare_a1a0_b1b0 > 0) // if ((a[1] << 16) + (a[0] << 8) > (b[1] << 16) + (b[0] << 8))
					continue;

				{
					__m128i md = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)d), 0);
					__m128i ma = load_color_GR(a);
					__m128i mb = load_color_GR(b);

					__m128i mc0 = _mm_adds_epu8(ma, md);
					__m128i mc1 = _mm_subs_epu8(ma, md);
					__m128i mc2 = _mm_adds_epu8(mb, md);
					__m128i mc3 = _mm_subs_epu8(mb, md);

					e1 = ComputeErrorFourGR(area, mc0, mc1, mc2, mc3, water - min2);
				}
				if (e1 + min2 >= water)
					continue;

				for (int i2 = 0; i2 < n2; i2++)
				{
					int e2 = err2[i2].Error + e1;
					if (e2 >= water)
						break;

					int c2 = err2[i2].Color;
					a[2] = (uint8_t)ExpandColor4(c2 >> 4);
					b[2] = (uint8_t)ExpandColor4(c2 & 0xF);

					if (compare_a1a0_b1b0 >= b[2] - a[2]) // if ((a[1] << 16) + (a[0] << 8) + a[2] >= (b[1] << 16) + (b[0] << 8) + b[2])
						continue;

					int egb = memGB[i2];
					if (egb < 0)
					{
						__m128i md = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)d), 0);
						__m128i ma = load_color_GRB(a);
						__m128i mb = load_color_GRB(b);

						__m128i mc0 = _mm_adds_epu8(ma, md);
						__m128i mc1 = _mm_subs_epu8(ma, md);
						__m128i mc2 = _mm_adds_epu8(mb, md);
						__m128i mc3 = _mm_subs_epu8(mb, md);

						egb = ComputeErrorFourGB(area, mc0, mc1, mc2, mc3, water - err1[i1].Error);
						memGB[i2] = egb;
					}
					if (egb + err1[i1].Error >= water)
						continue;

					int err;
					{
						__m128i md = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)d), 0);
						__m128i ma = load_color_GRB(a);
						__m128i mb = load_color_GRB(b);

						__m128i mc0 = _mm_adds_epu8(ma, md);
						__m128i mc1 = _mm_subs_epu8(ma, md);
						__m128i mc2 = _mm_adds_epu8(mb, md);
						__m128i mc3 = _mm_subs_epu8(mb, md);

						err = ComputeErrorFourGRB(area, mc0, mc1, mc2, mc3, water);
					}

					if (water > err)
					{
						water = err;
						memcpy(best_a, a, sizeof(a));
						memcpy(best_b, b, sizeof(b));
						best_q = q;
					}
				}
			}
		}
	}

	if (water >= input_error)
		return water;

	//

	if (best_q & 1)
	{
		std::swap(best_a, best_b);
	}

	{
		uint32_t d = best_q & 4;
		uint32_t a = (best_a[1] & 0xF) << 27;
		uint32_t b = (best_b[1] & 0xF) << 11;

		d += (best_q & 2) >> 1;
		a += (best_a[0] & 0xE) << 23;
		b += (best_b[0] & 0xF) << 7;

		a += (best_a[0] & 1) << 20;
		b += (best_b[2] & 0xF) << 3;
		a += (best_a[2] & 8) << 16;

		a += (best_a[2] & 7) << 15;

		uint32_t c = (a + b) + (d + 2);

		{
			uint32_t dR = (c >> 24) & 7;
			if (dR & 4)
			{
				uint32_t cR = (c >> 27) & 0xF;
				if (cR + dR < 8)
				{
					c += 1u << 31;
				}
			}
		}

		{
			uint32_t cG = (c >> 19) & 3;
			uint32_t dG = (c >> 16) & 3;

			if (cG + dG >= 4)
			{
				c += 7u << 21;
			}
			else
			{
				c += 1u << 18;
			}
		}

		*(uint32_t*)output = BSWAP(c);

		__m128i md = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)g_tableHT[(size_t)(uint32_t)best_q]), 0);
		__m128i ma = load_color_GRB(best_a);
		__m128i mb = load_color_GRB(best_b);

		__m128i mc0 = _mm_adds_epu8(ma, md);
		__m128i mc1 = _mm_subs_epu8(ma, md);
		__m128i mc2 = _mm_adds_epu8(mb, md);
		__m128i mc3 = _mm_subs_epu8(mb, md);

		uint32_t index = 0;
		ComputeTableColorFour(area, mc0, mc1, mc2, mc3, index);

		*(uint32_t*)(output + 4) = BSWAP(index);
	}

	return water;
}

static int CompressBlockColorT(uint8_t output[8], const Area& area, int input_error)
{
	int water = input_error;

	uint8_t best_a[4], best_b[4];
	int best_q = 0;

	alignas(16) int chunks0[0x10], chunks1[0x10], chunks2[0x10];
	alignas(16) Node err0[0x101], err1[0x101], err2[0x101];

	int memGB[0x100];

	for (int q = 0; q < 8; q++)
	{
		const auto stripes = g_stripesT[q];
		const auto errors = g_errorsT[q];

		CombineStripes(area, 0, chunks0, stripes, kGreen);
		CombineLevels(area, 0, err0, errors, chunks0, kGreen, water);
		int min0 = err0[0x100].Error;
		if (min0 >= water)
			continue;

		CombineStripes(area, 1, chunks1, stripes, kRed);
		CombineLevels(area, 1, err1, errors, chunks1, kRed, water - min0);
		int min1 = err1[0x100].Error;
		if (min0 + min1 >= water)
			continue;

		CombineStripes(area, 2, chunks2, stripes, kBlue);
		CombineLevels(area, 2, err2, errors, chunks2, kBlue, water - min0 - min1);
		int min2 = err2[0x100].Error;
		if (min0 + min1 + min2 >= water)
			continue;

		int n0 = Sort100(err0, water - min1 - min2);
		int n1 = Sort100(err1, water - min0 - min2);
		int n2 = Sort100(err2, water - min0 - min1);

		int d = g_tableHT[q];

		uint8_t a[4], b[4];

		for (int i0 = 0; i0 < n0; i0++)
		{
			int e0 = err0[i0].Error;
			if (e0 + min1 + min2 >= water)
				break;

			int c0 = err0[i0].Color;
			a[0] = (uint8_t)ExpandColor4(c0 >> 4);
			b[0] = (uint8_t)ExpandColor4(c0 & 0xF);

			memset(memGB, -1, n2 * sizeof(int));

			for (int i1 = 0; i1 < n1; i1++)
			{
				int e1 = err1[i1].Error + e0;
				if (e1 + min2 >= water)
					break;

				int c1 = err1[i1].Color;
				a[1] = (uint8_t)ExpandColor4(c1 >> 4);
				b[1] = (uint8_t)ExpandColor4(c1 & 0xF);

				{
					__m128i md = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)d), 0);
					__m128i ma = load_color_GR(a);
					__m128i mb = load_color_GR(b);

					__m128i mc0 = ma;
					__m128i mc1 = _mm_adds_epu8(mb, md);
					__m128i mc2 = mb;
					__m128i mc3 = _mm_subs_epu8(mb, md);

					e1 = ComputeErrorFourGR(area, mc0, mc1, mc2, mc3, water - min2);
				}
				if (e1 + min2 >= water)
					continue;

				for (int i2 = 0; i2 < n2; i2++)
				{
					int e2 = err2[i2].Error + e1;
					if (e2 >= water)
						break;

					int c2 = err2[i2].Color;
					a[2] = (uint8_t)ExpandColor4(c2 >> 4);
					b[2] = (uint8_t)ExpandColor4(c2 & 0xF);

					int egb = memGB[i2];
					if (egb < 0)
					{
						__m128i md = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)d), 0);
						__m128i ma = load_color_GRB(a);
						__m128i mb = load_color_GRB(b);

						__m128i mc0 = ma;
						__m128i mc1 = _mm_adds_epu8(mb, md);
						__m128i mc2 = mb;
						__m128i mc3 = _mm_subs_epu8(mb, md);

						egb = ComputeErrorFourGB(area, mc0, mc1, mc2, mc3, water - err1[i1].Error);
						memGB[i2] = egb;
					}
					if (egb + err1[i1].Error >= water)
						continue;

					int err;
					{
						__m128i md = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)d), 0);
						__m128i ma = load_color_GRB(a);
						__m128i mb = load_color_GRB(b);

						__m128i mc0 = ma;
						__m128i mc1 = _mm_adds_epu8(mb, md);
						__m128i mc2 = mb;
						__m128i mc3 = _mm_subs_epu8(mb, md);

						err = ComputeErrorFourGRB(area, mc0, mc1, mc2, mc3, water);
					}

					if (water > err)
					{
						water = err;
						memcpy(best_a, a, sizeof(a));
						memcpy(best_b, b, sizeof(b));
						best_q = q;
					}
				}
			}
		}
	}

	if (water >= input_error)
		return water;

	//

	{
		uint32_t d = (best_q & 6) << 1;
		uint32_t a = (best_a[1] & 0xC) << 25;
		uint32_t b = (best_b[1] & 0xF) << 12;

		d += best_q & 1;
		a += (best_a[1] & 3) << 24;
		b += (best_b[0] & 0xF) << 8;

		a += (best_a[0] & 0xF) << 20;
		b += (best_b[2] & 0xF) << 4;
		a += (best_a[2] & 0xF) << 16;

		uint32_t c = (a + b) + (d + 2);

		{
			uint32_t cR = (c >> 27) & 3;
			uint32_t dR = (c >> 24) & 3;

			if (cR + dR >= 4)
			{
				c += 7u << 29;
			}
			else
			{
				c += 1u << 26;
			}
		}

		*(uint32_t*)output = BSWAP(c);

		__m128i md = _mm_shuffle_epi32(_mm_cvtsi64_si128((size_t)(uint32_t)g_tableHT[(size_t)(uint32_t)best_q]), 0);
		__m128i ma = load_color_GRB(best_a);
		__m128i mb = load_color_GRB(best_b);

		__m128i mc0 = ma;
		__m128i mc1 = _mm_adds_epu8(mb, md);
		__m128i mc2 = mb;
		__m128i mc3 = _mm_subs_epu8(mb, md);

		uint32_t index = 0;
		ComputeTableColorFour(area, mc0, mc1, mc2, mc3, index);

		*(uint32_t*)(output + 4) = BSWAP(index);
	}

	return water;
}

static INLINED void PlanarCollectO(const Area& area, size_t offset, Surface& surface)
{
	memset(&surface, 0, sizeof(surface));

	size_t w = 0;

	for (size_t i = 0, n = area.Count; i < n; i++)
	{
		int pos = area.Shift[i];

		int x = pos >> 2;
		int y = pos & 3;

		int r = x + y;
		if (r >= 4)
			continue;

		surface.Mask[w] = -1;
		surface.U[w] = (short)r;
		surface.Data[w] = (short)area.Data[i * 4 + offset];
		w++;
	}
}

static INLINED int PlanarPyramidO(const Surface& surface, int c)
{
	__m128i mo = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128((c << 2) + 2), 0), 0);

	__m128i mmask0 = _mm_load_si128((const __m128i*)&surface.Mask[0]);
	__m128i mmask1 = _mm_load_si128((const __m128i*)&surface.Mask[8]);

	__m128i mo0 = _mm_and_si128(mmask0, mo);
	__m128i mo1 = _mm_and_si128(mmask1, mo);

	__m128i mmin = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(-c), 0), 0);
	__m128i mmax = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(255 - c), 0), 0);

	__m128i mr0 = _mm_load_si128((const __m128i*)&surface.U[0]);
	__m128i mr1 = _mm_load_si128((const __m128i*)&surface.U[8]);

	__m128i mmin0 = _mm_mullo_epi16(mmin, mr0);
	__m128i mmin1 = _mm_mullo_epi16(mmin, mr1);

	__m128i mmax0 = _mm_mullo_epi16(mmax, mr0);
	__m128i mmax1 = _mm_mullo_epi16(mmax, mr1);

	mmin0 = _mm_add_epi16(mmin0, mo0);
	mmin1 = _mm_add_epi16(mmin1, mo1);

	mmax0 = _mm_add_epi16(mmax0, mo0);
	mmax1 = _mm_add_epi16(mmax1, mo1);

	mmin0 = _mm_srai_epi16(mmin0, 2);
	mmin1 = _mm_srai_epi16(mmin1, 2);

	mmax0 = _mm_srai_epi16(mmax0, 2);
	mmax1 = _mm_srai_epi16(mmax1, 2);

	__m128i md0 = _mm_load_si128((const __m128i*)&surface.Data[0]);
	__m128i md1 = _mm_load_si128((const __m128i*)&surface.Data[8]);

	__m128i e0 = _mm_or_si128(_mm_subs_epu8(mmin0, md0), _mm_subs_epu8(md0, mmax0));
	__m128i e1 = _mm_or_si128(_mm_subs_epu8(mmin1, md1), _mm_subs_epu8(md1, mmax1));

	e0 = _mm_madd_epi16(e0, e0);
	e1 = _mm_madd_epi16(e1, e1);

	__m128i e = _mm_add_epi32(e0, e1);

	e = HorizontalSum4(e);

	return _mm_cvtsi128_si32(e);
}

static INLINED void PlanarCollect(const Area& area, size_t offset, Surface& surface)
{
	memset(&surface, 0, sizeof(surface));

	for (size_t i = 0, n = area.Count; i < n; i++)
	{
		int pos = area.Shift[i];

		int x = pos >> 2;
		int y = pos & 3;

		surface.Mask[i] = -1;
		surface.U[i] = (short)x;
		surface.V[i] = (short)y;
		surface.Data[i] = (short)area.Data[i * 4 + offset];
	}
}

static INLINED int PlanarStripeOH(const Surface& surface, int co, int chL, int chH)
{
	__m128i mo = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128((co << 2) + 2), 0), 0);

	__m128i mmask0 = _mm_load_si128((const __m128i*)&surface.Mask[0]);
	__m128i mmask1 = _mm_load_si128((const __m128i*)&surface.Mask[8]);

	__m128i mo0 = _mm_and_si128(mmask0, mo);
	__m128i mo1 = _mm_and_si128(mmask1, mo);

	__m128i mmin = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(-co), 0), 0);
	__m128i mmax = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(255 - co), 0), 0);

	__m128i my0 = _mm_load_si128((const __m128i*)&surface.V[0]);
	__m128i my1 = _mm_load_si128((const __m128i*)&surface.V[8]);

	__m128i mmin0 = _mm_mullo_epi16(mmin, my0);
	__m128i mmin1 = _mm_mullo_epi16(mmin, my1);

	__m128i mmax0 = _mm_mullo_epi16(mmax, my0);
	__m128i mmax1 = _mm_mullo_epi16(mmax, my1);

	mmin0 = _mm_add_epi16(mmin0, mo0);
	mmin1 = _mm_add_epi16(mmin1, mo1);

	mmax0 = _mm_add_epi16(mmax0, mo0);
	mmax1 = _mm_add_epi16(mmax1, mo1);

	//

	__m128i mhL = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(chL - co), 0), 0);
	__m128i mhH = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(chH - co), 0), 0);

	__m128i mx0 = _mm_load_si128((const __m128i*)&surface.U[0]);
	__m128i mx1 = _mm_load_si128((const __m128i*)&surface.U[8]);

	__m128i mt0L = _mm_mullo_epi16(mhL, mx0);
	__m128i mt1L = _mm_mullo_epi16(mhL, mx1);

	mmin0 = _mm_add_epi16(mmin0, mt0L);
	mmin1 = _mm_add_epi16(mmin1, mt1L);

	__m128i mt0H = _mm_mullo_epi16(mhH, mx0);
	__m128i mt1H = _mm_mullo_epi16(mhH, mx1);

	mmax0 = _mm_add_epi16(mmax0, mt0H);
	mmax1 = _mm_add_epi16(mmax1, mt1H);

	mmin0 = _mm_srai_epi16(mmin0, 2);
	mmin1 = _mm_srai_epi16(mmin1, 2);

	mmax0 = _mm_srai_epi16(mmax0, 2);
	mmax1 = _mm_srai_epi16(mmax1, 2);

	mmin0 = _mm_packus_epi16(mmin0, mmin1);
	mmax0 = _mm_packus_epi16(mmax0, mmax1);

	__m128i mzero = _mm_setzero_si128();

	mmin1 = _mm_unpackhi_epi8(mmin0, mzero);
	mmax1 = _mm_unpackhi_epi8(mmax0, mzero);
	mmin0 = _mm_unpacklo_epi8(mmin0, mzero);
	mmax0 = _mm_unpacklo_epi8(mmax0, mzero);

	__m128i md0 = _mm_load_si128((const __m128i*)&surface.Data[0]);
	__m128i md1 = _mm_load_si128((const __m128i*)&surface.Data[8]);

	__m128i e0 = _mm_or_si128(_mm_subs_epu8(mmin0, md0), _mm_subs_epu8(md0, mmax0));
	__m128i e1 = _mm_or_si128(_mm_subs_epu8(mmin1, md1), _mm_subs_epu8(md1, mmax1));

	e0 = _mm_madd_epi16(e0, e0);
	e1 = _mm_madd_epi16(e1, e1);

	__m128i e = _mm_add_epi32(e0, e1);

	e = HorizontalSum4(e);

	return _mm_cvtsi128_si32(e);
}

static INLINED int PlanarPyramidOH(const Surface& surface, int co, int ch)
{
	__m128i mo = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128((co << 2) + 2), 0), 0);

	__m128i mmask0 = _mm_load_si128((const __m128i*)&surface.Mask[0]);
	__m128i mmask1 = _mm_load_si128((const __m128i*)&surface.Mask[8]);

	__m128i mo0 = _mm_and_si128(mmask0, mo);
	__m128i mo1 = _mm_and_si128(mmask1, mo);

	__m128i mmin = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(-co), 0), 0);
	__m128i mmax = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(255 - co), 0), 0);

	__m128i my0 = _mm_load_si128((const __m128i*)&surface.V[0]);
	__m128i my1 = _mm_load_si128((const __m128i*)&surface.V[8]);

	__m128i mmin0 = _mm_mullo_epi16(mmin, my0);
	__m128i mmin1 = _mm_mullo_epi16(mmin, my1);

	__m128i mmax0 = _mm_mullo_epi16(mmax, my0);
	__m128i mmax1 = _mm_mullo_epi16(mmax, my1);

	mmin0 = _mm_add_epi16(mmin0, mo0);
	mmin1 = _mm_add_epi16(mmin1, mo1);

	mmax0 = _mm_add_epi16(mmax0, mo0);
	mmax1 = _mm_add_epi16(mmax1, mo1);

	//

	__m128i mh = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(ch - co), 0), 0);

	__m128i mx0 = _mm_load_si128((const __m128i*)&surface.U[0]);
	__m128i mx1 = _mm_load_si128((const __m128i*)&surface.U[8]);

	__m128i mt0 = _mm_mullo_epi16(mh, mx0);
	__m128i mt1 = _mm_mullo_epi16(mh, mx1);

	mmin0 = _mm_add_epi16(mmin0, mt0);
	mmin1 = _mm_add_epi16(mmin1, mt1);

	mmax0 = _mm_add_epi16(mmax0, mt0);
	mmax1 = _mm_add_epi16(mmax1, mt1);

	mmin0 = _mm_srai_epi16(mmin0, 2);
	mmin1 = _mm_srai_epi16(mmin1, 2);

	mmax0 = _mm_srai_epi16(mmax0, 2);
	mmax1 = _mm_srai_epi16(mmax1, 2);

	mmin0 = _mm_packus_epi16(mmin0, mmin1);
	mmax0 = _mm_packus_epi16(mmax0, mmax1);

	__m128i mzero = _mm_setzero_si128();

	mmin1 = _mm_unpackhi_epi8(mmin0, mzero);
	mmax1 = _mm_unpackhi_epi8(mmax0, mzero);
	mmin0 = _mm_unpacklo_epi8(mmin0, mzero);
	mmax0 = _mm_unpacklo_epi8(mmax0, mzero);

	__m128i md0 = _mm_load_si128((const __m128i*)&surface.Data[0]);
	__m128i md1 = _mm_load_si128((const __m128i*)&surface.Data[8]);

	__m128i e0 = _mm_or_si128(_mm_subs_epu8(mmin0, md0), _mm_subs_epu8(md0, mmax0));
	__m128i e1 = _mm_or_si128(_mm_subs_epu8(mmin1, md1), _mm_subs_epu8(md1, mmax1));

	e0 = _mm_madd_epi16(e0, e0);
	e1 = _mm_madd_epi16(e1, e1);

	__m128i e = _mm_add_epi32(e0, e1);

	e = HorizontalSum4(e);

	return _mm_cvtsi128_si32(e);
}

static INLINED int PlanarStripeOV(const Surface& surface, int co, int cvL, int cvH)
{
	__m128i mo = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128((co << 2) + 2), 0), 0);

	__m128i mmask0 = _mm_load_si128((const __m128i*)&surface.Mask[0]);
	__m128i mmask1 = _mm_load_si128((const __m128i*)&surface.Mask[8]);

	__m128i mo0 = _mm_and_si128(mmask0, mo);
	__m128i mo1 = _mm_and_si128(mmask1, mo);

	__m128i mmin = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(-co), 0), 0);
	__m128i mmax = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(255 - co), 0), 0);

	__m128i mx0 = _mm_load_si128((const __m128i*)&surface.U[0]);
	__m128i mx1 = _mm_load_si128((const __m128i*)&surface.U[8]);

	__m128i mmin0 = _mm_mullo_epi16(mmin, mx0);
	__m128i mmin1 = _mm_mullo_epi16(mmin, mx1);

	__m128i mmax0 = _mm_mullo_epi16(mmax, mx0);
	__m128i mmax1 = _mm_mullo_epi16(mmax, mx1);

	mmin0 = _mm_add_epi16(mmin0, mo0);
	mmin1 = _mm_add_epi16(mmin1, mo1);

	mmax0 = _mm_add_epi16(mmax0, mo0);
	mmax1 = _mm_add_epi16(mmax1, mo1);

	//

	__m128i mvL = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(cvL - co), 0), 0);
	__m128i mvH = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(cvH - co), 0), 0);

	__m128i my0 = _mm_load_si128((const __m128i*)&surface.V[0]);
	__m128i my1 = _mm_load_si128((const __m128i*)&surface.V[8]);

	__m128i mt0L = _mm_mullo_epi16(mvL, my0);
	__m128i mt1L = _mm_mullo_epi16(mvL, my1);

	mmin0 = _mm_add_epi16(mmin0, mt0L);
	mmin1 = _mm_add_epi16(mmin1, mt1L);

	__m128i mt0H = _mm_mullo_epi16(mvH, my0);
	__m128i mt1H = _mm_mullo_epi16(mvH, my1);

	mmax0 = _mm_add_epi16(mmax0, mt0H);
	mmax1 = _mm_add_epi16(mmax1, mt1H);

	mmin0 = _mm_srai_epi16(mmin0, 2);
	mmin1 = _mm_srai_epi16(mmin1, 2);

	mmax0 = _mm_srai_epi16(mmax0, 2);
	mmax1 = _mm_srai_epi16(mmax1, 2);

	mmin0 = _mm_packus_epi16(mmin0, mmin1);
	mmax0 = _mm_packus_epi16(mmax0, mmax1);

	__m128i mzero = _mm_setzero_si128();

	mmin1 = _mm_unpackhi_epi8(mmin0, mzero);
	mmax1 = _mm_unpackhi_epi8(mmax0, mzero);
	mmin0 = _mm_unpacklo_epi8(mmin0, mzero);
	mmax0 = _mm_unpacklo_epi8(mmax0, mzero);

	__m128i md0 = _mm_load_si128((const __m128i*)&surface.Data[0]);
	__m128i md1 = _mm_load_si128((const __m128i*)&surface.Data[8]);

	__m128i e0 = _mm_or_si128(_mm_subs_epu8(mmin0, md0), _mm_subs_epu8(md0, mmax0));
	__m128i e1 = _mm_or_si128(_mm_subs_epu8(mmin1, md1), _mm_subs_epu8(md1, mmax1));

	e0 = _mm_madd_epi16(e0, e0);
	e1 = _mm_madd_epi16(e1, e1);

	__m128i e = _mm_add_epi32(e0, e1);

	e = HorizontalSum4(e);

	return _mm_cvtsi128_si32(e);
}

static INLINED int PlanarPyramidOV(const Surface& surface, int co, int cv)
{
	__m128i mo = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128((co << 2) + 2), 0), 0);

	__m128i mmask0 = _mm_load_si128((const __m128i*)&surface.Mask[0]);
	__m128i mmask1 = _mm_load_si128((const __m128i*)&surface.Mask[8]);

	__m128i mo0 = _mm_and_si128(mmask0, mo);
	__m128i mo1 = _mm_and_si128(mmask1, mo);

	__m128i mmin = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(-co), 0), 0);
	__m128i mmax = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(255 - co), 0), 0);

	__m128i mx0 = _mm_load_si128((const __m128i*)&surface.U[0]);
	__m128i mx1 = _mm_load_si128((const __m128i*)&surface.U[8]);

	__m128i mmin0 = _mm_mullo_epi16(mmin, mx0);
	__m128i mmin1 = _mm_mullo_epi16(mmin, mx1);

	__m128i mmax0 = _mm_mullo_epi16(mmax, mx0);
	__m128i mmax1 = _mm_mullo_epi16(mmax, mx1);

	mmin0 = _mm_add_epi16(mmin0, mo0);
	mmin1 = _mm_add_epi16(mmin1, mo1);

	mmax0 = _mm_add_epi16(mmax0, mo0);
	mmax1 = _mm_add_epi16(mmax1, mo1);

	//

	__m128i mv = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(cv - co), 0), 0);

	__m128i my0 = _mm_load_si128((const __m128i*)&surface.V[0]);
	__m128i my1 = _mm_load_si128((const __m128i*)&surface.V[8]);

	__m128i mt0 = _mm_mullo_epi16(mv, my0);
	__m128i mt1 = _mm_mullo_epi16(mv, my1);

	mmin0 = _mm_add_epi16(mmin0, mt0);
	mmin1 = _mm_add_epi16(mmin1, mt1);

	mmax0 = _mm_add_epi16(mmax0, mt0);
	mmax1 = _mm_add_epi16(mmax1, mt1);

	mmin0 = _mm_srai_epi16(mmin0, 2);
	mmin1 = _mm_srai_epi16(mmin1, 2);

	mmax0 = _mm_srai_epi16(mmax0, 2);
	mmax1 = _mm_srai_epi16(mmax1, 2);

	mmin0 = _mm_packus_epi16(mmin0, mmin1);
	mmax0 = _mm_packus_epi16(mmax0, mmax1);

	__m128i mzero = _mm_setzero_si128();

	mmin1 = _mm_unpackhi_epi8(mmin0, mzero);
	mmax1 = _mm_unpackhi_epi8(mmax0, mzero);
	mmin0 = _mm_unpacklo_epi8(mmin0, mzero);
	mmax0 = _mm_unpacklo_epi8(mmax0, mzero);

	__m128i md0 = _mm_load_si128((const __m128i*)&surface.Data[0]);
	__m128i md1 = _mm_load_si128((const __m128i*)&surface.Data[8]);

	__m128i e0 = _mm_or_si128(_mm_subs_epu8(mmin0, md0), _mm_subs_epu8(md0, mmax0));
	__m128i e1 = _mm_or_si128(_mm_subs_epu8(mmin1, md1), _mm_subs_epu8(md1, mmax1));

	e0 = _mm_madd_epi16(e0, e0);
	e1 = _mm_madd_epi16(e1, e1);

	__m128i e = _mm_add_epi32(e0, e1);

	e = HorizontalSum4(e);

	return _mm_cvtsi128_si32(e);
}

static INLINED int PlanarStripe(const Surface& surface, int co, int chL, int chH, int cvL, int cvH)
{
	__m128i mo = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128((co << 2) + 2), 0), 0);

	__m128i mmask0 = _mm_load_si128((const __m128i*)&surface.Mask[0]);
	__m128i mmask1 = _mm_load_si128((const __m128i*)&surface.Mask[8]);

	__m128i mo0 = _mm_and_si128(mmask0, mo);
	__m128i mo1 = _mm_and_si128(mmask1, mo);

	__m128i mmin = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(chL - co), 0), 0);
	__m128i mmax = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(chH - co), 0), 0);

	__m128i mx0 = _mm_load_si128((const __m128i*)&surface.U[0]);
	__m128i mx1 = _mm_load_si128((const __m128i*)&surface.U[8]);

	__m128i mmin0 = _mm_mullo_epi16(mmin, mx0);
	__m128i mmin1 = _mm_mullo_epi16(mmin, mx1);

	__m128i mmax0 = _mm_mullo_epi16(mmax, mx0);
	__m128i mmax1 = _mm_mullo_epi16(mmax, mx1);

	mmin0 = _mm_add_epi16(mmin0, mo0);
	mmin1 = _mm_add_epi16(mmin1, mo1);

	mmax0 = _mm_add_epi16(mmax0, mo0);
	mmax1 = _mm_add_epi16(mmax1, mo1);

	//

	__m128i mvL = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(cvL - co), 0), 0);
	__m128i mvH = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(cvH - co), 0), 0);

	__m128i my0 = _mm_load_si128((const __m128i*)&surface.V[0]);
	__m128i my1 = _mm_load_si128((const __m128i*)&surface.V[8]);

	__m128i mt0L = _mm_mullo_epi16(mvL, my0);
	__m128i mt1L = _mm_mullo_epi16(mvL, my1);

	mmin0 = _mm_add_epi16(mmin0, mt0L);
	mmin1 = _mm_add_epi16(mmin1, mt1L);

	__m128i mt0H = _mm_mullo_epi16(mvH, my0);
	__m128i mt1H = _mm_mullo_epi16(mvH, my1);

	mmax0 = _mm_add_epi16(mmax0, mt0H);
	mmax1 = _mm_add_epi16(mmax1, mt1H);

	mmin0 = _mm_srai_epi16(mmin0, 2);
	mmin1 = _mm_srai_epi16(mmin1, 2);

	mmax0 = _mm_srai_epi16(mmax0, 2);
	mmax1 = _mm_srai_epi16(mmax1, 2);

	mmin0 = _mm_packus_epi16(mmin0, mmin1);
	mmax0 = _mm_packus_epi16(mmax0, mmax1);

	__m128i mzero = _mm_setzero_si128();

	mmin1 = _mm_unpackhi_epi8(mmin0, mzero);
	mmax1 = _mm_unpackhi_epi8(mmax0, mzero);
	mmin0 = _mm_unpacklo_epi8(mmin0, mzero);
	mmax0 = _mm_unpacklo_epi8(mmax0, mzero);

	__m128i md0 = _mm_load_si128((const __m128i*)&surface.Data[0]);
	__m128i md1 = _mm_load_si128((const __m128i*)&surface.Data[8]);

	__m128i e0 = _mm_or_si128(_mm_subs_epu8(mmin0, md0), _mm_subs_epu8(md0, mmax0));
	__m128i e1 = _mm_or_si128(_mm_subs_epu8(mmin1, md1), _mm_subs_epu8(md1, mmax1));

	e0 = _mm_madd_epi16(e0, e0);
	e1 = _mm_madd_epi16(e1, e1);

	__m128i e = _mm_add_epi32(e0, e1);

	e = HorizontalSum4(e);

	return _mm_cvtsi128_si32(e);
}

static INLINED int PlanarPyramid(const Surface& surface, int co, int ch, int cv)
{
	__m128i mo = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128((co << 2) + 2), 0), 0);

	__m128i mmask0 = _mm_load_si128((const __m128i*)&surface.Mask[0]);
	__m128i mmask1 = _mm_load_si128((const __m128i*)&surface.Mask[8]);

	__m128i mo0 = _mm_and_si128(mmask0, mo);
	__m128i mo1 = _mm_and_si128(mmask1, mo);

	__m128i mh = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(ch - co), 0), 0);

	__m128i mx0 = _mm_load_si128((const __m128i*)&surface.U[0]);
	__m128i mx1 = _mm_load_si128((const __m128i*)&surface.U[8]);

	__m128i mth0 = _mm_mullo_epi16(mx0, mh);
	__m128i mth1 = _mm_mullo_epi16(mx1, mh);

	mth0 = _mm_add_epi16(mth0, mo0);
	mth1 = _mm_add_epi16(mth1, mo1);

	//

	__m128i mv = _mm_shuffle_epi32(_mm_shufflelo_epi16(_mm_cvtsi32_si128(cv - co), 0), 0);

	__m128i my0 = _mm_load_si128((const __m128i*)&surface.V[0]);
	__m128i my1 = _mm_load_si128((const __m128i*)&surface.V[8]);

	__m128i mt0 = _mm_mullo_epi16(my0, mv);
	__m128i mt1 = _mm_mullo_epi16(my1, mv);

	mt0 = _mm_add_epi16(mt0, mth0);
	mt1 = _mm_add_epi16(mt1, mth1);

	mt0 = _mm_srai_epi16(mt0, 2);
	mt1 = _mm_srai_epi16(mt1, 2);

	mt0 = _mm_packus_epi16(mt0, mt1);

	__m128i mzero = _mm_setzero_si128();

	mt1 = _mm_unpackhi_epi8(mt0, mzero);
	mt0 = _mm_unpacklo_epi8(mt0, mzero);

	__m128i md0 = _mm_load_si128((const __m128i*)&surface.Data[0]);
	__m128i md1 = _mm_load_si128((const __m128i*)&surface.Data[8]);

	__m128i e0 = _mm_sub_epi16(mt0, md0);
	__m128i e1 = _mm_sub_epi16(mt1, md1);

	e0 = _mm_madd_epi16(e0, e0);
	e1 = _mm_madd_epi16(e1, e1);

	__m128i e = _mm_add_epi32(e0, e1);

	e = HorizontalSum4(e);

	return _mm_cvtsi128_si32(e);
}

static INLINED int Planar7(const Area& area, size_t offset, int c[3], int weight, int water)
{
	int top = (water + weight - 1) / weight;

	Surface surface;

	alignas(16) Node nodesO[0x81], nodesH[0x81], nodesV[0x81];

	alignas(16) int chunks[0x10 * 0x10];

	PlanarCollectO(area, offset, surface);

	{
		size_t w = 0;

		for (int i = 0; i < 0x80; i++)
		{
			int co = (i << 1) + (i >> 6);

			int err = PlanarPyramidO(surface, co);
			if (err < top)
			{
				nodesO[w].Error = err;
				nodesO[w].Color = co;
				w++;
			}
		}

		if (!w)
			return water;

		nodesO[0x80].Color = (int)w;
	}

	PlanarCollect(area, offset, surface);

	int best = top;

	for (int io = 0, no = nodesO[0x80].Color; io < no; io++)
	{
		int eo = nodesO[io].Error;
		if (eo >= best)
			continue;

		int co = nodesO[io].Color;

		{
			size_t w = 0;

			for (int i = 0; i < 0x80; i += 0x20)
			{
				int ch = (i << 1) + (i >> 6);

				int stripe32 = PlanarStripeOH(surface, co, ch, ch + (31 << 1));
				if (stripe32 < best)
				{
					for (int j = 8; --j >= 0;)
					{
						int stripe4 = PlanarStripeOH(surface, co, ch, ch + (3 << 1));
						if (stripe4 < best)
						{
							for (int k = 4; --k >= 0; ch += (1 << 1))
							{
								int err = PlanarPyramidOH(surface, co, ch);
								if (err < best)
								{
									nodesH[w].Error = err;
									nodesH[w].Color = ch;
									w++;
								}
							}
						}
						else
						{
							ch += (4 << 1);
						}
					}
				}
			}

			if (!w)
				continue;

			nodesH[0x80].Color = (int)w;
		}

		{
			size_t w = 0;

			for (int i = 0; i < 0x80; i += 0x20)
			{
				int cv = (i << 1) + (i >> 6);

				int stripe32 = PlanarStripeOV(surface, co, cv, cv + (31 << 1));
				if (stripe32 < best)
				{
					for (int j = 8; --j >= 0;)
					{
						int stripe4 = PlanarStripeOV(surface, co, cv, cv + (3 << 1));
						if (stripe4 < best)
						{
							for (int k = 4; --k >= 0; cv += (1 << 1))
							{
								int err = PlanarPyramidOV(surface, co, cv);
								if (err < best)
								{
									nodesV[w].Error = err;
									nodesV[w].Color = cv;
									w++;
								}
							}
						}
						else
						{
							cv += (4 << 1);
						}
					}
				}
			}

			if (!w)
				continue;

			nodesV[0x80].Color = (int)w;
		}

		memset(chunks, -1, sizeof(chunks));

		for (int ih = 0, nh = nodesH[0x80].Color; ih < nh; ih++)
		{
			int eh = nodesH[ih].Error;
			if (eh >= best)
				continue;

			int ch = nodesH[ih].Color;

			for (int iv = 0, nv = nodesV[0x80].Color; iv < nv; iv++)
			{
				int ev = nodesV[iv].Error;
				if (ev >= best)
					continue;

				int cv = nodesV[iv].Color;

				{
					size_t index = (uint32_t)(ch & 0xF0) + (uint32_t)(cv >> 4);

					int estimate = chunks[index];
					if (estimate < 0)
					{
						int ch0 = ch & 0xF1;
						int cv0 = cv & 0xF1;

						estimate = PlanarStripe(surface, co, ch0, ch0 + (7 << 1), cv0, cv0 + (7 << 1));
						chunks[index] = estimate;
					}
					if (estimate >= best)
						continue;
				}

				int sum = PlanarPyramid(surface, co, ch, cv);
				if (best > sum)
				{
					best = sum;

					c[0] = co;
					c[1] = ch;
					c[2] = cv;
				}
			}
		}
	}

	return best * weight;
}

static INLINED int Planar6(const Area& area, size_t offset, int c[3], int weight, int water)
{
	int top = (water + weight - 1) / weight;

	Surface surface;

	alignas(16) Node nodesO[0x41], nodesH[0x41], nodesV[0x41];

	alignas(16) int chunks[0x10 * 0x10];

	PlanarCollectO(area, offset, surface);

	{
		size_t w = 0;

		for (int i = 0; i < 0x40; i++)
		{
			int co = (i << 2) + (i >> 4);

			int err = PlanarPyramidO(surface, co);
			if (err < top)
			{
				nodesO[w].Error = err;
				nodesO[w].Color = co;
				w++;
			}
		}

		if (!w)
			return water;

		nodesO[0x40].Color = (int)w;
	}

	PlanarCollect(area, offset, surface);

	int best = top;

	for (int io = 0, no = nodesO[0x40].Color; io < no; io++)
	{
		int eo = nodesO[io].Error;
		if (eo >= best)
			continue;

		int co = nodesO[io].Color;

		{
			size_t w = 0;

			for (int i = 0; i < 0x40; i += 0x10)
			{
				int ch = (i << 2) + (i >> 4);

				int stripe16 = PlanarStripeOH(surface, co, ch, ch + (15 << 2));
				if (stripe16 < best)
				{
					for (int j = 4; --j >= 0;)
					{
						int stripe4 = PlanarStripeOH(surface, co, ch, ch + (3 << 2));
						if (stripe4 < best)
						{
							for (int k = 4; --k >= 0; ch += (1 << 2))
							{
								int err = PlanarPyramidOH(surface, co, ch);
								if (err < best)
								{
									nodesH[w].Error = err;
									nodesH[w].Color = ch;
									w++;
								}
							}
						}
						else
						{
							ch += (4 << 2);
						}
					}
				}
			}

			if (!w)
				continue;

			nodesH[0x40].Color = (int)w;
		}

		{
			size_t w = 0;

			for (int i = 0; i < 0x40; i += 0x10)
			{
				int cv = (i << 2) + (i >> 4);

				int stripe16 = PlanarStripeOV(surface, co, cv, cv + (15 << 2));
				if (stripe16 < best)
				{
					for (int j = 4; --j >= 0;)
					{
						int stripe4 = PlanarStripeOV(surface, co, cv, cv + (3 << 2));
						if (stripe4 < best)
						{
							for (int k = 4; --k >= 0; cv += (1 << 2))
							{
								int err = PlanarPyramidOV(surface, co, cv);
								if (err < best)
								{
									nodesV[w].Error = err;
									nodesV[w].Color = cv;
									w++;
								}
							}
						}
						else
						{
							cv += (4 << 2);
						}
					}
				}
			}

			if (!w)
				continue;

			nodesV[0x40].Color = (int)w;
		}

		memset(chunks, -1, sizeof(chunks));

		for (int ih = 0, nh = nodesH[0x40].Color; ih < nh; ih++)
		{
			int eh = nodesH[ih].Error;
			if (eh >= best)
				continue;

			int ch = nodesH[ih].Color;

			for (int iv = 0, nv = nodesV[0x40].Color; iv < nv; iv++)
			{
				int ev = nodesV[iv].Error;
				if (ev >= best)
					continue;

				int cv = nodesV[iv].Color;

				{
					size_t index = (uint32_t)(ch & 0xF0) + (uint32_t)(cv >> 4);

					int estimate = chunks[index];
					if (estimate < 0)
					{
						int ch0 = ch & 0xF3;
						int cv0 = cv & 0xF3;

						estimate = PlanarStripe(surface, co, ch0, ch0 + (3 << 2), cv0, cv0 + (3 << 2));
						chunks[index] = estimate;
					}
					if (estimate >= best)
						continue;
				}

				int sum = PlanarPyramid(surface, co, ch, cv);
				if (best > sum)
				{
					best = sum;

					c[0] = co;
					c[1] = ch;
					c[2] = cv;
				}
			}
		}
	}

	return best * weight;
}

static int CompressBlockColorP(uint8_t output[8], const Area& area, int input_error)
{
	int g[3], r[3], b[3];

	int water = input_error;

	int err = Planar7(area, 0, g, kGreen, water);
	if (err >= water)
		return water;

	err += Planar6(area, 1, r, kRed, water - err);
	if (err >= water)
		return water;

	err += Planar6(area, 2, b, kBlue, water - err);
	if (err >= water)
		return water;

	//

	{
		uint32_t co0 = ((r[0] >> 2) & 0x3F) << 25;
		uint32_t ch0 = ((r[1] >> 2) & 0x3E) << 1;
		uint32_t ch1 = ((g[1] >> 1) & 0x7F) << 25;
		uint32_t cv1 = ((r[2] >> 2) & 0x3F) << 13;

		co0 += ((g[0] >> 1) & 0x40) << 18;
		ch0 += (r[1] >> 2) & 1;
		ch1 += ((b[1] >> 2) & 0x3F) << 19;
		cv1 += ((g[2] >> 1) & 0x7F) << 6;

		co0 += ((g[0] >> 1) & 0x3F) << 17;
		cv1 += (b[2] >> 2) & 0x3F;
		co0 += ((b[0] >> 2) & 0x20) << 11;

		co0 += ((b[0] >> 2) & 0x18) << 8;
		co0 += ((b[0] >> 2) & 7) << 7;

		uint32_t d0 = (co0 + ch0) + 2;
		uint32_t d1 = ch1 + cv1;

		{
			uint32_t dR = (d0 >> 24) & 7;
			if (dR & 4)
			{
				uint32_t cR = (d0 >> 27) & 0xF;
				if (cR + dR < 8)
				{
					d0 += 1u << 31;
				}
			}
		}

		{
			uint32_t dG = (d0 >> 16) & 7;
			if (dG & 4)
			{
				uint32_t cG = (d0 >> 19) & 0xF;
				if (cG + dG < 8)
				{
					d0 += 1u << 23;
				}
			}
		}

		{
			uint32_t cB = (d0 >> 11) & 3;
			uint32_t dB = (d0 >> 8) & 3;

			if (cB + dB >= 4)
			{
				d0 += 7u << 13;
			}
			else
			{
				d0 += 1u << 10;
			}
		}

		*(uint32_t*)output = BSWAP(d0);
		*(uint32_t*)(output + 4) = BSWAP(d1);
	}

	return err;
}

static INLINED void FilterPixelsColor16(Area& area, uint64_t order)
{
	size_t w = 0;

	for (size_t i = 0; i < 16 * 4; i += 4)
	{
		__m128i m = _mm_load_si128((const __m128i*)&area.Data[i]);

		int a = _mm_extract_epi16(m, 6);

		_mm_store_si128((__m128i*)&area.Data[w * 4], m);

		area.Shift[w] = order & 0xF;

		order >>= 4;

		w += (a != 0) ? 1 : 0;
	}

	area.Count = (int)w;
}

static int CompressBlockColorEnhanced(uint8_t output[8], const uint8_t* __restrict cell, size_t stride, int input_error)
{
	Area area;

	{
		const uint8_t* src = cell;

		__m128i c00 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(area.Data + 0), c00);
		__m128i c01 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(area.Data + 4), c01);
		__m128i c02 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(area.Data + 8), c02);
		__m128i c03 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(area.Data + 12), c03);

		src += stride;

		__m128i c10 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(area.Data + 16), c10);
		__m128i c11 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(area.Data + 20), c11);
		__m128i c12 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(area.Data + 24), c12);
		__m128i c13 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(area.Data + 28), c13);

		src += stride;

		__m128i c20 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(area.Data + 32), c20);
		__m128i c21 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(area.Data + 36), c21);
		__m128i c22 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(area.Data + 40), c22);
		__m128i c23 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(area.Data + 44), c23);

		src += stride;

		__m128i c30 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(area.Data + 48), c30);
		__m128i c31 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(area.Data + 52), c31);
		__m128i c32 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(area.Data + 56), c32);
		__m128i c33 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(area.Data + 60), c33);
	}

	FilterPixelsColor16(area, 0xFB73EA62D951C840uLL);

	int err = input_error;
	if (err > 0)
	{
		int errP = CompressBlockColorP(output, area, err);
		if (err > errP)
			err = errP;

		if (err > 0)
		{
			int errH = CompressBlockColorH(output, area, err);
			if (err > errH)
				err = errH;

			if (err > 0)
			{
				int errT = CompressBlockColorT(output, area, err);
				if (err > errT)
					err = errT;
			}
		}
	}

	return err;
}


enum class PackMode
{
	CompressAlphaEnhanced, DecompressAlphaEnhanced,
	CompressColorEnhanced, DecompressColorEnhanced
};

class Worker
{
public:
	class Item
	{
	public:
		uint8_t* _Output;
		uint8_t* _Cell;

		Item()
		{
		}

		Item(uint8_t* output, uint8_t* cell)
		{
			_Output = output;
			_Cell = cell;
		}
	};

	class Job
	{
	public:
		Job* _Next;

	protected:
		int _Count, _Index;
		Item _Items[0x100];

	public:
		Job()
		{
			_Next = nullptr;

			_Count = 0;
			_Index = 0;
		}

		bool Add(const Item& item)
		{
			_Items[_Count++] = item;

			return _Count >= (int)(sizeof(_Items) / sizeof(_Items[0]));
		}

		Item* Take()
		{
			return (_Index < _Count) ? &_Items[_Index++] : nullptr;
		}
	};

protected:
#ifdef WIN32
	CRITICAL_SECTION _Sync;
	HANDLE _Done;
#else
	std::mutex _Sync;
#endif

	Job* _First;
	Job* _Last;

	int64_t _mse;
	double _ssim;

	std::atomic_int _Running;

	PackMode _Mode;

public:
	Worker()
	{
#ifdef WIN32
		if (!InitializeCriticalSectionAndSpinCount(&_Sync, 1000))
			throw std::runtime_error("init");

		_Done = CreateEvent(NULL, FALSE, FALSE, NULL);
		if (_Done == nullptr)
			throw std::runtime_error("init");
#endif

		_First = nullptr;
		_Last = nullptr;
	}

	~Worker()
	{
		for (Job* job; (job = _First) != nullptr;)
		{
			_First = job->_Next;

			delete job;
		}

		_Last = nullptr;

#ifdef WIN32
		if (_Done != nullptr)
			CloseHandle(_Done), _Done = nullptr;

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

	void Add(Job* job)
	{
		if (_Last)
			_Last->_Next = job;
		else
			_First = job;

		_Last = job;
	}

protected:
	Job* Take()
	{
		Lock();

		Job* job = _First;
		if (job)
		{
			_First = job->_Next;

			if (_First == nullptr)
				_Last = nullptr;
		}

		UnLock();

		return job;
	}

#ifdef WIN32
	static DWORD WINAPI ThreadProc(LPVOID lpParameter)
#else
	static int ThreadProc(Worker* lpParameter)
#endif
	{
		Worker* worker = static_cast<Worker*>(lpParameter);

		int64_t mse = 0;
		double ssim = 0;

		for (Job* job; (job = worker->Take()) != nullptr;)
		{
			switch (worker->_Mode)
			{
			case PackMode::CompressAlphaEnhanced:
				while (Item* item = job->Take())
				{
					alignas(16) uint8_t temp[4 * 4 * 4];
					uint8_t* output = item->_Output;
					output[1] = (uint8_t)Max(output[1], 0x10);
					DecompressBlockAlphaEnhanced(output, temp, 4 * 4);

					int input_error = CompareBlocksAlpha(item->_Cell, Stride, temp, 4 * 4);
					if (input_error > 0)
					{
						mse += CompressBlockAlphaEnhanced(item->_Output, item->_Cell, Stride, input_error);

						DecompressBlockAlphaEnhanced(item->_Output, temp, 4 * 4);
						ssim += CompareBlocksAlphaSSIM(item->_Cell, Stride, temp, 4 * 4);
					}
					else
					{
						ssim += 1.0;
					}
				}
				break;

			case PackMode::DecompressAlphaEnhanced:
				while (Item* item = job->Take())
				{
					DecompressBlockAlphaEnhanced(item->_Output, item->_Cell, Stride);
				}
				break;

			case PackMode::CompressColorEnhanced:
				while (Item* item = job->Take())
				{
					alignas(16) uint8_t temp[4 * 4 * 4];
					DecompressBlockColor(item->_Output, temp, 4 * 4);

					int input_error = CompareBlocksColor(item->_Cell, Stride, temp, 4 * 4);
					if (input_error > 0)
					{
						input_error = CompressBlockColor(item->_Output, item->_Cell, Stride, input_error);
						if (input_error > 0)
						{
							input_error = CompressBlockColorEnhanced(item->_Output, item->_Cell, Stride, input_error);
						}
						mse += input_error;

						DecompressBlockColor(item->_Output, temp, 4 * 4);
						ssim += CompareBlocksColorSSIM(item->_Cell, Stride, temp, 4 * 4);
					}
					else
					{
						ssim += 1.0;
					}
				}
				break;

			case PackMode::DecompressColorEnhanced:
				while (Item* item = job->Take())
				{
					DecompressBlockColor(item->_Output, item->_Cell, Stride);
				}
				break;
			}

			delete job;
		}

		worker->Lock();

		worker->_mse += mse;
		worker->_ssim += ssim;

		worker->UnLock();

		worker->_Running--;

#ifdef WIN32
		if (worker->_Running <= 0)
		{
			SetEvent(worker->_Done);
		}
#endif

		return 0;
	}

public:
	__m128i Run(PackMode mode)
	{
		_Mode = mode;

		_mse = 0;
		_ssim = 0;

		int n = std::thread::hardware_concurrency();
		_Running = n;

		for (int i = 0; i < n; i++)
		{
#ifdef WIN32
			HANDLE hthread = CreateThread(NULL, WorkerThreadStackSize, ThreadProc, this, 0, NULL);
			if (hthread == nullptr)
				throw std::runtime_error("fork");
			CloseHandle(hthread);
#else
			boost::thread::attributes attrs;
			attrs.set_stack_size(WorkerThreadStackSize);
			boost::thread thread(attrs, boost::bind(ThreadProc, this));
			thread.detach();
#endif
		}

#ifdef WIN32
		WaitForSingleObject(_Done, INFINITE);
#else
		for (;; )
		{
			std::this_thread::yield();

			if (_Running <= 0)
				break;
		}
#endif

		return _mm_unpacklo_epi64(_mm_cvtsi64_si128(_mse), _mm_castpd_si128(_mm_load_sd((double*)&_ssim)));
	}
};

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

static void LoadEtc2(const char* name, uint8_t* buffer, int size)
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

static void SaveEtc2(const char* name, const uint8_t* buffer, int size)
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

static __m128i PackTexture(uint8_t* dst_etc1, uint8_t* src_bgra, int src_w, int src_h, PackMode mode, size_t block_size)
{
	auto start = std::chrono::high_resolution_clock::now();

	int64_t mse = 0;
	double ssim = 0;

	{
		uint8_t* output = dst_etc1;

		Worker* worker = new Worker();

		Worker::Job* job = new Worker::Job();

		for (int y = 0; y < src_h; y += 4)
		{
			uint8_t* cell = src_bgra + y * Stride;

			for (int x = 0; x < src_w; x += 4)
			{
				if (job->Add(Worker::Item(output, cell)))
				{
					worker->Add(job);

					job = new Worker::Job();
				}

				output += block_size;
				cell += 16;
			}
		}

		worker->Add(job);

		__m128i v = worker->Run(mode);
		mse += _mm_cvtsi128_si64(v);
		ssim += _mm_cvtsd_f64(_mm_castsi128_pd(_mm_unpackhi_epi64(v, v)));

		delete worker;
	}

	auto finish = std::chrono::high_resolution_clock::now();

	if ((mode == PackMode::CompressAlphaEnhanced) || (mode == PackMode::CompressColorEnhanced))
	{
		int n = (src_h * src_w) >> 4;

		int span = Max((int)std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count(), 1);

		printf("    Compressed %d blocks, elapsed %i ms, %i bps\n", n, span, (int)(n * 1000LL / span));
	}

	return _mm_unpacklo_epi64(_mm_cvtsi64_si128(mse), _mm_castpd_si128(_mm_load_sd(&ssim)));
}

static INLINED void OutlineAlpha(uint8_t* src_bgra, int src_w, int src_h, int radius)
{
	if (radius <= 0)
		return;

	int full_w = 1 + radius + src_w + radius;
	int full_h = 1 + radius + src_h + radius;

	uint8_t* data = new uint8_t[full_h * full_w];
	memset(data, 0, full_h * full_w);

	for (int y = 0; y < src_h; y++)
	{
		const uint8_t* r = &src_bgra[y * Stride + 3];
		uint8_t* w = &data[(y + radius + 1) * full_w + (radius + 1)];

		for (int x = 0; x < src_w; x++)
		{
			w[x] = (r[x * 4] != 0) ? 1 : 0;
		}
	}

	int* sum = new int[full_h * full_w];
	memset(sum, 0, full_h * full_w * sizeof(int));

	int from_py = (radius + 1) * full_w;
	int to_py = full_h * full_w;
	for (int py = from_py; py < to_py; py += full_w)
	{
		int prev_py = py - full_w;

		for (int x = radius + 1; x < full_w; x++)
		{
			sum[py + x] = sum[prev_py + x] - sum[prev_py + x - 1] + data[py + x] + sum[py + x - 1];
		}
	}

	int a = radius + radius + 1;
	for (int y = 0; y < src_h; y++)
	{
		uint8_t* w = &src_bgra[y * Stride + 3];
		const int* rL = &sum[y * full_w];
		const int* rH = &sum[(y + a) * full_w];

		for (int x = 0; x < src_w; x++, rL++, rH++)
		{
			int v = rH[a] - *rH + *rL - rL[a];

			w[x * 4] = (v != 0) ? 1 : 0;
		}
	}

	delete[] sum, sum = nullptr;

	delete[] data, data = nullptr;
}

int Etc2MainWithArgs(const std::vector<std::string>& args)
{
	bool flip = false;
	int border = 1;

	const char* src_name = nullptr;
	const char* dst_name = nullptr;
	const char* result_name = nullptr;

	for (int i = 0, n = (int)args.size(); i < n; i++)
	{
		const char* arg = args[i].c_str();

		if (arg[0] == '/')
		{
			if (strcmp(arg, "/retina") == 0)
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
		else if (dst_name == nullptr)
		{
			dst_name = arg;
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

	int src_texture_w = (src_image_w + 3) & ~3;
	int src_texture_h = (src_image_h + 3) & ~3;

	if (src_texture_w < 4)
		src_texture_w = 4;
	if (src_texture_h < 4)
		src_texture_h = 4;

	if (Max(src_texture_w, src_texture_h) > 8192)
	{
		printf("Huge image %s\n", src_name);
		return 1;
	}

	int c = 4;
	int src_image_stride = src_image_w * c;
	int src_texture_stride = src_texture_w * c;

	uint8_t* src_texture_bgra = new uint8_t[src_texture_h * src_texture_stride];

	for (int i = 0; i < src_image_h; i++)
	{
		memcpy(&src_texture_bgra[i * src_texture_stride], &src_image_bgra[i * src_image_stride], src_image_stride);

		for (int j = src_image_stride; j < src_texture_stride; j += c)
		{
			memcpy(&src_texture_bgra[i * src_texture_stride + j], &src_image_bgra[i * src_image_stride + src_image_stride - c], c);
		}
	}

	for (int i = src_image_h; i < src_texture_h; i++)
	{
		memcpy(&src_texture_bgra[i * src_texture_stride], &src_texture_bgra[(src_image_h - 1) * src_texture_stride], src_texture_stride);
	}

	printf("  Image %dx%d, Texture %dx%d\n", src_image_w, src_image_h, src_texture_w, src_texture_h);

	Stride = src_texture_stride;

	uint8_t* dst_texture_bgra = new uint8_t[src_texture_h * src_texture_stride];

	int SizeEnhanced = src_texture_h * src_texture_w;

	InitLevelErrors();

	memcpy(dst_texture_bgra, src_texture_bgra, src_texture_h * src_texture_stride);

	if ((dst_name != nullptr) && dst_name[0])
	{
		uint8_t* dst_etc2 = new uint8_t[SizeEnhanced];
		memset(dst_etc2, 0, SizeEnhanced);

		LoadEtc2(dst_name, dst_etc2, SizeEnhanced);

		__m128i v2 = PackTexture(dst_etc2, dst_texture_bgra, src_texture_w, src_texture_h, PackMode::CompressAlphaEnhanced, 8 * 2);
		int64_t mse_alpha = _mm_cvtsi128_si64(v2);
		double ssim_alpha = _mm_cvtsd_f64(_mm_castsi128_pd(_mm_unpackhi_epi64(v2, v2)));

		if (mse_alpha > 0)
		{
			printf("      SubTexture A PSNR = %f, SSIM_4x4 = %.8f\n",
				10.0 * log((255.0 * 255.0) * (src_texture_h * src_texture_w) / mse_alpha) / log(10.0),
				ssim_alpha * 16.0 / (src_texture_h * src_texture_w));
		}
		else
		{
			printf("      Exactly\n");
		}

		PackTexture(dst_etc2, dst_texture_bgra, src_texture_w, src_texture_h, PackMode::DecompressAlphaEnhanced, 8 * 2);

		uint8_t* dst_texture_color = new uint8_t[src_texture_h * src_texture_stride];

		memcpy(dst_texture_color, dst_texture_bgra, src_texture_h * src_texture_stride);

		OutlineAlpha(dst_texture_color, src_texture_w, src_texture_h, border);

		v2 = PackTexture(dst_etc2 + 8, dst_texture_color, src_texture_w, src_texture_h, PackMode::CompressColorEnhanced, 8 * 2);
		int64_t mse_color = _mm_cvtsi128_si64(v2);
		double ssim_color = _mm_cvtsd_f64(_mm_castsi128_pd(_mm_unpackhi_epi64(v2, v2)));

		if (mse_color > 0)
		{
			printf("      SubTexture RGB wPSNR = %f, wSSIM_4x4 = %.8f\n",
				10.0 * log((255.0 * 255.0) * kColor * (src_texture_h * src_texture_w) / mse_color) / log(10.0),
				ssim_color * 16.0 / (src_texture_h * src_texture_w));
		}
		else
		{
			printf("      Exactly\n");
		}

		SaveEtc2(dst_name, dst_etc2, SizeEnhanced);

		PackTexture(dst_etc2 + 8, dst_texture_color, src_texture_w, src_texture_h, PackMode::DecompressColorEnhanced, 8 * 2);

		size_t delta_dst = dst_texture_bgra - dst_texture_color;

		for (int y = 0; y < src_texture_h; y++)
		{
			uint8_t* cell = dst_texture_color + y * src_texture_stride;

			for (int x = 0; x < src_texture_w; x++)
			{
				uint32_t c1 = *(uint32_t*)(cell + delta_dst);
				uint32_t c2 = *(uint32_t*)cell;

				c1 &= ~0xFFFFFFu;
				c2 &= 0xFFFFFFu;

				*(uint32_t*)(cell + delta_dst) = c1 | c2;

				cell += 4;
			}
		}

		delete[] dst_texture_color;
		delete[] dst_etc2;
	}

	if ((result_name != nullptr) && result_name[0])
	{
		WriteImage(result_name, dst_texture_bgra, src_texture_w, src_texture_h, flip);
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
		printf("Usage: Etc2Compress [/retina] src [dst] [/debug result.png]\n");
		return 1;
	}

	std::vector<std::string> args;
	args.reserve(argc);

	for (int i = 1; i < argc; i++)
	{
		args.emplace_back(argv[i]);
	}

	return Etc2MainWithArgs(args);
}
