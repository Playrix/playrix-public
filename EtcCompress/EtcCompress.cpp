// Copyright(c) 2016 Playrix LLC
//
// LICENSE: https://mit-license.org

#include <windows.h>
#pragma warning(push)
#pragma warning(disable : 4458)
#include <gdiplus.h>
#pragma warning(pop)

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

#pragma comment(lib, "gdiplus.lib")

#define M128I_I32(mm, index) ((mm).m128i_i32[index])

typedef unsigned char BYTE;

typedef __declspec(align(16)) struct { int Data[8 * 4]; int Count, unused; BYTE Shift[8]; } Half;
typedef __declspec(align(16)) struct { Half A, B; } Elem;

typedef __declspec(align(8)) struct { int Error, Color; } Node;

// http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html sRGB
enum { kGreen = 715, kRed = 213, kBlue = 72, kUnknownError = (255 * 255) * 1000 * (4 * 4) + 1 };

static const __declspec(align(16)) int g_table[8][2] = { { 2, 8 },{ 5, 17 },{ 9, 29 },{ 13, 42 },{ 18, 60 },{ 24, 80 },{ 33, 106 },{ 47, 183 } };

static const __declspec(align(16)) short g_GRB_I16[8] = { kGreen, kRed, kBlue, 0, kGreen, kRed, kBlue, 0 };
static const __declspec(align(16)) short g_GR_I16[8] = { kGreen, kRed, kGreen, kRed, kGreen, kRed, kGreen, kRed };

static const __declspec(align(16)) int g_colors4[0x10] =
{
	0x00, 0x11, 0x22, 0x33,
	0x44, 0x55, 0x66, 0x77,
	0x88, 0x99, 0xAA, 0xBB,
	0xCC, 0xDD, 0xEE, 0xFF
};

static const __declspec(align(16)) int g_colors5[0x20] =
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

static __m128i g_errors4[8][0x10 >> 2][0x100];
static __m128i g_errors5[8][0x20 >> 2][0x100];

static const double g_ssim_nk1L = (0.01 * 255 * 8) * (0.01 * 255 * 8);
static const double g_ssim_nk2L = (0.03 * 255 * 8) * (0.03 * 255 * 8);

static const int WorkerThreadStackSize = 3 * 1024 * 1024;

static int Stride;

static __forceinline uint32_t BSWAP(uint32_t x)
{
	return _byteswap_ulong(x);
}

static __forceinline uint32_t BROR(uint32_t x)
{
	return _rotr(x, 8);
}

static __forceinline int Min(int x, int y)
{
	return (x < y) ? x : y;
}

static __forceinline int Max(int x, int y)
{
	return (x > y) ? x : y;
}

static __forceinline int ExpandColor5(int c)
{
	return (c << 3) ^ (c >> 2);
}

static __forceinline int ExpandColor4(int c)
{
	return (c << 4) ^ c;
}

static __forceinline __m128i HorizontalMinimum4(__m128i me4)
{
	__m128i me2 = _mm_min_epi32(me4, _mm_shuffle_epi32(me4, _MM_SHUFFLE(2, 3, 0, 1)));
	__m128i me1 = _mm_min_epi32(me2, _mm_shuffle_epi32(me2, _MM_SHUFFLE(0, 1, 2, 3)));
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

				M128I_I32(g_errors4[q][i >> 2][x], i & 3) = v * v;
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

				M128I_I32(g_errors5[q][i >> 2][x], i & 3) = v * v;
			}
		}
	}
}

static __forceinline __m128i ComputeLevel(const Half& half, int offset, const __m128i errors[0x100])
{
	__m128i sum = _mm_setzero_si128();

	int k = half.Count, j = offset;
	if (k & 8)
	{
		sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 0]]));
		sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 4]]));
		sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 8]]));
		sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 12]]));
		sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 16]]));
		sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 20]]));
		sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 24]]));
		sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 28]]));
	}
	else
	{
		if (k & 4)
		{
			sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 0]]));
			sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 4]]));
			sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 8]]));
			sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 12]]));

			j += 16;
		}

		if (k & 2)
		{
			sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 0]]));
			sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 4]]));

			j += 8;
		}

		if (k & 1)
		{
			sum = _mm_add_epi32(sum, _mm_load_si128(&errors[half.Data[j + 0]]));
		}
	}

	return sum;
}

static __forceinline void AdjustLevels(const Half& half, int offset, Node nodes[0x20 + 1], int weight, int water, int q)
{
	int top = (water + weight - 1) / weight;

	__m128i mweight = _mm_shuffle_epi32(_mm_cvtsi32_si128(weight), 0);

	__m128i mtop = _mm_shuffle_epi32(_mm_cvtsi32_si128(top), 0);

	__m128i level = _mm_mullo_epi32(mtop, mweight);

	int w = 0;

	for (int c = 0; c < 0x20; c += 4)
	{
		__m128i sum = ComputeLevel(half, offset, g_errors5[q][c >> 2]);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(mtop, sum)) != 0)
		{
			sum = _mm_mullo_epi32(_mm_min_epi32(sum, mtop), mweight);

			__m128i mc = _mm_load_si128((__m128i*)&g_colors5[c]);

			__m128i me1 = HorizontalMinimum4(sum);

			__m128i mL = _mm_unpacklo_epi32(sum, mc);
			__m128i mH = _mm_unpackhi_epi32(sum, mc);

			_mm_store_si128((__m128i*)&nodes[w + 0], mL);
			_mm_store_si128((__m128i*)&nodes[w + 2], mH);

			level = _mm_min_epi32(level, me1);

			w += 4;
		}
	}

	nodes[0x20].Error = _mm_cvtsi128_si32(level);
	nodes[0x20].Color = w;
}

static __forceinline void GuessLevels(const Half& half, int offset, Node nodes[0x10 + 1], int weight, int water, int q)
{
	int top = (water + weight - 1) / weight;

	__m128i mweight = _mm_shuffle_epi32(_mm_cvtsi32_si128(weight), 0);

	__m128i mtop = _mm_shuffle_epi32(_mm_cvtsi32_si128(top), 0);

	__m128i level = _mm_mullo_epi32(mtop, mweight);

	int w = 0;

	for (int c = 0; c < 0x10; c += 4)
	{
		__m128i sum = ComputeLevel(half, offset, g_errors4[q][c >> 2]);
		if (_mm_movemask_epi8(_mm_cmpgt_epi32(mtop, sum)) != 0)
		{
			sum = _mm_mullo_epi32(_mm_min_epi32(sum, mtop), mweight);

			__m128i mc = _mm_load_si128((__m128i*)&g_colors4[c]);

			__m128i me1 = HorizontalMinimum4(sum);

			__m128i mL = _mm_unpacklo_epi32(sum, mc);
			__m128i mH = _mm_unpackhi_epi32(sum, mc);

			_mm_store_si128((__m128i*)&nodes[w + 0], mL);
			_mm_store_si128((__m128i*)&nodes[w + 2], mH);

			level = _mm_min_epi32(level, me1);

			w += 4;
		}
	}

	nodes[0x10].Error = _mm_cvtsi128_si32(level);
	nodes[0x10].Color = w;
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

#define SSIM_CLOSE() \
	sab = _mm_slli_epi32(sab, 3 + 1); \
	saa_sbb = _mm_slli_epi32(saa_sbb, 3); \
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

#define SSIM_FINAL(dst) \
	__m128d dst; \
	{ \
		__m128d mc1 = _mm_load1_pd(&g_ssim_nk1L); \
		__m128d mc2 = _mm_load1_pd(&g_ssim_nk2L); \
		dst = _mm_div_pd( \
			_mm_mul_pd(_mm_add_pd(_mm_cvtepi32_pd(sasb), mc1), _mm_add_pd(_mm_cvtepi32_pd(sab), mc2)), \
			_mm_mul_pd(_mm_add_pd(_mm_cvtepi32_pd(sasa_sbsb), mc1), _mm_add_pd(_mm_cvtepi32_pd(saa_sbb), mc2))); \
	} \


static __forceinline void DecompressHalfAlpha(BYTE pL[4 * 4 - 3], BYTE pH[4 * 4 - 3], int alpha, int q, uint32_t data, int shift)
{
	static const __declspec(align(16)) int g_delta[8][2] =
	{
		{ 2, (8 ^ 2) },
		{ 5, (17 ^ 5) },
		{ 9, (29 ^ 9) },
		{ 13, (42 ^ 13) },
		{ 18, (60 ^ 18) },
		{ 24, (80 ^ 24) },
		{ 33, (106 ^ 33) },
		{ 47, (183 ^ 47) }
	};

	static const __declspec(align(16)) int g_mask[16][4] =
	{
		{ 0, 0, 0, 0 },{ -1, 0, 0, 0 },{ 0, -1, 0, 0 },{ -1, -1, 0, 0 },
		{ 0, 0, -1, 0 },{ -1, 0, -1, 0 },{ 0, -1, -1, 0 },{ -1, -1, -1, 0 },
		{ 0, 0, 0, -1 },{ -1, 0, 0, -1 },{ 0, -1, 0, -1 },{ -1, -1, 0, -1 },
		{ 0, 0, -1, -1 },{ -1, 0, -1, -1 },{ 0, -1, -1, -1 },{ -1, -1, -1, -1 }
	};

	__m128i mt0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_delta[q][0]), 0);
	__m128i mt10 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_delta[q][1]), 0);

	__m128i mMaskL = _mm_load_si128((const __m128i*)&((const BYTE*)g_mask)[(shift < 4 ? data << (4 - shift) : data >> (shift - 4)) & 0xF0]);
	__m128i mMaskH = _mm_load_si128((const __m128i*)&((const BYTE*)g_mask)[(data >> (shift + 4 - 4)) & 0xF0]);

	__m128i mtL = _mm_xor_si128(_mm_and_si128(mMaskL, mt10), mt0);
	__m128i mtH = _mm_xor_si128(_mm_and_si128(mMaskH, mt10), mt0);

	mMaskL = _mm_load_si128((const __m128i*)&((const BYTE*)g_mask)[(data >> (shift + 16 + 0 - 4)) & 0xF0]);
	mMaskH = _mm_load_si128((const __m128i*)&((const BYTE*)g_mask)[(data >> (shift + 16 + 4 - 4)) & 0xF0]);

	__m128i mc = _mm_shuffle_epi32(_mm_cvtsi32_si128(alpha), 0);

	__m128i mcL = _mm_or_si128(_mm_and_si128(mMaskL, _mm_subs_epu8(mc, mtL)), _mm_andnot_si128(mMaskL, _mm_adds_epu8(mc, mtL)));
	__m128i mcH = _mm_or_si128(_mm_and_si128(mMaskH, _mm_subs_epu8(mc, mtH)), _mm_andnot_si128(mMaskH, _mm_adds_epu8(mc, mtH)));

	pL[0] = (BYTE)_mm_extract_epi16(mcL, 0);
	pL[4] = (BYTE)_mm_extract_epi16(mcL, 2);
	pL[8] = (BYTE)_mm_extract_epi16(mcL, 4);
	pL[12] = (BYTE)_mm_extract_epi16(mcL, 6);

	pH[0] = (BYTE)_mm_extract_epi16(mcH, 0);
	pH[4] = (BYTE)_mm_extract_epi16(mcH, 2);
	pH[8] = (BYTE)_mm_extract_epi16(mcH, 4);
	pH[12] = (BYTE)_mm_extract_epi16(mcH, 6);
}

static __forceinline void DecompressBlockAlpha(const BYTE input[8], BYTE* __restrict cell)
{
	cell += 3;

	int a, b;

	uint32_t c = *(const uint32_t*)input;

	if ((c & (2 << 24)) == 0)
	{
		a = c & 0xF0; a |= a >> 4;
		b = c & 0x0F; b |= b << 4;
	}
	else
	{
		a = c & 0xF8;
		b = (((((c & 0x07) ^ 0x24) - 0x04) << 3) + a) & 0xF8;

		a |= (a >> 5) & 0x07;
		b |= (b >> 5) & 0x07;
	}

	uint32_t way = BSWAP(*(const uint32_t*)&input[4]);

	if ((c & (1 << 24)) == 0)
	{
		BYTE buffer[4 * 4];

		int qa = (c >> (5 + 24)) & 7;
		DecompressHalfAlpha(&buffer[0], &buffer[1], a, qa, way, 0);

		int qb = (c >> (2 + 24)) & 7;
		DecompressHalfAlpha(&buffer[2], &buffer[3], b, qb, way, 8);

		BYTE* dst = cell;

		dst[0] = buffer[0];
		dst[4] = buffer[1];
		dst[8] = buffer[2];
		dst[12] = buffer[3];

		dst += Stride;

		dst[0] = buffer[4];
		dst[4] = buffer[5];
		dst[8] = buffer[6];
		dst[12] = buffer[7];

		dst += Stride;

		dst[0] = buffer[8];
		dst[4] = buffer[9];
		dst[8] = buffer[10];
		dst[12] = buffer[11];

		dst += Stride;

		dst[0] = buffer[12];
		dst[4] = buffer[13];
		dst[8] = buffer[14];
		dst[12] = buffer[15];
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
		DecompressHalfAlpha(cell, cell + Stride, a, qa, way, 0);

		cell += Stride + Stride;

		int qb = (c >> (2 + 24)) & 7;
		DecompressHalfAlpha(cell, cell + Stride, b, qb, way, 8);
	}
}


struct BlockStateAlpha
{
	int a, b, qa, qb, mode;
};

struct GuessStateAlpha
{
	__declspec(align(16)) Node node[0x11];

	__forceinline GuessStateAlpha()
	{
	}

	__forceinline int Init(const Half& half, int water, int q)
	{
		GuessLevels(half, 0, node, 1, water, q);

		return node[0x10].Error;
	}

	__forceinline int Find(int error)
	{
		for (int c = 0, n = node[0x10].Color; c < n; c++)
		{
			if (node[c].Error <= error)
			{
				return node[c].Color;
			}
		}

		return 0x10;
	}
};

struct AdjustStateAlpha
{
	__declspec(align(16)) Node node[0x21];

	int swap[0x20];

	int stop;

	bool flag_swap;

	__forceinline AdjustStateAlpha()
	{
	}

	__forceinline void Init(const Half& half, int water, int q)
	{
		flag_swap = false;

		AdjustLevels(half, 0, node, 1, water, q);

		stop = node[0x20].Error;
	}

	__forceinline void Index()
	{
		if (flag_swap)
			return;
		flag_swap = true;

		memset(swap, -1, sizeof(swap));

		for (int i = 0, n = node[0x20].Color; i < n; i++)
		{
			int c = node[i].Color >> 3;
			swap[c] = i;
		}
	}
};

struct AdjustStateAlphaGroup
{
	AdjustStateAlpha S[8];

	__forceinline AdjustStateAlphaGroup()
	{
	}

	__forceinline void Init(const Half& half, int water)
	{
		for (int q = 0; q < 8; q++)
		{
			S[q].Init(half, water, q);
		}
	}

	__forceinline int Best(int water) const
	{
		int best = water;

		for (int q = 0; q < 8; q++)
		{
			best = Min(best, S[q].stop);
		}

		return best;
	}
};

static __forceinline double ComputeTableAlpha(const Half& half, int alpha, int q, uint32_t& index, uint32_t order)
{
	__m128i ma = _mm_cvtsi32_si128(alpha);
	ma = _mm_unpacklo_epi32(ma, ma);

	__m128i mt10 = _mm_loadl_epi64((const __m128i*)&g_table[q][0]);

	__m128i mt3210 = _mm_unpacklo_epi64(_mm_adds_epu8(ma, mt10), _mm_subs_epu8(ma, mt10));

	int good = 0xF;
	if (M128I_I32(mt3210, 0) == M128I_I32(mt3210, 1)) good &= ~2;
	if (M128I_I32(mt3210, 2) == M128I_I32(mt3210, 3)) good &= ~8;

	int ways[8];

	__m128i mc = mt3210;

	for (int i = 0; i < 8; i++)
	{
		__m128i mb = _mm_shuffle_epi32(_mm_cvtsi32_si128(half.Data[i << 2]), 0);

		__m128i me4 = _mm_abs_epi16(_mm_sub_epi16(mb, mc));

		__m128i me2 = _mm_min_epi16(me4, _mm_shuffle_epi32(me4, _MM_SHUFFLE(2, 3, 0, 1)));
		__m128i me1 = _mm_min_epi16(me2, _mm_shuffle_epi32(me2, _MM_SHUFFLE(0, 1, 2, 3)));

		int way = _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(me4, me1)));
		ways[i] = (way & good) | (1 << 4);
	}

	int loops[8];

	for (int i = 0; i < 8; i++)
	{
		int k = 0;
		while ((ways[i] & (1 << k)) == 0) k++;
		loops[i] = k;
	}

	double best = -1.1;
	uint32_t codes = 0;

	for (;; )
	{
		SSIM_INIT();

		for (int i = 0; i < 8; i++)
		{
			__m128i mt = _mm_cvtsi32_si128(M128I_I32(mt3210, loops[i]));

			__m128i mb = _mm_cvtsi32_si128(half.Data[i << 2]);

			SSIM_UPDATE(mt, mb);
		}

		SSIM_CLOSE();

		SSIM_FINAL(mssim);

		double ssim = _mm_cvtsd_f64(mssim);

		if (best < ssim)
		{
			best = ssim;

			uint32_t v = 0;
			for (int j = 0; j < 8; j++)
			{
				v |= ((uint32_t)loops[j]) << (j + j);
			}
			codes = v;

			if (best >= 1.0)
				break;
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

			i++;
			if (i & 8)
				break;
		}
		if (i & 8)
			break;
	}

	for (int i = 8; --i >= 0;)
	{
		int shift = order & 0xF;

		order >>= 4;

		uint32_t code = ((codes & 2u) << (16 - 1)) | (codes & 1u);

		codes >>= 2;

		index |= code << shift;
	}

	return best;
}

static __forceinline int GuessAlpha4(const Half& half, int& alpha, int water, int& table)
{
	GuessStateAlpha SA;

	for (int q = 0; q < 8; q++)
	{
		int error = SA.Init(half, water, q);

		if (water > error)
		{
			water = error;

			alpha = SA.Find(error);
			table = q;

			if (water <= 0)
				break;
		}
	}

	return water;
}

static int CompressBlockAlpha44(BlockStateAlpha& s, const Elem& elem, int water, int mode)
{
	int a, b, qa, qb;

	int err = GuessAlpha4(elem.A, a, water, qa);
	if (err >= water)
		return water;

	err += GuessAlpha4(elem.B, b, water - err, qb);
	if (err >= water)
		return water;

	s.a = a;
	s.b = b;
	s.qa = qa;
	s.qb = qb;
	s.mode = mode;

	return err;
}

static __forceinline int AdjustAlphas53(int& alpha, int& other, int water, AdjustStateAlpha& SA, AdjustStateAlpha& SB)
{
	SB.Index();

	for (int a0 = 0; a0 < SA.node[0x20].Color; a0++)
	{
		int e0 = SA.node[a0].Error;
		if (e0 + SB.stop >= water)
			continue;

		int a = SA.node[a0].Color;

		int Id = a >> 3;
		int Ld = Max(Id - 4, 0);
		int Hd = Min(Id + 3, 31);

		for (int d0 = Ld; d0 <= Hd; d0++)
		{
			int b0 = SB.swap[d0];
			if (b0 < 0)
				continue;

			int e1 = SB.node[b0].Error + e0;
			if (e1 >= water)
				continue;

			water = e1;

			alpha = a;
			other = SB.node[b0].Color;

			if (water <= e0 + SB.stop)
				break;
		}
	}

	return water;
}

static int CompressBlockAlpha53(BlockStateAlpha& s, const Elem& elem, int water, int mode)
{
	AdjustStateAlphaGroup GB;

	GB.Init(elem.B, water);

	int stopB = GB.Best(water);
	if (stopB >= water)
		return water;

	AdjustStateAlphaGroup GA;

	GA.Init(elem.A, water - stopB);

	for (int qa = 0; qa < 8; qa++)
	{
		int bestA = GA.Best(water - stopB);
		if (bestA + stopB >= water)
			return water;

		AdjustStateAlpha& SA = GA.S[qa];

		for (int qb = 0; qb < 8; qb++)
		{
			AdjustStateAlpha& SB = GB.S[qb];

			if (SA.stop + stopB >= water)
				break;

			if (SA.stop + SB.stop >= water)
				continue;

			int error = AdjustAlphas53(s.a, s.b, water, SA, SB);

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

static __forceinline int MeasureHalfAlpha(const int pL[4 * 4], const int pH[4 * 4], int alpha, int q, uint32_t dataL, uint32_t dataH, int slice, int shift)
{
	__m128i ma = _mm_cvtsi32_si128(alpha);
	ma = _mm_unpacklo_epi32(ma, ma);

	__m128i mt10 = _mm_loadl_epi64((const __m128i*)&g_table[q][0]);

	__m128i mt3210 = _mm_unpacklo_epi64(_mm_adds_epu8(ma, mt10), _mm_subs_epu8(ma, mt10));

	__m128i sum = _mm_setzero_si128();

	for (int i = 0; i < 4; i++)
	{
		__m128i m = _mm_cvtsi32_si128(*pL);

		__m128i mt = mt3210;

		if ((dataL & 0x10000u) != 0)
		{
			mt = _mm_shuffle_epi32(mt, _MM_SHUFFLE(1, 0, 3, 2));
		}

		if ((dataL & 1u) != 0)
		{
			mt = _mm_shuffle_epi32(mt, _MM_SHUFFLE(2, 3, 0, 1));
		}

		m = _mm_sub_epi16(m, mt);
		m = _mm_mullo_epi16(m, m);

		sum = _mm_add_epi32(sum, m);

		pL += slice;

		dataL >>= shift;
	}

	for (int i = 0; i < 4; i++)
	{
		__m128i m = _mm_cvtsi32_si128(*pH);

		__m128i mt = mt3210;

		if ((dataH & 0x10000u) != 0)
		{
			mt = _mm_shuffle_epi32(mt, _MM_SHUFFLE(1, 0, 3, 2));
		}

		if ((dataH & 1u) != 0)
		{
			mt = _mm_shuffle_epi32(mt, _MM_SHUFFLE(2, 3, 0, 1));
		}

		m = _mm_sub_epi16(m, mt);
		m = _mm_mullo_epi16(m, m);

		sum = _mm_add_epi32(sum, m);

		pH += slice;

		dataH >>= shift;
	}

	return _mm_cvtsi128_si32(sum);
}

static __m128i CompressBlockAlpha(BYTE output[8], const BYTE* __restrict cell)
{
	Elem norm, flip;

	{
		const BYTE* src = cell;

		int c00 = src[3]; flip.A.Data[0] = c00; norm.A.Data[0] = c00;
		int c01 = src[7]; flip.A.Data[4] = c01; norm.A.Data[4] = c01;
		int c02 = src[11]; flip.A.Data[8] = c02; norm.B.Data[0] = c02;
		int c03 = src[15]; flip.A.Data[12] = c03; norm.B.Data[4] = c03;

		src += Stride;

		int c10 = src[3]; flip.A.Data[16] = c10; norm.A.Data[8] = c10;
		int c11 = src[7]; flip.A.Data[20] = c11; norm.A.Data[12] = c11;
		int c12 = src[11]; flip.A.Data[24] = c12; norm.B.Data[8] = c12;
		int c13 = src[15]; flip.A.Data[28] = c13; norm.B.Data[12] = c13;

		src += Stride;

		int c20 = src[3]; flip.B.Data[0] = c20; norm.A.Data[16] = c20;
		int c21 = src[7]; flip.B.Data[4] = c21; norm.A.Data[20] = c21;
		int c22 = src[11]; flip.B.Data[8] = c22; norm.B.Data[16] = c22;
		int c23 = src[15]; flip.B.Data[12] = c23; norm.B.Data[20] = c23;

		src += Stride;

		int c30 = src[3]; flip.B.Data[16] = c30; norm.A.Data[24] = c30;
		int c31 = src[7]; flip.B.Data[20] = c31; norm.A.Data[28] = c31;
		int c32 = src[11]; flip.B.Data[24] = c32; norm.B.Data[24] = c32;
		int c33 = src[15]; flip.B.Data[28] = c33; norm.B.Data[28] = c33;
	}

	BlockStateAlpha s;

	int err;
	{
		int c0 = output[0];

		int f = output[3];

		s.mode = f & 3;

		if ((f & 2) == 0)
		{
			s.a = (BYTE)ExpandColor4(c0 >> 4);
			s.b = (BYTE)ExpandColor4(c0 & 0xF);
		}
		else
		{
			s.a = (BYTE)ExpandColor5(c0 >> 3);
			s.b = (BYTE)ExpandColor5((c0 >> 3) + (int(uint32_t(c0) << 29) >> 29));
		}

		s.qa = (f >> 5) & 7;
		s.qb = (f >> 2) & 7;

		uint32_t way = BSWAP(*(const uint32_t*)&output[4]);

		if ((f & 1) == 0)
		{
			err = MeasureHalfAlpha(norm.A.Data, norm.A.Data + 1 * 4, s.a, s.qa, way, way >> 4, 2 * 4, 1);
			err += MeasureHalfAlpha(norm.B.Data, norm.B.Data + 1 * 4, s.b, s.qb, way >> 8, way >> 12, 2 * 4, 1);
		}
		else
		{
			err = MeasureHalfAlpha(flip.A.Data, flip.A.Data + 4 * 4, s.a, s.qa, way, way >> 1, 1 * 4, 4);
			err += MeasureHalfAlpha(flip.B.Data, flip.B.Data + 4 * 4, s.b, s.qb, way >> 2, way >> 3, 1 * 4, 4);
		}
	}

	norm.A.Count = 8;
	norm.B.Count = 8;

	flip.A.Count = 8;
	flip.B.Count = 8;

	if (err > 0)
	{
		int err_norm44 = CompressBlockAlpha44(s, norm, err, 0);
		if (err > err_norm44)
			err = err_norm44;

		if (err > 0)
		{
			int err_flip44 = CompressBlockAlpha44(s, flip, err, 1);
			if (err > err_flip44)
				err = err_flip44;

			if (err > 0)
			{
				int err_norm53 = CompressBlockAlpha53(s, norm, err, 2);
				if (err > err_norm53)
					err = err_norm53;

				if (err > 0)
				{
					int err_flip53 = CompressBlockAlpha53(s, flip, err, 3);
					if (err > err_flip53)
						err = err_flip53;
				}
			}
		}
	}

	double ssim;
	{
		uint32_t index = 0;

		bool f = (s.mode & 1) != 0;
		ssim = ComputeTableAlpha(f ? flip.A : norm.A, s.a, s.qa, index, f ? 0xD951C840u : 0x73625140u);
		ssim += ComputeTableAlpha(f ? flip.B : norm.B, s.b, s.qb, index, f ? 0xD951C840u + 0x22222222u : 0x73625140u + 0x88888888u);

		BYTE c = (s.mode & 2) ?
			(BYTE)((s.a & 0xF8) ^ (((s.b >> 3) - (s.a >> 3)) & 7)) :
			(BYTE)((s.a & 0xF0) ^ (s.b & 0x0F));

		output[0] = c;
		output[1] = c;
		output[2] = c;

		output[3] = (BYTE)((s.qa << 5) ^ (s.qb << 2) ^ s.mode);

		*(uint32_t*)&output[4] = BSWAP(index);
	}

	return _mm_unpacklo_epi64(_mm_cvtsi32_si128(err), _mm_castpd_si128(_mm_load_sd(&ssim)));
}


static __forceinline void DecompressHalfColor(int pL[4], int pH[4], int color, int q, uint32_t data, int shift)
{
	static const __declspec(align(16)) int g_delta[8][2] =
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

	static const __declspec(align(16)) int g_mask[16][4] =
	{
		{ 0, 0, 0, 0 },{ -1, 0, 0, 0 },{ 0, -1, 0, 0 },{ -1, -1, 0, 0 },
		{ 0, 0, -1, 0 },{ -1, 0, -1, 0 },{ 0, -1, -1, 0 },{ -1, -1, -1, 0 },
		{ 0, 0, 0, -1 },{ -1, 0, 0, -1 },{ 0, -1, 0, -1 },{ -1, -1, 0, -1 },
		{ 0, 0, -1, -1 },{ -1, 0, -1, -1 },{ 0, -1, -1, -1 },{ -1, -1, -1, -1 }
	};

	__m128i mt0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_delta[q][0]), 0);
	__m128i mt10 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_delta[q][1]), 0);

	__m128i mMaskL = _mm_load_si128((const __m128i*)&((const BYTE*)g_mask)[(shift < 4 ? data << (4 - shift) : data >> (shift - 4)) & 0xF0]);
	__m128i mMaskH = _mm_load_si128((const __m128i*)&((const BYTE*)g_mask)[(data >> (shift + 4 - 4)) & 0xF0]);

	__m128i mtL = _mm_xor_si128(_mm_and_si128(mMaskL, mt10), mt0);
	__m128i mtH = _mm_xor_si128(_mm_and_si128(mMaskH, mt10), mt0);

	mMaskL = _mm_load_si128((const __m128i*)&((const BYTE*)g_mask)[(data >> (shift + 16 + 0 - 4)) & 0xF0]);
	mMaskH = _mm_load_si128((const __m128i*)&((const BYTE*)g_mask)[(data >> (shift + 16 + 4 - 4)) & 0xF0]);

	__m128i mc = _mm_shuffle_epi32(_mm_cvtsi32_si128(color), 0);

	__m128i mcL = _mm_or_si128(_mm_and_si128(mMaskL, _mm_subs_epu8(mc, mtL)), _mm_andnot_si128(mMaskL, _mm_adds_epu8(mc, mtL)));
	__m128i mcH = _mm_or_si128(_mm_and_si128(mMaskH, _mm_subs_epu8(mc, mtH)), _mm_andnot_si128(mMaskH, _mm_adds_epu8(mc, mtH)));

	_mm_storeu_si128((__m128i*)pL, mcL);
	_mm_storeu_si128((__m128i*)pH, mcH);
}

static __forceinline void DecompressBlockColor(const BYTE input[8], BYTE* __restrict cell)
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
		b = (((((c & 0x070707) ^ 0x242424) - 0x040404) << 3) + a) & 0xF8F8F8;

		a |= (a >> 5) & 0x070707;
		b |= (b >> 5) & 0x070707;
	}

	a |= 0xFFu << 24;
	b |= 0xFFu << 24;

	uint32_t way = BSWAP(*(const uint32_t*)&input[4]);

	if ((c & (1 << 24)) == 0)
	{
		__declspec(align(16)) int buffer[4][4];

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
		_mm_storeu_si128((__m128i*)(cell + Stride), _mm_unpackhi_epi64(tmp0, tmp2));

		cell += Stride + Stride;

		_mm_storeu_si128((__m128i*)cell, _mm_unpacklo_epi64(tmp1, tmp3));
		_mm_storeu_si128((__m128i*)(cell + Stride), _mm_unpackhi_epi64(tmp1, tmp3));
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
		DecompressHalfColor((int*)cell, (int*)(cell + Stride), a, qa, way, 0);

		cell += Stride + Stride;

		int qb = (c >> (2 + 24)) & 7;
		DecompressHalfColor((int*)cell, (int*)(cell + Stride), b, qb, way, 8);
	}
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

static __forceinline void SortNodes10(Node nodes[0x10 + 1], int water)
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

static __forceinline void SortNodes20(Node nodes[0x20 + 1], int water)
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


static __forceinline __m128i load_color_BGRA(const BYTE color[4])
{
	__m128i margb = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int*)color));

	return _mm_shuffle_epi32(margb, _MM_SHUFFLE(3, 0, 2, 1));
}

static __forceinline __m128i load_color_GRB(const BYTE color[4])
{
	return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int*)color));
}

static __forceinline __m128i load_color_GR(const BYTE color[2])
{
	__m128i mrg = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(uint16_t*)color));

	return _mm_unpacklo_epi64(mrg, mrg);
}

static __forceinline int ComputeErrorGRB(const Half& half, const BYTE color[4], int water, int q)
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

static __forceinline int ComputeErrorGR(const Half& half, const BYTE color[2], int water, int q)
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
	BYTE a[4], b[4];
	int qa, qb, mode;
};

struct GuessStateColor
{
	__declspec(align(16)) Node node0[0x12], node1[0x12], node2[0x12];

	int stop;

	__forceinline GuessStateColor()
	{
	}

	__forceinline void Init(const Half& half, int water, int q)
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

	__forceinline void Sort(int water)
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

	__declspec(align(16)) Node node0[0x22], node1[0x22], node2[0x22];

	int ErrorsG[0x20];
	int ErrorsGR[0x20 * 0x20];
	bool LazyGR[0x20 * 0x20];
	int ErrorsGRB[0x20 * 0x20 * 0x20];

	int swap0[0x20], swap1[0x20], swap2[0x20];
	int part0[0x20], part1[0x20], part2[0x20];

	__forceinline AdjustStateColor()
	{
	}

	__forceinline void Init(const Half& half, int water, int q)
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

	__forceinline void Sort(int water)
	{
		if (flag_sort)
			return;
		flag_sort = true;

		SortNodes20(node0, water - node1[0x20].Error - node2[0x20].Error);
		SortNodes20(node1, water - node0[0x20].Error - node2[0x20].Error);
		SortNodes20(node2, water - node0[0x20].Error - node1[0x20].Error);
	}

	__declspec(noinline) void DoWalk(const Half& half, int water, int q)
	{
		BYTE c[4];

		int min01 = water - node2[0x20].Error;

		for (int c0 = 0; c0 < node0[0x20].Color; c0++)
		{
			int e0 = node0[c0].Error;
			if (e0 + node1[0x20].Error + node2[0x20].Error >= water)
				break;

			c[0] = (BYTE)node0[c0].Color;

			int min1 = water - node2[0x20].Error;

			for (int c1 = 0; c1 < node1[0x20].Color; c1++)
			{
				int e1 = node1[c1].Error + e0;
				if (e1 + node2[0x20].Error >= water)
					break;

				c[1] = (BYTE)node1[c1].Color;

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

	__forceinline void Walk(const Half& half, int water, int q)
	{
		if (flag_error)
			return;
		flag_error = true;

		DoWalk(half, water, q);
	}

	__declspec(noinline) void DoBottom(const Half& half, int water, int q)
	{
		BYTE c[4];

		int blue_max = 0xFF;
		int minimum = Min(water, stop - node2[0x20].Error + (blue_max * blue_max * kBlue) * half.Count);

		for (int c0 = 0; c0 < node0[0x20].Color; c0++)
		{
			int e0 = node0[c0].Error;
			if (e0 + node1[0x20].Error + node2[0x20].Error >= minimum)
				break;

			if (ErrorsG[c0] + node2[0x20].Error >= minimum)
				continue;

			c[0] = (BYTE)node0[c0].Color;

			for (int c1 = 0; c1 < node1[0x20].Color; c1++)
			{
				int e1 = node1[c1].Error + e0;
				if (e1 + node2[0x20].Error >= minimum)
					break;

				int originGR = (c0 << 5) + c1;
				e1 = ErrorsGR[originGR];
				if (e1 + node2[0x20].Error >= minimum)
					continue;

				c[1] = (BYTE)node1[c1].Color;

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

					c[2] = (BYTE)node2[c2].Color;

					e2 = ComputeErrorGRB(half, c, water, q);
					ErrorsGRB[c2 + origin] = e2;

					if (minimum > e2)
						minimum = e2;
				}
			}
		}

		stop = minimum;
	}

	__forceinline void Bottom(const Half& half, int water, int q)
	{
		if (flag_minimum)
			return;
		flag_minimum = true;

		DoBottom(half, water, q);
	}

	__forceinline void Index()
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

	__forceinline AdjustStateColorGroup()
	{
	}

	__forceinline void Init(const Half& half, int water)
	{
		for (int q = 0; q < 8; q++)
		{
			S[q].Init(half, water, q);
		}
	}

	__forceinline int Best(int water) const
	{
		int best = water;

		for (int q = 0; q < 8; q++)
		{
			best = Min(best, S[q].stop);
		}

		return best;
	}
};

static __forceinline double ComputeTableColor(const Half& half, const BYTE color[4], int q, uint32_t& index)
{
	int halfSize = half.Count;
	if (halfSize <= 0)
		return 1.0;

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

	for (int k = 0, i = 0; k < halfSize; k++, i += 4)
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

	for (int i = 0; i < halfSize; i++)
	{
		int k = 0;
		while ((ways[i] & (1 << k)) == 0) k++;
		loops[i] = k;
	}

	double best = -1000.1;
	uint32_t codes = 0;

	for (;; )
	{
		SSIM_INIT();

		for (int i = 0; i < halfSize; i++)
		{
			__m128i mt = _mm_load_si128(&vals[loops[i]]);

			__m128i mb = _mm_load_si128((const __m128i*)&half.Data[i << 2]);

			SSIM_UPDATE(mt, mb);
		}

		SSIM_CLOSE();

		SSIM_FINAL(mssim_rg);
		SSIM_OTHER();
		SSIM_FINAL(mssim_b);

		double ssim =
			_mm_cvtsd_f64(mssim_rg) * kGreen +
			_mm_cvtsd_f64(_mm_unpackhi_pd(mssim_rg, mssim_rg)) * kRed +
			_mm_cvtsd_f64(mssim_b) * kBlue;

		if (best < ssim)
		{
			best = ssim;

			uint32_t v = 0;
			for (int j = 0; j < halfSize; j++)
			{
				v |= ((uint32_t)loops[j]) << (j + j);
			}
			codes = v;

			if (best >= 1000.0)
				break;
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

			i++;
			if (i >= halfSize)
				break;
		}
		if (i >= halfSize)
			break;
	}

	for (int i = halfSize, j = 0; --i >= 0; j++)
	{
		int shift = half.Shift[j];

		uint32_t code = ((codes & 2u) << (16 - 1)) | (codes & 1u);

		codes >>= 2;

		index |= code << shift;
	}

	return best * (1.0 / 1000.0);
}

static __forceinline int GuessColor4(const Half& half, BYTE color[4], int water, int& table)
{
	GuessStateColor S;

	for (int q = 0; q < 8; q++)
	{
		S.Init(half, water, q);

		if (S.stop >= water)
			continue;

		S.Sort(water);

		BYTE c[4];

		for (int c0 = 0; c0 < S.node0[0x10].Color; c0++)
		{
			int e0 = S.node0[c0].Error;
			if (e0 + S.node1[0x10].Error + S.node2[0x10].Error >= water)
				break;

			c[0] = (BYTE)S.node0[c0].Color;

			for (int c1 = 0; c1 < S.node1[0x10].Color; c1++)
			{
				int e1 = S.node1[c1].Error + e0;
				if (e1 + S.node2[0x10].Error >= water)
					break;

				c[1] = (BYTE)S.node1[c1].Color;

				e1 = ComputeErrorGR(half, c, water - S.node2[0x10].Error, q);
				if (e1 + S.node2[0x10].Error >= water)
					continue;

				for (int c2 = 0; c2 < S.node2[0x10].Color; c2++)
				{
					int e2 = S.node2[c2].Error + e1;
					if (e2 >= water)
						break;

					c[2] = (BYTE)S.node2[c2].Color;

					e2 = ComputeErrorGRB(half, c, water, q);

					if (water > e2)
					{
						water = e2;

						memcpy(color, c, 4);

						table = q;

						if (water <= e1 + S.node2[0x10].Error)
							break;
					}
				}

				if (water <= e0 + S.node1[0x10].Error + S.node2[0x10].Error)
					break;
			}
		}
	}

	return water;
}

static int CompressBlockColor44(BlockStateColor &s, const Elem& elem, int water, int mode)
{
	BYTE a[4], b[4];
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

static __forceinline int AdjustColors53(const Elem& elem, BYTE color[4], BYTE other[4], int water, int qa, int qb, AdjustStateColor& SA, AdjustStateColor& SB, int bestA, int bestB)
{
	BYTE a[4], b[4];

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

		a[0] = (BYTE)SA.node0[a0].Color;

		int Id0 = a[0] >> 3;
		int Ld0 = Max(Id0 - 4, 0);
		int Hd0 = Min(Id0 + 3, 31);

		if ((SB.flag0 & (1u << Id0)) == 0)
		{
			SB.flag0 |= (1u << Id0);

			int min0 = kUnknownError;
			for (int d0 = Ld0; d0 <= Hd0; d0++)
			{
				int b0 = SB.swap0[d0];
				if (b0 < 0)
					continue;

				min0 = Min(min0, SB.node0[b0].Error);
			}
			SB.part0[Id0] = min0;
		}
		int min0 = SB.part0[Id0];

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

			a[1] = (BYTE)SA.node1[a1].Color;

			int Id1 = a[1] >> 3;
			int Ld1 = Max(Id1 - 4, 0);
			int Hd1 = Min(Id1 + 3, 31);

			if ((SB.flag1 & (1u << Id1)) == 0)
			{
				SB.flag1 |= (1u << Id1);

				int min1 = kUnknownError;
				for (int d1 = Ld1; d1 <= Hd1; d1++)
				{
					int b1 = SB.swap1[d1];
					if (b1 < 0)
						continue;

					min1 = Min(min1, SB.node1[b1].Error);
				}
				SB.part1[Id1] = min1;
			}
			int min1 = SB.part1[Id1];

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

				a[2] = (BYTE)SA.node2[a2].Color;

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
				int Ld2 = Max(Id2 - 4, 0);
				int Hd2 = Min(Id2 + 3, 31);

				if ((SB.flag2 & (1u << Id2)) == 0)
				{
					SB.flag2 |= (1u << Id2);

					int min2 = kUnknownError;
					for (int d2 = Ld2; d2 <= Hd2; d2++)
					{
						int b2 = SB.swap2[d2];
						if (b2 < 0)
							continue;

						min2 = Min(min2, SB.node2[b2].Error);
					}
					SB.part2[Id2] = min2;
				}
				int min2 = SB.part2[Id2];

				if (e2 + min0 + min1 + min2 >= water)
					continue;

				for (int d0 = Ld0; d0 <= Hd0; d0++)
				{
					int b0 = SB.swap0[d0];
					if (b0 < 0)
						continue;

					int e3 = SB.node0[b0].Error + e2;
					if (e3 + min1 + min2 >= water)
						continue;

					if (e2 + SB.ErrorsG[b0] + min2 >= water)
						continue;

					for (int d1 = Ld1; d1 <= Hd1; d1++)
					{
						int b1 = SB.swap1[d1];
						if (b1 < 0)
							continue;

						int e4 = SB.node1[b1].Error + e3;
						if (e4 + min2 >= water)
							continue;

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

						for (int d2 = Ld2; d2 <= Hd2; d2++)
						{
							int b2 = SB.swap2[d2];
							if (b2 < 0)
								continue;

							int e5 = SB.node2[b2].Error + e4;
							if (e5 >= water)
								continue;

							int eb = SB.ErrorsGRB[b2 + b_origin];
							if (eb < 0)
							{
								b[0] = (BYTE)SB.node0[b0].Color;
								b[1] = (BYTE)SB.node1[b1].Color;
								b[2] = (BYTE)SB.node2[b2].Color;

								eb = ComputeErrorGRB(elem.B, b, water - bestA, qb);
								SB.ErrorsGRB[b2 + b_origin] = eb;
							}

							e5 = eb + e2;

							if (water > e5)
							{
								water = e5;
								most = water - SB.stop;

								memcpy(color, a, 4);

								other[0] = (BYTE)SB.node0[b0].Color;
								other[1] = (BYTE)SB.node1[b1].Color;
								other[2] = (BYTE)SB.node2[b2].Color;

								if (water <= e4 + min2)
									break;
							}
						}

						if (water <= e2 + SB.ErrorsG[b0] + min2)
							break;
					}

					if (water <= e2 + min0 + min1 + min2)
						break;
				}

				if (most <= e1 + SA.node2[0x20].Error)
					break;
			}

			if (most <= SA.ErrorsG[a0] + SA.node2[0x20].Error)
				break;
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

static __forceinline int MeasureHalfColor(const int pL[4 * 4], const int pH[4 * 4], const BYTE color[4], int q, uint32_t dataL, uint32_t dataH, int slice, int shift)
{
	__m128i mc = load_color_GRB(color);

	__m128i mt0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_table[q][0]), 0);
	__m128i mt1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(g_table[q][1]), 0);

	__m128i mt2 = _mm_subs_epu8(mc, mt0);
	__m128i mt3 = _mm_subs_epu8(mc, mt1);

	mt0 = _mm_adds_epu8(mc, mt0);
	mt1 = _mm_adds_epu8(mc, mt1);

	__m128i sum = _mm_setzero_si128();

	for (int i = 0; i < 4; i++)
	{
		__m128i m = _mm_load_si128((const __m128i*)pL);

		__m128i mtU = mt0;
		__m128i mtV = mt1;
		if ((dataL & 0x10000u) != 0)
		{
			mtU = mt2;
			mtV = mt3;
		}

		__m128i ma = _mm_shuffle_epi32(m, _MM_SHUFFLE(3, 3, 3, 3));

		__m128i mt = mtU;
		if ((dataL & 1u) != 0)
		{
			mt = mtV;
		}

		ma = _mm_cmpgt_epi16(ma, _mm_setzero_si128());
		m = _mm_sub_epi16(m, mt);
		m = _mm_mullo_epi16(m, m);
		m = _mm_and_si128(m, ma);

		sum = _mm_add_epi32(sum, m);

		pL += slice;

		dataL >>= shift;
	}

	for (int i = 0; i < 4; i++)
	{
		__m128i m = _mm_load_si128((const __m128i*)pH);

		__m128i mtU = mt0;
		__m128i mtV = mt1;
		if ((dataH & 0x10000u) != 0)
		{
			mtU = mt2;
			mtV = mt3;
		}

		__m128i ma = _mm_shuffle_epi32(m, _MM_SHUFFLE(3, 3, 3, 3));

		__m128i mt = mtU;
		if ((dataH & 1u) != 0)
		{
			mt = mtV;
		}

		ma = _mm_cmpgt_epi16(ma, _mm_setzero_si128());
		m = _mm_sub_epi16(m, mt);
		m = _mm_mullo_epi16(m, m);
		m = _mm_and_si128(m, ma);

		sum = _mm_add_epi32(sum, m);

		pH += slice;

		dataH >>= shift;
	}

	__m128i mgrb = _mm_loadl_epi64((const __m128i*)g_GRB_I16);
	mgrb = _mm_cvtepi16_epi32(mgrb);

	sum = _mm_mullo_epi32(sum, mgrb);

	sum = _mm_hadd_epi32(sum, sum);
	sum = _mm_hadd_epi32(sum, sum);

	return _mm_cvtsi128_si32(sum);
}

static __forceinline void FilterPixelsColor(Half& half, uint32_t order)
{
	int w = 0;

	for (int i = 0; i < 8 * 4; i += 4)
	{
		__m128i m = _mm_load_si128((const __m128i*)&half.Data[i]);

		int a = _mm_extract_epi16(m, 6);

		_mm_store_si128((__m128i*)&half.Data[w * 4], m);

		half.Shift[w] = order & 0xF;

		order >>= 4;

		w += (a != 0) ? 1 : 0;
	}

	half.Count = w;
}

static __m128i CompressBlockColor(BYTE output[8], const BYTE* __restrict cell)
{
	Elem norm, flip;

	{
		const BYTE* src = cell;

		__m128i c00 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(flip.A.Data + 0), c00); _mm_store_si128((__m128i*)(norm.A.Data + 0), c00);
		__m128i c01 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(flip.A.Data + 4), c01); _mm_store_si128((__m128i*)(norm.A.Data + 4), c01);
		__m128i c02 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(flip.A.Data + 8), c02); _mm_store_si128((__m128i*)(norm.B.Data + 0), c02);
		__m128i c03 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(flip.A.Data + 12), c03); _mm_store_si128((__m128i*)(norm.B.Data + 4), c03);

		src += Stride;

		__m128i c10 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(flip.A.Data + 16), c10); _mm_store_si128((__m128i*)(norm.A.Data + 8), c10);
		__m128i c11 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(flip.A.Data + 20), c11); _mm_store_si128((__m128i*)(norm.A.Data + 12), c11);
		__m128i c12 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(flip.A.Data + 24), c12); _mm_store_si128((__m128i*)(norm.B.Data + 8), c12);
		__m128i c13 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(flip.A.Data + 28), c13); _mm_store_si128((__m128i*)(norm.B.Data + 12), c13);

		src += Stride;

		__m128i c20 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(flip.B.Data + 0), c20); _mm_store_si128((__m128i*)(norm.A.Data + 16), c20);
		__m128i c21 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(flip.B.Data + 4), c21); _mm_store_si128((__m128i*)(norm.A.Data + 20), c21);
		__m128i c22 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(flip.B.Data + 8), c22); _mm_store_si128((__m128i*)(norm.B.Data + 16), c22);
		__m128i c23 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(flip.B.Data + 12), c23); _mm_store_si128((__m128i*)(norm.B.Data + 20), c23);

		src += Stride;

		__m128i c30 = load_color_BGRA(src + 0); _mm_store_si128((__m128i*)(flip.B.Data + 16), c30); _mm_store_si128((__m128i*)(norm.A.Data + 24), c30);
		__m128i c31 = load_color_BGRA(src + 4); _mm_store_si128((__m128i*)(flip.B.Data + 20), c31); _mm_store_si128((__m128i*)(norm.A.Data + 28), c31);
		__m128i c32 = load_color_BGRA(src + 8); _mm_store_si128((__m128i*)(flip.B.Data + 24), c32); _mm_store_si128((__m128i*)(norm.B.Data + 24), c32);
		__m128i c33 = load_color_BGRA(src + 12); _mm_store_si128((__m128i*)(flip.B.Data + 28), c33); _mm_store_si128((__m128i*)(norm.B.Data + 28), c33);
	}

	BlockStateColor s;

	int err;
	{
		int c1 = output[0];
		int c0 = output[1];
		int c2 = output[2];

		int f = output[3];

		s.mode = f & 3;

		if ((f & 2) == 0)
		{
			s.a[0] = (BYTE)ExpandColor4(c0 >> 4);
			s.a[1] = (BYTE)ExpandColor4(c1 >> 4);
			s.a[2] = (BYTE)ExpandColor4(c2 >> 4);

			s.b[0] = (BYTE)ExpandColor4(c0 & 0xF);
			s.b[1] = (BYTE)ExpandColor4(c1 & 0xF);
			s.b[2] = (BYTE)ExpandColor4(c2 & 0xF);
		}
		else
		{
			s.a[0] = (BYTE)ExpandColor5(c0 >> 3);
			s.a[1] = (BYTE)ExpandColor5(c1 >> 3);
			s.a[2] = (BYTE)ExpandColor5(c2 >> 3);

			s.b[0] = (BYTE)ExpandColor5((c0 >> 3) + (int(uint32_t(c0) << 29) >> 29));
			s.b[1] = (BYTE)ExpandColor5((c1 >> 3) + (int(uint32_t(c1) << 29) >> 29));
			s.b[2] = (BYTE)ExpandColor5((c2 >> 3) + (int(uint32_t(c2) << 29) >> 29));
		}

		s.qa = (f >> 5) & 7;
		s.qb = (f >> 2) & 7;

		uint32_t way = BSWAP(*(const uint32_t*)&output[4]);

		if ((f & 1) == 0)
		{
			err = MeasureHalfColor(norm.A.Data, norm.A.Data + 1 * 4, s.a, s.qa, way, way >> 4, 2 * 4, 1);
			err += MeasureHalfColor(norm.B.Data, norm.B.Data + 1 * 4, s.b, s.qb, way >> 8, way >> 12, 2 * 4, 1);
		}
		else
		{
			err = MeasureHalfColor(flip.A.Data, flip.A.Data + 4 * 4, s.a, s.qa, way, way >> 1, 1 * 4, 4);
			err += MeasureHalfColor(flip.B.Data, flip.B.Data + 4 * 4, s.b, s.qb, way >> 2, way >> 3, 1 * 4, 4);
		}
	}

	FilterPixelsColor(norm.A, 0x73625140u);
	FilterPixelsColor(norm.B, 0x73625140u + 0x88888888u);

	FilterPixelsColor(flip.A, 0xD951C840u);
	FilterPixelsColor(flip.B, 0xD951C840u + 0x22222222u);

	if (err > 0)
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

	double ssim;
	{
		uint32_t index = 0;

		bool f = (s.mode & 1) != 0;
		ssim = ComputeTableColor(f ? flip.A : norm.A, s.a, s.qa, index);
		ssim += ComputeTableColor(f ? flip.B : norm.B, s.b, s.qb, index);

		if (s.mode & 2)
		{
			output[0] = (BYTE)((s.a[1] & 0xF8) ^ (((s.b[1] >> 3) - (s.a[1] >> 3)) & 7));
			output[1] = (BYTE)((s.a[0] & 0xF8) ^ (((s.b[0] >> 3) - (s.a[0] >> 3)) & 7));
			output[2] = (BYTE)((s.a[2] & 0xF8) ^ (((s.b[2] >> 3) - (s.a[2] >> 3)) & 7));
		}
		else
		{
			output[0] = (s.a[1] & 0xF0) ^ (s.b[1] & 0x0F);
			output[1] = (s.a[0] & 0xF0) ^ (s.b[0] & 0x0F);
			output[2] = (s.a[2] & 0xF0) ^ (s.b[2] & 0x0F);
		}

		output[3] = (BYTE)((s.qa << 5) ^ (s.qb << 2) ^ s.mode);

		*(uint32_t*)&output[4] = BSWAP(index);
	}

	return _mm_unpacklo_epi64(_mm_cvtsi32_si128(err), _mm_castpd_si128(_mm_load_sd(&ssim)));
}


enum class PackMode
{
	CompressAlpha, DecompressAlpha,
	CompressColor, DecompressColor
};

class Worker
{
public:
	class Item
	{
	public:
		BYTE* _Output;
		BYTE* _Cell;

		Item()
		{
		}

		Item(BYTE* output, BYTE* cell)
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
	CRITICAL_SECTION _Sync;
	HANDLE _Done;

	Job* _First;
	Job* _Last;

	int64_t _mse;
	double _ssim;

	std::atomic_int _Running;

	PackMode _Mode;

public:
	Worker()
	{
		if (!InitializeCriticalSectionAndSpinCount(&_Sync, 1000))
			throw std::runtime_error("init");

		_Done = CreateEvent(NULL, FALSE, FALSE, NULL);
		if (_Done == NULL)
			throw std::runtime_error("init");

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

		if (_Done != nullptr)
			CloseHandle(_Done), _Done = nullptr;

		DeleteCriticalSection(&_Sync);
	}

	void Lock()
	{
		EnterCriticalSection(&_Sync);
	}

	void UnLock()
	{
		LeaveCriticalSection(&_Sync);
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

	static DWORD WINAPI ThreadProc(LPVOID lpParameter)
	{
		Worker* worker = static_cast<Worker*>(lpParameter);

		int64_t mse = 0;
		double ssim = 0;

		for (Job* job; (job = worker->Take()) != nullptr;)
		{
			switch (worker->_Mode)
			{
			case PackMode::CompressAlpha:
				while (Item* item = job->Take())
				{
					__m128i v = CompressBlockAlpha(item->_Output, item->_Cell);
					mse += _mm_cvtsi128_si64(v);
					ssim += _mm_cvtsd_f64(_mm_castsi128_pd(_mm_unpackhi_epi64(v, v)));
				}
				break;

			case PackMode::DecompressAlpha:
				while (Item* item = job->Take())
				{
					DecompressBlockAlpha(item->_Output, item->_Cell);
				}
				break;

			case PackMode::CompressColor:
				while (Item* item = job->Take())
				{
					__m128i v = CompressBlockColor(item->_Output, item->_Cell);
					mse += _mm_cvtsi128_si64(v);
					ssim += _mm_cvtsd_f64(_mm_castsi128_pd(_mm_unpackhi_epi64(v, v)));
				}
				break;

			case PackMode::DecompressColor:
				while (Item* item = job->Take())
				{
					DecompressBlockColor(item->_Output, item->_Cell);
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

		if (worker->_Running <= 0)
		{
			SetEvent(worker->_Done);
		}

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
			CreateThread(NULL, WorkerThreadStackSize, ThreadProc, this, 0, NULL);
		}

		WaitForSingleObject(_Done, INFINITE);

		return _mm_unpacklo_epi64(_mm_cvtsi64_si128(_mse), _mm_castpd_si128(_mm_load_sd((double*)&_ssim)));
	}
};

static bool ReadImage(const char* src_name, BYTE* &pixels, int &width, int &height, bool flip)
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

			pixels = new BYTE[height * stride];

			BYTE* w = pixels;
			for (int y = 0; y < height; y++)
			{
				const BYTE* r = (const BYTE*)data.Scan0 + (flip ? height - 1 - y : y) * data.Stride;
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

static void WriteImage(const char* dst_name, const BYTE* pixels, int w, int h, bool flip)
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
				memcpy((BYTE*)data.Scan0 + (flip ? h - 1 - y : y) * data.Stride, pixels + y * w * 4, w * 4);
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
				Gdiplus::ImageCodecInfo* pArray = (Gdiplus::ImageCodecInfo*)new BYTE[size];
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

				delete[](BYTE*)pArray;
			}
		}
		if (ok)
		{
			std::wstring wide_dst_name;
			wide_dst_name.resize(std::mbstowcs(nullptr, dst_name, MAX_PATH));
			std::mbstowcs(&wide_dst_name.front(), dst_name, MAX_PATH);

			ok = (bitmap.Save(wide_dst_name.c_str(), &format) == Gdiplus::Ok);
		}

		printf(ok ? "Saved %s\n" : "Lost %s\n", dst_name);
	}
	Gdiplus::GdiplusShutdown(gdiplusToken);
}

static void LoadEtc1(const char* name, BYTE* buffer, int size)
{
	HANDLE file = CreateFile(name, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, 0, NULL);
	if (file != INVALID_HANDLE_VALUE)
	{
		DWORD transferred;
		BOOL ok = ReadFile(file, buffer, size, &transferred, NULL);

		CloseHandle(file);

		if (ok)
		{
			printf("Loaded %s\n", name);
		}
	}
}

static void SaveEtc1(const char* name, const BYTE* buffer, int size)
{
	HANDLE file = CreateFile(name, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	if (file != INVALID_HANDLE_VALUE)
	{
		DWORD transferred;
		WriteFile(file, buffer, size, &transferred, NULL);

		CloseHandle(file);

		printf("Saved %s\n", name);
	}
}

static __m128i PackTexture(BYTE* dst_etc1, BYTE* src_bgra, int src_w, int src_h, PackMode mode)
{
	auto start = std::chrono::high_resolution_clock::now();

	int64_t mse = 0;
	double ssim = 0;

	{
		BYTE* output = dst_etc1;

		Worker* worker = new Worker();

		Worker::Job* job = new Worker::Job();

		for (int y = 0; y < src_h; y += 4)
		{
			BYTE* cell = src_bgra + y * Stride;

			for (int x = 0; x < src_w; x += 4)
			{
				if (job->Add(Worker::Item(output, cell)))
				{
					worker->Add(job);

					job = new Worker::Job();
				}

				output += 8;
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

	if ((mode == PackMode::CompressAlpha) || (mode == PackMode::CompressColor))
	{
		int n = (src_h * src_w) >> 4;

		int span = Max((int)std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count(), 1);

		printf("Compressed %d blocks, elapsed %i ms, %i bps\n", n, span, (int)(n * 1000LL / span));
	}

	return _mm_unpacklo_epi64(_mm_cvtsi64_si128(mse), _mm_castpd_si128(_mm_load_sd(&ssim)));
}

static __forceinline void OutlineAlpha(BYTE* src_bgra, int src_w, int src_h, int radius)
{
	if (radius <= 0)
		return;

	int full_w = 1 + radius + src_w + radius;
	int full_h = 1 + radius + src_h + radius;

	BYTE* data = new BYTE[full_h * full_w];
	memset(data, 0, full_h * full_w);

	for (int y = 0; y < src_h; y++)
	{
		const BYTE* r = &src_bgra[y * Stride + 3];
		BYTE* w = &data[(y + radius + 1) * full_w + (radius + 1)];

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
		BYTE* w = &src_bgra[y * Stride + 3];
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

int EtcMainWithArgs(const std::vector<std::string>& args)
{
	bool flip = true;
	int border = 1;

	const char* src_name = nullptr;
	const char* dst_color_name = nullptr;
	const char* dst_alpha_name = nullptr;
	const char* result_name = nullptr;

	for (int i = 0, n = (int)args.size(); i < n; i++)
	{
		const char* arg = args[i].c_str();

		if (arg[0] == '/')
		{
			if (strcmp(arg, "/retina") == 0)
			{
				border = 2;
			}
			else if (strcmp(arg, "/debug") == 0)
			{
				if (++i < n)
				{
					result_name = args[i].c_str();
				}
			}
			else
			{
				printf("Unknown %s\n", arg);
			}

			continue;
		}

		if (src_name == nullptr)
		{
			src_name = arg;
		}
		else if (dst_color_name == nullptr)
		{
			dst_color_name = arg;
		}
		else if (dst_alpha_name == nullptr)
		{
			dst_alpha_name = arg;
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

	BYTE* src_image_bgra;
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

	BYTE* src_texture_bgra = new BYTE[src_texture_h * src_texture_stride];

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

	printf("Image %dx%d, Texture %dx%d\n\n", src_image_w, src_image_h, src_texture_w, src_texture_h);

	Stride = src_texture_stride;

	BYTE* dst_texture_bgra = new BYTE[src_texture_h * src_texture_stride];

	int Size = (src_texture_h * src_texture_w) >> 1;

	InitLevelErrors();

	memcpy(dst_texture_bgra, src_texture_bgra, src_texture_h * src_texture_stride);

	if ((dst_alpha_name != nullptr) && dst_alpha_name[0])
	{
		BYTE* dst_alpha_etc1 = new BYTE[Size];
		memset(dst_alpha_etc1, 0, Size);

		LoadEtc1(dst_alpha_name, dst_alpha_etc1, Size);

		__m128i v2 = PackTexture(dst_alpha_etc1, dst_texture_bgra, src_texture_w, src_texture_h, PackMode::CompressAlpha);
		int64_t mse_alpha = _mm_cvtsi128_si64(v2);
		double ssim_alpha = _mm_cvtsd_f64(_mm_castsi128_pd(_mm_unpackhi_epi64(v2, v2)));

		SaveEtc1(dst_alpha_name, dst_alpha_etc1, Size);

		if (mse_alpha > 0)
		{
			printf("Texture A PSNR = %f, SSIM_4x2 = %.8f\n\n",
				10.0 * log((255.0 * 255.0) * (src_texture_h * src_texture_w) / mse_alpha) / log(10.0),
				ssim_alpha * 8.0 / (src_texture_h * src_texture_w));
		}
		else
		{
			printf("Exactly\n\n");
		}

		PackTexture(dst_alpha_etc1, dst_texture_bgra, src_texture_w, src_texture_h, PackMode::DecompressAlpha);

		delete[] dst_alpha_etc1;
	}

	if ((dst_color_name != nullptr) && dst_color_name[0])
	{
		BYTE* dst_texture_color = new BYTE[src_texture_h * src_texture_stride];

		memcpy(dst_texture_color, dst_texture_bgra, src_texture_h * src_texture_stride);

		OutlineAlpha(dst_texture_color, src_texture_w, src_texture_h, border);

		BYTE* dst_color_etc1 = new BYTE[Size];
		memset(dst_color_etc1, 0, Size);

		LoadEtc1(dst_color_name, dst_color_etc1, Size);

		__m128i v2 = PackTexture(dst_color_etc1, dst_texture_color, src_texture_w, src_texture_h, PackMode::CompressColor);
		int64_t mse_color = _mm_cvtsi128_si64(v2);
		double ssim_color = _mm_cvtsd_f64(_mm_castsi128_pd(_mm_unpackhi_epi64(v2, v2)));

		SaveEtc1(dst_color_name, dst_color_etc1, Size);

		if (mse_color > 0)
		{
			printf("Texture RGB wPSNR = %f, wSSIM_4x2 = %.8f\n\n",
				10.0 * log((255.0 * 255.0) * 1000.0 * (src_texture_h * src_texture_w) / mse_color) / log(10.0),
				ssim_color * 8.0 / (src_texture_h * src_texture_w));
		}
		else
		{
			printf("Exactly\n\n");
		}

		PackTexture(dst_color_etc1, dst_texture_color, src_texture_w, src_texture_h, PackMode::DecompressColor);

		size_t delta_dst = dst_texture_bgra - dst_texture_color;

		for (int y = 0; y < src_texture_h; y++)
		{
			BYTE* cell = dst_texture_color + y * src_texture_stride;

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

		delete[] dst_color_etc1;
		delete[] dst_texture_color;
	}

	if ((result_name != nullptr) && result_name[0])
	{
		if ((dst_alpha_name == nullptr) || !dst_alpha_name[0])
		{
			for (int y = 0; y < src_texture_h; y++)
			{
				BYTE* cell = dst_texture_bgra + y * src_texture_stride;

				for (int x = 0; x < src_texture_w; x++)
				{
					*(cell + 3) = 0xFF;

					cell += 4;
				}
			}
		}

		WriteImage(result_name, dst_texture_bgra, src_texture_w, src_texture_h, true);
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
		printf("Usage: EtcCompress [/retina] src [dst_color] [dst_alpha] [/debug result.png]\n");
		return 1;
	}

	std::vector<std::string> args;
	args.reserve(argc);

	for (int i = 1; i < argc; i++)
	{
		args.emplace_back(argv[i]);
	}

	return EtcMainWithArgs(args);
}
