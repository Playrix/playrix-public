#include "ObjectStorageBase.h"
#include "DefaultObjectsStorage.h"

#include <time.h>
#include <functional>
#include <iostream>

using namespace Patterns;

namespace GlobalObjectGetterTests
{

	struct Dummy : public GameGlobalObject
	{
		bool x;
	};
	struct Dummy1 : public GameGlobalObject
	{
		bool x;
	};
	struct Dummy2 : public GameGlobalObject
	{
		bool x;
	};
	struct Dummy3 : public GameGlobalObject
	{
		bool x;
	};
	struct Dummy4 : public GameGlobalObject
	{
		bool x;
	};

	struct First : public GameGlobalObject
	{
		int x;
		virtual void SetX()
		{
			x = rand()%1;
		}
	};

	struct First0 : public GameGlobalObject
	{
		int x;
		void SetX(bool val)
		{
			x = rand() % 1;
		}
	};

	struct First1 : public GameGlobalObject
	{
		int x;
		void SetX(bool val)
		{
			x = rand() % 1;
		}
	};

	struct Second : public GameGlobalObject
	{
		int x;
		void SetX(bool val)
		{
			x = rand() % 1;
		}
	};
	struct Second0 : public GameGlobalObject
	{
		int x;
		void SetX(bool val)
		{
			x = rand() % 1;
		}
	};
	struct Second1 : public GameGlobalObject
	{
		int x;
		void SetX(bool val)
		{
			x = rand() % 1;
		}
	};
	constexpr size_t NUMBER = 15;
	size_t indices[NUMBER];

	First g_first[NUMBER / 3];
	Second g_second[NUMBER / 3];
	First0 g_first_0[NUMBER / 3];
	Second0 g_second_0[NUMBER / 3];
	First1 g_first_1[NUMBER / 3];
	Second1 g_second_1[NUMBER / 3];

	void ConstructIndices()
	{
		for (size_t i = 0; i < NUMBER; ++i)
			indices[i] = rand() % NUMBER;
	}

	void TestGlobalVars()
	{
		constexpr static size_t NUMBER_3 = NUMBER / 3;
		for (size_t i = 0; i < NUMBER_3; ++i)
		{
			g_first[i].SetX();
			g_second[i].SetX(false);
		}
		for (size_t i = 0; i < NUMBER_3; ++i)
		{
			g_first_0[i].SetX(true);
			g_second_0[i].SetX(false);
		}
		for (size_t i = 0; i < NUMBER_3; ++i)
		{
			g_first_1[i].SetX(true);
			g_second_1[i].SetX(false);
		}
	}

	void TestGetter(ObjectStorageBase<GameGlobalObject>& i_getter)
	{
		constexpr static size_t NUMBER_3 = NUMBER / 3;
		for (size_t i = 0; i < NUMBER_3; ++i)
		{
			i_getter.GetGlobalObject<First>()->SetX();
			i_getter.GetGlobalObject<Second>()->SetX(false);
		}
		for (size_t i = 0; i < NUMBER_3; ++i)
		{
			i_getter.GetGlobalObject<First0>()->SetX(true);
			i_getter.GetGlobalObject<Second0>()->SetX(false);
		}
		for (size_t i = 0; i < NUMBER_3; ++i)
		{
			i_getter.GetGlobalObject<First1>()->SetX(true);
			i_getter.GetGlobalObject<Second1>()->SetX(false);
		}
	}

	clock_t Runner(std::function<void()> func, size_t i_nums)
	{
		clock_t begin = clock();
		for (size_t i = 0; i < i_nums; ++i)
		{
			func();
		}
		return clock() - begin;
	}

	void TestReplacement()
	{
		std::cout << "\tTest replacement" << std::endl;

		struct FirstReplace : public First
		{
			virtual void SetX() override
			{
				x = 5;
			}

			virtual void RecalcHashCode() { m_hash_code = typeid(First).hash_code(); }
		};
		DefaultObjectsStorage g_getter;

		// this will be the default in programm
		g_getter.AddGlobalObject<First>();
		// in test we should delete first
		g_getter.RemoveGlobalObject<First>();
		// and add our replacement
		g_getter.AddGlobalObject<FirstReplace>();
		
		g_getter.GetGlobalObject<First>()->SetX();
		std::cout << "Expected \"5\"; Actual is \"" << g_getter.GetGlobalObject<First>()->x <<  "\"" << std::endl << std::endl;
	}

	void Test()
	{
		std::cout << "==========================================================" << std::endl
			<< "\t\tGlobal objects getter tests" << std::endl;
		TestReplacement();

		ConstructIndices();
		DefaultObjectsStorage g_getter;
		{
			constexpr static size_t NUMBER_3 = NUMBER / 3;
			g_getter.AddGlobalObject<First>();
			g_getter.AddGlobalObject<Dummy>();
			g_getter.AddGlobalObject<Second>();
			g_getter.AddGlobalObject<Dummy1>();
			g_getter.AddGlobalObject<First0>();
			g_getter.AddGlobalObject<Dummy2>();
			g_getter.AddGlobalObject<Second0>();
			g_getter.AddGlobalObject<Dummy3>();
			g_getter.AddGlobalObject<First1>();
			g_getter.AddGlobalObject<Dummy4>();
			g_getter.AddGlobalObject<Second1>();
		}

		constexpr static size_t CALL_COUNT = 1000000;
		constexpr static size_t CYCLES = 10;
		clock_t av_gl_obj = 0;
		clock_t av_getter = 0;
		for (size_t i = 0; i < CYCLES; ++i)
		{
			av_gl_obj += Runner(&TestGlobalVars, CALL_COUNT);
			av_getter += Runner([&g_getter]() { TestGetter(g_getter); }, CALL_COUNT);
		}
		av_gl_obj /= CYCLES;
		av_getter /= CYCLES;
		std::cout << "Result for " << CYCLES << " cycles" << std::endl;
		std::cout << "\tGlobal object: " << av_gl_obj << std::endl
			<< "\tGetter: " << av_getter << std::endl;
	}

} // GlobalObjectGetterTests