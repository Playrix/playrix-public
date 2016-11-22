#pragma once

#include <typeinfo>

namespace Patterns
{
	// Базовый класс-метка для глобальных объектов
	class GlobalObjectBase
	{
	protected:
		size_t m_hash_code;

	public:		
		virtual ~GlobalObjectBase() {}

		size_t GetTypeHashCode() const { return m_hash_code; }
		virtual void RecalcHashCode() { m_hash_code = typeid(*this).hash_code(); }
	};

	// Базовый класс для объектов, которым нужно 
	//	обновление Update(float dt)
	//	инициализация/деинициализация - Init/Release
	class GameGlobalObject : public GlobalObjectBase
	{
	public:
		virtual ~GameGlobalObject() {}

		virtual void Update(float dt) {}
		virtual void Init() {}
		virtual void Release() {}
	};

} // Patterns