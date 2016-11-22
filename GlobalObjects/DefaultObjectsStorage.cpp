#include "DefaultObjectsStorage.h"

#include <algorithm>

namespace Patterns
{

	DefaultObjectsStorage::DefaultObjectsStorage()
	{
		// Recalculate hash

		// Add to cache
	}

	GameGlobalObject* DefaultObjectsStorage::GetGlobalObjectImpl(size_t i_type_code) const
	{
		for (auto* p_object : m_cache_objects)
		{
			if (p_object->GetTypeHashCode() == i_type_code)
				return p_object;
		}
		return nullptr;
	}

	void DefaultObjectsStorage::AddGlobalObjectImpl(std::unique_ptr<GameGlobalObject> ip_object)
	{
#if defined(_DEBUG)
		if (GetGlobalObjectImpl(ip_object->GetTypeHashCode()) != nullptr)
		{
			assert(false);
			return;
		}
#endif
		// cache object and add it to pool
		m_cache_objects.push_back(ip_object.get());
		m_dynamic_objects.emplace_back(std::move(ip_object));

	}

	void DefaultObjectsStorage::RemoveGlobalObjectImpl(size_t i_type_code)
	{	
		const auto it_dyn = std::find_if(m_dynamic_objects.begin(), m_dynamic_objects.end(), [i_type_code](ObjPtr& p_obj)
		{
			return p_obj->GetTypeHashCode() == i_type_code;
		});
		// we can delete from cache only dynamic objects - not static which are defined inside this class
		//	and will be removed with destruction of getter
		if (it_dyn != m_dynamic_objects.end())
		{
			// remove from cache
			const auto it = std::find_if(m_cache_objects.begin(), m_cache_objects.end(), [i_type_code](GameGlobalObject* p_obj)
			{
				return p_obj->GetTypeHashCode() == i_type_code;
			});

			it_dyn->reset(nullptr);

			if (it != m_cache_objects.end())
				m_cache_objects.erase(it);

			m_dynamic_objects.erase(it_dyn);
		}
	}

} // Patterns