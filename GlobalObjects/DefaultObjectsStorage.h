#pragma once

#include "GlobalObjectBase.h"
#include "ObjectStorageBase.h"

#include <vector>
#include <memory>

namespace Patterns
{

	class DefaultObjectsStorage : public ObjectStorageBase<GameGlobalObject>
	{
	private:
		std::vector<GameGlobalObject*> m_cache_objects;
		using ObjPtr = std::unique_ptr<GameGlobalObject>;
		std::vector<ObjPtr> m_dynamic_objects;


	private:
		virtual GameGlobalObject* GetGlobalObjectImpl(size_t i_type_code) const override;
		virtual void AddGlobalObjectImpl(std::unique_ptr<GameGlobalObject> ip_object) override;
		virtual void RemoveGlobalObjectImpl(size_t i_type_code) override;

	public:
		DefaultObjectsStorage();

		virtual std::vector<GameGlobalObject*> GetStoredObjects() override { return m_cache_objects; }
	};

} // Patterns