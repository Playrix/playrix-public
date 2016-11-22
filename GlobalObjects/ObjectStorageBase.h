#pragma once

#include <vector>
#include <memory>
#include <cassert>

namespace Patterns
{
	template <typename BaseObject>
	class ObjectStorageBase
	{
	private:
		virtual BaseObject* GetGlobalObjectImpl(size_t i_type_code) const = 0;
		virtual void AddGlobalObjectImpl(std::unique_ptr<BaseObject> ip_object) = 0;
		virtual void RemoveGlobalObjectImpl(size_t i_type_code) = 0;

	public:
		virtual ~ObjectStorageBase() {}
		
		template <typename ObjectType, typename... Args>
		void AddGlobalObject(Args... args)
		{
			auto p_object = std::unique_ptr<ObjectType>(new ObjectType(args...));
			p_object->RecalcHashCode();
			AddGlobalObjectImpl(std::move(p_object));
		}

		template <typename ObjectType>
		ObjectType* GetGlobalObject() const
		{
#if defined(_DEBUG)
			auto p_obj = GetGlobalObjectImpl(typeid(ObjectType).hash_code());
			assert(p_obj != nullptr && "There is no registered global object");
			assert(dynamic_cast<ObjectType*>(p_obj) != nullptr && "Cannot convert type.");
#endif
			// cache object hash_code, because type will be always the same
			static const size_t obj_type = typeid(ObjectType).hash_code();
			return static_cast<ObjectType*>( GetGlobalObjectImpl(obj_type) );
		}

		template <typename ObjectType>
		void RemoveGlobalObject()
		{
			// cache object hash_code, because type will be always the same
			static const size_t obj_type = typeid(ObjectType).hash_code();
			RemoveGlobalObjectImpl(obj_type);
		}

		virtual std::vector<BaseObject*> GetStoredObjects() = 0;
	};
} // Patterns