# -*- coding: utf-8 -*-


class VerificationContext(object):
    def __init__(self, app_path, build_type, platform, changed_files=None):
        self.__app_path = app_path
        self.__build_type = build_type
        self.__platform = platform
        # Заполняются запускающимися тестами
        self.__modified_resources = set()
        self.__expected_resources = set()
        # Если запуск происходит из прекомитного хука, тогда в этом списке будут изменённые файлы
        self.__changed_files = changed_files
        # Мета-данные о ресурсах, которые нашли тесты
        self.__resources = {}

    @property
    def app_path(self):
        return self.__app_path

    @property
    def build_type(self):
        return self.__build_type

    @property
    def platform(self):
        return self.__platform

    @property
    def changed_files(self):
        return self.__changed_files

    def is_build(self, build):
        return build == self.__build_type

    def modify_resources(self, resources):
        for resource in resources:
            self.__modified_resources.add(resource)

    def expect_resources(self, resources):
        self.__expected_resources.update(resources)

    def is_resource_expected(self, resource):
        return resource in self.__expected_resources

    def register_resource(self, resource_type, resource_id, resource_data=None):
        self.__resources.setdefault(resource_type, {})[resource_id] = resource_data

    def get_resource(self, resource_type, resource_id):
        if resource_type not in self.__resources or resource_id not in self.__resources[resource_type]:
            return None, None
        return resource_id, self.__resources[resource_type][resource_id]
