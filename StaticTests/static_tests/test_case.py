# -*- coding: utf-8 -*-


class TestCase(object):
    def __init__(self, name, context, build_types=None, platforms=None, predicate=None,
                 expected_resources=None, modified_resources=None):
        self.__name = name
        self.__context = context
        self.__build_types = build_types
        self.__platforms = platforms
        self.__predicate = predicate
        self.__expected_resources = expected_resources
        self.__modified_resources = modified_resources

        # Подходит ли тип сборки и платформы для запуска теста
        # Изменились ли ресурсы, за которые отвечает предикат
        self.__need_run = self.__check_run()
        self.__need_resource_run = False

    @property
    def context(self):
        return self.__context

    def fail(self, message):
        print('Fail: {}'.format(message))

    def __log_run(self, stage):
        name = self.__name if stage == 'run' else '{}.Prepare'.format(self.__name)
        print("Start test: {}".format(name))

    def __log_finish(self, stage):
        name = self.__name if stage == 'run' else '{}.Prepare'.format(self.__name)
        print("Finish test: {}".format(name))

    def __check_run(self):
        build_success = self.__build_types is None or self.__context.build_type in self.__build_types
        platform_success = self.__platforms is None or self.__context.platform in self.__platforms
        hook_success = build_success
        if build_success and self.__context.is_build('hook') and self.__predicate:
            hook_success = any(self.__predicate(changed_file) for changed_file in self.__context.changed_files)

        return build_success and platform_success and hook_success

    def __set_context_resources(self):
        if not self.__need_run:
            return
        if self.__modified_resources:
            self.__context.modify_resources(self.__modified_resources)
        if self.__expected_resources:
            self.__context.expect_resources(self.__expected_resources)

    def init(self):
        """
        Запускается после того, как создались все тесты и в контекст записана информация
        об изменённых ресурсах и тех ресурах, которые нужны другим тестам
        """
        self.__need_resource_run = self.__modified_resources and any(self.__context.is_resource_expected(resource)
                                                                     for resource in self.__modified_resources)

    def _prepare_impl(self):
        pass

    def prepare(self):
        if not self.__need_run and not self.__need_resource_run:
            return
        self.__log_run('prepare')
        self._prepare_impl()
        self.__log_finish('prepare')

    def _run_impl(self):
        pass

    def run(self):
        if self.__need_run:
            self.__log_run('run')
            self._run_impl()
            self.__log_finish('run')
