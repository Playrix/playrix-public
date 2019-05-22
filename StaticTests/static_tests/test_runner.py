# -*- coding: utf-8 -*-

from .verification_context import VerificationContext
from configparser import RawConfigParser


def get_changed_files():
    '''
    :return: Возвращает список изменённых файлов. Команда зависит от используемой CVS
    '''
    return None


class Runner(object):
    def __init__(self, config_str, setup_function):
        self.__tests = []

        config_parser = RawConfigParser()
        config_parser.read_string(config_str)

        app_path = config_parser.get('main', 'app_path')
        build_type = config_parser.get('main', 'build_type')
        platform = config_parser.get('main', 'platform')

        '''
        get_changed_files возвращает список изменённых файлов и зависит от используемой CVS  
        '''
        changed_files = None if build_type != 'hook' else get_changed_files()
        self.__context = VerificationContext(app_path, build_type, platform, changed_files)
        setup_function(self)

    @property
    def context(self):
        return self.__context

    def add_test(self, test):
        self.__tests.append(test)

    def run(self):
        for test in self.__tests:
            test.init()

        for test in self.__tests:
            test.prepare()

        for test in self.__tests:
            test.run()
