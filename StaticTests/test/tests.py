# -*- coding: utf-8 -*-

from static_tests.test_case import TestCase
from static_tests.test_runner import Runner
import os
from lxml import etree


class VerifyTexture(TestCase):
    def __init__(self, context):
        super(VerifyTexture, self).__init__('VerifyTexture', context,
                                            build_types=['production', 'hook'],
                                            platforms=['windows', 'ios'],
                                            expected_resources=None,
                                            modified_resources=['Texture'],
                                            predicate=lambda file_path: os.path.splitext(file_path)[1] == '.png')

    def _prepare_impl(self):
        texture_dir = os.path.join(self.context.app_path, 'resources', 'textures')
        for root, dirs, files in os.walk(texture_dir):
            for tex_file in files:
                self.context.register_resource('Texture', tex_file)


class VerifyModels(TestCase):
    def __init__(self, context):
        super(VerifyModels, self).__init__('VerifyModels', context,
                                           expected_resources=['Texture'],
                                           predicate=lambda file_path: os.path.splitext(file_path)[1] == '.obj')

    def _run_impl(self):
        models_descriptions = etree.parse(os.path.join(self.context.app_path, 'resources', 'models.xml'))
        for model_xml in models_descriptions.findall('.//Model'):
            texture_id = model_xml.get('texture')
            texture_id, data = self.context.get_resource('Texture', texture_id)
            if texture_id is None:
                self.fail('Texture for model {} was not found: {}'.format(model_xml.get('id'), texture_id))


def run_validator(runner_setup_func, config_raw):
    runner = Runner(config_raw, runner_setup_func)
    runner.run()


def setup(runner):
    runner.add_test(VerifyTexture(runner.context))
    runner.add_test(VerifyModels(runner.context))


raw_config = '''
    [main]
    app_path = {app_path}
    build_type = production
    platform = ios
    '''.format(app_path=os.path.join(os.path.dirname(__file__), '..', 'test_app'))


run_validator(setup, raw_config)
