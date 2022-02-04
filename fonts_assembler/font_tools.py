import os
import codecs
from typing import List, Dict
import shutil
from copy import deepcopy
from collections import namedtuple

from fontTools.ttLib import TTFont, newTable
from fontTools.subset import Subsetter
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._k_e_r_n import KernTable_format_0


self_dir = os.path.dirname(__file__)
out_dir = f'{self_dir}/output'
source_dir = f'{self_dir}/sources'


def __ensure_output_dir():
    os.makedirs(out_dir, exist_ok=True)


def get_normalized_names(font):
    cmap = font['cmap']
    cmap_prepared = {}
    for table in cmap.tables:
        for sym, sym_nm in table.cmap.items():
            sym_hex = hex(sym)
            hex_name = f'{sym_hex}'.replace('0x', '').upper()
            # Исправляем имена символов, так как почему-то в некоторых шрифтав решили сделать несовпадающими
            # ord(unicode) и glyph_name. Действительно, почему бы и нет
            # Но чтобы в текущем шрифте их найти нужно сохранить и первоначальное имя - будет хранится в synonims[0]
            true_name = f'u{hex_name}'
            syms_codes = cmap_prepared.setdefault(sym, {'name': true_name, 'synonims': []})
            if sym_nm not in syms_codes:
                syms_codes['synonims'].append(sym_nm)
    return cmap_prepared


def get_name(sym_name, cmap_prepared):
    for normalized_name, info in cmap_prepared.items():
        if sym_name in info['synonims']:
            return info['name']
    return None


def erase_not_used(font_path, source_file, symbols):
    shutil.copyfile(source_file, font_path)

    subsetter = Subsetter()
    subsetter.populate(text=''.join(symbols))

    font = TTFont(font_path)
    subsetter.subset(font)
    font.save(font_path)
    font.close()

    if subsetter.unicodes_missing:
        # Это символы сердечек, пик и подобных. Они встречаются у нас в тексте, но их нет в исходных шрифтах
        service_symbols = [(65024, 65039), (13, 13)]
        missed_symbols = [sym for sym in subsetter.unicodes_missing
                          if not any(beg <= sym <= end for beg, end in service_symbols)]
        print(f'Found missing glyphs: {missed_symbols}')


FontMetrics = namedtuple('FontMetrics', ['os2', 'hhea'])


class ParsedFont(object):
    def __init__(self, font_path):
        self.__file_path = font_path
        # Открытый текст - TTFont
        self.font = None
        # Интересующие нас таблицы
        self.hmtx = None
        self.vmtx = None
        self.glyph_set = None
        self.glyf = None
        self.cmap = None
        self.kern = None
        self.os2 = None
        # Таблица имён глифов, связь нового имени и имени в шрифте
        self.cmap_prepared = None
        # Связь имени глифа в шрифте и его unicode
        self.name_uni = None

        self._is_otf = False

        self.__prepare()

    def __prepare(self):
        self.font = TTFont(self.__file_path)
        # Пробел в конце не ошибка, таблица имеет тэг  "CFF "
        self._is_otf = 'CFF ' in self.font
        # Для otf-шрифтов другие таблицы, мы их не сливаем в один
        if self._is_otf:
            return

        self.hmtx = self.font['hmtx']
        self.vmtx = self.font['vmtx'] if 'vmtx' in self.font else None
        self.vhea = self.font['vhea'] if 'vhea' in self.font else None
        self.glyph_set = self.font.getGlyphSet()
        self.glyf = self.font['glyf']
        self.cmap = self.font['cmap']
        self.kern = self.font['kern'] if 'kern' in self.font else None
        self.os2 = self.font['OS/2']

        self.cmap_prepared = {}
        self.name_uni = {}
        for table in self.cmap.tables:
            for sym, sym_nm in table.cmap.items():
                sym_hex = hex(sym)
                hex_name = f'{sym_hex}'.replace('0x', '').upper()
                # Исправляем имена символов, так как почему-то в некоторых шрифтав решили сделать несовпадающими
                # ord(unicode) и glyph_name. Действительно, почему бы и нет
                # Но чтобы в текущем шрифте их найти нужно сохранить и первоначальное имя - будет хранится в synonims[0]
                true_name = f'u{hex_name}'
                syms_codes = self.cmap_prepared.setdefault(sym, {'name': true_name, 'synonims': []})
                if sym_nm not in syms_codes:
                    syms_codes['synonims'].append(sym_nm)
                self.name_uni[sym_nm] = sym

    @property
    def is_otf(self):
        return self._is_otf

    def clear(self):
        self.hmtx = None
        self.vmtx = None
        self.glyph_set = None
        self.glyf = None
        self.cmap = None
        self.kern = None
        self.os2 = None
        self.cmap_prepared = None
        self.name_uni = None
        self.font.close()


def get_metrics(metrics_font) -> FontMetrics:
    # Значения по-умолчанию которые достались в наследство от fontforge
    hhea = {'ascent': 1000, 'ascender': 1000, 'descent': -300, 'descender': -300, 'lineGap': 92}
    os2 = {'sCapHeight': 710, 'sTypoAscender': 819, 'sTypoDescender': -205, 'sTypoLineGap': 92,
           'sxHeight': 532, 'usWinAscent': 1000, 'usWinDescent': 300, 'yStrikeoutPosition': 265,
           'yStrikeoutSize': 51, 'ySubscriptXOffset': 0, 'ySubscriptXSize': 665, 'ySubscriptYOffset': 143,
           'ySubscriptYSize': 716, 'ySuperscriptXOffset': 0, 'ySuperscriptXSize': 665, 'ySuperscriptYOffset': 491,
           'ySuperscriptYSize': 716, 'version': 4, 'fsSelection': 64
           }

    if metrics_font is None:
        return FontMetrics(os2=os2, hhea=hhea)

    if 'hhea' in metrics_font.font:
        font_hhea = metrics_font.font['hhea']
        for attr_id in hhea:
            hhea[attr_id] = getattr(font_hhea, attr_id)

    for attr_id in os2:
        os2[attr_id] = getattr(metrics_font.os2, attr_id)

    return FontMetrics(os2=os2, hhea=hhea)


def setup_hints(metrics_font, font, font_name):
    if metrics_font is None:
        return

    fpgm = metrics_font.font['fpgm'] if 'fpgm' in metrics_font.font else None
    prep = metrics_font.font['prep'] if 'prep' in metrics_font.font else None
    cvt = metrics_font.font['cvt '] if 'cvt ' in metrics_font.font else None
    maxp_src_table = metrics_font.font['maxp'] if 'maxp' in metrics_font.font else None

    # Строгое условие, всё хорошо - нам просто не надо ничего копировать
    if fpgm is None and prep is None and cvt is None:
        return

    if fpgm is None or prep is None or cvt is None:
        print(f'[WARNING] Expected prep+cvt+fpgm tables in source font: fpgm="{fpgm is not None}"; '
              f'cvt="{cvt is not None}"; prep="{prep is not None}"')
        return

    fpgm_table = newTable('fpgm')
    fpgm_table.program = fpgm.program
    font['fpgm'] = fpgm_table

    cvt_table = newTable('cvt ')
    cvt_table.values = cvt.values
    font['cvt '] = cvt_table

    prep_table = newTable('prep')
    prep_table.program = prep.program
    font['prep'] = prep_table

    maxp_table = font['maxp']
    maxp_table.maxZones = maxp_src_table.maxZones
    maxp_table.maxStorage = maxp_src_table.maxStorage
    maxp_table.maxFunctionDefs = maxp_src_table.maxFunctionDefs
    maxp_table.maxTwilightPoints = maxp_src_table.maxTwilightPoints
    maxp_table.maxStackElements = maxp_src_table.maxStackElements
    maxp_table.maxInstructionDefs = maxp_src_table.maxInstructionDefs

    if maxp_table.maxFunctionDefs == 0:
        print(f'[WARNING] [{font_name}] There are instructions in font but functionDefs in "maxp" table == 0')

    if maxp_table.maxStackElements == 0:
        print(f'[WARNING] [{font_name}] There are instructions in font but maxStackElements in "maxp" table == 0')


def setup_kern_table(font, kerns):
    ttf_kern_table = newTable('kern')
    ttf_kern_table.version = 0
    ttf_kern_table.kernTables = []
    for coverage, kern_table in kerns.items():
        ttf_kern_value = KernTable_format_0(apple=False)
        ttf_kern_value.coverage = coverage
        ttf_kern_value.format = 0
        ttf_kern_value.kernTable = kern_table
        ttf_kern_value.tupleIndex = None
        ttf_kern_value.version = 0

        ttf_kern_table.kernTables.append(ttf_kern_value)
    font['kern'] = ttf_kern_table


def setup_name_table(metrics_font, font_builder, font_name):
    style_name = "TauStyle"
    if metrics_font is not None and 'name' in metrics_font.font:
        # Берём имя стиля из таблицы шрифта для метрик, если он есть
        style_name_record = metrics_font.font['name'].getName(nameID=2, platformID=3, platEncID=1, langID=1033) \
                            or metrics_font.font['name'].getName(nameID=2, platformID=3, platEncID=1, langID=None)
        if style_name_record is not None:
            style_name = style_name_record.toUnicode()
    name_strings = dict(familyName=dict(en=font_name),
                        styleName=dict(en=style_name),
                        uniqueFontIdentifier=f'tau_empire:{font_name}',
                        fullName=f'{font_name}',
                        psName=f'{font_name}',
                        version='Version 0.1')
    font_builder.setupNameTable(name_strings)


def get_symbol_kerns(glyph_name, kern, name_uni):
    if kern is None:
        return None
    kerns = []
    for kern_table in kern.kernTables:
        # Поддерживаем только формат 0: упорядоченные пары
        if kern_table.format != 0:
            continue
        pairs = []
        for kern_pair in kern_table.kernTable:
            if kern_pair[0] == glyph_name:
                pairs.append((name_uni.get(kern_pair[1]), kern_table[kern_pair]))

        if pairs:
            kerns.append({'coverage': kern_table.coverage, 'pairs': pairs})
    return kerns


def fallback_composite_add(gl, glyf):
    coords, end_pts, flags = gl.getCoordinates(glyf)
    gl.coordinates = coords
    gl.endPtsOfContours = end_pts
    gl.flags = flags
    del gl.components
    gl.numberOfContours = len(end_pts)


def add_symbol(sym_ord: int, glyphs_info: Dict, font: ParsedFont):
    glyph_link = font.cmap_prepared.get(sym_ord)
    if glyph_link is None:
        return None
    glyph_name_font = glyph_link['synonims'][0]
    glyph = font.glyph_set.get(glyph_name_font)

    pen = TTGlyphPen(font.glyf)
    glyph.draw(pen)
    gl = pen.glyph()

    if hasattr(glyph._glyph, 'program'):
        gl.program = deepcopy(glyph._glyph.program)
    else:
        gl.program = ttProgram.Program()
        gl.program.fromBytecode([])

    if gl.isComposite():
        add_error = True
        compound_names = {}  # component names
        for gl_component in gl.components:
            comp_sym_ord = font.name_uni.get(gl_component.glyphName)
            if comp_sym_ord is None:
                break

            sym_name = add_symbol(comp_sym_ord, glyphs_info, font)
            if sym_name is None:
                break
            compound_names[gl_component.glyphName] = sym_name
        else:
            add_error = False
            # В новом шрифте возможно будет друге имя, заменяем его на него
            for gl_component in gl.components:
                gl_component.glyphName = compound_names[gl_component.glyphName]
        if add_error:
            fallback_composite_add(gl, font.glyf)

    new_name = glyph_link['name']
    glyphs_info[sym_ord] = {
        'name': glyph_link['name'],
        'hmtx': font.hmtx[glyph_name_font],
        'vmtx': font.vmtx[glyph_name_font] if font.vmtx is not None else (1200, 200),
        'kern': get_symbol_kerns(glyph_name_font, font.kern, font.name_uni),
        'glyph': gl
    }
    return new_name


def merge_fonts(font_path: str, source_files: List[str], symbols, metrics_source, font_name):
    glyphs_info = {}
    found_symbols = []

    for source_path in source_files:
        font_add = ParsedFont(source_path)

        if font_add.is_otf:
            print(f'[ERROR] Merge of otf fonts does not supported.')
            return

        for symbol in symbols:
            sym_ord = ord(symbol)
            if add_symbol(sym_ord, glyphs_info, font_add):
                found_symbols.append(symbol)

    glyphs = []
    character_map = {}
    h_metrics = {}
    glyphs_setup = {}
    kerns = {}
    ord_name_links = {}
    for sym_id, gl_info in glyphs_info.items():
        gl_name = gl_info['name']
        ord_name_links[gl_name] = sym_id
        # Таблицы cmap, glyf и htmx
        character_map[sym_id] = gl_name
        h_metrics[gl_name] = gl_info['hmtx']
        glyphs.append(gl_name)
        glyphs_setup[gl_name] = gl_info['glyph']

        # Генерирования информация для kern-таблицы, которая отдельно записывается
        kern_info = gl_info['kern']
        if kern_info is not None:
            for kern_coverage in kern_info:
                coverage = kern_coverage['coverage']
                kern_table = kerns.setdefault(coverage, {})
                for pair in kern_coverage['pairs']:
                    # возможно, что символ для кернинга не нужен в результирующем шрифте
                    if pair[0] not in glyphs_info:
                        continue
                    pair_gl_name = glyphs_info[pair[0]]['name']
                    kern_table.update({(gl_name, pair_gl_name): pair[1]})
    fb = FontBuilder(unitsPerEm=1024, isTTF=True)
    fb.setupGlyphOrder(glyphs)
    fb.setupCharacterMap(character_map)
    fb.setupGlyf(glyphs_setup)

    fb.setupHorizontalMetrics(h_metrics)

    metrics_font = ParsedFont(metrics_source) if metrics_source is not None else None
    metrics = get_metrics(metrics_font)
    fb.setupHorizontalHeader(**metrics.hhea)

    font_name = font_name.replace(' ', '')
    setup_name_table(metrics_font=metrics_font, font_builder=fb, font_name=font_name)
    fb.setupOS2(**metrics.os2)
    setup_kern_table(fb.font, kerns)
    setup_hints(metrics_font=metrics_font, font=fb.font, font_name=font_name)

    fb.setupPost()
    fb.save(font_path)


def load_symbols(text_file):
    chars = []
    with codecs.open(text_file, 'r', encoding='utf-8') as f:
        chars.extend(f.read())
    unique_chars = sorted(set(chars))
    return unique_chars


__ensure_output_dir()

print('Try to erase symbol from font...')
erase_not_used(font_path=os.path.join(out_dir, 'test.ttf'),
               source_file=os.path.join(source_dir, 'GosmickSans Bold.ttf'),
               symbols=['Ć'])

print('Try merge fonts...')
merge_fonts(font_path=os.path.join(out_dir, 'test_merge.ttf'),
            source_files=[os.path.join(source_dir, 'GosmickSans Bold.ttf'),
                          os.path.join(source_dir, 'SourceHanSans-Normal.ttf')],
            symbols=load_symbols(os.path.join(self_dir, 'general_FilmotypeMajor.txt')),
            metrics_source=None,
            font_name='My font')
