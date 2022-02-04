import os
from multiprocessing import Process

import fontforge


def copy_glyph_info(old, new):
    points = old.anchorPoints
    for point in points:
        new.addAnchorPoint(point[0], point[1], point[2], point[3])


def append_fonts(dst_font_path, add_fonts, chars, family_name):
    path = os.path.normpath(dst_font_path)
    if not os.path.exists(path):
        font_base = fontforge.font()
    else:
        font_base = fontforge.open(path)

    for src_font_path in add_fonts:
        font_add = fontforge.open(os.path.normpath(src_font_path))

        for ch in chars:
            glyphname = fontforge.nameFromUnicode(ord(ch))
            font_base.createChar(ord(ch), glyphname)
            font_add.selection.select(('more', 'unicode',), ord(ch))
            font_base.selection.select(('more', 'unicode',), ord(ch))

        font_add.copy()
        font_base.paste()

        font_add.selection.none()
        font_base.selection.none()

        for glyph in font_base.glyphs():
            if glyph.glyphname in font_add:
                oldGlyph = font_add[glyph.glyphname]
                copy_glyph_info(oldGlyph, glyph)

    font_base.familyname = family_name
    font_base.fullname = family_name
    font_base.fontname = family_name
    font_base.generate(os.path.normpath(dst_font_path))


def assemble_font(dst_font_path, name, chars, src_fonts):
    count_in_chunk = 2                    # по сколько шрифтов за раз можно сливать
    fonts_chunks = [src_fonts[name][:4]]  # Первые 4 шрифта лёгкие, можно пачкой обработать
    fonts_chunks.extend([src_fonts[name][i:i + count_in_chunk] for i in range(4, len(src_fonts[name]), count_in_chunk)])

    for add_fonts in fonts_chunks:
        p = Process(target=append_fonts, args=(dst_font_path, add_fonts, chars, name))
        p.start()
        p.join()


def use_in_other_glyph(glyph, chars):
    # altuni -  Tuple of alternate encodings.
    # Each alternate encoding is a tuple of (unicode-value, variation-selector, reserved-field)
    # https://fontforge.github.io/python.html#Glyph
    if glyph.altuni is None:
        return False
    for alts in glyph.altuni:
        if alts[0] in chars:
            return True
    return False


def exclude_unused_glyphs(font, chars):
    for g in font.glyphs():
        if g.unicode in chars:
            continue

        if use_in_other_glyph(g, chars):
            # глиф используется в других глифах
            continue

        if g.glyphname != ".notdef":
            g.unlinkThisGlyph()
            g.clear()


def build_with_erasing(dst_font_path, source_font_path, chars):
    font = fontforge.open(source_font_path)
    exclude_unused_glyphs(font, chars)
    font.generate(dst_font_path, flags=('old-kern', 'short-post'))
    font.close()
