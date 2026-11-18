"""Register a valid TrueType form of installed Palatino-compatible P052 fonts.

Matplotlib's PDF Type42 backend cannot embed CFF OpenType as TrueType. Convert
cubic outlines to quadratic outlines (maximum error 0.5 font units), preserving
all advance widths, character maps, names/licence notices and layout tables.
The converted files are temporary build-cache assets, not research inputs.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import tempfile
from fontTools.ttLib import TTFont
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen
from matplotlib import font_manager

FAMILY='P052 TrueType'

def configure_palatino_font():
    # P052_FONT_DIR: optional folder holding the URW base35 P052 OpenType files when they are not installed system-wide
    if os.environ.get('P052_FONT_DIR'):
        for otf in sorted(Path(os.environ['P052_FONT_DIR']).glob('P052-*.otf')):font_manager.fontManager.addfont(str(otf))
    cache=Path(tempfile.gettempdir())/'congressional-trading-p052-truetype'
    cache.mkdir(exist_ok=True)
    records={}
    for label,style,weight in [('Regular','normal','normal'),('Italic','italic','normal'),('Bold','normal','bold'),('BoldItalic','italic','bold')]:
        source=Path(font_manager.findfont(font_manager.FontProperties(family='P052',style=style,weight=weight),fallback_to_default=False))
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        target=cache/f'P052-{label}-{digest[:12]}.ttf'
        if not target.exists():
            old=TTFont(source);glyph_set=old.getGlyphSet();glyphs={}
            for name in old.getGlyphOrder():
                pen=TTGlyphPen(None);quadratic=Cu2QuPen(pen,max_err=.5,reverse_direction=True)
                glyph_set[name].draw(quadratic);glyphs[name]=pen.glyph()
            fb=FontBuilder(old['head'].unitsPerEm,isTTF=True)
            fb.setupGlyphOrder(old.getGlyphOrder());fb.setupCharacterMap(old.getBestCmap())
            fb.setupGlyf(glyphs);fb.setupHorizontalMetrics(old['hmtx'].metrics)
            fb.setupHorizontalHeader(ascent=old['hhea'].ascent,descent=old['hhea'].descent,lineGap=old['hhea'].lineGap)
            fb.font['name']=deepcopy(old['name'])
            names={1:FAMILY,2:label.replace('BoldItalic','Bold Italic'),3:FAMILY+' '+label+' '+digest[:12],4:FAMILY+' '+label,6:'P052TrueType-'+label,16:FAMILY,17:label.replace('BoldItalic','Bold Italic')}
            for record in list(fb.font['name'].names):
                if record.nameID in names:fb.font['name'].setName(names[record.nameID],record.nameID,record.platformID,record.platEncID,record.langID)
            fb.font['OS/2']=deepcopy(old['OS/2'])
            for tag in ('GPOS','GSUB'):
                if tag in old:fb.font[tag]=deepcopy(old[tag])
            fb.setupPost(italicAngle=old['post'].italicAngle,underlinePosition=old['post'].underlinePosition,underlineThickness=old['post'].underlineThickness,isFixedPitch=old['post'].isFixedPitch)
            fb.setupMaxp()
            for field in ('created','modified','macStyle'):setattr(fb.font['head'],field,getattr(old['head'],field))
            fb.save(target)
            converted=TTFont(target)
            assert converted['hmtx'].metrics==old['hmtx'].metrics
            assert converted.getBestCmap()==old.getBestCmap()
            assert 'glyf' in converted and 'CFF ' not in converted
        font_manager.fontManager.addfont(str(target))
        records[label]={'source_font':source.name,'source_sha256':digest,'conversion':'cubic to quadratic; max outline error 0.5 font units; unchanged advance widths and cmap'}
    return FAMILY,records
