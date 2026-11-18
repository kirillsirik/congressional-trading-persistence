"""The alternative benchmark decompositions of Table F1, from data/v4/ext/decomposition.csv.

writing_sample_appendix_tables.py reads the table body this writes to paper/coherence_generated/.
"""
from pathlib import Path
import csv
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/coherence_generated'
def num(v):return f'{float(v):,.0f}'
OUT.mkdir(exist_ok=True)
# Frozen alternative decompositions previously printed in the main table.
d=list(csv.DictReader((ROOT/'data/v4/ext/decomposition.csv').open()))
lines=[r'\footnotesize\setlength{\tabcolsep}{4pt}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}>{\raggedright\arraybackslash}p{4.5cm}*{5}{S[table-format=-4.0]}@{}}',r'\toprule',r'Alternative benchmark & {$K(Y^0,Y^0)$} & {$K(D,D)$} & {$K(Y^0,D)$} & {$K(D,Y^0)$} & {$K(Y^1,Y^1)$} \\',r'\midrule',r'\multicolumn{6}{l}{\textit{Panel A: Winsorized trade-date returns}} \\']
terms=['base','adj','cross_base_adj','cross_adj_base','total']
for variant,label in [('no_alpha_refit','Factor model refitted without an intercept'),('market_adjusted','Market-adjusted returns')]:
 vals=[num(next(x['value'] for x in d if x['clock']=='T' and x['variant']==variant and x['statistic']==f'tau2_{term}_wins')) for term in terms]
 lines.append(label+' & '+' & '.join(vals)+r' \\')
lines += [r'\addlinespace',r'\multicolumn{6}{l}{\textit{Panel B: Unwinsorized returns; factor model refitted without an intercept}} \\']
for clock,label in [('T','Trade date'),('P','Disclosure date')]:
 vals=[num(next(x['value'] for x in d if x['clock']==clock and x['variant']=='no_alpha_refit' and x['statistic']==f'tau2_{term}_raw')) for term in terms]
 lines.append(label+' & '+' & '.join(vals)+r' \\')
lines += [r'\bottomrule',r'\end{tabular*}']
(OUT/'appendix_alternative_decompositions.tex').write_text('% Generated from frozen data/v4/ext/decomposition.csv.\n'+'\n'.join(lines)+'\n')
