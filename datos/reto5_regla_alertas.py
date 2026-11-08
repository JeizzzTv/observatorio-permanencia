"""
Observatorio de Permanencia - Universidad Andina del Norte
Reto 5: mejora de la alerta temprana (reproducible sobre los CSV originales)
Uso: python reto5_regla_alertas.py  (ajustar RUTA si los CSV estan en otra carpeta)
"""
import os
import pandas as pd, numpy as np

RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Datos_Universidad_Andina(1)', '')  # carpeta con los CSV originales
CAPACIDAD = 120                           # consejeros disponibles
K_ACC = 5                                 # umbral de accesos acumulados semanas 6-8
RATIO = 0.4                               # caida relativa de accesos (s6-8 vs s1-5)

# ---------- 1. Cohorte (limpieza minima necesaria para el reto) ----------
s = pd.read_csv(RUTA+'sis_matriculas.csv', encoding='utf-8-sig', dtype=str)
s['documento'] = s.documento.str.replace('.', '', regex=False)      # 1.017.055.478 -> 1017055478
s['periodo'] = s.periodo.replace({'2025-II': '2025-2'})              # estandarizar periodo
s = s.drop_duplicates()                                              # 17 filas repetidas
coh = s[(s.periodo == '2025-1') & (s.tipo_ingreso == 'Nuevo')].copy()
assert len(coh) == 1200, len(coh)
coh['desertor'] = (~coh.documento.isin(set(s[s.periodo == '2025-2'].documento))).astype(int)
coh['programa'] = coh.programa.replace({
    'ADMON EMPRESAS': 'Administración de Empresas', 'PSICOLOGIA': 'Psicología',
    'ING. INFORMATICA': 'Ingeniería Informática', 'CONTADURIA': 'Contaduría Pública',
    'TRABAJO SOCIAL': 'Trabajo Social', 'ING INDUSTRIAL': 'Ingeniería Industrial'})

# ---------- 2. Moodle: 2 correos compartidos por 2 estudiantes ----------
# El archivo viene ordenado por codigo: el 1er bloque de 16 semanas es del menor codigo.
m = pd.read_csv(RUTA+'moodle_actividad_2025_1.csv', encoding='utf-8-sig')
m['uid'] = m.correo
m['bloque'] = m.groupby(['correo', 'semana']).cumcount()
coh['cod_n'] = coh.codigo.str[-4:].astype(int)
dup = coh[coh.correo_institucional.duplicated(keep=False)].sort_values('cod_n')
for c, g in dup.groupby('correo_institucional'):
    for b, (_, r) in enumerate(g.iterrows()):
        m.loc[(m.correo == c) & (m.bloque == b), 'uid'] = r.codigo
coh['uid'] = np.where(coh.correo_institucional.isin(dup.correo_institucional),
                      coh.codigo, coh.correo_institucional)
acc = m.pivot_table(index='uid', columns='semana', values='accesos', aggfunc='sum').add_prefix('acc_w')
nota = m[m.nota_parcial_0_100.notna()].pivot_table(index='uid', columns='semana',
        values='nota_parcial_0_100').add_prefix('nota_w')
d = coh.merge(acc, left_on='uid', right_index=True, how='left') \
       .merge(nota, left_on='uid', right_index=True, how='left')

adm = pd.read_csv(RUTA+'admisiones.csv', encoding='utf-8-sig', dtype=str)
adm['documento'] = adm.documento.str.replace('.', '', regex=False)
d = d.merge(adm.drop_duplicates('documento')[['documento', 'estrato']], on='documento', how='left')

# ---------- 3. Regla actual (definicion oficial) ----------
d['nota_s4_s8_5'] = d[['nota_w4', 'nota_w8']].mean(axis=1) / 20      # 0-100 -> 0-5
d['alerta_actual'] = ((d.nota_s4_s8_5 < 3.0) | ((d.acc_w7 + d.acc_w8) == 0)).astype(int)

# ---------- 4. Regla nueva (solo Moodle semanas 1-8) ----------
d['acc_s6_8'] = d[['acc_w6', 'acc_w7', 'acc_w8']].sum(axis=1)
d['prom_s1_5'] = d[[f'acc_w{i}' for i in range(1, 6)]].mean(axis=1)
d['prom_s6_8'] = d[['acc_w6', 'acc_w7', 'acc_w8']].mean(axis=1)
d['alerta_nueva'] = ((d.acc_s6_8 <= K_ACC) | (d.prom_s6_8 < RATIO * d.prom_s1_5)).astype(int)

# ---------- 5. Metricas ----------
def reporte(col):
    tp = int(((d[col] == 1) & (d.desertor == 1)).sum()); fp = int(((d[col] == 1) & (d.desertor == 0)).sum())
    fn = int(((d[col] == 0) & (d.desertor == 1)).sum()); tn = int(((d[col] == 0) & (d.desertor == 0)).sum())
    print(f'\n== {col} ==  alertados={tp+fp}  (capacidad {CAPACIDAD})')
    print(pd.DataFrame([[tp, fp], [fn, tn]], index=['Alertó', 'No alertó'], columns=['Desertó', 'No desertó']))
    print(f'precision={tp/(tp+fp):.1%}  cobertura={tp/(tp+fn):.1%}')
print('Cohorte:', len(d), '| desertores:', d.desertor.sum(), f'| desercion={d.desertor.mean():.1%}')
reporte('alerta_actual'); reporte('alerta_nueva')
n = d.alerta_nueva.sum(); cob = ((d.alerta_nueva == 1) & (d.desertor == 1)).sum() / d.desertor.sum()
print('\nCUMPLE Reto 5:', n <= CAPACIDAD and cob >= 0.35)

# ---------- 6. Salida seudonimizada para Looker Studio (sin nombre ni documento) ----------
out = d[['codigo', 'programa', 'estrato', 'desertor', 'alerta_actual', 'alerta_nueva',
         'acc_s6_8', 'prom_s1_5', 'prom_s6_8']].copy()
out.to_csv('alertas_cohorte_looker.csv', index=False, encoding='utf-8-sig')
print('\nCarga por programa (alertas nuevas):')
print(d.groupby('programa').agg(cohorte=('codigo', 'size'), alertados=('alerta_nueva', 'sum'),
      desertores=('desertor', 'sum')).assign(tasa_alerta=lambda x: (x.alertados / x.cohorte).round(3)))
print(d.groupby('estrato').agg(cohorte=('codigo', 'size'), alertados=('alerta_nueva', 'sum'),
      desertores=('desertor', 'sum')).assign(tasa_alerta=lambda x: (x.alertados / x.cohorte).round(3)))
