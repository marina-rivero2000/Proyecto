import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Cargar datos
def cargar_datos(ventas, articulos):
    # Cargar las ventas
    # ventas = pd.read_csv('data/ventas.csv', usecols=['Codigo_Fac', 'Año', 'Codigo_Art', 'Descripcion_Art'])
    ventas['Codigo_Art'] = ventas['Codigo_Art'].astype(str)
    # Cargar los artículos
    # articulos = pd.read_csv('data/articulos2.csv', usecols=['Codigo_Art', 'Familia'])
    articulos['Codigo_Art'] = articulos['Codigo_Art'].astype(str)
    # Selector múltiple de productos
    mapeo = dict(zip(articulos['Codigo_Art'], articulos['Familia']))
    ventas['Familia'] = ventas['Codigo_Art'].map(mapeo)
    ventas.dropna(subset=['Familia'], inplace=True)
    return ventas

# Entrenar el modelo
def entrenar_modelo(ventas, soporte_minimo=0.02):
    #Agrupamos las transacciones por código de factura y año y creamos una lista de familias asegurando que no haya duplicados
    transacciones = (ventas.groupby(['Codigo_Fac', 'Año'])['Familia'].apply(lambda fams: list(dict.fromkeys(fams))).tolist())
    # Separamos el 80% de las transacciones para entrenamiento y el 20% para test
    x_train, x_test = train_test_split(transacciones, test_size=0.2, random_state=42)
    # Convertimos todos los elementos de las transacciones de entrenamiento y test a string
    x_train = [[str(f) for f in compra] for compra in x_train]
    x_test  = [[str(f) for f in compra] for compra in x_test]
    # Aplicamos el one-hot encoding a las transacciones de entrenamiento
    encoder = TransactionEncoder()
    encoder_ary = encoder.fit(x_train).transform(x_train)
    df_train = pd.DataFrame(encoder_ary, columns=encoder.columns_)
    # Modelo de Apriori
    frequency_ap = apriori(df_train, min_support=soporte_minimo, use_colnames=True)
    return frequency_ap, df_train

# Generar reglas de asociación
def generar_reglas(freq, lift_min, conf_min):
    reglas = association_rules(freq, metric='lift', min_threshold=lift_min)
    # Filtrar las reglas por confianza mínima
    reglas = reglas[reglas['confidence'] >= conf_min]
    # Devolver las reglas ordenadas por lift y confianza 
    return reglas.sort_values(by=['confidence','lift'], ascending=False).reset_index(drop=True)

# Recomendar reglas
def recomendar_reglas(cesta, reglas, top_n):
    # Filtrar reglas cuyo antecedente esté contenido completamente en la cesta
    reglas_filtradas = reglas[reglas['antecedents'].apply(lambda ant: ant.issubset(set(cesta)))]
    # Ordenar por confianza descendente
    reglas_ordenadas = reglas_filtradas.sort_values(by=['confidence', 'lift'], ascending=False)
    # Si no hay reglas, devolver vacio
    if reglas_ordenadas.empty:
        return []
    # Si hay reglas, devolver los productos recomendados
    return list(reglas_ordenadas['consequents'].explode().unique())[:top_n]

# Recomendador
def recomendar_integrado(cesta, reglas_estrictas, reglas_no_estrictas, populares, top_n=2):
    # reglas estrictas
    recomendaciones_estrictas = recomendar_reglas(cesta, reglas_estrictas, top_n)
    if recomendaciones_estrictas:
        return recomendaciones_estrictas
    # reglas no estrictas
    recomendaciones_no_estrictas = recomendar_reglas(cesta, reglas_no_estrictas, top_n)
    if recomendaciones_no_estrictas:
        return recomendaciones_no_estrictas
    # Fallback de popularidad
    return populares[:top_n]

def recomendar_por_productos(descripciones_art, ventas, reglas_estrictas, reglas_no_estrictas, populares, top_n=3):
    # Verificamos que se pasen descripciones válidas
    if not descripciones_art:
        return ["⚠️ No se han proporcionado descripciones de productos."]
    # Filtramos las filas del dataframe que contienen las descripciones dadas
    familias = ventas[ventas['Descripcion_Art'].isin(descripciones_art)]['Familia'].unique()
    if len(familias) == 0:
        return ["⚠️ Ninguno de los productos fue encontrado."]
    # Convertimos a lista y eliminamos duplicados
    cesta = list(familias)
    # Aplicar recomendación
    return recomendar_integrado(cesta, reglas_estrictas, reglas_no_estrictas, populares, top_n=3)