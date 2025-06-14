import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
#import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# Cargar los datos
def preparar_datos(ventas):
    # Cargar las ventas
    # ventas = pd.read_csv(url, parse_dates=['Mes_Año'])
    # Transformar la columna 'Mes_Año' en índice
    ventas.set_index('Mes_Año', inplace=True)
    # Eliminar las ventas que no tienen código o descripción
    ventas = ventas.dropna(subset=['Descripcion_Art', 'Codigo_Art'])
    return ventas


# Agrupar las ventas
def agregar_mensual (ventas):
    # Agrupar las variables por mes
    ventas_mensuales = ventas['Cantidad'].resample('M').sum()
    ventas_mensuales_regre = ventas.resample('M').agg({
    'Cantidad': 'sum',
    'Precio': 'mean',
    'Precio_Costo': 'mean',
    'Ganancia': 'mean',
    }).fillna(0)
    return ventas_mensuales, ventas_mensuales_regre

# Funciones que recalcula 2020
def recalcular_2020_ventas_mensuales(ventas_mensuales):
    # Extrae los datos mensuales de 2019 y 2021 asegurando que todos los meses estén presentes
    ventas_2019 = ventas_mensuales[ventas_mensuales.index.year==2019].copy()
    ventas_2021 = ventas_mensuales[ventas_mensuales.index.year==2021].copy()
    # Reindexa por mes (1-12) para ambos años
    ventas_2019.index = ventas_2019.index.month
    ventas_2021.index = ventas_2021.index.month
    ventas_2019 = ventas_2019.reindex(range(1,13), fill_value=0)
    ventas_2021 = ventas_2021.reindex(range(1,13), fill_value=0)
    # Calcula la media por mes
    media_meses = (ventas_2019 + ventas_2021) / 2
    # Sustituye en 2020 usando la media de cada mes, buscando cualquier fecha de 2020 con ese mes
    for mes in range(1, 13):
        # Busca todas las fechas de 2020 para ese mes en el índice
        mask = (ventas_mensuales.index.year == 2020) & (ventas_mensuales.index.month == mes)
        ventas_mensuales.loc[mask] = media_meses[mes]
    return ventas_mensuales

def recalcular_2020_ventas_mensuales_regr(ventas_mensuales_regr):
    # Calcula la suma mensual de cada año para 'Cantidad'
    cantidad_2019 = ventas_mensuales_regr[ventas_mensuales_regr.index.year==2019]['Cantidad']
    cantidad_2021 = ventas_mensuales_regr[ventas_mensuales_regr.index.year==2021]['Cantidad']
    # Asegura que ambos años tienen todos los meses (1-12)
    cantidad_2019.index = cantidad_2019.index.month
    cantidad_2021.index = cantidad_2021.index.month
    cantidad_2019 = cantidad_2019.reindex(range(1,13), fill_value=0)
    cantidad_2021 = cantidad_2021.reindex(range(1,13), fill_value=0)
    # Calcula la media por mes
    media_meses = (cantidad_2019 + cantidad_2021) / 2
    # Sustituye en 2020
    for mes in range(1, 13):
        mask = (ventas_mensuales_regr.index.year == 2020) & (ventas_mensuales_regr.index.month == mes)
        ventas_mensuales_regr.loc[mask] = media_meses[mes]
    return ventas_mensuales_regr

# Separar los conjuntos de entrenamiento y test
def dividir_train_test (df, corte='2024-12'):
    # Dividir el conjunto de datos en train y test
    train = df[:corte].copy()
    test = df[pd.to_datetime(corte) + pd.offsets.MonthBegin():].copy()
    return train, test

# Entrenamiento y predicciones de modelos sin regresores
def entrenar_prophet(train):
    df_pro = pd.DataFrame({'ds':train.index, 'y':train.values})
    modelo_pro = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
    modelo_pro.fit(df_pro)
    return modelo_pro

def predecir_prophet(modelo_pro, test):
    fc_pro = modelo_pro.predict(pd.DataFrame({'ds':test.index})).set_index('ds')
    pred_pro, ci_pro = fc_pro['yhat'], fc_pro[['yhat_lower','yhat_upper']]
    return pred_pro,ci_pro

def entrenar_sarima(train, order=(0,1,1), seasonal_order=(0,0,0,12)):
    modelo_sarima = SARIMAX(train, order=order, r=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    modelo_sarima = modelo_sarima.fit(disp=False)
    return modelo_sarima

def predecir_sarima(modelo_sarima, steps):
    fc_sarima = modelo_sarima.get_forecast(steps=steps)
    pred_sarima, ci_sarima = fc_sarima.predicted_mean, fc_sarima.conf_int(alpha=0.05)
    return pred_sarima, ci_sarima

def entrenar_ets(train, seasonal_periods=12):
    modelo_hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit()
    return modelo_hw

def predecir_ets(modelo_hw, steps, train):
    pred_hw = modelo_hw.forecast(steps)
    rmse_m = np.sqrt(((train - modelo_hw.fittedvalues)**2).mean())
    ci_lower = pred_hw - 1.96 * rmse_m
    ci_upper = pred_hw + 1.96 * rmse_m
    return pred_hw, ci_lower, ci_upper

# Entrenamiento y predicciones de modelos con regresores
def entrenar_prophet_regresores(train, regresores):
    df_train = train.reset_index().rename(columns={'Mes_Año': 'ds', 'Cantidad': 'y'})
    modelo_pro_regr = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
    for r in regresores:
        modelo_pro_regr.add_regressor(r)
    modelo_pro_regr.fit(df_train)
    return modelo_pro_regr

def predecir_prophet_regresores(modelo_pro_regr, test, regresores):
    df_test = test.reset_index().rename(columns={'Mes_Año': 'ds'})
    df_test_reg = df_test[['ds'] + regresores]
    fc_pro_regr = modelo_pro_regr.predict(df_test_reg).set_index('ds')
    pred_pro_regr = fc_pro_regr['yhat'] 
    ci_pro_regr = fc_pro_regr[['yhat_lower', 'yhat_upper']]
    return pred_pro_regr, ci_pro_regr

def entrenar_sarima_regresores(train, regresores, order=(0,1,1), seasonal_order=(1,0,0,12)):
    modelo_sarima_regr = SARIMAX(train['Cantidad'], exog=train[regresores],
                     order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    modelo_sarima_regr = modelo_sarima_regr.fit(disp=False)
    return modelo_sarima_regr

def predecir_sarima_regresores(modelo_sarima_regr, test, regresores):
    fc_sarima_regr = modelo_sarima_regr.get_forecast(steps=len(test), exog=test[regresores])
    pred_sarima_regr = fc_sarima_regr.predicted_mean
    ci_sarima_regr = fc_sarima_regr.conf_int(alpha=0.05)
    return pred_sarima_regr, ci_sarima_regr

def evaluar_y_seleccionar_mejor_modelo_completo(modelos):
    """
    Evalúa los modelos usando MAE, RMSE y amplitud promedio del intervalo de confianza.
    Cada modelo debe ser una tupla: (nombre, pred, ci, test_df, columna_real)
    """
    resultados = []
    print("Evaluación de Modelos:")
    
    for nombre, pred, ci, test_df, col_real in modelos:
        # Si test_df es una Serie, úsala directamente
        if isinstance(test_df, pd.Series):
            y_true = test_df
        else:
            y_true = test_df[col_real]
        mae = mean_absolute_error(y_true, pred)
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        if ci is not None:
            amplitud_prom = (ci.iloc[:, 1] - ci.iloc[:, 0]).mean()
        else:
            amplitud_prom = np.nan
        
        print(f"{nombre}: MAE={mae:.2f}, RMSE={rmse:.2f}, Amplitud Prom. Intervalo={amplitud_prom:.2f}")
        
        resultados.append({
            'nombre': nombre,
            'mae': mae,
            'rmse': rmse,
            'amplitud': amplitud_prom,
            'pred': pred,
            'ci': ci
        })

    df_resultados = pd.DataFrame(resultados)
    df_resultados['mae_norm'] = df_resultados['mae'] / df_resultados['mae'].max()
    df_resultados['rmse_norm'] = df_resultados['rmse'] / df_resultados['rmse'].max()
    max_amplitud = df_resultados['amplitud'].dropna().max()
    df_resultados['amplitud_norm'] = df_resultados['amplitud'].fillna(max_amplitud) / max_amplitud
    df_resultados['score'] = df_resultados[['mae_norm', 'rmse_norm', 'amplitud_norm']].mean(axis=1)

    mejor_fila = df_resultados.loc[df_resultados['score'].idxmin()]
    print(f"\n📈 Mejor modelo: {mejor_fila['nombre']} (Score={mejor_fila['score']:.3f})")
    mejor_modelo = mejor_fila['nombre']
    mejor_pred = mejor_fila['pred']
    mejor_ci = mejor_fila['ci']
    return mejor_modelo, mejor_pred, mejor_ci

# Dibujar las ventas y sus predicciones
# def graficar_predicciones(valores_reales, test, nombre_modelo, pred, ci, color='b'):
#     fig, ax = plt.subplots(figsize=(10, 3))  # ✅ Crear figura y eje explícitamente
#     ax.plot(valores_reales.index, valores_reales.values, 'k-o', label='Real')
#     ax.plot(test.index, pred.values, f'{color}--', label=nombre_modelo)
#     if ci is not None:
#         ax.fill_between(test.index, ci.iloc[:, 0], ci.iloc[:, 1], color=color, alpha=0.3)
#     ax.set_title(f'{nombre_modelo}: Predicción')
#     ax.set_ylabel('Cantidad')
#     ax.legend()
#     ax.grid(True)
#     return fig

# Predecir
def ajustar_modelo(datos, periodo_estacional):
    if len(datos) < 2 * periodo_estacional:
        modelo = ExponentialSmoothing(datos, seasonal=None)
    else:
        modelo = ExponentialSmoothing(datos, seasonal='add', seasonal_periods=periodo_estacional)
    modelo_fit = modelo.fit()
    return modelo_fit

periodo_estacional = 12

def predecir(
    ventas_df,
    fecha_objetivo,
    codigo_art=None,
    regresores=['Precio', 'Precio_Costo', 'Ganancia'],
    forward_fill_regresores=False,
    graficar=False
):

    fecha_objetivo = pd.to_datetime(fecha_objetivo)
    fecha_mes = fecha_objetivo.to_period('M').to_timestamp()

    # Filtrar por código si se especifica
    if codigo_art:
        ventas_df = ventas_df[ventas_df['Descripcion_Art'] == codigo_art]
        if ventas_df.empty:
            raise ValueError(f"No hay datos para el Código_Art: {codigo_art}")

    modelos = []

    # --- Modelos sin regresores ---
    # Sin ajuste
    ventas_mensuales, _ = agregar_mensual(ventas_df)
    train, test = dividir_train_test(ventas_mensuales)
    pasos = (fecha_mes.to_period('M') - train.index[-1].to_period('M')).n
    if pasos <= 0:
        raise ValueError("La fecha objetivo debe ser posterior al último mes del conjunto de entrenamiento.")

    modelo_pro = entrenar_prophet(train)
    pred_pro, ci_pro = predecir_prophet(modelo_pro, test)
    modelos.append(('Prophet', pred_pro, ci_pro, test, 'Cantidad'))

    modelo_sarima = entrenar_sarima(train)
    pred_sarima, ci_sarima = predecir_sarima(modelo_sarima, steps=len(test))
    modelos.append(('SARIMA', pred_sarima, ci_sarima, test, 'Cantidad'))

    # modelo_hw = entrenar_ets(train)
    modelo_hw = ajustar_modelo(train, periodo_estacional)
    pred_hw, ci_lo, ci_hi = predecir_ets(modelo_hw, len(test), train)
    ci_hw = pd.concat([ci_lo, ci_hi], axis=1)
    ci_hw.columns = ['yhat_lower', 'yhat_upper']
    modelos.append(('ETS', pred_hw, ci_hw, test, 'Cantidad'))

    # Con ajuste
    ventas_mensuales2 = recalcular_2020_ventas_mensuales(ventas_mensuales.copy())
    train2, test2 = dividir_train_test(ventas_mensuales2)
    modelo_pro2 = entrenar_prophet(train2)
    pred_pro2, ci_pro2 = predecir_prophet(modelo_pro2, test2)
    modelos.append(('Prophet ajustado', pred_pro2, ci_pro2, test2, 'Cantidad'))

    modelo_sarima2 = entrenar_sarima(train2)
    pred_sarima2, ci_sarima2 = predecir_sarima(modelo_sarima2, steps=len(test2))
    modelos.append(('SARIMA ajustado', pred_sarima2, ci_sarima2, test2, 'Cantidad'))

    # modelo_hw2 = entrenar_ets(train2)
    modelo_hw2 = ajustar_modelo(train2, periodo_estacional)
    pred_hw2, ci_lo2, ci_hi2 = predecir_ets(modelo_hw2, len(test2), train2)
    ci_hw2 = pd.concat([ci_lo2, ci_hi2], axis=1)
    ci_hw2.columns = ['yhat_lower', 'yhat_upper']
    modelos.append(('ETS ajustado', pred_hw2, ci_hw2, test2, 'Cantidad'))

    # --- Modelos con regresores ---
    # Sin ajuste
    _, ventas_mensuales_regre = agregar_mensual(ventas_df)
    train_regr, test_regr = dividir_train_test(ventas_mensuales_regre)
    if len(train_regr) > 0 and len(test_regr) > 0:
        modelo_pro_regr = entrenar_prophet_regresores(train_regr, regresores)
        pred_pro_regr, ci_pro_regr = predecir_prophet_regresores(modelo_pro_regr, test_regr, regresores)
        modelos.append(('Prophet regresores', pred_pro_regr, ci_pro_regr, test_regr, 'Cantidad'))

        modelo_sarima_regr = entrenar_sarima_regresores(train_regr, regresores)
        pred_sarima_regr, ci_sarima_regr = predecir_sarima_regresores(modelo_sarima_regr, test_regr, regresores)
        modelos.append(('SARIMA regresores', pred_sarima_regr, ci_sarima_regr, test_regr, 'Cantidad'))

    # Con ajuste
    ventas_mensuales_regre2 = recalcular_2020_ventas_mensuales_regr(ventas_mensuales_regre.copy())
    train_regr2, test_regr2 = dividir_train_test(ventas_mensuales_regre2)
    if len(train_regr2) > 0 and len(test_regr2) > 0:
        modelo_pro_regr2 = entrenar_prophet_regresores(train_regr2, regresores)
        pred_pro_regr2, ci_pro_regr2 = predecir_prophet_regresores(modelo_pro_regr2, test_regr2, regresores)
        modelos.append(('Prophet regresores ajustado', pred_pro_regr2, ci_pro_regr2, test_regr2, 'Cantidad'))

        modelo_sarima_regr2 = entrenar_sarima_regresores(train_regr2, regresores)
        pred_sarima_regr2, ci_sarima_regr2 = predecir_sarima_regresores(modelo_sarima_regr2, test_regr2, regresores)
        modelos.append(('SARIMA regresores ajustado', pred_sarima_regr2, ci_sarima_regr2, test_regr2, 'Cantidad'))

    # --- Evaluar y seleccionar el mejor modelo ---
    mejor_nombre, mejor_pred, mejor_ci = evaluar_y_seleccionar_mejor_modelo_completo(modelos)

    # --- Predecir para la fecha objetivo con el mejor modelo ---
    # Modelos sin regresores
    if mejor_nombre == 'Prophet':
        df_pred = pd.DataFrame({'ds': [fecha_mes]})
        fc = modelo_pro.predict(df_pred).iloc[0]
        prediccion = {'modelo': mejor_nombre, 'prediccion': fc['yhat'], 'intervalo': (fc['yhat_lower'], fc['yhat_upper'])}
        valores_reales = ventas_mensuales
        test_plot = test
        pred_plot = pred_pro
        ci_plot = ci_pro
    elif mejor_nombre == 'SARIMA':
        pred, ci = predecir_sarima(modelo_sarima, steps=pasos)
        prediccion = {'modelo': mejor_nombre, 'prediccion': pred.iloc[-1], 'intervalo': tuple(ci.iloc[-1])}
        valores_reales = ventas_mensuales
        test_plot = test
        pred_plot = pred_sarima
        ci_plot = ci_sarima
    elif mejor_nombre == 'ETS':
        pred, ci_lo, ci_hi = predecir_ets(modelo_hw, steps=pasos, train=train)
        prediccion = {'modelo': mejor_nombre, 'prediccion': pred.iloc[-1], 'intervalo': (ci_lo.iloc[-1], ci_hi.iloc[-1])}
        valores_reales = ventas_mensuales
        test_plot = test
        pred_plot = pred_hw
        ci_plot = pd.concat([ci_lo, ci_hi], axis=1)
        ci_plot.columns = ['yhat_lower', 'yhat_upper']
    # Modelos sin regresores con ajuste
    elif mejor_nombre == 'Prophet ajustado':
        pasos_aj = (fecha_mes.to_period('M') - train2.index[-1].to_period('M')).n
        df_pred = pd.DataFrame({'ds': [fecha_mes]})
        fc = modelo_pro2.predict(df_pred).iloc[0]
        prediccion = {'modelo': mejor_nombre, 'prediccion': fc['yhat'], 'intervalo': (fc['yhat_lower'], fc['yhat_upper'])}
        valores_reales = ventas_mensuales2
        test_plot = test2
        pred_plot = pred_pro2
        ci_plot = ci_pro2
    elif mejor_nombre == 'SARIMA ajustado':
        pasos_aj = (fecha_mes.to_period('M') - train2.index[-1].to_period('M')).n
        pred, ci = predecir_sarima(modelo_sarima2, steps=pasos_aj)
        prediccion = {'modelo': mejor_nombre, 'prediccion': pred.iloc[-1], 'intervalo': tuple(ci.iloc[-1])}
        valores_reales = ventas_mensuales2
        test_plot = test2
        pred_plot = pred_sarima2
        ci_plot = ci_sarima2
    elif mejor_nombre == 'ETS ajustado':
        pasos_aj = (fecha_mes.to_period('M') - train2.index[-1].to_period('M')).n
        pred, ci_lo, ci_hi = predecir_ets(modelo_hw2, steps=pasos_aj, train=train2)
        prediccion = {'modelo': mejor_nombre, 'prediccion': pred.iloc[-1], 'intervalo': (ci_lo.iloc[-1], ci_hi.iloc[-1])}
        valores_reales = ventas_mensuales2
        test_plot = test2
        pred_plot = pred_hw2
        ci_plot = pd.concat([ci_lo, ci_hi], axis=1)
        ci_plot.columns = ['yhat_lower', 'yhat_upper']
    # Modelos con regresores
    elif mejor_nombre == 'Prophet regresores':
        if fecha_mes not in train_regr.index:
            if not forward_fill_regresores:
                raise ValueError("No hay valores de regresores para la fecha objetivo en Prophet regresores. Usa forward_fill_regresores=True para usar el último valor conocido.")
            fila = train_regr.iloc[[-1]][regresores]
        else:
            fila = train_regr.loc[[fecha_mes], regresores]
        df_pred = pd.DataFrame({'ds': [fecha_mes], **{reg: fila[reg].values for reg in regresores}})
        fc = modelo_pro_regr.predict(df_pred).iloc[0]
        prediccion = {'modelo': mejor_nombre, 'prediccion': fc['yhat'], 'intervalo': (fc['yhat_lower'], fc['yhat_upper'])}
        valores_reales = ventas_mensuales_regre['Cantidad']
        test_plot = test_regr
        pred_plot = pred_pro_regr
        ci_plot = ci_pro_regr
    elif mejor_nombre == 'SARIMA regresores':
        if fecha_mes not in train_regr.index:
            if not forward_fill_regresores:
                raise ValueError("No hay valores de regresores para la fecha objetivo en SARIMA regresores. Usa forward_fill_regresores=True para usar el último valor conocido.")
            fila = train_regr.iloc[[-1]][regresores]
        else:
            fila = train_regr.loc[[fecha_mes], regresores]
        pred, ci = predecir_sarima_regresores(modelo_sarima_regr, fila, regresores)
        prediccion = {'modelo': mejor_nombre, 'prediccion': pred.iloc[-1], 'intervalo': tuple(ci.iloc[-1])}
        valores_reales = ventas_mensuales_regre['Cantidad']
        test_plot = test_regr
        pred_plot = pred_sarima_regr
        ci_plot = ci_sarima_regr
    # Modelos con regresores y ajuste
    elif mejor_nombre == 'Prophet regresores ajustado':
        if fecha_mes not in train_regr2.index:
            if not forward_fill_regresores:
                raise ValueError("No hay valores de regresores para la fecha objetivo en Prophet regresores ajustado. Usa forward_fill_regresores=True para usar el último valor conocido.")
            fila = train_regr2.iloc[[-1]][regresores]
        else:
            fila = train_regr2.loc[[fecha_mes], regresores]
        df_pred = pd.DataFrame({'ds': [fecha_mes], **{reg: fila[reg].values for reg in regresores}})
        fc = modelo_pro_regr2.predict(df_pred).iloc[0]
        prediccion = {'modelo': mejor_nombre, 'prediccion': fc['yhat'], 'intervalo': (fc['yhat_lower'], fc['yhat_upper'])}
        valores_reales = ventas_mensuales_regre2['Cantidad']
        test_plot = test_regr2
        pred_plot = pred_pro_regr2
        ci_plot = ci_pro_regr2
    elif mejor_nombre == 'SARIMA regresores ajustado':
        if fecha_mes not in train_regr2.index:
            if not forward_fill_regresores:
                raise ValueError("No hay valores de regresores para la fecha objetivo en SARIMA regresores ajustado. Usa forward_fill_regresores=True para usar el último valor conocido.")
            fila = train_regr2.iloc[[-1]][regresores]
        else:
            fila = train_regr2.loc[[fecha_mes], regresores]
        pred, ci = predecir_sarima_regresores(modelo_sarima_regr2, fila, regresores)
        prediccion = {'modelo': mejor_nombre, 'prediccion': pred.iloc[-1], 'intervalo': tuple(ci.iloc[-1])}
        valores_reales = ventas_mensuales_regre2['Cantidad']
        test_plot = test_regr2
        pred_plot = pred_sarima_regr2
        ci_plot = ci_sarima_regr2
    else:
        raise ValueError("Modelo no reconocido por la función de selección.")

    # --- Graficar si se solicita ---
    # figura = None
    # if graficar:
    #     figura = graficar_predicciones(valores_reales, test_plot, mejor_nombre, pred_plot, ci_plot, color='b') 

    return {
        'modelo': mejor_nombre,
        'prediccion': prediccion['prediccion'],
        'min': prediccion['intervalo'][0],
        'max': prediccion['intervalo'][1],
        # 'figura': figura
    }



    