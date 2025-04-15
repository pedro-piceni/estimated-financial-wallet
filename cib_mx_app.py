import streamlit as st
import altair as alt
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression


# Listas originales
sectores = [
    #"Energía y Combustibles", 
    #"Tecnología y Servicios", 
    #"Construcción e Infraestructura", 
    "Alimentos y Bebidas",
    "Embotelladora", 
    #"Automotriz y Transporte", 
    #"Materiales e Industria", 
    "Comercio y Retail", 
    #"Otros"
]

pagos = [
    "Tarjetas de débito/crédito",
    "Efectivo",
    "Pagos electrónicos",
    "Transferencias",
    "Cheque"
]



# Crear un diccionario para almacenar los datos
# La estructura será: {(sector, tipo_pago): (porcentaje_recaudo, margen)}
# Diccionario con todas las combinaciones
datos_sector_pago = {
    # Energía y Combustibles
    ("Energía y Combustibles", "Tarjetas de débito/crédito"): (50, 0.5),
    ("Energía y Combustibles", "Efectivo"): (50, 0.085),

    # Tecnología y Servicios
    ("Tecnología y Servicios", "Efectivo"): (40, 0.085),
    ("Tecnología y Servicios", "Pagos electrónicos"): (35, 0.8),
    ("Tecnología y Servicios", "Tarjetas de débito/crédito"): (15, 0.6),
    ("Tecnología y Servicios", "Transferencias"): (10, 0),

    # Construcción e Infraestructura
    ("Construcción e Infraestructura", "Transferencias"): (45, 0),
    ("Construcción e Infraestructura", "Pagos electrónicos"): (35, 0.02),
    ("Construcción e Infraestructura", "Cheque"): (20, 0.5),

    # Alimentos y Bebidas
    ("Alimentos y Bebidas", "Tarjetas de débito/crédito"): (40, 0.008),
    ("Alimentos y Bebidas", "Efectivo"): (30, 0.07),
    ("Alimentos y Bebidas", "Transferencias"): (20, 0),
    ("Alimentos y Bebidas", "Pagos electrónicos"): (10, 0.02),

    # Embotelladora
    ("Embotelladora", "Tarjetas de débito/crédito"): (15, 0.08),
    ("Embotelladora", "Efectivo"): (85, 0.07),
    ("Embotelladora", "Transferencias"): (0, 0),
    ("Embotelladora", "Pagos electrónicos"): (0, 0.02),

    # Automotriz y Transporte
    ("Automotriz y Transporte", "Pagos electrónicos"): (35, 0.02),
    ("Automotriz y Transporte", "Transferencias"): (30, 0),
    ("Automotriz y Transporte", "Financiamiento"): (30, 0),
    ("Automotriz y Transporte", "Tarjetas de débito/crédito"): (5, 0.9),

    # Materiales e Industria
    ("Materiales e Industria", "Pagos electrónicos"): (35, 0.02),
    ("Materiales e Industria", "Efectivo"): (25, 0.085),
    ("Materiales e Industria", "Transferencias"): (25, 0),
    ("Materiales e Industria", "Tarjetas de débito/crédito"): (10, 1.9),
    ("Materiales e Industria", "Cheque"): (5, 0.5),

    # Comercio y Retail
    ("Comercio y Retail", "Efectivo"): (45, 0.085),
    ("Comercio y Retail", "Tarjetas de débito/crédito"): (40, 1.2),
    ("Comercio y Retail", "Pagos electrónicos"): (15, 0.02),

    # Otros
    ("Otros", "Tarjetas de débito/crédito"): (55, 1.0),
    ("Otros", "Efectivo"): (40, 0.085),
    ("Otros", "Financiamiento"): (5, 0)
}

# Crear listas para almacenar los datos
sector_list = []
tipo_pago_list = []
prct_recaudo_list = []
margen_list = []

# Iterar sobre todas las combinaciones posibles
for sector in sectores:
    for pago in pagos:
        # Si existe la combinación en el diccionario, usar esos valores
        if (sector, pago) in datos_sector_pago:
            prct_recaudo, margen = datos_sector_pago[(sector, pago)]
        else:
            # Si no existe, usar valores por defecto (puedes ajustar esto)
            prct_recaudo, margen = 0, 0
        
        # Añadir a las listas
        sector_list.append(sector)
        tipo_pago_list.append(pago)
        prct_recaudo_list.append(prct_recaudo)
        margen_list.append(margen)

# Crear el DataFrame
df_sector_payments = pd.DataFrame({
    'sector': sector_list,
    'tipo_de_pago': tipo_pago_list,
    'prct_recaudo': prct_recaudo_list,
    'margen': margen_list
})

script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "financial_data_mx.csv"

file_path = os.path.join(script_dir, file_name)
print(file_path)

df = pd.read_csv(file_path)
print(df.columns)

# Aquí inicia la app

st.set_page_config(page_title="CIB Estimated Financial Services Wallet",
                   layout="wide",
                   initial_sidebar_state="expanded")

with st.sidebar:
    st.title("CIB Estimated Financial Services Wallet")
    
        # Multiselect for Year
    year_list = df["Year"].unique()
    selected_years = st.multiselect("Año", year_list, default=year_list)

    # Color theme fijo en 'blues'
    selected_color_theme = 'blues'

    # Selectbox for Sector
    sector_list = df_sector_payments["sector"].unique()
    selected_sector = st.selectbox("Sector", ["None"] + sorted(list(sector_list)))

    # Filter companies based on selected sector
    if selected_sector == "None":
        filtered_companies = df["Company"].unique()
    else:
        filtered_companies = df[df["Sector"] == selected_sector]["Company"].unique()

    # Selectbox for Company
    selected_company = st.selectbox("Empresa", ["None"] + sorted(list(filtered_companies)))

    # Añadir cajas de texto para modificar los porcentajes de recaudo según el sector seleccionado
if selected_sector != "None":
    st.sidebar.markdown("## Porcentajes de recaudo para " + selected_sector)
    
    # Filtrar las claves del diccionario datos_sector_pago para el sector seleccionado
    sector_payment_keys = [key for key in datos_sector_pago.keys() if key[0] == selected_sector]
    
    # Crear una caja de texto para cada tipo de pago del sector seleccionado
    for key in sector_payment_keys:
        sector, payment_type = key
        original_percentage, margin = datos_sector_pago[key]
        
        # Crear una caja de texto para este tipo de pago
        new_percentage = st.sidebar.text_input(
            f"% Recaudo para {payment_type}", 
            value=str(original_percentage),
            key=f"input_{sector}_{payment_type}"
        )
        
        # Actualizar el valor en el diccionario si se ingresó algo
        try:
            new_percentage_value = float(new_percentage)
            # Verificar que el porcentaje esté entre 0 y 100
            if 0 <= new_percentage_value <= 100:
                datos_sector_pago[key] = (new_percentage_value, margin)
            else:
                st.sidebar.warning(f"El porcentaje para {payment_type} debe estar entre 0 y 100")
        except ValueError:
            # Si no se puede convertir a float, mantener el valor original
            pass

# Recrear el DataFrame después de las modificaciones del usuario
sector_list = []
tipo_pago_list = []
prct_recaudo_list = []
margen_list = []

# Iterar sobre todas las combinaciones posibles
for sector in sectores:
    for pago in pagos:
        # Si existe la combinación en el diccionario, usar esos valores
        if (sector, pago) in datos_sector_pago:
            prct_recaudo, margen = datos_sector_pago[(sector, pago)]
        else:
            # Si no existe, usar valores por defecto
            prct_recaudo, margen = 0, 0
        
        # Añadir a las listas
        sector_list.append(sector)
        tipo_pago_list.append(pago)
        prct_recaudo_list.append(prct_recaudo)
        margen_list.append(margen)

# Recrear el DataFrame
df_sector_payments = pd.DataFrame({
    'sector': sector_list,
    'tipo_de_pago': tipo_pago_list,
    'prct_recaudo': prct_recaudo_list,
    'margen': margen_list
})

# Filter the dataframe based on selections
if selected_company != "None":
    filtered_df = df[(df["Company"] == selected_company) & (df["Year"].isin(selected_years))]
elif selected_sector != "None":
    filtered_df = df[(df["Sector"] == selected_sector) & (df["Year"].isin(selected_years))]
else:
    filtered_df = df[df["Year"].isin(selected_years)]

# Divide numeric columns by 1000 in filtered_df
numeric_columns = ['Ingresos_netos_ventas', 'Activos_financieros_corto_plazo', 'Efectivo_Equivalentes', 
                  'Gastos_de_venta_y_distribución', 'Costo_mercancia_vendida']
filtered_df[numeric_columns] = filtered_df[numeric_columns] / 1000

# Merge df_sector_payments with filtered_df on 'Sector'
collected_income_df = df_sector_payments.merge(df[['Company', 'Sector', 'Year', 'Ingresos_netos_ventas']], 
                                               left_on='sector', right_on='Sector', how='left')

# Calculate Ingreso_Recaudado
collected_income_df['Ingreso_Recaudado'] = collected_income_df['prct_recaudo'] / 100 * collected_income_df['Ingresos_netos_ventas'] * collected_income_df['margen']/100

# Divide Ingreso_Recaudado by 1000
collected_income_df['Ingreso_Recaudado'] = collected_income_df['Ingreso_Recaudado'] / 1000

# Select the required columns
collected_income_df = collected_income_df[['Company', 'Sector', 'Year', 'tipo_de_pago', 'Ingreso_Recaudado']]  

# Apply the same filtering logic to collected_income_df
if selected_company != "None":
    collected_income_df = collected_income_df[collected_income_df["Company"] == selected_company]
elif selected_sector != "None":
    collected_income_df = collected_income_df[collected_income_df["Sector"] == selected_sector]

collected_income_df = collected_income_df[collected_income_df["Year"].isin(selected_years)]


def histogram_chart(collected_income_df, selected_color_theme):
    # Group by Company and tipo_de_pago, summing Ingreso_Recaudado
    company_totals = collected_income_df.groupby(['Company', 'tipo_de_pago'])['Ingreso_Recaudado'].sum().reset_index()
    
    # Create a stacked bar chart
    chart = alt.Chart(company_totals).mark_bar().encode(
        x=alt.X('Company:N', sort='-y'),
        y=alt.Y('Ingreso_Recaudado:Q', title='Total Ingreso Recaudado'),
        color=alt.Color('tipo_de_pago:N', scale=alt.Scale(scheme=selected_color_theme)),
        tooltip=['Company', 'tipo_de_pago', 'Ingreso_Recaudado']
    ).properties(
        width=600,
        height=400
    ).interactive()
    
    return chart

def plot_revenue_pool_altair(revenue_pivot_df):
    """
    Crea un gráfico de barras apiladas basado en el DataFrame `revenue_pivot_df` utilizando Altair.

    Args:
        revenue_pivot_df (pd.DataFrame): DataFrame con los datos de Revenue Pool.
    """
    # Convertir las columnas a valores numéricos (eliminar formato de moneda si es necesario)
    revenue_pivot_df_numeric = revenue_pivot_df.copy()
    for col in revenue_pivot_df_numeric.columns:
        revenue_pivot_df_numeric[col] = revenue_pivot_df_numeric[col].replace('[\$,]', '', regex=True).astype(float)

    # Transformar el DataFrame para que Altair pueda trabajar con él
    revenue_pivot_df_melted = revenue_pivot_df_numeric.reset_index().melt(
        id_vars='Year', 
        var_name='Tipo de Pago', 
        value_name='Ingreso Recaudado'
    )

    # Excluir la columna "Revenue Pool Total" del gráfico
    revenue_pivot_df_melted = revenue_pivot_df_melted[revenue_pivot_df_melted['Tipo de Pago'] != "Revenue Pool Total"]

    # Crear el gráfico de barras apiladas
    chart = alt.Chart(revenue_pivot_df_melted).mark_bar().encode(
        x=alt.X('Year:N', title='Año', sort='ascending', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('sum(Ingreso Recaudado):Q', title='Revenue Pool Total (en millones)'),
        color=alt.Color('Tipo de Pago:N', scale=alt.Scale(scheme=selected_color_theme), title='Tipo de Pago'),
        tooltip=['Year', 'Tipo de Pago', 'sum(Ingreso Recaudado)']
    ).properties(
        width=800,
        height=400
    )

    return chart

# Example layout with columns
col1, col2 = st.columns((1, 2))

# Define color map
color_map = {
    'blues': '#1f77b4',
    'cividis': '#e5e500',
    'greens': '#2ca02c',
    'reds': '#d62728',
    'rainbow': '#ff7f0e',
    'inferno': '#ff7f0e',
    'magma': '#d62728',
    'plasma': '#9467bd',
    'turbo': '#ff7f0e',
    'viridis': '#2ca02c'
}

# Get the color for the selected theme
selected_color = color_map.get('#1f77b4', '#000000')  # Default to black if theme not found

with col1:
    st.markdown("## Resumen de Totales")
    st.markdown("##### Millones de Pesos MXN")
    
    # Calculate sums for selected columns
    total_ingresos_netos_ventas = filtered_df["Ingresos_netos_ventas"].sum()
    total_activos_financieros_corto_plazo = filtered_df["Activos_financieros_corto_plazo"].sum()
    total_efectivo_equivalentes = filtered_df["Efectivo_Equivalentes"].sum()
    total_gastos_venta_distribucion = filtered_df["Gastos_de_venta_y_distribución"].sum()
    total_costo_mercancia_vendida = filtered_df["Costo_mercancia_vendida"].sum()
    
    # Display totals with titles and color in a more compact format
    st.markdown("""
    <div style='font-size:20px;'>
        <p style='margin:6px 0;'>Ingresos Netos de Ventas: 
            <span style='color:{}; font-size:20px; font-weight:bold;'>${:,.2f}</span>
        </p>
        <p style='margin:6px 0;'>Activos Financieros a Corto Plazo: 
            <span style='color:{}; font-size:20px; font-weight:bold;'>${:,.2f}</span>
        </p>
        <p style='margin:6px 0;'>Efectivo y Equivalentes: 
            <span style='color:{}; font-size:20px; font-weight:bold;'>${:,.2f}</span>
        </p>
        <p style='margin:6px 0;'>Gastos de venta y distribución: 
            <span style='color:{}; font-size:20px; font-weight:bold;'>${:,.2f}</span>
        </p>
        <p style='margin:6px 0;'>Costo de Mercancía Vendida: 
            <span style='color:{}; font-size:20px; font-weight:bold;'>${:,.2f}</span>
        </p>
    </div>
    """.format(
        '#1f77b4', total_ingresos_netos_ventas,
        '#1f77b4', total_activos_financieros_corto_plazo,
        '#1f77b4', total_efectivo_equivalentes,
        '#1f77b4', total_gastos_venta_distribucion,
        '#1f77b4', total_costo_mercancia_vendida
    ), unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("## Márgenes (%)")

    # Inject CSS to control slider width
    css = """
    <style>
        /* Target the container holding the slider to limit its width */
        div[data-testid="stSlider"] {
            max-width: 400px; /* You can adjust this pixel value */
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    st.markdown("#### Recaudo")
    # Create sliders for each payment type
    margin_debito_credito = st.slider("Tarjetas de débito/crédito", 
                                      min_value=0.0, max_value=2.0, 
                                      value=0.8, step=0.01,
                                      format="%.2f%%")
    margin_efectivo = st.slider("Efectivo", 
                                min_value=0.0, max_value=0.1, 
                                value=0.085, step=0.001, 
                                format="%.3f%%")
    margin_pagos_electronicos = st.slider("Pagos electrónicos", 
                                          min_value=0.0, max_value=2.0, 
                                          value=0.20, step=0.001, 
                                          format="%.2f%%")
    margin_transferencias = st.slider("Transferencias", 
                                      min_value=0.0, max_value=2.0, 
                                      value=0.0, step=0.001, 
                                      format="%.2f%%")
    margin_cheque = st.slider("Cheque", 
                              min_value=0.0, max_value=2.0, 
                              value=0.0, step=0.001, 
                              format="%.2f%%")
    
    st.markdown("#### Pagos")
    margin_proveedores = st.slider("Proveedores", 
                                   min_value=0.0, max_value=5.0, 
                                   value=0.01, step=0.01, 
                                   format="%.2f%%")
    margin_nomina = st.slider("Nómina", 
                              min_value=0.0, max_value=5.0, 
                              value=0.01, step=0.01, 
                              format="%.2f%%")

    # Update margins in df_sector_payments based on slider values
    df_sector_payments.loc[df_sector_payments['tipo_de_pago'] == 'Tarjetas de débito/crédito', 'margen'] = margin_debito_credito
    df_sector_payments.loc[df_sector_payments['tipo_de_pago'] == 'Efectivo', 'margen'] = margin_efectivo
    df_sector_payments.loc[df_sector_payments['tipo_de_pago'] == 'Pagos electrónicos', 'margen'] = margin_pagos_electronicos
    df_sector_payments.loc[df_sector_payments['tipo_de_pago'] == 'Transferencias', 'margen'] = margin_transferencias
    df_sector_payments.loc[df_sector_payments['tipo_de_pago'] == 'Cheque', 'margen'] = margin_cheque
    df_sector_payments.loc[df_sector_payments['tipo_de_pago'] == 'Pagos de proveedores', 'margen'] = margin_proveedores
    df_sector_payments.loc[df_sector_payments['tipo_de_pago'] == 'Pagos de nómina', 'margen'] = margin_nomina

# Calcular Soluciones de Pago a Proveedores y Soluciones de Pagos de Nómina
filtered_df['Soluciones_de_Pago'] = (filtered_df['Costo_mercancia_vendida'] * margin_proveedores / 100) * -1
filtered_df['Soluciones_de_Nomina'] = (filtered_df['Gastos_de_venta_y_distribución'] * margin_nomina / 100) * -1

# Crear un DataFrame con Pagos de Proveedores y Nómina por año
pagos_por_año = filtered_df.groupby('Year')[['Soluciones_de_Pago', 'Soluciones_de_Nomina']].sum().reset_index()
pagos_por_año.columns = ['Year', 'Pago a Proveedores', 'Pagos de Nómina']

pagos_por_año['Year'] = pagos_por_año['Year'].astype(str)  # Convert Year to string for display

pagos_por_año = pagos_por_año.set_index('Year')

pagos_por_año_display = pagos_por_año.copy()
pagos_por_año_display['Pago a Proveedores'] = pagos_por_año_display['Pago a Proveedores'].apply(lambda x: f"${x:,.2f}")
pagos_por_año_display['Pagos de Nómina'] = pagos_por_año_display['Pagos de Nómina'].apply(lambda x: f"${x:,.2f}")

# Recalculate collected_income_df with updated margins
collected_income_df = df_sector_payments.merge(df[['Company', 'Sector', 'Year', 'Ingresos_netos_ventas']], 
                                               left_on='sector', right_on='Sector', how='left')

collected_income_df['Ingreso_Recaudado'] = (collected_income_df['prct_recaudo'] / 100) * (collected_income_df['Ingresos_netos_ventas'] / 1000) * (collected_income_df['margen'] / 100)

collected_income_df = collected_income_df[['Company', 'Sector', 'Year', 'tipo_de_pago', 'Ingreso_Recaudado']]

# Apply the same filtering logic to collected_income_df
if selected_company != "None":
    collected_income_df = collected_income_df[collected_income_df["Company"] == selected_company]
elif selected_sector != "None":
    collected_income_df = collected_income_df[collected_income_df["Sector"] == selected_sector]

collected_income_df = collected_income_df[collected_income_df["Year"].isin(selected_years)]

#################
future_years = 3
predicted_rows = []

grouped = collected_income_df.dropna().groupby(['Company', 'tipo_de_pago'])

for (company, tipo_pago), group in grouped:
    group = group.sort_values('Year')
   
    if group['Ingreso_Recaudado'].nunique() < 2:
        continue  # Evitar problemas con datos constantes o insuficientes

    X = group[['Year']]
    y = group['Ingreso_Recaudado']

    # Ajustar modelo lineal
    model = LinearRegression()
    model.fit(X, y)

    last_year = group['Year'].max()
    last_value = y.iloc[-1]

    for i in range(1, future_years + 1):
        future_year = last_year + i
        predicted_value = model.predict([[future_year]])[0]

        if predicted_value < 0:
            predicted_value = last_value

        sector = group['Sector'].iloc[0]

        predicted_rows.append({
            'Company': company,
            'Sector': sector,
            'Year': future_year,
            'tipo_de_pago': tipo_pago,
            'Ingreso_Recaudado': predicted_value
        })

# Convertir predicciones a DataFrame y añadirlas
predicted_df = pd.DataFrame(predicted_rows)
collected_income_df = pd.concat([collected_income_df, predicted_df], ignore_index=True)
#################

# Filtrar para excluir "Financiamiento"
collected_income_df = collected_income_df[collected_income_df['tipo_de_pago'] != 'Financiamiento']

with col2:
    # Format numeric columns as currency in filtered_df for display
    filtered_df_display = filtered_df.copy()
    for col in ['Ingresos_netos_ventas', 'Activos_financieros_corto_plazo', 'Efectivo_Equivalentes', 
                 'Gastos_de_venta_y_distribución', 'Costo_mercancia_vendida']:
        filtered_df_display[col] = filtered_df_display[col].apply(lambda x: f"${x:,.2f}")

    # Format numeric columns as currency in collected_income_df for display
    collected_income_df_display = collected_income_df.copy()
    for col in ['Ingreso_Recaudado']:
        collected_income_df_display[col] = collected_income_df_display[col].apply(lambda x: f"${x:,.2f}")

    # Transform filtered_df for display
    indicators = ['Ingresos_netos_ventas', 'Activos_financieros_corto_plazo', 'Efectivo_Equivalentes', 
                 'Gastos_de_venta_y_distribución', 'Costo_mercancia_vendida', 'Soluciones_de_Pago', 'Soluciones_de_Nomina']
    
    # Create a new dataframe with indicators as rows and years as columns
    pivot_data = {}
    years = sorted(filtered_df['Year'].unique())  # Sort years to ensure chronological order
    for year in years:
        pivot_data[str(year)] = {}  # Convert year to string to use as column name
        for indicator in indicators:
            if indicator in filtered_df.columns:
                pivot_data[str(year)][indicator] = filtered_df[filtered_df['Year'] == year][indicator].sum()

    # Convert to dataframe and set index
    pivot_df = pd.DataFrame(pivot_data)
    pivot_df.index.name = 'Years'  # Name the index
    
    # Format numeric values as currency
    for col in pivot_df.columns:
        pivot_df[col] = pivot_df[col].apply(lambda x: f"${x:,.2f}")

    # Eliminar las últimas dos filas de pivot_df
    pivot_df = pivot_df.iloc[:-2]

    # Create pivot table for Revenue Pool
    revenue_pivot_data = {}
    years = sorted(collected_income_df['Year'].unique())  # Sort years to ensure chronological order
    payment_types = collected_income_df['tipo_de_pago'].unique()
    
    for year in years:
        revenue_pivot_data[str(year)] = {}
        for payment in payment_types:
            revenue_pivot_data[str(year)][payment] = collected_income_df[
                (collected_income_df['Year'] == year) & 
                (collected_income_df['tipo_de_pago'] == payment)
            ]['Ingreso_Recaudado'].sum()

    # Convert to dataframe
    revenue_pivot_df = pd.DataFrame(revenue_pivot_data).T
    revenue_pivot_df.index.name = 'Year'
    revenue_pivot_df["Revenue Pool Total"] = revenue_pivot_df.select_dtypes(include="number").sum(axis=1)

    # Format numeric values as currency
    for col in revenue_pivot_df.columns:
        revenue_pivot_df[col] = revenue_pivot_df[col].apply(lambda x: f"${x:,.2f}")

    st.markdown("## Revenue Pool Anual")
    # Create and display the bar chart
    #evolution_chart = histogram_chart(collected_income_df, selected_color_theme)
    #st.altair_chart(evolution_chart)
    revenue_pool_chart = plot_revenue_pool_altair(revenue_pivot_df)
    st.altair_chart(revenue_pool_chart, use_container_width = True)

    st.markdown("## Revenue Pool")
    st.dataframe(revenue_pivot_df)

    st.markdown("## Soluciones de Pago")
    st.dataframe(pagos_por_año_display)

    st.markdown("## Cuentas Financieras")
    st.dataframe(pivot_df)

    st.markdown("### Revenue Pool por Empresa")
    st.dataframe(collected_income_df_display)
