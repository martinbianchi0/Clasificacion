def label_encode(data, column, mapping=None):
    """
    Codifica una columna categórica a valores numéricos.

    Params:
        data (pd.DataFrame): DataFrame de entrada.
        column (str): Columna a codificar.
        mapping (dict, opcional): Diccionario de mapeo personalizado.

    Returns:
        pd.DataFrame: DataFrame con la columna codificada.
    """
    if mapping is not None:
        data[column] = data[column].map(mapping)
    else:
        data[column] = data[column].astype('category').cat.codes
    return data