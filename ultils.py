import json
import csv
import pandas as pd

def json_to_dataframe(json_data):
    """
    Chuyển dữ liệu JSON thành DataFrame.
    
    Args:
        json_data (list): Danh sách dữ liệu JSON.

    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu từ JSON.
    """
    # Tạo DataFrame từ JSON
    df = pd.DataFrame(json_data)
    return df