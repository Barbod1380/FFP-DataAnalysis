import pandas as pd
import numpy as np

def float_to_clock(time_float):
    if pd.isna(time_float):
        return None  # or return "NaN" or ""

    total_minutes = time_float * 24 * 60
    hours = int(total_minutes // 60)
    minutes = int(round(total_minutes % 60))
    return f"{hours:02d}:{minutes:02d}"


def parse_clock(clock_str):
    """
    Parse clock string format (e.g. "5:30") to decimal hours (e.g. 5.5)
    """
    try:
        hours, minutes = map(int, clock_str.split(":"))
        return hours + minutes / 60
    except Exception:
        return np.nan

def decimal_to_clock_str(decimal_hours):
    """
    Convert decimal hours to clock format string.
    Example: 5.9 → "5:54"
    
    Parameters:
    - decimal_hours: Clock position in decimal format
    
    Returns:
    - String in clock format "H:MM"
    """
    if pd.isna(decimal_hours):
        return "Unknown"
    
    # Ensure the value is between 1 and 12
    if decimal_hours < 1:
        decimal_hours += 12
    elif decimal_hours > 12:
        decimal_hours = decimal_hours % 12
        if decimal_hours == 0:
            decimal_hours = 12
    
    hours = int(decimal_hours)
    minutes = int((decimal_hours - hours) * 60)
    
    return f"{hours}:{minutes:02d}"