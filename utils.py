import pandas as pd
import numpy as np

def float_to_clock(time_float):
    """
    Convert a floating point number to clock format string (HH:MM).
    Assumes the input is a decimal representation of hours.
    
    Parameters:
    - time_float: Clock position in decimal format (e.g. 11.4 for 11:24)
    
    Returns:
    - String in clock format "HH:MM"
    """
    if pd.isna(time_float):
        return None  # or return "NaN" or ""
    
    # Ensure the value is in the range 1-12
    if time_float < 1:
        time_float += 12
    elif time_float > 12:
        time_float = time_float % 12
        if time_float == 0:
            time_float = 12
    
    # Calculate hours and minutes
    hours = int(time_float)
    minutes = int(round((time_float - hours) * 60))
    
    # Handle case where rounding minutes results in 60
    if minutes == 60:
        minutes = 0
        hours += 1
        if hours > 12:
            hours = 1
    
    return f"{hours:01d}:{minutes:02d}"

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