import os
import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr, linregress
from scipy.special import wofz
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pandas as pd
import sqlite3
import h5py

# Database path resolves to LIBSApp/databases/LIBS_data.db relative to this file.
# Allow override via APP_BASE_PATH environment variable (used by run_app.py /
# PyInstaller bundles).
_DEFAULT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_DIR = os.environ.get("APP_BASE_PATH", _DEFAULT_BASE)
DATABASE_PATH = os.path.join(_BASE_DIR, "databases", "LIBS_data.db")
PARTITION_FUNCTION_PATH = DATABASE_PATH  # For backward compatibility
EION_PATH = DATABASE_PATH  # For backward compatibility

def snv(data):
    """
    Standard Normal Variate (SNV) normalization.
    Normalizes each row to have zero mean and unit variance.
    
    Handles zero variance cases (rows with all identical values) by adding
    a small epsilon to prevent division by zero, which would produce inf/nan values.
    """
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_std = np.std(data, axis=1, keepdims=True)
    
    # Add small epsilon to prevent division by zero when variance is zero
    # When std is zero, mean-centered data is already zero, so result will be zero
    epsilon = 1e-10
    data_std = np.maximum(data_std, epsilon)
    
    data_snv = (data - data_mean) / data_std
    return data_snv

def movingMinimum(listSpectra, m=200, n=50):
    """
    Background subtraction using moving minimum method.
    
    This function performs background subtraction by:
    1. Calculating a moving minimum with window width m
    2. Calculating a moving average of the minimum with window width n
    3. Subtracting the smoothed background from the original spectra
    
    Parameters:
    -----------
    listSpectra : array-like
        Array of spectral data (1D array)
    m : int, optional
        Window width for minimum calculation (default: 200)
    n : int, optional
        Window width for averaging (default: 50)
    
    Returns:
    --------
    numpy.ndarray
        Background-subtracted spectra (listSpectra - meanBackground)
    
    Notes:
    ------
    Edge conditions are handled by using available data:
    - At the beginning: uses data from index 0 to j+(m/2)
    - At the end: uses data from index j-(m/2) to end
    - In the middle: uses symmetric window around current point
    """
    listSpectra = np.array(listSpectra, dtype=np.float64)
    length = len(listSpectra)
    
    minBackground = np.zeros(length, dtype=np.float64)
    meanBackground = np.zeros(length, dtype=np.float64)
    
    # Calculate moving minimum
    m_half = int(m / 2)
    for j in range(length):
        if j <= m_half:  # Edge condition at the beginning
            minPoint = np.min(listSpectra[0:j + m_half + 1])
        elif j > (length - m_half - 1):  # Edge condition at the end
            minPoint = np.min(listSpectra[j - m_half:length])
        else:  # Middle condition
            minPoint = np.min(listSpectra[j - m_half:j + m_half + 1])
        minBackground[j] = minPoint
    
    # Calculate moving average of minimum
    n_half = int(n / 2)
    for k in range(length):
        if k <= n_half:  # Edge condition at the beginning
            meanPoint = np.mean(minBackground[0:k + n_half + 1])
        elif k > (length - n_half - 1):  # Edge condition at the end
            meanPoint = np.mean(minBackground[k - n_half:length])
        else:  # Middle condition
            meanPoint = np.mean(minBackground[k - n_half:k + n_half + 1])
        meanBackground[k] = meanPoint
    
    return (listSpectra - meanBackground)

def triangular_function(b1,a1,b2):
    left = np.linspace(0, 1, a1-b1)
    right = np.linspace(1, 0, b2-a1)
    return np.concatenate((left, right))

def calculate_signal_area(data, b1,b2):
    baseline = np.trapezoid([data[b1], data[b2]])
    signal_area = np.trapezoid(data[b1:b2]) - baseline
    return signal_area

def voigt(x, x0_fit, amplitude_fit, gamma_fit, sigma_fit):
    z = (x - x0_fit + 1j * gamma_fit) / (sigma_fit * np.sqrt(2))
    voigt_fit = amplitude_fit * wofz(z).real / (sigma_fit * np.sqrt(2 * np.pi))
    return voigt_fit

def peak_intensity(data, wavelengths, b1_w, b2_w):
    # Find the indices of the wavelengths
    b1 = np.argmin(np.abs(wavelengths - b1_w))
    #a1 = np.argmin(np.abs(wavelengths - a1_w))
    b2 = np.argmin(np.abs(wavelengths - b2_w))
    max_data = np.max(data[:,b1:b2], axis=0)
    a1 = b1 + np.argmax(max_data)

    
    # Create triangular function
    triangle = triangular_function(b1, a1, b2)

    # Calculate the correlation
    correlation = np.apply_along_axis(correlate,1,data[:,b1:b2], triangle, mode='valid') #correlate(data[b1:b2], triangle, mode='valid')

    # Calculate the pearsor correlation coefficient and p-value
    def pearsonr_wrapper(row, triangle):
        r, p = pearsonr(row, triangle)
        return r, p

    results = np.apply_along_axis(pearsonr_wrapper,1,data[:,b1:b2], triangle)
    r_value = results[:,0]
    p_value = results[:,1]

    # Calculate the signal area
    signal_area = np.apply_along_axis(calculate_signal_area,1,data,b1,b2) #calculate_signal_area(data, b1, b2)

    # Determine the peak intensity on significance level
    correlation = np.array(correlation)
    signal_area = np.array(signal_area)
    peak_intensity = correlation[:,0] * signal_area
    peak_intensity[(r_value < -0.2) & (p_value > 0.1)] = 0
    print (f"r_value max: {np.max(r_value)}")
    print (f"p_value max: {np.max(p_value)}")
    print (f"r_valu_min: {np.min(r_value)}")
    print (f"p_value_min: {np.min(p_value)}")
    
    return peak_intensity

def simple_sum(data, wavelengths, b1_w, b2_w):
    # Find the indices of the wavelengths
    b1 = np.argmin(np.abs(wavelengths - b1_w))
    b2 = np.argmin(np.abs(wavelengths - b2_w))

    signal_area = np.apply_along_axis(calculate_signal_area, 1, data, b1, b2)
    return signal_area

def max_intensity(data, wavelengths, b1_w, b2_w):
    # Find the indices of the wavelengths
    b1 = np.argmin(np.abs(wavelengths - b1_w))
    b2 = np.argmin(np.abs(wavelengths - b2_w))
    # Calculate the max intensity of the data minus the mean of data[b1] and data[b2]
    mean_data = (data[:,b1] + data[:,b2]) / 2
    max_intensity = np.apply_along_axis(np.max, 1, data[:,b1:b2]) - mean_data
    return max_intensity

def voigt_fit(data, wavelengths, b1_w, b2_w):
    # Find the indices of the wavelengths
    b1 = np.argmin(np.abs(wavelengths - b1_w))
    b2 = np.argmin(np.abs(wavelengths - b2_w))
    gamma = 0.1
    sigma = 0.006
    x = wavelengths[b1:b2]
    data_slice = data[:,b1:b2]

    max_data = np.max(data_slice, axis=0)
    a1 = b1 + np.argmax(max_data)
    x0 = wavelengths[a1]
    
    def voigt_fit_wrapper(vector, x, x0, gamma, sigma):
        try:
            popt, _ = curve_fit(voigt, x, vector, p0=[x0, np.max(vector), gamma, sigma])
            
            # Validate fitted parameters are finite
            if not np.all(np.isfinite(popt)):
                return 0
            
            # Calculate fitted curve
            real_fit = voigt(x, *popt)
            
            # Validate fitted curve is finite (no NaN or inf values)
            if not np.all(np.isfinite(real_fit)):
                return 0
            
            # 6. Check RMSE and normalized RMSE (fit quality)
            RMSE = np.sqrt(np.mean((vector - real_fit)**2))
            
            # Validate RMSE is finite
            if not np.isfinite(RMSE):
                return 0
            
            # Calculate normalized RMSE (NRMSE) as percentage of data range
            data_range = np.max(vector) - np.min(vector)
            if data_range > 0:
                NRMSE = (RMSE / data_range) * 100  # Percentage
            else:
                NRMSE = float('inf')  # No variation in data
            
            # Calculate R² (coefficient of determination)
            ss_res = np.sum((vector - real_fit)**2)
            ss_tot = np.sum((vector - np.mean(vector))**2)
            
            # Validate ss_res and ss_tot are finite
            if not np.isfinite(ss_res) or not np.isfinite(ss_tot):
                return 0
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = -float('inf')
            
            # Validate r_squared is finite before comparison
            if not np.isfinite(r_squared):
                return 0
            
            print(f"R²: {r_squared:.4f}")
            
            if r_squared < 0.85:
                return 0
            else:# Calculate signal area and return R²
                signal_area = np.trapz(real_fit, x)
                # Validate signal_area is finite before returning
                if not np.isfinite(signal_area):
                    return 0
                return signal_area
        except (RuntimeError, ValueError, TypeError):
            return 0
    signal_area = np.apply_along_axis(voigt_fit_wrapper, 1, data_slice, x, x0, gamma, sigma)
    return signal_area

def simple_voigt_fit(data, wavelengths, b1_w, b2_w):
    # Find the indices of the wavelengths
    b1 = np.argmin(np.abs(wavelengths - b1_w))
    b2 = np.argmin(np.abs(wavelengths - b2_w))
    # Use same parameters as voigt_fit for consistency
    gamma = 0.1
    sigma = 0.006
    x = wavelengths[b1:b2]
    data_slice = data[:,b1:b2]
    max_data = np.max(data_slice, axis=0)
    a1 = b1 + np.argmax(max_data)
    x0 = wavelengths[a1]
    basis = voigt(x, x0, np.max(max_data), gamma, sigma)
    denom = np.dot(basis, basis)
    amplitudes = data_slice @ basis / denom
    signal_area = amplitudes * np.trapz(basis, x)
    return signal_area  # Return the signal area for each row

def partition_function(elem, T):
    """
    Calculate partition function for an element at temperature T.
    
    Parameters:
    -----------
    elem : str
        Element symbol
    T : float
        Temperature in Kelvin (must be > 0)
    
    Returns:
    --------
    U_I : float
        Partition function for neutral state (I)
    U_II : float
        Partition function for singly ionized state (II)
    
    Raises:
    -------
    ValueError
        If temperature T is zero or negative
    """
    if T <= 0:
        raise ValueError(f"Temperature must be positive, got T={T} K. Temperature must be > 0 to avoid division by zero.")
    
    with sqlite3.connect(PARTITION_FUNCTION_PATH) as conn:
        cursor = conn.cursor()  
        cursor.execute("SELECT Elem_name, ion_state, Ei, gi FROM PartF_var WHERE Elem_name = ?", (elem,))
        df = pd.DataFrame(cursor.fetchall(), columns=['Elem_name', 'ion_state', 'Ei', 'gi'])
        df_I = df[(df['ion_state'] == 'I')]
        df_II = df[(df['ion_state'] == 'II')]
        kb_eV = 8.617333262e-5  # eV/K
        
        if df_I.empty:
            U_I = 0.0
        else:
            U_I = np.sum(df_I['gi'] * np.exp(-df_I['Ei'] / (kb_eV * T)))
            U_I = float(U_I)
        
        if df_II.empty:
            U_II = 0.0
        else:
            U_II = np.sum(df_II['gi'] * np.exp(-df_II['Ei'] / (kb_eV * T)))
            U_II = float(U_II)
    
    return U_I, U_II


# ---------------------------------------------------------------------------
# Cached version – avoids re-opening the DB on every call
# ---------------------------------------------------------------------------
_partf_data_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
_partf_conn = None


def _get_partf_connection(db_path: str):
    """Reuse a single connection for partition-function queries."""
    global _partf_conn
    if _partf_conn is None:
        _partf_conn = sqlite3.connect(db_path)
    return _partf_conn


def _load_partf_data(elem: str, db_path: str):
    """Load gi/Ei arrays for neutral (I) and ionised (II) states once."""
    if elem not in _partf_data_cache:
        conn = _get_partf_connection(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT ion_state, Ei, gi FROM PartF_var WHERE Elem_name = ?",
            (elem,),
        )
        rows = cur.fetchall()
        gi_I, Ei_I, gi_II, Ei_II = [], [], [], []
        for ion_state, Ei, gi in rows:
            if ion_state == "I":
                gi_I.append(gi)
                Ei_I.append(Ei)
            elif ion_state == "II":
                gi_II.append(gi)
                Ei_II.append(Ei)
        _partf_data_cache[elem] = (
            np.array(gi_I, dtype=np.float64),
            np.array(Ei_I, dtype=np.float64),
            np.array(gi_II, dtype=np.float64),
            np.array(Ei_II, dtype=np.float64),
        )
    return _partf_data_cache[elem]


def partition_function_cached(elem: str, T: float, db_path: str) -> tuple[float, float]:
    """
    Same physics as partition_function() but:
      - accepts an explicit db_path (no hard-coded path)
      - caches gi/Ei arrays so the DB is hit only once per element
    """
    if T <= 0:
        raise ValueError(f"Temperature must be positive, got T={T} K.")

    gi_I, Ei_I, gi_II, Ei_II = _load_partf_data(elem, db_path)
    kb_eV = 8.617333262e-5  # eV/K

    U_I = float(np.sum(gi_I * np.exp(-Ei_I / (kb_eV * T)))) if gi_I.size else 0.0
    U_II = float(np.sum(gi_II * np.exp(-Ei_II / (kb_eV * T)))) if gi_II.size else 0.0
    return U_I, U_II
