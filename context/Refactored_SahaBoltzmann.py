# Refactored Python script for Saha-Boltzmann plasma temperature estimation
# Translated from MultiElemSBPlot.txt

import os
import json
import pandas as pd
import numpy as np
from numpy.polynomial import Polynomial
from scipy.constants import Boltzmann, speed_of_light, Planck, electron_mass, elementary_charge
from scipy.special import wofz
from readData import get_spectra


# -----------------------------
# File paths (modify as needed)
# -----------------------------
EXPERIMENT_FOLDER = "/home/LIBS/prochazka/data/Running_projects/24_0011_CF_spark/CF SPARK/Data/Chameleon_OptiCal_DB_Calibration_Data_Fe_20211126/REMUS-9951892/FECRNI"
SAMPLE_LIST_PATH = "/home/LIBS/prochazka/data/Running_projects/24_0011_CF_spark/CF SPARK/Methods/Git/CF_OES/Samples_Fe_matrix_2021_12_10.txt"
PARTITION_FUNCTION_PATH = "/home/LIBS/prochazka/data/Running_projects/24_0011_CF_spark/CF SPARK/Methods/Git/CF_OES/PartF_var.txt"
EION_PATH = "/home/LIBS/prochazka/data/Running_projects/24_0011_CF_spark/CF SPARK/Methods/Git/CF_OES/E_ion.txt"
QUANT_PAR_PATH = "/home/LIBS/prochazka/data/Running_projects/24_0011_CF_spark/CF SPARK/Methods/Git/CF_OES/Quant_par_vacuum.txt"
LINE_DATA_PATH = "/home/LIBS/prochazka/data/Running_projects/24_0011_CF_spark/CF SPARK/Methods/Code/R/Boltzmann_lines_v15.txt"

# -----------------------------
# Constants
# -----------------------------
kb = 1.380649e-16 #erg/K #Boltzmann  # J/K
c = speed_of_light  # m/s
h = 6.62607015e-27 #erg*s #Planck  # J.s
me = electron_mass*1000  # g
e = elementary_charge  # C

Te = 12000  # Initial electron temperature in K
Ne = 1.79e+18  # Initial electron density in cm^-3	
#------------------------------
# Voigt profile function
#------------------------------
def voigt_profile(wavelength, x0, sigma, gamma):
    z = ((wavelength - x0) + 1j*gamma) / (sigma * np.sqrt(2))
    return np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

# -----------------------------
# Function to read JSON spectral data
# -----------------------------


# -----------------------------
# Function to calculate partition function
# -----------------------------
def partition_function(elem, T):
    df = pd.read_csv(PARTITION_FUNCTION_PATH, sep='\t', decimal='.')
    kb_eV = 8.617333262e-5  # eV/K

    df_I = df[(df['Element'] == elem) & (df['ionState'] == 'I')]
    df_II = df[(df['Element'] == elem) & (df['ionState'] == 'II')]

    U_I = np.sum(df_I['gi'] * np.exp(-df_I['Ei'] / (kb_eV * T)))
    U_II = np.sum(df_II['gi'] * np.exp(-df_II['Ei'] / (kb_eV * T)))
    #Change U_I and U_II to float
    U_I = float(U_I)
    U_II = float(U_II)

    return U_I, U_II

# -----------------------------
# Load experiment metadata
# -----------------------------
def load_metadata():
    sample_list = pd.read_csv(SAMPLE_LIST_PATH, sep='\t', decimal='.')
    eion = pd.read_csv(EION_PATH, sep='\t', decimal='.')
    quant_par = pd.read_csv(QUANT_PAR_PATH, sep='\t', decimal='.')
    line_data = pd.read_csv(LINE_DATA_PATH, sep='\t', decimal='.')
    line_data.dropna(inplace=True)
    return sample_list, eion, quant_par, line_data

# -----------------------------
# Saha-Boltzmann analysis function
# -----------------------------
def saha_boltzmann_analysis(inp_data_idx, experimentList, sample_list, lineData, Eion, QparFinal):
    # Prepare file path and read spectra
    json_path = os.path.join(EXPERIMENT_FOLDER, experimentList[inp_data_idx])
    w1, spectraData_1 = get_spectra(json_path, 1, 1, 1)
    w2, spectraData_2 = get_spectra(json_path, 1, 1, 2)
    wavelengths = np.concatenate([w1, w2])
    spectraAll = np.concatenate([spectraData_1, spectraData_2])

       

    # Sample metadata
    sample_name = experimentList[inp_data_idx].replace(".json", "").replace("/", "_")
    Cont = sample_list[sample_list['Sample Name'].str.replace("/", "_") == sample_name].copy()
    Cont = Cont.fillna(0)
    SahaBoltzmann_reduced = pd.DataFrame()

    for q in range(1, 3):
        SahaBoltzmann = pd.DataFrame()
        finalData = pd.DataFrame()
        SahBol = pd.DataFrame()
        N = 1e-5
        cnt = 1

        elements = lineData['Elem_name'].unique()
        elements = elements[[5,6,7,0,1,2,3,4]]  # Python is 0-based
        #print(f"Processing elements: {elements}")

        for elem in elements:#loop over elements
            Eion_temp = Eion[Eion['Element'] == f"{elem}+I"]
            lineData_temp = lineData[lineData['Elem_name'] == elem]
            Qpar_temp = QparFinal[QparFinal['Elem_name'] == elem]
            Cc = float(Cont[elem]) if elem in Cont.columns else 0
            print(f"Element: {elem}, Concentration: {Cc}")

            elem_selector = SahaBoltzmann_reduced['Element'] == elem if not SahaBoltzmann_reduced.empty else None
            if elem_selector is not None and elem_selector.sum() != 0:
                C0 = 1 if q == 1 else SahaBoltzmann_reduced.loc[elem_selector, 'C.calc'].values[0]
            else:
                C0 = 0.00001

            data_lines = np.zeros((2, lineData_temp.shape[0]))
            SahBol_data = pd.DataFrame(columns=["Experiment", "Line", "Y", "X"], index=range(lineData_temp.shape[0]))

            for u in range(lineData_temp.shape[0]):#determine spectral line intensities for an element (loop over lines)
                # Select wavelength region
                selection = (wavelengths > lineData_temp.iloc[u, 8]) & (wavelengths < lineData_temp.iloc[u, 9])
                bckgrnd = (wavelengths > lineData_temp.iloc[u, 10]) & (wavelengths < lineData_temp.iloc[u, 11])

                background_average = np.mean(spectraAll[bckgrnd])
                data_sum = abs(np.sum(spectraAll[selection])) - (np.sum(selection) * background_average)
                data_max = np.max(spectraAll[selection]) - background_average
                
                data_lines[0, u] = data_sum
                data_lines[1, u] = data_max

                SahBol_data.loc[u, "Experiment"] = experimentList[inp_data_idx]
                SahBol_data.loc[u, "Line"] = f"{lineData_temp.iloc[u,0]} @ {lineData_temp.iloc[u,2]}"
                ion_st = 1 if lineData_temp.iloc[u,1] == "II" else 0

                # Voigt fit placeholder (not implemented)
                dL = 0.09  # Replace with actual fit result if available

                SahBol_data.loc[u, "Y"] = float(dL)
                SahBol_data.loc[u, "X"] = (lineData_temp.iloc[u,4]*1.60217e-12) + (ion_st * Eion_temp['Eion'].values[0]*1.60217e-12 if not Eion_temp.empty else 0)
                print(f"Y={SahBol_data.loc[u, 'Y']}, X={SahBol_data.loc[u, 'X']}")
                print(f"Dimensions of SahBol_data: {SahBol_data.shape}")

            dT = 1000
            T0 = 9175
            if SahBol_data.shape[0] > 1 and Cc != 0:
                while dT > 50:#determine temperature for an element
                    PF_I, PF_II = partition_function(Qpar_temp['Elem_name'].values[0], T0)
                    #Ne = 1  # Placeholder, set electron density
                    S10 = (((2*PF_II)/(Ne*PF_I))*((me*kb*T0)/((h**2)/(2*np.pi)))**(1.5))*np.exp(-(Eion_temp['Eion'].values[0]*1.60217e-12)/(kb*T0)) if not Eion_temp.empty else 1
                    # Transition coefficient
                    kt = ((lineData_temp['Wl']**4)/(8*np.pi*c)) * (lineData_temp['Ak']*lineData_temp['gk']*np.exp(-(lineData_temp['Ei']*1.60217e-12)/(kb*T0))) * (1-np.exp(-1.60217e-12*(lineData_temp['Ek']-lineData_temp['Ei'])/(kb*T0))) / np.where(lineData_temp['ion.state']=="I", PF_I, PF_II)
                    # Ionization ratio
                    ri = np.where(lineData_temp['ion.state']=="I", 1/(1+S10), S10/(1+S10))
                    # Ionization state
                    ion_st = np.where(lineData_temp.iloc[:,1]=="II", 1, 0)
                    # Saha-Boltzmann equations
                    Lp = ((8*np.pi*h*c)/(10*lineData_temp['Wl']**3))*N*np.exp((-1.60217e-12*(lineData_temp['Ek']-lineData_temp['Ei']))/(kb*T0))*(lineData_temp['gk']/lineData_temp['gi'])
                   
                    #print shape of data_lines and Lp and SahBol_data["Y"]
                    print(f"data_lines shape: {len(data_lines[1,:])}, Lp shape: {len(Lp)}, SahBol_data['Y'] shape: {len(SahBol_data['Y'])}")
                    # Calculate Cs_y and Cs_x
                    Cs_y = np.array(data_lines[0,:], dtype=float)/(np.array(Lp,dtype=float)*np.array(SahBol_data["Y"], dtype=float))
                    Cs_x = (0.01*Cc*kt*ri*N)/np.array(SahBol_data["Y"], dtype=float)
                    print(f"data_lines: {data_lines}")
                    


                    SahBol_y = np.log((np.array(data_lines[1,:], dtype=float)*lineData_temp['Wl']*1e-9)/(lineData_temp['gk']*lineData_temp['Ak'])) - (ion_st*np.log(2*(2*np.pi*me*kb*T0)**1.5/((h**3)*Ne))) - np.log(Cc/(100*PF_I*(1+S10)))
                    SahBol_x = (lineData_temp['Ek']*1.60217e-12)+(ion_st*Eion_temp['Eion'].values[0]*1.60217e-12 if not Eion_temp.empty else 0)

                    Kl = (kt*ri*N*0.01*Cc)/np.array(SahBol_data["Y"], dtype=float)
                    corr_int = 1 if q == 1 else (1-np.exp(-Kl))/Kl
                    SahBol_x_corr = SahBol_x
                    SahBol_y_corr = np.log(((np.array(data_lines[1,:], dtype=float)/corr_int**0.5)*lineData_temp.iloc[:,2]*1e-9)/(lineData_temp.iloc[:,7].astype(float)*lineData_temp.iloc[:,5].astype(float))) - (ion_st*np.log(2*(2*np.pi*me*kb*T0)**1.5/((h**3)*Ne))) - np.log(Cc/(100*PF_I*(1+S10)))
                    print(f"SahBol_x_corr={SahBol_x_corr}, SahBol_y_corr={SahBol_y_corr}")

                    # Linear fit
                    fit = np.polyfit(SahBol_x_corr, SahBol_y_corr, 1)
                    Ti = 1/(-fit[0]*kb)
                    print(f"T0={T0} and Ti={Ti}")
                    dT = abs(T0-Ti)
                    if Ti < 0:
                        dT = 0
                        Ti = 10000
                        print(f"Element: {elem} error")
                    b = fit[1]
                    # R2 calculation
                    y_pred = np.polyval(fit, SahBol_x_corr)
                    R2 = 1 - np.sum((SahBol_y_corr - y_pred)**2) / np.sum((SahBol_y_corr - np.mean(SahBol_y_corr))**2)
                    T0 = Ti
                    cnt += 1
                    if cnt > 10:
                        dT = 10

                SahBol_temp = pd.DataFrame({
                    "Experiment": [experimentList[inp_data_idx]]*len(SahBol_x),
                    "Element": [elem]*len(SahBol_x),
                    "x": SahBol_x,
                    "y": SahBol_y,
                    "Cs.x": Cs_x,
                    "Cs.y": Cs_y,
                    "dL": np.array(SahBol_data["Y"],dtype=float),
                    "x.corr": SahBol_x_corr,
                    "y.corr": SahBol_y_corr,
                    "Line.max": data_lines[1,:]/corr_int**0.5
                })

                print(f"SahBol_temp shape: {SahBol_temp.shape}")

                SahBol = pd.concat([SahBol_temp, SahBol], ignore_index=True)
                SahaBoltzmann_temp = pd.DataFrame({
                    "Sample": [experimentList[inp_data_idx]],
                    "Element": [elem],
                    "Temp": [Ti],
                    "R2": [R2],
                    "Intercept": [b],
                    "U_I": [PF_I]
                })
                SahaBoltzmann = pd.concat([SahaBoltzmann, SahaBoltzmann_temp], ignore_index=True)
                finalData = pd.concat([finalData, SahBol_data], ignore_index=True)
                print(f"SahBol: {SahBol}")

        # Concentration calculation
        C_u = (SahaBoltzmann['U_I']*np.exp(SahaBoltzmann['Intercept']))/np.sum(SahaBoltzmann['U_I']*np.exp(SahaBoltzmann['Intercept']))
        SahaBoltzmann['C'] = C_u
        print(f"SahaBoltzmann: {SahaBoltzmann}")

        SahaBoltzmann_reduced = SahaBoltzmann[(SahaBoltzmann['Temp'] < 16000) & (SahaBoltzmann['Temp'] > 8800) & (SahaBoltzmann['R2'] > 0.945)]
        C_calc = (SahaBoltzmann_reduced['U_I']*np.exp(SahaBoltzmann_reduced['Intercept']))/np.sum(SahaBoltzmann_reduced['U_I']*np.exp(SahaBoltzmann_reduced['Intercept']))
        SahaBoltzmann_reduced['C.calc'] = C_calc

    return SahaBoltzmann_reduced

# -----------------------------
# Main execution block
# -----------------------------
def main():
    inp_data_idx = 39  # Index of the experiment to process
    sample_list, Eion, QparFinal, lineData = load_metadata()
    print(f"Sample list: {sample_list}") 

    experimentList = [f for f in os.listdir(EXPERIMENT_FOLDER) if f.endswith('.json')]
    #experimentList.sort()   

    SB = saha_boltzmann_analysis(inp_data_idx, experimentList, sample_list, lineData, Eion, QparFinal)
    print(f"SB={SB}")

if __name__ == '__main__':
    main()
