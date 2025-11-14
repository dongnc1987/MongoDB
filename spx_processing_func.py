import streamlit as st
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import base64
import struct
from typing import Dict, List, Tuple, Optional

st.set_page_config(page_title="SPX & XRF Data Integration", layout="wide")

# ============================================================================
# CONSTANTS
# ============================================================================

ELEMENT_SYMBOLS = {
    13: 'Al', 22: 'Ti', 24: 'Cr', 26: 'Fe', 27: 'Co', 28: 'Ni', 
    29: 'Cu', 42: 'Mo', 45: 'Rh', 46: 'Pd', 47: 'Ag', 74: 'W', 79: 'Au'
}

# ============================================================================
# XML PARSING UTILITIES
# ============================================================================

def _localname(tag: str) -> str:
    """Extract local name from XML tag (remove namespace)"""
    return tag.split('}', 1)[-1] if '}' in tag else tag

def get_text(element, path: str, default=''):
    """Safely extract text from XML element"""
    if element is None:
        return default
    found = element.find(path)
    return found.text.strip() if found is not None and found.text else default

def get_float(element, path: str, default=0.0):
    """Safely extract float from XML element"""
    try:
        text = get_text(element, path, str(default))
        return float(text)
    except:
        return default

def get_int(element, path: str, default=0):
    """Safely extract integer from XML element"""
    try:
        text = get_text(element, path, str(default))
        return int(float(text))
    except:
        return default

# ============================================================================
# COORDINATE EXTRACTION
# ============================================================================

def _length_to_mm(val: float, unit: str = None) -> float:
    """Convert length value to millimeters"""
    if unit is None:
        return float(val)
    u = unit.strip().lower()
    if u in ("mm",):
        return float(val)
    if u in ("µm", "μm", "um"):
        return float(val) / 1000.0
    if u in ("cm",):
        return float(val) * 10.0
    if u in ("m",):
        return float(val) * 1000.0
    return float(val)

def find_xyz_from_blob(blob: bytes):
    """Extract XYZ coordinates from binary blob"""
    best = None
    for off in range(0, len(blob) - 24 + 1, 1):
        try:
            x, y, z = struct.unpack("<ddd", blob[off:off+24])
        except struct.error:
            continue
        if all(np.isfinite([x, y, z])) and all(0 <= v <= 1000 for v in (x, y, z)):
            if all(v > 0 for v in (x, y, z)):
                best = (x, y, z)
    return best

def extract_xyz_positions_from_spx(root: ET.Element):
    """Extract X, Y, Z positions from SPX file in millimeters"""
    positions = {}
    
    # First try: look for Axis elements
    for elem in root.iter():
        tag_local = _localname(elem.tag)
        if tag_local.startswith("Axis"):
            name = elem.attrib.get("AxisName", tag_local)
            unit = elem.attrib.get("AxisUnit", "")
            pos = elem.attrib.get("AxisPosition")
            if pos is not None:
                try:
                    pos_val = float(pos)
                    positions[name] = {"position": pos_val, "unit": unit}
                except ValueError:
                    pass
    
    if positions:
        x_data = positions.get("X") or next((v for k, v in positions.items() if k.upper().startswith("X")), None)
        y_data = positions.get("Y") or next((v for k, v in positions.items() if k.upper().startswith("Y")), None)
        z_data = positions.get("Z") or next((v for k, v in positions.items() if k.upper().startswith("Z")), None)
        
        x_mm = _length_to_mm(x_data["position"], x_data.get("unit")) if x_data else None
        y_mm = _length_to_mm(y_data["position"], y_data.get("unit")) if y_data else None
        z_mm = _length_to_mm(z_data["position"], z_data.get("unit")) if z_data else None
        
        return x_mm, y_mm, z_mm
    
    # Second try: look for Data blob
    for node in root.iter():
        if _localname(node.tag) != "Data":
            continue
        b64 = (node.text or "").strip()
        if len(b64) < 8:
            continue
        try:
            blob = base64.b64decode(b64, validate=True)
        except Exception:
            continue
        xyz = find_xyz_from_blob(blob)
        if xyz:
            x, y, z = xyz
            return float(x), float(y), float(z)
    
    return None, None, None

# ============================================================================
# SPECTRUM DATA PARSING
# ============================================================================

def parse_date_time(date_str: str, time_str: str):
    """Parse date and time strings to datetime object"""
    try:
        day, month, year = date_str.split('.')
        hour, minute, second = time_str.split(':')
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    except:
        return datetime.now()

def parse_spectrum_data(channels_text: str):
    """Parse spectrum channel data from comma-separated string"""
    try:
        return np.array([int(x) for x in channels_text.split(',')])
    except:
        return np.array([])

def parse_spx_file(file_path: Path) -> Dict:
    """Parse SPX file and extract all relevant information"""
    with open(file_path, 'rb') as f:
        file_content = f.read()
    
    root = ET.fromstring(file_content)
    spectrum = root.find(".//ClassInstance[@Type='TRTSpectrum']")
    
    if spectrum is None:
        raise ValueError("Invalid SPX file")
    
    header = spectrum.find(".//ClassInstance[@Type='TRTSpectrumHeader']")
    hardware = spectrum.find(".//ClassInstance[@Type='TRTSpectrumHardwareHeader']")
    xrf_header = spectrum.find(".//ClassInstance[@Type='TRTXrfHeader']")
    
    # Date and time
    date_str = get_text(header, 'Date')
    time_str = get_text(header, 'Time')
    measurement_date = parse_date_time(date_str, time_str)
    
    # Spectrum data
    channels_element = spectrum.find('Channels')
    spectrum_data = parse_spectrum_data(channels_element.text if channels_element is not None else '')
    
    # Calibration
    calib_abs = get_float(header, 'CalibAbs', -0.956)
    calib_lin = get_float(header, 'CalibLin', 0.01)
    channel_count = get_int(header, 'ChannelCount', 4096)
    
    # X-ray tube parameters
    tube_voltage = get_float(xrf_header, 'Voltage', 0) if xrf_header is not None else 0
    tube_current = get_float(xrf_header, 'Current', 0) if xrf_header is not None else 0
    tube_anode = get_int(xrf_header, 'Anode', 45) if xrf_header is not None else 45
    
    xray_tube_target = ELEMENT_SYMBOLS.get(tube_anode, 'Unknown')
    chassis_type = get_text(xrf_header, 'ChassisType') if xrf_header is not None else ''
    
    # Position
    x_position, y_position, z_position = extract_xyz_positions_from_spx(root)
    
    # Ensure spectrum data matches channel count
    if len(spectrum_data) > channel_count:
        spectrum_data = spectrum_data[:channel_count]
    elif len(spectrum_data) < channel_count:
        padded = np.zeros(channel_count, dtype=int)
        padded[:len(spectrum_data)] = spectrum_data
        spectrum_data = padded
    
    result = {
        'file_name': file_path.name,
        'date': date_str,
        'time': time_str,
        'measurement_date': measurement_date,
        'real_time_ms': int(get_int(hardware, 'RealTime', 0)),
        'live_time_ms': int(get_int(hardware, 'LifeTime', 0)),
        'dead_time_percent': float(get_float(hardware, 'DeadTime', 0)),
        'xray_tube_target': xray_tube_target,
        'voltage_kV': float(tube_voltage),
        'current_uA': float(tube_current),
        'calib_abs_kev': float(calib_abs),
        'calib_lin_kev_per_channel': float(calib_lin),
        'channel_count': int(channel_count),
        'detector_count': int(get_int(hardware, 'DetectorCount', 0)),
        'selected_detectors': str(get_text(hardware, 'SelectedDetectors')).replace('\t', '&').replace(' ', '&').replace('\n', '&'),
        'total_counts': int(spectrum_data.sum()),
        'max_counts': int(spectrum_data.max()),
        'spectrum_data': spectrum_data,
        'chassis_type': chassis_type,
        'x_position_mm': x_position,
        'y_position_mm': y_position,
        'z_position_mm': z_position,
    }
    
    # Calculate energy at max counts
    energy_axis = calib_abs + calib_lin * np.arange(channel_count)
    result['max_counts_energy_kev'] = float(energy_axis[spectrum_data.argmax()])
    
    return result

# ============================================================================
# XRF EXCEL FILE PARSING (NEW)
# ============================================================================

def extract_xrf_excel_data(uploaded_file) -> Dict:
    """
    Extract XRF data from Excel file and detect multiple layers
    Handles multi-level headers where layer names are on one row and column names below
    Returns dictionary with layer information and dataframes
    """
    df = pd.read_excel(uploaded_file, header=None)
    
    # Find header rows
    layer_header_row = None
    column_header_row = None
    spectrum_col_row = None
    
    # Find row with "Spectrum" and layer information
    for idx, row in df.iterrows():
        row_str = ' '.join([str(x) for x in row.values if pd.notna(x)])
        if "Spectrum" in row_str:
            if "Layer" in row_str or any("Thickn" in str(x) for x in row.values):
                # This might be layer header or column header
                if "Layer" in row_str:
                    layer_header_row = idx
                else:
                    column_header_row = idx
                    spectrum_col_row = idx
            elif spectrum_col_row is None:
                spectrum_col_row = idx
    
    # If layer names are in a separate row above column names
    if layer_header_row is not None and column_header_row is None:
        column_header_row = layer_header_row + 1
    elif layer_header_row is None and column_header_row is not None:
        layer_header_row = column_header_row - 1
    elif layer_header_row is None and column_header_row is None:
        # Single row header
        layer_header_row = spectrum_col_row
        column_header_row = spectrum_col_row
    
    if column_header_row is None:
        raise ValueError("Could not find header rows in Excel file")
    
    # Extract layer names and build column structure
    layer_row = df.iloc[layer_header_row].values
    column_row = df.iloc[column_header_row].values
    
    # Parse layer structure
    layers_columns = {}
    current_layer = None
    spectrum_idx = None
    
    for idx, (layer_name, col_name) in enumerate(zip(layer_row, column_row)):
        # Check if this is the Spectrum column
        if "Spectrum" in str(col_name):
            spectrum_idx = idx
            continue
        
        # Detect layer change
        if pd.notna(layer_name) and "Layer" in str(layer_name):
            layer_num = str(layer_name).replace("Layer", "").strip()
            current_layer = f"layer{layer_num}"
            if current_layer not in layers_columns:
                layers_columns[current_layer] = []
        
        # Add column to current layer
        if current_layer is not None and pd.notna(col_name):
            col_name_str = str(col_name).strip()
            if col_name_str and col_name_str != 'nan':
                # Create unique column name by adding layer prefix
                unique_col_name = f"{current_layer}_{col_name_str}"
                layers_columns[current_layer].append((idx, col_name_str, unique_col_name))
    
    if spectrum_idx is None:
        raise ValueError("Could not find Spectrum column")
    
    # Extract data starting from the row after column headers
    data_start_row = column_header_row + 1
    data_df = df.iloc[data_start_row:].reset_index(drop=True)
    
    # Filter valid data rows (Grid measurements)
    if len(data_df) > 0:
        spectrum_data = data_df.iloc[:, spectrum_idx].astype(str)
        valid_rows = spectrum_data.str.startswith("Grid", na=False)
        valid_rows = valid_rows & ~spectrum_data.str.contains("Mean|Std|Grid_0", na=False)
        data_df = data_df[valid_rows].reset_index(drop=True)
    
    # Create separate dataframe for each layer
    layers_data = {}
    
    for layer_name, columns_info in layers_columns.items():
        if len(columns_info) == 0:
            continue
        
        # Extract columns for this layer
        layer_df_data = {
            'Spectrum': data_df.iloc[:, spectrum_idx]
        }
        
        for col_idx, original_name, unique_name in columns_info:
            layer_df_data[original_name] = data_df.iloc[:, col_idx]
        
        # Create dataframe
        layer_df = pd.DataFrame(layer_df_data)
        
        # Convert numeric columns
        for col in layer_df.columns[1:]:
            try:
                layer_df[col] = pd.to_numeric(layer_df[col], errors='coerce')
            except Exception:
                pass
        
        # Remove rows with all NaN values (except Spectrum column)
        layer_df = layer_df.dropna(how='all', subset=layer_df.columns[1:])
        
        if len(layer_df) > 0:
            layers_data[layer_name] = layer_df
    
    if len(layers_data) == 0:
        raise ValueError("No valid layer data found in Excel file")
    
    return {
        'num_layers': len(layers_data),
        'layers': layers_data
    }


def parse_xrf_excel_to_dict(xrf_df: pd.DataFrame, layer_name: str) -> List[Dict]:
    """
    Convert XRF Excel DataFrame to list of dictionaries for a specific layer
    Each dict contains: spectrum_name, thickness_nm, composition, layer_name
    """
    xrf_data_list = []
    
    spectrum_col = [col for col in xrf_df.columns if "Spectrum" in str(col)][0]
    thickness_cols = [col for col in xrf_df.columns if "Thick" in str(col)]
    ratio_cols = [col for col in xrf_df.columns if "%" in str(col)]
    
    for _, row in xrf_df.iterrows():
        spectrum_name = str(row[spectrum_col])
        
        # Extract thickness
        thickness_nm = row[thickness_cols[0]] if thickness_cols else None
        
        # Extract composition
        composition = {}
        for col in ratio_cols:
            element = col.replace('%', '').strip()
            value = row[col]
            if pd.notna(value):
                composition[element] = float(value)
        
        xrf_data_list.append({
            'spectrum_name': spectrum_name,
            'thickness_nm': thickness_nm,
            'composition': composition,
            'layer_name': layer_name
        })
    
    return xrf_data_list

# ============================================================================
# SPECTRUM MATCHING (NEW)
# ============================================================================

def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Normalize spectrum to sum = 1"""
    total = spectrum.sum()
    if total > 0:
        return spectrum / total
    return spectrum

def compare_spectra_correlation(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """
    Compare two spectra using correlation coefficient
    Returns correlation value (0 to 1, higher is better match)
    """
    if spectrum1 is None or spectrum2 is None:
        return 0.0
    
    # Ensure same length
    min_len = min(len(spectrum1), len(spectrum2))
    s1 = spectrum1[:min_len]
    s2 = spectrum2[:min_len]
    
    # Normalize
    s1_norm = normalize_spectrum(s1)
    s2_norm = normalize_spectrum(s2)
    
    # Calculate correlation
    try:
        correlation = np.corrcoef(s1_norm, s2_norm)[0, 1]
        return float(correlation) if np.isfinite(correlation) else 0.0
    except:
        return 0.0

def match_spx_with_xrf_excel(spx_data_list: List[Dict], xrf_data_list: List[Dict], 
                                        layer_name: str) -> List[Dict]:
    """
    Match SPX files with XRF Excel data for a specific layer
    """
    combined_data = []
    
    # Create lookup by spectrum name
    xrf_lookup = {xrf['spectrum_name']: xrf for xrf in xrf_data_list}
    
    for spx_data in spx_data_list:
        spx_name = spx_data['file_name']
        spx_base = Path(spx_name).stem
        
        # Try to find match
        xrf_match = None
        match_method = None
        correlation = 0.0
        
        # Method 1: Direct name match
        for xrf_name in xrf_lookup.keys():
            if spx_base in xrf_name or xrf_name in spx_base:
                xrf_match = xrf_lookup[xrf_name]
                match_method = "name"
                correlation = 1.0
                break
        
        # Method 2: Extract Grid number from SPX filename
        if xrf_match is None:
            import re
            grid_match = re.search(r'Grid[_\s]*(\d+)', spx_base, re.IGNORECASE)
            if grid_match:
                grid_num = grid_match.group(0)
                for xrf_name in xrf_lookup.keys():
                    if grid_num in xrf_name:
                        xrf_match = xrf_lookup[xrf_name]
                        match_method = "grid_number"
                        correlation = 1.0
                        break
        
        # Combine data
        combined = {
            'spx_name': spx_name,
            'xrf_spectrum_name': xrf_match['spectrum_name'] if xrf_match else None,
            'layer_name': layer_name,
            'date': spx_data['date'],
            'time': spx_data['time'],
            'x_position_mm': spx_data['x_position_mm'],
            'y_position_mm': spx_data['y_position_mm'],
            'z_position_mm': spx_data['z_position_mm'],
            'thickness_nm': xrf_match['thickness_nm'] if xrf_match else None,
            'composition': xrf_match['composition'] if xrf_match else {},
            'spectrum_data_spx': spx_data['spectrum_data'],
            'calib_abs_kev': spx_data['calib_abs_kev'],
            'calib_lin_kev_per_channel': spx_data['calib_lin_kev_per_channel'],
            'matched': xrf_match is not None,
            'match_method': match_method,
            'correlation': correlation,
            'real_time_ms': spx_data['real_time_ms'],
            'live_time_ms': spx_data['live_time_ms'],
            'dead_time_percent': spx_data['dead_time_percent'],
            'xray_tube_target': spx_data['xray_tube_target'],
            'voltage_kV': spx_data['voltage_kV'],
            'current_uA': spx_data['current_uA'],
            'total_counts': spx_data['total_counts'],
            'max_counts': spx_data['max_counts'],
        }
        
        combined_data.append(combined)
    
    return combined_data


# ============================================================================
# COORDINATE CONVERSION
# ============================================================================

def extract_xrf_bounds(combined_data: List[Dict]) -> Optional[Dict]:
    """Extract first and last XRF measurement coordinates"""
    if not combined_data:
        return None
    
    first_data = combined_data[0]
    x_1 = first_data['x_position_mm']
    y_1 = first_data['y_position_mm']
    
    last_data = combined_data[-1]
    x_2 = last_data['x_position_mm']
    y_2 = last_data['y_position_mm']
    
    if x_1 is not None and y_1 is not None and x_2 is not None and y_2 is not None:
        return {'x_1': x_1, 'y_1': y_1, 'x_2': x_2, 'y_2': y_2}
    return None

def convert_xrf_to_optical(combined_data: List[Dict], xrf_bounds: Dict, 
                          optical_bounds: Dict) -> List[Dict]:
    """Convert XRF coordinates to optical coordinates (X and Y only)"""
    A_x_1, A_y_1 = xrf_bounds['x_1'], xrf_bounds['y_1']
    A_x_2, A_y_2 = xrf_bounds['x_2'], xrf_bounds['y_2']
    
    B_x_1, B_y_1 = optical_bounds['x_1'], optical_bounds['y_1']
    B_x_2, B_y_2 = optical_bounds['x_2'], optical_bounds['y_2']
    
    converted_data = []
    
    for data in combined_data:
        x_xrf = data['x_position_mm']
        y_xrf = data['y_position_mm']
        
        if x_xrf is None or y_xrf is None:
            converted_data.append({
                'spx_name': data['spx_name'],
                'x_xrf': 0, 'y_xrf': 0,
                'x_optical': 0, 'y_optical': 0
            })
            continue
        
        # Linear mapping
        t_x = (x_xrf - A_x_1) / (A_x_2 - A_x_1) if A_x_2 != A_x_1 else 0
        t_y = (y_xrf - A_y_1) / (A_y_2 - A_y_1) if A_y_2 != A_y_1 else 0
        
        x_optical = B_x_1 + t_x * (B_x_2 - B_x_1)
        y_optical = B_y_1 + t_y * (B_y_2 - B_y_1)
        
        converted_data.append({
            'spx_name': data['spx_name'],
            'x_xrf': x_xrf, 'y_xrf': y_xrf,
            'x_optical': x_optical, 'y_optical': y_optical
        })
    
    return converted_data

# ============================================================================
# CSV EXPORT WITH CONVERTED COORDINATES
# ============================================================================

def create_combined_csv(combined_data: List[Dict], metadata: Dict, 
                                   converted_data: Optional[List[Dict]] = None,
                                   layer_name: str = "layer1") -> str:
    """
    Create CSV file with combined SPX and XRF data for specific layer
    """
    csv_lines = []
    
    # Metadata section
    csv_lines.append("Sample ID," + metadata['sample_id'])
    csv_lines.append("Layer," + layer_name)
    csv_lines.append("Substrate," + metadata['substrate'])
    csv_lines.append("Sample Description," + metadata['sample_description'])
    csv_lines.append("Sample Size," + metadata['sample_size'])
    csv_lines.append("Fabrication Method," + metadata['fabrication_method'])
    csv_lines.append("Treatment Method," + metadata['treatment_method'])
    csv_lines.append("Treatment Sequence," + metadata['treatment_sequence'])
    csv_lines.append("Air exposure Duration (min)," + metadata['air_exposure_duration'])
    csv_lines.append("Operator," + metadata['operator'])
    csv_lines.append("Institution," + metadata['institution'])
    csv_lines.append("Measurement Type," + metadata['measurement_type'])
    csv_lines.append("Spectrometer," + metadata.get('spectrometer', 'Bruker M4 Tornado'))


    if combined_data:
        first_data = combined_data[0]
        
        # Calculate median values
        medians = {
            'real_time': int(np.median([d['real_time_ms'] for d in combined_data])),
            'live_time': int(np.median([d['live_time_ms'] for d in combined_data])),
            'dead_time': float(np.median([d['dead_time_percent'] for d in combined_data])),
            'voltage': float(np.median([d['voltage_kV'] for d in combined_data])),
            'current': float(np.median([d['current_uA'] for d in combined_data])),
            'calib_abs': float(np.median([d['calib_abs_kev'] for d in combined_data])),
            'calib_lin': float(np.median([d['calib_lin_kev_per_channel'] for d in combined_data])),
        }
        
        csv_lines.append("real_time_ms," + str(medians['real_time']))
        csv_lines.append("live_time_ms," + str(medians['live_time']))
        csv_lines.append("dead_time_percent," + str(medians['dead_time']))
        csv_lines.append("voltage_kV," + str(medians['voltage']))
        csv_lines.append("current_micro_A," + str(medians['current']))
        csv_lines.append("calib_abs_kev," + str(medians['calib_abs']))
        csv_lines.append("calib_lin_kev_per_channel," + str(medians['calib_lin']))
        csv_lines.append("material layer number," + str(metadata.get('total_layers', 1)))
        csv_lines.append("material layer," + layer_name)
        
        # Build header row - remove z position
        header_cols = [
            "x position (mm)", "y position (mm)",
            "spectrum", "date", "time",
            "thickness (nm)"
        ]
        
        # Add composition columns
        all_elements = set()
        for data in combined_data:
            if data['composition']:
                all_elements.update(data['composition'].keys())
        
        element_cols = sorted(list(all_elements))
        for elem in element_cols:
            clean_elem = elem.replace('[]', '').replace('[', '').replace(']', '').strip()
            header_cols.append(f"{clean_elem} (%)")
        
        # Add spectrum energy columns
        channel_count = len(first_data['spectrum_data_spx'])
        energy_axis = medians['calib_abs'] + medians['calib_lin'] * np.arange(channel_count)
        energy_headers = [f"{e:.6f}" for e in energy_axis]
        
        header_row = ",".join(header_cols) + "," + ",".join(energy_headers)
        csv_lines.append(header_row)
        
        # Create coordinate lookup if converted_data provided
        converted_dict = {}
        if converted_data is not None:
            converted_dict = {c['spx_name']: c for c in converted_data}
        
        # Data rows
        for data in combined_data:
            # Use converted coordinates if available
            if data['spx_name'] in converted_dict:
                x_pos = converted_dict[data['spx_name']]['x_optical']
                y_pos = converted_dict[data['spx_name']]['y_optical']
            else:
                x_pos = data['x_position_mm'] if data['x_position_mm'] is not None else 0
                y_pos = data['y_position_mm'] if data['y_position_mm'] is not None else 0
            
            spectrum_name = data['spx_name']
            
            row_data = [
                f"{x_pos:.3f}",
                f"{y_pos:.3f}",
                spectrum_name,
                data['date'],
                data['time'],
                f"{data['thickness_nm']:.2f}" if data['thickness_nm'] is not None else 'N/A'
            ]
            
            # Add composition values
            for elem in element_cols:
                value = data['composition'].get(elem, 0)
                row_data.append(f"{value:.2f}")
            
            # Add spectrum data
            spectrum_data = data['spectrum_data_spx'][:channel_count]
            counts_str = ",".join([str(int(c)) for c in spectrum_data])
            
            data_row = ",".join(row_data) + "," + counts_str
            csv_lines.append(data_row)
    
    return "\n".join(csv_lines)

# ============================================================================
# VISUALIZATION FOR COORDINATE CONVERSION
# ============================================================================

def plot_coordinate_comparison(xrf_bounds: Dict, optical_bounds: Dict, converted_data: List[Dict]):
    """Create visualization comparing XRF and optical coordinates"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("XRF Coordinates", "Optical Coordinates"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # XRF plot
    x_1_xrf, y_1_xrf = xrf_bounds['x_1'], xrf_bounds['y_1']
    x_2_xrf, y_2_xrf = xrf_bounds['x_2'], xrf_bounds['y_2']
    
    x_min_xrf, x_max_xrf = min(x_1_xrf, x_2_xrf), max(x_1_xrf, x_2_xrf)
    y_min_xrf, y_max_xrf = min(y_1_xrf, y_2_xrf), max(y_1_xrf, y_2_xrf)
    
    x_rect_xrf = [x_min_xrf, x_max_xrf, x_max_xrf, x_min_xrf, x_min_xrf]
    y_rect_xrf = [y_min_xrf, y_min_xrf, y_max_xrf, y_max_xrf, y_min_xrf]
    
    fig.add_trace(go.Scatter(
        x=x_rect_xrf, y=y_rect_xrf, mode="lines",
        line=dict(color="black", width=2), name="XRF Bounds", showlegend=True
    ), row=1, col=1)
    
    x_points_xrf = [d['x_xrf'] for d in converted_data]
    y_points_xrf = [d['y_xrf'] for d in converted_data]
    
    fig.add_trace(go.Scatter(
        x=x_points_xrf, y=y_points_xrf, mode="markers+lines",
        marker=dict(size=8, color=np.arange(len(converted_data)), colorscale="Viridis"),
        line=dict(width=1.5), name="XRF Points", showlegend=True
    ), row=1, col=1)
    
    # Add First and Last Point markers for XRF
    fig.add_trace(go.Scatter(
        x=[x_1_xrf], y=[y_1_xrf], mode="markers+text",
        marker=dict(size=15, color="red", symbol="circle"),
        text=["First Point"], textposition="top center",
        name="First Point (XRF)", showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[x_2_xrf], y=[y_2_xrf], mode="markers+text",
        marker=dict(size=15, color="blue", symbol="circle"),
        text=["Last Point"], textposition="top center",
        name="Last Point (XRF)", showlegend=True
    ), row=1, col=1)
    
    # Optical plot
    x_1_opt, y_1_opt = optical_bounds['x_1'], optical_bounds['y_1']
    x_2_opt, y_2_opt = optical_bounds['x_2'], optical_bounds['y_2']
    
    x_min_opt, x_max_opt = min(x_1_opt, x_2_opt), max(x_1_opt, x_2_opt)
    y_min_opt, y_max_opt = min(y_1_opt, y_2_opt), max(y_1_opt, y_2_opt)
    
    x_rect_opt = [x_min_opt, x_max_opt, x_max_opt, x_min_opt, x_min_opt]
    y_rect_opt = [y_min_opt, y_min_opt, y_max_opt, y_max_opt, y_min_opt]
    
    fig.add_trace(go.Scatter(
        x=x_rect_opt, y=y_rect_opt, mode="lines",
        line=dict(color="black", width=2), name="Optical Bounds", showlegend=True
    ), row=1, col=2)
    
    x_points_opt = [d['x_optical'] for d in converted_data]
    y_points_opt = [d['y_optical'] for d in converted_data]
    
    fig.add_trace(go.Scatter(
        x=x_points_opt, y=y_points_opt, mode="markers+lines",
        marker=dict(size=8, color=np.arange(len(converted_data)), colorscale="Viridis"),
        line=dict(width=1.5), name="Optical Points", showlegend=True
    ), row=1, col=2)
    
    # Add First and Last Point markers for Optical
    fig.add_trace(go.Scatter(
        x=[x_1_opt], y=[y_1_opt], mode="markers+text",
        marker=dict(size=15, color="red", symbol="circle"),
        text=["First Point"], textposition="top center",
        name="First Point (Optical)", showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[x_2_opt], y=[y_2_opt], mode="markers+text",
        marker=dict(size=15, color="blue", symbol="circle"),
        text=["Last Point"], textposition="top center",
        name="Last Point (Optical)", showlegend=True
    ), row=1, col=2)
    
    # XRF axes: reversed
    fig.update_xaxes(title_text="X (mm)", autorange="reversed", scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="Y (mm)", autorange="reversed", row=1, col=1)
    
    # Optical axes: normal
    fig.update_xaxes(title_text="X (mm)", scaleanchor="y2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(title_text="Y (mm)", row=1, col=2)
    
    fig.update_layout(height=500, template="plotly_white")
    
    return fig



# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_spectrum(data: Dict, title: str):
    """Plot SPX spectrum"""
    calib_abs = data['calib_abs_kev']
    calib_lin = data['calib_lin_kev_per_channel']
    spectrum_spx = data['spectrum_data_spx']
    
    energy_axis = calib_abs + calib_lin * np.arange(len(spectrum_spx))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=energy_axis,
        y=spectrum_spx,
        mode='lines',
        name='SPX',
        line=dict(color='royalblue', width=1.5)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Energy (keV)",
        yaxis_title="Counts",
        height=400,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

# ============================================================================
# METADATA UI
# ============================================================================

def render_metadata_section():
    """Render metadata input section"""
    st.subheader(" Sample Metadata")
    
    if 'sample_id' not in st.session_state:
        st.session_state.sample_id = "3716-15"
    if 'operator' not in st.session_state:
        st.session_state.operator = ""
    if 'operator_valid' not in st.session_state:
        st.session_state.operator_valid = False
    if 'institution' not in st.session_state:
        st.session_state.institution = "HZB"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sample_id = st.text_input("Sample ID", value=st.session_state.sample_id) or st.session_state.sample_id
        
        substrate_options = ["Silicon", "Glass", "Quartz", "Plastic", "Metal", "Other"]
        substrate = st.selectbox("Substrate", substrate_options, index=0)
        
        sample_description = st.text_area(
            "Sample Description", 
            value="Perovskite solar cell with full stacks",
            max_chars=500,
            placeholder="Describe components, materials, layers of your samples",
            help="Maximum 500 characters"
        )
        
        sample_size = st.text_input("Sample Size", value="5x5")
        
        st.session_state.sample_id = sample_id
        st.session_state.substrate = substrate
        st.session_state.sample_description = sample_description
        st.session_state.sample_size = sample_size
    
    with col2:
        fab_method_options = [
            "CVD", "PVD", "PLD", "Sputtering", "Thermal Evaporation",
            "E-beam", "Spin Coating", "Sol-gel", "Slot-Die Coating", "Inkjet Printing"
        ]
        fabrication_method = st.selectbox("Fabrication Method", fab_method_options, index=1)
        
        treat_method_options = ["As-deposited", "Annealing", "UV-Ozone", "Irradiation", "Plasma"]
        treatment_method = st.selectbox("Treatment Method", treat_method_options, index=0)
        
        if treatment_method == "As-deposited":
            treatment_sequence = "0"
            st.info(f"Treatment sequence: {treatment_sequence}")
        else:
            treatment_sequence = str(st.number_input(
                "Treatment Sequence",
                min_value=1, max_value=10, value=1, step=1,
                help="Specify the order of this treatment process"
            ))
        
        air_exposure_duration = st.text_input("Air exposure Duration (min)", value="30")
        
        st.session_state.fabrication_method = fabrication_method
        st.session_state.treatment_method = treatment_method
        st.session_state.treatment_sequence = treatment_sequence
        st.session_state.air_exposure_duration = air_exposure_duration
    
    with col3:
        operator = st.text_input(
            "Operator (First Name Last Name)",
            value=st.session_state.operator,
            placeholder="e.g., Dong Nguyen",
            help="Please enter first name and last name"
        )
        
        operator_valid = True
        if operator and len(operator.split()) < 2:
            st.error("Please enter both first name and last name for Operator")
            operator_valid = False
        elif not operator:
            st.warning("Operator name is required")
            operator_valid = False
        
        institution = st.text_input("Institution", value=st.session_state.institution) or st.session_state.institution
        
        measurement_type = st.text_input("Measurement Type", value="Mapping XRF")
        
        st.session_state.operator = operator
        st.session_state.operator_valid = operator_valid
        st.session_state.institution = institution
        st.session_state.measurement_type = measurement_type
    
    st.markdown("---")
    
    return {
        'sample_id': sample_id,
        'substrate': substrate,
        'sample_description': sample_description,
        'sample_size': sample_size,
        'fabrication_method': fabrication_method,
        'treatment_method': treatment_method,
        'treatment_sequence': treatment_sequence,
        'air_exposure_duration': air_exposure_duration,
        'operator': operator,
        'operator_valid': operator_valid,
        'institution': institution,
        'measurement_type': measurement_type
    }


def render_layer_selection_ui(layers_info: Dict):
    """
    Render UI for layer selection when multiple layers detected
    """
    st.markdown("---")
    st.subheader("Multiple Layers Detected")
    
    num_layers = layers_info['num_layers']
    st.info(f"Found {num_layers} layers in XRF Excel file")
    
    # Display layer information
    for layer_key, layer_df in layers_info['layers'].items():
        with st.expander(f"{layer_key.upper()} - {len(layer_df)} measurements"):
            st.dataframe(layer_df.head(), use_container_width=True)
    
    st.warning("Please provide separate SPX folder paths for each layer")
    
    # Create input fields for each layer
    layer_inputs = {}
    
    for i in range(1, num_layers + 1):
        layer_name = f"layer{i}"
        st.markdown(f"### Layer {i}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            folder_path = st.text_input(
                f"SPX Folder Path for Layer {i}",
                value="",
                placeholder=f"/path/to/layer{i}/spx/files",
                key=f"spx_folder_layer{i}"
            )
        
        with col2:
            process_layer = st.checkbox(
                f"Process",
                value=True,
                key=f"process_layer{i}"
            )
        
        if folder_path:
            folder = Path(folder_path)
            if folder.exists() and folder.is_dir():
                spx_files = sorted(folder.glob('*.spx'))
                if spx_files:
                    st.success(f"Found {len(spx_files)} SPX files")
                else:
                    st.warning("No SPX files found")
            else:
                st.error("Invalid folder path")
        
        layer_inputs[layer_name] = {
            'folder_path': folder_path,
            'process': process_layer
        }
    
    return layer_inputs


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    st.title("SPX & XRF Data Integration System - Multilayer Support")
    st.markdown("Process and combine SPX files with XRF Excel data for multiple layers")
    st.markdown("---")
    
    # Metadata section
    metadata_dict = render_metadata_section()
    
    # File input section
    st.subheader("Input Files")
    
    st.markdown("##### XRF Data (Excel)")
    xrf_excel_file = st.file_uploader("Upload XRF Excel File", 
                                      type=["xls", "xlsx"],
                                      help="Upload the XRF analysis results Excel file")
    
    if xrf_excel_file is not None:
        st.success(f"Uploaded: {xrf_excel_file.name}")
        
        try:
            # Extract and detect layers
            layers_info = extract_xrf_excel_data(xrf_excel_file)
            
            st.session_state.layers_info = layers_info

            # Display all layers information
            st.markdown("##### Available Layers")

            for layer_key, layer_df in layers_info['layers'].items():
                with st.expander(f"{layer_key.capitalize()} - {len(layer_df)} measurements"):
                    st.dataframe(layer_df, use_container_width=True)
            
            st.markdown("---")
            
            # Layer selection
            st.subheader("Select Layer to Combine XRF and SPX")
            layer_options = list(layers_info['layers'].keys())
            selected_layer = st.selectbox(
                "Choose which layer to combine with SPX files",
                layer_options,
                format_func=lambda x: x.capitalize()
            )
            
            # Get selected layer data
            xrf_df = layers_info['layers'][selected_layer]
            
            # SPX folder input
            st.markdown("##### SPX Files")
            spx_folder_path = st.text_input(
                f"SPX Folder Path for {selected_layer.upper()}", 
                value=r"D:\High-throughput program\Mongo DB\ver10-newstructure\Full data\SPX", 
                placeholder=f"/path/to/{selected_layer}/spx/files"
            )
            
            if spx_folder_path:
                spx_folder = Path(spx_folder_path)
                if spx_folder.exists() and spx_folder.is_dir():
                    spx_files = sorted(spx_folder.glob('*.spx'))
                    if spx_files:
                        st.success(f"Found {len(spx_files)} SPX files")
                    else:
                        st.warning("No SPX files found")
                else:
                    st.error("Invalid folder path")
            
            # Process button
            if st.button("Combine XRF and SPX", type="primary"):
                if not st.session_state.operator_valid:
                    st.error("Please fix the operator name before processing")
                elif not spx_folder_path:
                    st.error("Please provide SPX folder path")
                else:
                    spx_folder = Path(spx_folder_path)
                    
                    if not spx_folder.exists():
                        st.error("SPX folder path is invalid")
                    else:
                        spx_files = sorted(spx_folder.glob('*.spx'))
                        
                        if len(spx_files) == 0:
                            st.error("No SPX files found in folder")
                        else:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            try:
                                # Process SPX files
                                status_text.text(f"Processing SPX files for {selected_layer}...")
                                spx_data_list = []
                                
                                for i, spx_file in enumerate(spx_files):
                                    spx_data = parse_spx_file(spx_file)
                                    spx_data_list.append(spx_data)
                                    progress_bar.progress((i + 1) / (len(spx_files) + 1))
                                
                                # Process XRF data
                                xrf_data_list = parse_xrf_excel_to_dict(xrf_df, selected_layer)
                                
                                progress_bar.progress(1.0)
                                
                                # Match files
                                status_text.text(f"Matching SPX and XRF files for {selected_layer}...")
                                combined_data = match_spx_with_xrf_excel(
                                    spx_data_list, xrf_data_list, selected_layer
                                )
                                
                                # Create metadata
                                metadata = {
                                    'sample_id': metadata_dict['sample_id'],
                                    'substrate': metadata_dict['substrate'],
                                    'sample_description': metadata_dict['sample_description'],
                                    'sample_size': metadata_dict['sample_size'],
                                    'fabrication_method': metadata_dict['fabrication_method'],
                                    'treatment_method': metadata_dict['treatment_method'],
                                    'treatment_sequence': metadata_dict['treatment_sequence'],
                                    'air_exposure_duration': metadata_dict['air_exposure_duration'],
                                    'operator': metadata_dict['operator'],
                                    'institution': metadata_dict['institution'],
                                    'measurement_type': metadata_dict['measurement_type'],
                                    'spectrometer': 'Bruker M4 Tornado',
                                    'total_layers': st.session_state.layers_info['num_layers'],
                                }
                                
                                # Generate CSV
                                csv_content = create_combined_csv(
                                    combined_data, metadata, None, selected_layer
                                )
                                
                                # Generate filename
                                created_date = datetime.now().strftime("%Y%m%d")
                                created_time = datetime.now().strftime("%H%M%S")
                                csv_filename = f"{metadata_dict['sample_id']}_{metadata_dict['institution']}_{metadata_dict['operator'].replace(' ', '_')}_{metadata_dict['treatment_method']}_{metadata_dict['treatment_sequence']}_mapping_xrf_{selected_layer}_{created_date}_{created_time}.csv"
                                
                                # Store results
                                st.session_state.processed_data = {
                                    selected_layer: {
                                        'combined_data': combined_data,
                                        'csv_content': csv_content,
                                        'csv_filename': csv_filename,
                                        'metadata': metadata,
                                        'xrf_df': xrf_df
                                    }
                                }

                                st.session_state.current_metadata = metadata

                                status_text.empty()
                                progress_bar.empty()
                                
                                matched_count = sum(1 for d in combined_data if d['matched'])
                                st.success(f"Processed {len(combined_data)} files successfully! ({matched_count} matched)")
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
        
        except Exception as e:
            st.error(f"Error reading XRF file: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.processed_data is not None:
        all_results = st.session_state.processed_data
        
        for layer_name, processed in all_results.items():
            st.markdown(f"##### {layer_name.upper()}")
            with st.expander("XRF Excel Data & Combined Data Table", expanded=False):
                
                combined_data = processed['combined_data']
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                matched_count = sum(1 for d in combined_data if d['matched'])
                
                with col1:
                    st.metric("Total SPX Files", len(combined_data))
                with col2:
                    st.metric("Matched with XRF", matched_count)
                with col3:
                    avg_thickness = np.mean([d['thickness_nm'] for d in combined_data 
                                            if d['thickness_nm'] is not None])
                    st.metric("Avg Thickness (nm)", f"{avg_thickness:.1f}" if not np.isnan(avg_thickness) else "N/A")
                with col4:
                    st.metric("Match Rate", f"{matched_count/len(combined_data)*100:.0f}%")
                
                # Display XRF data
                st.markdown("#### XRF Excel Data")
                st.dataframe(processed['xrf_df'], use_container_width=True, height=300)
                
                # Data table
                st.markdown("#### Combined Data Table")
                
                df_display = pd.DataFrame([{
                    'SPX Name': d['spx_name'],
                    'XRF Spectrum': d['xrf_spectrum_name'] if d['xrf_spectrum_name'] else 'N/A',
                    'X (mm)': f"{d['x_position_mm']:.3f}" if d['x_position_mm'] else 'N/A',
                    'Y (mm)': f"{d['y_position_mm']:.3f}" if d['y_position_mm'] else 'N/A',
                    'Z (mm)': f"{d['z_position_mm']:.3f}" if d['z_position_mm'] else 'N/A',
                    'Thickness (nm)': f"{d['thickness_nm']:.2f}" if d['thickness_nm'] is not None else 'N/A',
                    'Date': d['date'],
                    'Time': d['time'],
                    'Matched': 'Yes' if d['matched'] else 'No',
                } for d in combined_data])
                
                st.dataframe(df_display, use_container_width=True, height=400, hide_index=True)
                
                # Spectrum viewer
                st.markdown("#### Spectrum Viewer")
                
                if combined_data:
                    selected_file = st.selectbox(
                        "Select file to view spectrum",
                        [d['spx_name'] for d in combined_data],
                        index=0,
                        key=f"spectrum_select_{layer_name}"
                    )
                    
                    selected_data = next(d for d in combined_data if d['spx_name'] == selected_file)
                    
                    # Show metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Position:** ({selected_data['x_position_mm']:.3f}, {selected_data['y_position_mm']:.3f}) mm")
                    with col2:
                        st.info(f"**Thickness:** {selected_data['thickness_nm']:.2f} nm" if selected_data['thickness_nm'] else "**Thickness:** N/A")
                    with col3:
                        st.info(f"**Matched:** {'Yes' if selected_data['matched'] else 'No'}")
                    
                    fig = plot_spectrum(selected_data, f"Spectrum: {selected_file}")
                    st.plotly_chart(fig, use_container_width=True)
                
                # # ====================Download button for CSV fies with original xyz position in XRF measurement===================
                # st.markdown("---")
                # st.markdown("#### Download CSV")
                # st.download_button(
                #     label=f"Download CSV File - {layer_name.upper()}",
                #     data=processed['csv_content'],
                #     file_name=processed['csv_filename'],
                #     mime="text/csv",
                #     key=f"download_{layer_name}"
                # )


            # Coordinate conversion section (add after Download CSV button)
            st.markdown("---")
            st.subheader("Converting Coordinates: XRF Meaurement to Optical Mesurement")
            
            xrf_bounds = extract_xrf_bounds(combined_data)
            
            if xrf_bounds is None:
                st.error("Could not extract XRF coordinate bounds from the data")
            else:
                st.success("XRF coordinate bounds extracted from files")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("##### XRF Measurement Coordinates")
                    st.info(f"""
                    **First point (x1, y1):** ({xrf_bounds['x_1']:.3f}, {xrf_bounds['y_1']:.3f}) mm  
                    **Last point (x2, y2):** ({xrf_bounds['x_2']:.3f}, {xrf_bounds['y_2']:.3f}) mm
                    """)
                
                with col_right:
                    st.markdown("##### Optical Measurement Coordinates")
                    col5, col6 = st.columns(2)
                    x_1_opt = col5.number_input("x1 (mm)", value=5.0, format="%.3f", key=f"opt_x1_{layer_name}")
                    y_1_opt = col6.number_input("y1 (mm)", value=5.0, format="%.3f", key=f"opt_y1_{layer_name}")
                    col7, col8 = st.columns(2)
                    x_2_opt = col7.number_input("x2 (mm)", value=45.0, format="%.3f", key=f"opt_x2_{layer_name}")
                    y_2_opt = col8.number_input("y2 (mm)", value=45.0, format="%.3f", key=f"opt_y2_{layer_name}")
                
                optical_bounds = {
                    'x_1': x_1_opt, 'y_1': y_1_opt,
                    'x_2': x_2_opt, 'y_2': y_2_opt
                }
                
                if st.button("Convert Coordinates", type="primary", key=f"convert_{layer_name}"):
                    converted_data_result = convert_xrf_to_optical(
                        combined_data, xrf_bounds, optical_bounds
                    )
                    
                    st.session_state[f'converted_data_{layer_name}'] = converted_data_result
                    st.success(f"Converted {len(converted_data_result)} coordinates successfully!")
                
                if f'converted_data_{layer_name}' in st.session_state:
                    converted_data_result = st.session_state[f'converted_data_{layer_name}']
                    
                    # Visualization
                    st.markdown("---")
                    fig = plot_coordinate_comparison(xrf_bounds, optical_bounds, converted_data_result)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export with converted coordinates
                    st.markdown("---")
                    st.subheader("Download CSV with Converted Coordinates")
                    
                    # Get metadata from session state or from processed data
                    if 'current_metadata' in st.session_state:
                        metadata = st.session_state.current_metadata
                    else:
                        metadata = processed['metadata']
                    
                    csv_content_converted = create_combined_csv(
                        combined_data, metadata, converted_data_result, layer_name
                    )
                    
                    created_date = datetime.now().strftime("%Y%m%d")
                    created_time = datetime.now().strftime("%H%M%S")
                    csv_filename_converted = f"{metadata['sample_id']}_{metadata['institution']}_{metadata['operator'].replace(' ', '_')}_{metadata['treatment_method']}_{metadata['treatment_sequence']}_mapping_xrf_{layer_name}_{created_date}_{created_time}.csv"
                    
                    st.download_button(
                        label="Download CSV File",
                        data=csv_content_converted,
                        file_name=csv_filename_converted,
                        mime="text/csv",
                        type="primary",
                        key=f"download_optical_{layer_name}"
                    )

if __name__ == "__main__":
    main()