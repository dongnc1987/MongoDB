import streamlit as st
from datetime import datetime, timezone
import re
import pandas as pd
from typing import Tuple, Optional, Dict, List
from io import StringIO


# ==================== HELPER FUNCTIONS ====================

def normalize_treatment_method(method: str) -> str:
    """Normalize treatment method name"""
    return method.lower().replace(' ', '_').replace('-', '_')


def parse_datetime_from_strings(date_str: str, time_str: str, fallback: str) -> str:
    """Parse datetime from date and time strings with multiple format attempts"""
    try:
        date_formats = ["%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"]
        
        for date_format in date_formats:
            try:
                dt = datetime.strptime(f"{date_str} {time_str}", date_format)
                dt_utc = dt.replace(tzinfo=timezone.utc)
                return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
    except Exception:
        pass
    
    return fallback


def safe_float_extract(value, default=0):
    """Safely extract float from value, return default if invalid"""
    try:
        if value and str(value).strip().upper() not in ['N/A', 'NAN', '']:
            return float(value)
    except (ValueError, TypeError):
        pass
    return default


def safe_int_extract(value, default=0):
    """Safely extract int from value, return default if invalid"""
    try:
        if value and str(value).strip().upper() not in ['N/A', 'NAN', '']:
            return int(float(value))
    except (ValueError, TypeError):
        pass
    return default


# ==================== FILENAME PARSING ====================

def parse_measurement_filename(filename: str) -> Optional[Dict]:
    """
    Parse measurement mapping filename to extract metadata.
    
    Formats:
    - PL:           {Sample}_{Institution}_{Operator}_{Treatment}_{Sequence}_mapping_pl_{YYYYMMDD}_{HHMMSS}.csv
    - UV-Vis:       {Sample}_{Institution}_{Operator}_{Treatment}_{Sequence}_mapping_{MeasurementType}_{YYYYMMDD}_{HHMMSS}.csv
    - Resistance:   {Sample}_{Institution}_{Operator}_{Treatment}_{Sequence}_mapping_resistance_{YYYYMMDD}_{HHMMSS}.csv
    - MPL:          {Sample}_{Institution}_{Operator}_{Treatment}_{Sequence}_mapping_mpl_{attenuation|phaseshift}_{YYYYMMDD}_{HHMMSS}.csv
    - XRF:          {Sample}_{Institution}_{Operator}_{Treatment}_{Sequence}_mapping_xrf_{layer}_{YYYYMMDD}_{HHMMSS}.csv
    """
    
    patterns = {
        'pl': (r'(.+?)_(.+?)_(.+?)_(.+?)_(\d+)_mapping_pl_(\d{8})_(\d{6})\.csv', 
               'pl', 'Mapping PL'),
        'uv_vis': (r'(.+?)_(.+?)_(.+?)_(.+?)_(\d+)_mapping_(transmittance|reflectance|absorptance)_(\d{8})_(\d{6})\.csv', 
                   'uv_vis', None),
        'resistance': (r'(.+?)_(.+?)_(.+?)_(.+?)_(\d+)_mapping_resistance_(\d{8})_(\d{6})\.csv', 
                       'resistance', 'Mapping Resistance'),
        'mpl': (r'(.+?)_(.+?)_(.+?)_(.+?)_(\d+)_mapping_mpl_(attenuation|phaseshift)_(\d{8})_(\d{6})\.csv',
                'mpl', None),
        'xrf': (r'(.+?)_(.+?)_(.+?)_(.+?)_(\d+)_mapping_xrf_(.+?)_(\d{8})_(\d{6})\.csv',
                'xrf', 'Mapping XRF')
    }
    
    for tech_key, (pattern, technique, meas_type) in patterns.items():
        match = re.match(pattern, filename)
        
        if match:
            groups = match.groups()
            
            # Extract common fields
            sample_number = groups[0]
            institution = groups[1]
            operator = groups[2]
            treatment_method_raw = groups[3]
            treatment_sequence = groups[4]
            
            # Clean operator name from treatment_method if it got captured together
            # This handles cases where operator has underscores like "Dong_Nguyen"
            treatment_parts = treatment_method_raw.split('_')
            
            # Find where treatment actually starts by looking for common treatment keywords
            treatment_keywords = ['as-deposited', 'as_deposited', 'annealing', 'anneal', 'aging', 
                                 'treated', 'pristine', 'fresh', 'stored']
            
            treatment_start_idx = 0
            for i, part in enumerate(treatment_parts):
                if any(keyword in part.lower() for keyword in treatment_keywords):
                    treatment_start_idx = i
                    break
            
            # Reconstruct treatment method from the correct starting point
            if treatment_start_idx > 0:
                treatment_method = '_'.join(treatment_parts[treatment_start_idx:])
            else:
                treatment_method = treatment_method_raw
            
            # Handle UV-Vis, MPL, and XRF special cases
            if tech_key in ['uv_vis', 'mpl']:
                measurement_type = groups[5]
                date_str = groups[6]
                time_str = groups[7]
            elif tech_key == 'xrf':
                material_layer = groups[5]
                date_str = groups[6]
                time_str = groups[7]
                measurement_type = f"XRF {material_layer}"
            else:
                measurement_type = meas_type
                date_str = groups[5]
                time_str = groups[6]
            
            # Parse datetime
            try:
                dt = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                dt_utc = dt.replace(tzinfo=timezone.utc)
                acq_time_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            except:
                return None
            
            result = {
                'sample_number': sample_number,
                'institution': institution,
                'operator': operator,
                'treatment_method': treatment_method,
                'treatment_sequence': treatment_sequence,
                'date_str': date_str,
                'time_str': time_str,
                'acq_time_iso': acq_time_iso,
                'technique': technique,
                'measurement_type': measurement_type,
            }
            
            if tech_key == 'xrf':
                result['material_layer'] = material_layer
            
            return result
    
    return None

# ==================== CSV PARSING ====================

def parse_csv_metadata_and_data(content: str, is_resistance: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Parse CSV file with metadata in first rows and data below.
    Handles both spectrum (PL/UV-Vis) and resistance files.
    """
    try:
        lines = content.strip().split('\n')
        metadata = {}
        data_start_idx = 0
        wavelengths = []
        header_row_idx = None
        
        # Find header row (contains 'x position')
        for idx, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            
            # Check if ANY column contains 'x position' (not just first column)
            has_position = any('x position' in p.lower() for p in parts)
            
            if has_position:
                header_row_idx = idx
                
                # For spectrum files, extract wavelengths from header
                if not is_resistance:
                    for i, part in enumerate(parts):
                        # Skip position columns and date/time columns
                        if 'position' not in part.lower() and 'date' not in part.lower() and 'time' not in part.lower():
                            try:
                                wl = float(part)
                                wavelengths.append(wl)
                            except ValueError:
                                pass
                    
                    if wavelengths:
                        metadata['wavelengths'] = wavelengths
                    else:
                        st.warning("Could not find wavelengths in header")
                
                # Data starts on the next row after header
                data_start_idx = idx
                break
        
        if header_row_idx is None:
            st.error("Could not find header row with position columns")
            return None, None
        
        # Extract metadata from rows before header
        for idx in range(header_row_idx):
            line = lines[idx]
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                
                if key and value and value.lower() != 'nan':
                    metadata[key] = value
        
        # Parse data section starting from header row
        if data_start_idx < len(lines):
            data_lines = '\n'.join(lines[data_start_idx:])
            df = pd.read_csv(StringIO(data_lines), sep=',')
            df.columns = df.columns.str.strip()
            return df, metadata
        
        return None, metadata
    
    except Exception as e:
        st.error(f"Error parsing CSV: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


def parse_spectrum_csv_data(content: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Parse spectrum CSV file (PL or UV-Vis)"""
    return parse_csv_metadata_and_data(content, is_resistance=False)


def parse_resistance_csv_data(content: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """Parse resistance CSV file"""
    return parse_csv_metadata_and_data(content, is_resistance=True)








# ==================== VALIDATION ====================

def validate_measurement_csv(filename: str, db) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate measurement CSV file format and check if it already exists in database.
    Returns: (is_valid, error_msg, warning_msg)
    """
    try:
        metadata = parse_measurement_filename(filename)
        if not metadata:
            return False, "Invalid filename format", None
        
        technique = metadata['technique']
        sample_number = metadata['sample_number']
        treatment_seq = metadata['treatment_sequence']
        treatment_method = metadata['treatment_method']
        measurement_type = metadata['measurement_type']
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        
        validation_rules = {
            'pl': ('mapping_pl', "PL file must contain 'mapping_pl' in filename"),
            'uv_vis': (None, None),
            'resistance': ('mapping_resistance', "Resistance file must contain 'mapping_resistance' in filename"),
            'mpl': ('mapping_mpl', "MPL file must contain 'mapping_mpl' in filename"),
            'xrf': ('mapping_xrf', "XRF file must contain 'mapping_xrf' in filename")
        }
        
        if technique in validation_rules:
            required_text, error_msg = validation_rules[technique]
            
            if technique == 'uv_vis':
                valid_types = ['transmittance', 'reflectance', 'absorptance']
                if metadata['measurement_type'] not in valid_types:
                    return False, f"Invalid UV-Vis type. Must be one of: {', '.join(valid_types)}", None
            
            elif technique == 'mpl':
                valid_types = ['attenuation', 'phaseshift']
                if metadata['measurement_type'] not in valid_types:
                    return False, f"Invalid MPL type. Must be one of: {', '.join(valid_types)}", None
            
            elif required_text and required_text not in filename.lower():
                return False, error_msg, None
        
        # Check if acquisition already exists in database
        if technique == 'xrf':
            material_layer = metadata.get('material_layer', 'unknown')
            acq_id = f"{sample_number}_treat{treatment_seq}_{treatment_method_normalized}_acq_xrf_{material_layer}"
        elif technique == 'mpl':
            acq_id = f"{sample_number}_treat{treatment_seq}_{treatment_method_normalized}_acq_mpl"
        elif technique == 'resistance':
            acq_id = f"{sample_number}_treat{treatment_seq}_{treatment_method_normalized}_acq_resistance"
        elif technique == 'pl':
            acq_id = f"{sample_number}_treat{treatment_seq}_{treatment_method_normalized}_acq_pl"
        else:  # uv_vis
            acq_id = f"{sample_number}_treat{treatment_seq}_{treatment_method_normalized}_acq_uv_vis_{measurement_type}"
        
        existing_acq = db['acquisitions'].find_one({"_id": acq_id})
        
        warning_msg = None
        if existing_acq:
            if technique == 'mpl':
                existing_filenames = existing_acq.get('filenames', {})
                if measurement_type in existing_filenames:
                    warning_msg = f"MPL {measurement_type} already exists (will be updated)"
            else:
                warning_msg = f"Measurement already exists (will be updated)"
        
        return True, None, warning_msg
    
    except Exception as e:
        return False, f"Validation error: {str(e)}", None


# ==================== ACQUISITION CREATION ====================

def create_spectrum_acquisition(db, sample_number: str, treatment_sequence: str, 
                                treatment_method: str, technique: str, measurement_type: str,
                                metadata: Dict, acq_time: str, df: pd.DataFrame, 
                                filename: str) -> Tuple[bool, str, Optional[str]]:
    """Create acquisition document for spectrum measurement (PL or UV-Vis)"""
    try:
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        
        # Generate acquisition ID
        if technique == 'pl':
            acq_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}_acq_pl"
        else:
            acq_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}_acq_uv_vis_{measurement_type}"
        
        # Extract wavelength range
        wavelengths = [float(wl) for wl in metadata.get('wavelengths', []) if wl is not None]
        wavelength_min = float(wavelengths[0]) if wavelengths else 0
        wavelength_max = float(wavelengths[-1]) if wavelengths else 0
        measurement_count = len(df)
        
        # Build acquisition document
        acq_doc = {
            "_id": acq_id,
            "sample_number": sample_number,
            "treatment_id": f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}",
            "treatment_method": treatment_method,
            "treatment_sequence": treatment_sequence,
            "air_exposure_duration_min": safe_int_extract(metadata.get('Air exposure Duration (min)', 0)),
            "technique": technique,
            "measurement_type": measurement_type,
            "instrument": {
                "model": metadata.get('Spectrometer', 'CCS200'),
                "type": "spectrometer"
            },
            "wavelength_range_nm": [wavelength_min, wavelength_max],
            "measurement_count": measurement_count,
            "filename": filename,
            "measured_date(y/m/d)_time": acq_time,
            "created_date_time": current_time,
            "updated_date_time": current_time
        }
        
        # Technique-specific configuration
        if technique == 'pl':
            acq_doc["config"] = {
                "laser_wavelength_nm": 532,
                "laser_power_mw": safe_float_extract(metadata.get('Laser Power (mW)', 0)),
                "laser_irradiance_w_cm2": safe_float_extract(metadata.get('Laser Irradiance (W/cm2)', 0)),
                "laser_diameter_mm": safe_float_extract(metadata.get('Laser Diameter (mm)', 0.2)),
                "exposure_time_ms": safe_int_extract(metadata.get('Exposure Time (ms)', 1500)),
                "laser_current_ma": safe_int_extract(metadata.get('Laser Current (mA)', 200)),
                "filter_wheel_position": safe_int_extract(metadata.get('Filter wheel position', 1))
            }
        else:  # UV-Vis
            exposure_time = safe_float_extract(metadata.get('Exposure Time (ms)', None), default=None)
            
            acq_doc["config"] = {
                "light_diameter_mm": safe_float_extract(metadata.get('Light Diameter (mm)', 1.0), default=1.0),
                "exposure_time_ms": exposure_time
            }
            acq_doc["sample_info"] = {
                "substrate": metadata.get('Substrate', ''),
                "sample_size": metadata.get('Sample Size', '')
            }
        
        # Insert or update
        existing_acq = db['acquisitions'].find_one({"_id": acq_id})
        if existing_acq:
            acq_doc["created_date_time"] = existing_acq.get("created_date_time", current_time)
            db['acquisitions'].replace_one({"_id": acq_id}, acq_doc)
        else:
            db['acquisitions'].insert_one(acq_doc)
        
        return True, "Acquisition created successfully", acq_id
    
    except Exception as e:
        return False, f"Error creating acquisition: {str(e)}", None


def create_resistance_acquisition(db, sample_number: str, treatment_sequence: str,
                                  treatment_method: str, metadata: Dict, acq_time: str,
                                  df: pd.DataFrame, filename: str) -> Tuple[bool, str, Optional[str]]:
    """Create acquisition document for resistance measurement"""
    try:
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        acq_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}_acq_resistance"
        
        acq_doc = {
            "_id": acq_id,
            "sample_number": sample_number,
            "treatment_id": f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}",
            "treatment_method": treatment_method,
            "treatment_sequence": treatment_sequence,
            "air_exposure_duration_min": safe_int_extract(metadata.get('Air exposure Duration (min)', 0)),
            "technique": "resistance",
            "measurement_type": "Mapping Resistance",
            "instrument": {
                "model": metadata.get('Instrument', 'Keithley2601B'),
                "type": "source_meter"
            },
            "sample_info": {
                "substrate": metadata.get('Substrate', ''),
                "sample_size": metadata.get('Sample Size', ''),
                "fabrication_method": metadata.get('Fabrication Method', ''),
                "description": metadata.get('Sample Description', '')
            },
            "measurement_count": len(df),
            "filename": filename,
            "measured_date(y/m/d)_time": acq_time,
            "created_date_time": current_time,
            "updated_date_time": current_time
        }
        
        existing_acq = db['acquisitions'].find_one({"_id": acq_id})
        if existing_acq:
            acq_doc["created_date_time"] = existing_acq.get("created_date_time", current_time)
            db['acquisitions'].replace_one({"_id": acq_id}, acq_doc)
        else:
            db['acquisitions'].insert_one(acq_doc)
        
        return True, "Acquisition created successfully", acq_id
    
    except Exception as e:
        return False, f"Error creating acquisition: {str(e)}", None



# ==================== CREATION OF SPECTRA/PIXELS COLLECTIONS ====================

def extract_position(row, df, idx, axis='x'):
    """Extract x or y position from row, with fallback to index"""
    col_name = f'{axis} position (mm)'
    
    if col_name not in df.columns:
        return (int(idx) + 1) if axis == 'x' else 1
    
    try:
        val = row[col_name]
        return int(float(val)) if val and str(val).upper() != 'N/A' else ((int(idx) + 1) if axis == 'x' else 1)
    except (ValueError, TypeError, KeyError):
        return (int(idx) + 1) if axis == 'x' else 1


def create_spectrum_spectra(db, acq_id: str, sample_number: str, treatment_sequence: str,
                           treatment_method: str, technique: str, measurement_type: str,
                           df: pd.DataFrame, metadata: Dict, acq_time: str) -> Tuple[int, List[str]]:
    """Create spectrum documents for each (x,y) position. Z position ignored."""
    try:
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        treatment_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}"
        
        wavelength_values = [float(wl) for wl in metadata.get('wavelengths', []) if wl is not None]
        
        if not wavelength_values:
            return 0, ["No wavelength values found in metadata"]
        
        spectra_count = 0
        errors = []
        position_cols = ['x position (mm)', 'y position (mm)', 'date', 'time']
        spectrum_prefix = 'pl' if technique == 'pl' else 'uv_vis'
        y_quantity = 'pl_intensity' if technique == 'pl' else measurement_type
        
        for idx, row in df.iterrows():
            try:
                x_pos = extract_position(row, df, idx, 'x')
                y_pos = extract_position(row, df, idx, 'y')
                
                # Parse measurement time
                measurement_time = acq_time
                if 'date' in df.columns and 'time' in df.columns:
                    date_str = str(row['date']).strip()
                    time_str = str(row['time']).strip()
                    measurement_time = parse_datetime_from_strings(date_str, time_str, acq_time)
                
                # Extract intensities
                intensities = []
                for col in df.columns:
                    if col not in position_cols and 'z position' not in col.lower():
                        try:
                            val = row[col]
                            intensities.append(float(val) if val and str(val).upper() != 'N/A' else 0)
                        except (ValueError, TypeError):
                            intensities.append(0)
                
                if not intensities:
                    errors.append(f"Row {idx}: No intensity values found")
                    continue
                
                intensities = intensities[:len(wavelength_values)]
                spectrum_id = f"{acq_id}_{spectrum_prefix}_{x_pos}_{y_pos}"
                
                spectrum_doc = {
                    "_id": spectrum_id,
                    "acq_id": acq_id,
                    "treatment_id": treatment_id,
                    "technique": technique,
                    "measurement_type": measurement_type,
                    "position": {"x": x_pos, "y": y_pos, "unit": "mm"},
                    "x_axis": {"quantity": "wavelength_nm", "values": wavelength_values},
                    "y": {"quantity": y_quantity, "values": intensities},
                    "measurement_date_time": measurement_time,
                    "measured_date(y/m/d)_time": acq_time,
                    "created_date_time": current_time,
                    "updated_date_time": current_time
                }
                
                existing_spectrum = db['spectra'].find_one({"_id": spectrum_id})
                if existing_spectrum:
                    spectrum_doc["created_date_time"] = existing_spectrum.get("created_date_time", current_time)
                    db['spectra'].replace_one({"_id": spectrum_id}, spectrum_doc)
                else:
                    db['spectra'].insert_one(spectrum_doc)
                
                spectra_count += 1
            
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                continue
        
        return spectra_count, errors
    
    except Exception as e:
        return 0, [f"Error creating spectra: {str(e)}"]


def create_resistance_pixels(db, acq_id: str, sample_number: str, treatment_sequence: str,
                             treatment_method: str, df: pd.DataFrame, acq_time: str) -> Tuple[int, List[str]]:
    """Create pixel documents for each (x,y) position. Z position ignored."""
    try:
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        treatment_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}"
        
        pixels_count = 0
        errors = []
        
        # Find resistance column
        resistance_column = None
        for col in df.columns:
            if 'resistance' in col.lower():
                resistance_column = col
                break
        
        if not resistance_column:
            return 0, [f"Resistance column not found. Available columns: {df.columns.tolist()}"]
        
        for idx, row in df.iterrows():
            try:
                x_pos = extract_position(row, df, idx, 'x')
                y_pos = extract_position(row, df, idx, 'y')
                
                # Parse measurement time
                measurement_time = acq_time
                if 'date' in df.columns and 'time' in df.columns:
                    date_str = str(row['date']).strip()
                    time_str = str(row['time']).strip()
                    measurement_time = parse_datetime_from_strings(date_str, time_str, acq_time)
                
                # Extract resistance
                resistance = safe_float_extract(row.get(resistance_column))
                if resistance == 0 and row.get(resistance_column) is None:
                    errors.append(f"Row {idx}: No resistance value found")
                    continue
                
                pixel_id = f"{acq_id}_resistance_{x_pos}_{y_pos}"
                
                pixel_doc = {
                    "_id": pixel_id,
                    "acq_id": acq_id,
                    "treatment_id": treatment_id,
                    "technique": "resistance",
                    "measurement_type": "Mapping Resistance",
                    "position": {"x": x_pos, "y": y_pos, "unit": "mm"},
                    "resistance_ohm": resistance,
                    "measurement_date_time": measurement_time,
                    "measured_date(y/m/d)_time": acq_time,
                    "created_date_time": current_time,
                    "updated_date_time": current_time
                }
                
                existing_pixel = db['pixels'].find_one({"_id": pixel_id})
                if existing_pixel:
                    pixel_doc["created_date_time"] = existing_pixel.get("created_date_time", current_time)
                    db['pixels'].replace_one({"_id": pixel_id}, pixel_doc)
                else:
                    db['pixels'].insert_one(pixel_doc)
                
                pixels_count += 1
            
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                continue
        
        return pixels_count, errors
    
    except Exception as e:
        return 0, [f"Error creating pixels: {str(e)}"]


# ========================================
# mPL Measurement
# ========================================

def create_mpl_acquisition(db, sample_number: str, treatment_sequence: str, 
                          treatment_method: str, measurement_type: str,
                          metadata: Dict, acq_time: str, df: pd.DataFrame,
                          filename: str) -> Tuple[bool, str, Optional[str]]:
    """Create or update acquisition document for MPL measurement (attenuation OR phaseshift)"""
    try:
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        
        # Generate acquisition ID (same for both attenuation and phaseshift)
        acq_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}_acq_mpl"
        
        # Extract frequency range from metadata
        frequency_min = safe_float_extract(metadata.get('frequency_min_hz', metadata.get('Start Frequency (Hz)', 0)))
        frequency_max = safe_float_extract(metadata.get('frequency_max_hz', metadata.get('Stop Frequency (Hz)', 0)))
        frequency_points = safe_int_extract(metadata.get('frequency_points', metadata.get('Number of Points', 0)))
        measurement_count = len(df)
        
        # Check if acquisition already exists
        existing_acq = db['acquisitions'].find_one({"_id": acq_id})
        
        if existing_acq:
            # Update existing acquisition
            update_fields = {
                "updated_date_time": current_time,
                f"filenames.{measurement_type}": filename
            }
            
            # Update measurement count if this file has different count
            if measurement_count != existing_acq.get('measurement_count', 0):
                update_fields['measurement_count'] = max(measurement_count, existing_acq.get('measurement_count', 0))
            
            db['acquisitions'].update_one(
                {"_id": acq_id},
                {"$set": update_fields}
            )
            
            return True, f"MPL Acquisition updated with {measurement_type} data", acq_id
        
        else:
            # Create new acquisition document
            acq_doc = {
                "_id": acq_id,
                "sample_number": sample_number,
                "treatment_id": f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}",
                "treatment_method": treatment_method,
                "treatment_sequence": treatment_sequence,
                "air_exposure_duration_min": safe_int_extract(metadata.get('Air exposure Duration (min)', 0)),
                "technique": "mpl",
                "measurement_type": metadata.get('Measurement Type', 'Mapping mPL'),
                "instrument": {
                    "model": metadata.get('Spectrometer', 'Moku-Lab'),
                    "type": "Lock-in Amplifier"
                },
                "frequency_range_hz": [frequency_min, frequency_max],
                "frequency_points": frequency_points,
                "measurement_count": measurement_count,
                "filenames": {
                    measurement_type: filename
                },
                "config": {
                    "laser_wavelength_nm": 532,
                    "laser_power_mw": safe_float_extract(metadata.get('Laser Power (mW)', 0)),
                    "laser_irradiance_w_cm2": safe_float_extract(metadata.get('Laser Irradiance (W/cm2)', 0)),
                    "laser_diameter_mm": safe_float_extract(metadata.get('Laser Diameter (mm)', 0.2)),
                    "laser_current_ma": safe_int_extract(metadata.get('Laser Current (mA)', 80)),
                    "filter_wheel_position": safe_int_extract(metadata.get('Filter wheel position', 1)),
                    "averaging_time_ms": safe_int_extract(metadata.get('Averaging Time (ms)', 20)),
                    "averaging_cycles": safe_int_extract(metadata.get('Averaging Cycles', 5)),
                    "settling_cycles": safe_int_extract(metadata.get('Settling Cycles', 5)),
                    "settling_time_ms": safe_int_extract(metadata.get('Settling Time (ms)', 20))
                },
                "sample_info": {
                    "substrate": metadata.get('Substrate', ''),
                    "sample_size": metadata.get('Sample Size', ''),
                    "fabrication_method": metadata.get('Fabrication Method', ''),
                    "description": metadata.get('Sample Description', '')
                },
                "measured_date(y/m/d)_time": acq_time,
                "created_date_time": current_time,
                "updated_date_time": current_time
            }
            
            db['acquisitions'].insert_one(acq_doc)
            
            return True, f"MPL Acquisition created with {measurement_type} data", acq_id
    
    except Exception as e:
        return False, f"Error creating MPL acquisition: {str(e)}", None


#==================== CREATION OF SPECTRA FOR MODULATED PHOTOLUMINESCENCE ====================

def create_mpl_spectra(db, acq_id: str, sample_number: str, treatment_sequence: str,
                      treatment_method: str, measurement_type: str, df: pd.DataFrame,
                      metadata: Dict, acq_time: str) -> Tuple[int, List[str]]:
    """Create or update MPL spectrum documents - combines attenuation and phaseshift at each (x,y) position"""
    try:
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        treatment_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}"
        
        frequency_values = [float(freq) for freq in metadata.get('frequencies', []) if freq is not None]
        
        if not frequency_values:
            return 0, ["No frequency values found in metadata"]
        
        spectra_count = 0
        errors = []
        position_cols = ['x position (mm)', 'y position (mm)', 'date', 'time']
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                x_pos = extract_position(row, df, idx, 'x')
                y_pos = extract_position(row, df, idx, 'y')
                
                # Parse measurement time
                measurement_time = acq_time
                if 'date' in df.columns and 'time' in df.columns:
                    date_str = str(row['date']).strip()
                    time_str = str(row['time']).strip()
                    measurement_time = parse_datetime_from_strings(date_str, time_str, acq_time)
                
                # Extract values (attenuation or phaseshift)
                values = []
                for col in df.columns:
                    if col not in position_cols and 'z position' not in col.lower() and 'tau' not in col.lower() and 'r2' not in col.lower():
                        try:
                            val = row[col]
                            values.append(float(val) if val and str(val).upper() != 'N/A' else 0)
                        except (ValueError, TypeError):
                            values.append(0)
                
                if not values:
                    errors.append(f"Row {idx}: No {measurement_type} values found")
                    continue
                
                # Trim to match frequency length
                values = values[:len(frequency_values)]
                
                # Create unified spectrum ID (without measurement_type)
                spectrum_id = f"{acq_id}_{x_pos}_{y_pos}"
                
                # Check if spectrum already exists
                existing_spectrum = db['spectra'].find_one({"_id": spectrum_id})
                
                if existing_spectrum:
                    # Update existing spectrum with new measurement type data
                    if measurement_type == 'attenuation':
                        db['spectra'].update_one(
                            {"_id": spectrum_id},
                            {"$set": {
                                "attenuation": {
                                    "quantity": "attenuation",
                                    "values": values
                                },
                                "updated_date_time": current_time
                            }}
                        )
                    elif measurement_type == 'phaseshift':
                        db['spectra'].update_one(
                            {"_id": spectrum_id},
                            {"$set": {
                                "phase_shift": {
                                    "quantity": "phase_shift_degrees",
                                    "values": values
                                },
                                "updated_date_time": current_time
                            }}
                        )
                else:
                    # Create new spectrum document
                    spectrum_doc = {
                        "_id": spectrum_id,
                        "acq_id": acq_id,
                        "treatment_id": treatment_id,
                        "technique": "mpl",
                        "measurement_type": "Mapping mPL",
                        "position": {"x": x_pos, "y": y_pos, "unit": "mm"},
                        "x_axis": {"quantity": "frequency_hz", "values": frequency_values},
                        "measurement_date_time": measurement_time,
                        "measured_date(y/m/d)_time": acq_time,
                        "created_date_time": current_time,
                        "updated_date_time": current_time
                    }
                    
                    # Add attenuation or phaseshift data
                    if measurement_type == 'attenuation':
                        spectrum_doc["attenuation"] = {
                            "quantity": "attenuation",
                            "values": values
                        }
                    elif measurement_type == 'phaseshift':
                        spectrum_doc["phase_shift"] = {
                            "quantity": "phase_shift_degrees",
                            "values": values
                        }
                    
                    db['spectra'].insert_one(spectrum_doc)
                
                spectra_count += 1
            
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                continue
        
        return spectra_count, errors
    
    except Exception as e:
        return 0, [f"Error creating MPL spectra: {str(e)}"]


#==================== CREATION OF PIXELS COLLECTIONS FOR MODULATED PHOTOLUMINESCENCE ====================

def create_mpl_pixels(db, acq_id: str, sample_number: str, treatment_sequence: str,
                     treatment_method: str, measurement_type: str, df: pd.DataFrame,
                     acq_time: str) -> Tuple[int, List[str]]:
    """Create or update MPL pixel documents with fitted lifetime parameters from attenuation OR phaseshift"""
    try:
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        treatment_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}"
        
        pixels_count = 0
        errors = []
        
        # Helper function to extract valid lifetime values (null if N/A or negative)
        def extract_lifetime_value(row, column_name):
            """Extract lifetime parameter, return None if N/A, empty, or negative"""
            try:
                val = row.get(column_name)
                if val is None or str(val).strip().upper() in ['N/A', 'NAN', '']:
                    return None
                float_val = float(val)
                # Return None if negative
                if float_val < 0:
                    return None
                return float_val
            except (ValueError, TypeError):
                return None
        
        # Process each position
        for idx, row in df.iterrows():
            try:
                x_pos = extract_position(row, df, idx, 'x')
                y_pos = extract_position(row, df, idx, 'y')
                
                # Parse measurement time
                measurement_time = acq_time
                if 'date' in df.columns and 'time' in df.columns:
                    date_str = str(row['date']).strip()
                    time_str = str(row['time']).strip()
                    measurement_time = parse_datetime_from_strings(date_str, time_str, acq_time)
                
                pixel_id = f"{acq_id}_mpl_{x_pos}_{y_pos}"
                
                # Check if pixel already exists
                existing_pixel = db['pixels'].find_one({"_id": pixel_id})
                
                if measurement_type == 'attenuation':
                    # Extract attenuation parameters (None if N/A or negative)
                    tau_amplitude = extract_lifetime_value(row, 'tau_amplitude (s)')
                    r2_attenuation = extract_lifetime_value(row, 'R2_attenuation (%)')
                    
                    if existing_pixel:
                        # Update existing pixel with attenuation data
                        db['pixels'].update_one(
                            {"_id": pixel_id},
                            {"$set": {
                                "lifetime_parameters.tau_amplitude_s": tau_amplitude,
                                "lifetime_parameters.r2_attenuation_percent": r2_attenuation,
                                "updated_date_time": current_time
                            }}
                        )
                    else:
                        # Create new pixel with attenuation data
                        pixel_doc = {
                            "_id": pixel_id,
                            "acq_id": acq_id,
                            "treatment_id": treatment_id,
                            "technique": "mpl",
                            "measurement_type": "Mapping mPL",
                            "position": {"x": x_pos, "y": y_pos, "unit": "mm"},
                            "lifetime_parameters": {
                                "tau_amplitude_s": tau_amplitude,
                                "r2_attenuation_percent": r2_attenuation
                            },
                            "measurement_date_time": measurement_time,
                            "measured_date(y/m/d)_time": acq_time,
                            "created_date_time": current_time,
                            "updated_date_time": current_time
                        }
                        db['pixels'].insert_one(pixel_doc)
                
                elif measurement_type == 'phaseshift':
                    # Extract phaseshift parameters (None if N/A or negative)
                    tau_phase = extract_lifetime_value(row, 'tau_phase (s)')
                    r2_phase = extract_lifetime_value(row, 'R2_phase (%)')
                    
                    if existing_pixel:
                        # Update existing pixel with phaseshift data
                        db['pixels'].update_one(
                            {"_id": pixel_id},
                            {"$set": {
                                "lifetime_parameters.tau_phase_s": tau_phase,
                                "lifetime_parameters.r2_phase_percent": r2_phase,
                                "updated_date_time": current_time
                            }}
                        )
                    else:
                        # Create new pixel with phaseshift data
                        pixel_doc = {
                            "_id": pixel_id,
                            "acq_id": acq_id,
                            "treatment_id": treatment_id,
                            "technique": "mpl",
                            "measurement_type": "Mapping mPL",
                            "position": {"x": x_pos, "y": y_pos, "unit": "mm"},
                            "lifetime_parameters": {
                                "tau_phase_s": tau_phase,
                                "r2_phase_percent": r2_phase
                            },
                            "measurement_date_time": measurement_time,
                            "measured_date(y/m/d)_time": acq_time,
                            "created_date_time": current_time,
                            "updated_date_time": current_time
                        }
                        db['pixels'].insert_one(pixel_doc)
                
                pixels_count += 1
            
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                continue
        
        return pixels_count, errors
    
    except Exception as e:
        return 0, [f"Error creating MPL pixels: {str(e)}"]


#==================== PARSE MPL CSV DATA ====================

def parse_mpl_csv_data(content: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Parse MPL CSV file (attenuation or phaseshift).
    Similar to spectrum parsing but extracts frequencies instead of wavelengths.
    """
    try:
        lines = content.strip().split('\n')
        metadata = {}
        data_start_idx = 0
        frequencies = []
        header_row_idx = None
        
        # Find header row (contains 'x position')
        for idx, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            
            # Check if ANY column contains 'x position'
            has_position = any('x position' in p.lower() for p in parts)
            
            if has_position:
                header_row_idx = idx
                
                # Extract frequencies from header (skip position, tau, R2, date/time columns)
                for i, part in enumerate(parts):
                    # Skip non-frequency columns
                    if ('position' in part.lower() or 'date' in part.lower() or 
                        'time' in part.lower() or 'tau' in part.lower() or 
                        'r2' in part.lower()):
                        continue
                    
                    try:
                        freq = float(part)
                        frequencies.append(freq)
                    except ValueError:
                        pass
                
                if frequencies:
                    metadata['frequencies'] = frequencies
                    metadata['frequency_min_hz'] = min(frequencies)
                    metadata['frequency_max_hz'] = max(frequencies)
                    metadata['frequency_points'] = len(frequencies)
                else:
                    st.warning("Could not find frequencies in header")
                
                # Data starts on the next row after header
                data_start_idx = idx
                break
        
        if header_row_idx is None:
            st.error("Could not find header row with position columns")
            return None, None
        
        # Extract metadata from rows before header
        for idx in range(header_row_idx):
            line = lines[idx]
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                
                if key and value and value.lower() != 'nan':
                    metadata[key] = value
        
        # Parse data section starting from header row
        if data_start_idx < len(lines):
            data_lines = '\n'.join(lines[data_start_idx:])
            df = pd.read_csv(StringIO(data_lines), sep=',')
            df.columns = df.columns.str.strip()
            return df, metadata
        
        return None, metadata
    
    except Exception as e:
        st.error(f"Error parsing MPL CSV: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None
    



# ========================================
# XRF Measurement
# ========================================

# ==================== XRF CSV PARSING ====================

def parse_xrf_csv_data(content: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    Parse XRF CSV file.
    Extracts metadata, energy values from header and identifies material composition columns.
    """
    try:
        lines = content.strip().split('\n')
        metadata = {}
        data_start_idx = 0
        energy_values = []
        header_row_idx = None
        material_columns = []
        
        # Detect separator (tab or comma)
        first_line = lines[0] if lines else ""
        separator = '\t' if '\t' in first_line else ','
        
        # Find header row (contains 'x position')
        for idx, line in enumerate(lines):
            parts = [p.strip() for p in line.split(separator)]
            
            # Check if ANY column contains 'x position'
            has_position = any('x position' in p.lower() for p in parts)
            
            if has_position:
                header_row_idx = idx
                
                # Identify material composition columns and energy values
                for i, part in enumerate(parts):
                    # Skip empty columns
                    if not part or part.upper() == 'NAN':
                        continue
                    
                    # Skip position, spectrum, date, time, thickness columns
                    if any(kw in part.lower() for kw in ['position', 'spectrum', 'date', 'time', 'thickness']):
                        continue
                    
                    # Check if this is a material composition column (contains % or parentheses with %)
                    if '(%)' in part or part.endswith('%'):
                        # Extract material name from column like "Cs (%)" or "Pb (%)" or "Cs%"
                        material_name = part.replace('(%)', '').replace('(', '').replace(')', '').replace('%', '').strip()
                        material_columns.append({
                            'name': material_name,
                            'column': part
                        })
                    else:
                        # Try to parse as energy value (numeric)
                        try:
                            energy = float(part)
                            energy_values.append(energy)
                        except ValueError:
                            pass
                
                if energy_values:
                    metadata['energy_values'] = energy_values
                    metadata['energy_min_kev'] = min(energy_values)
                    metadata['energy_max_kev'] = max(energy_values)
                
                if material_columns:
                    metadata['material_columns'] = material_columns
                
                # Data starts on the next row after header
                data_start_idx = idx
                break
        
        if header_row_idx is None:
            st.error("Could not find header row with position columns")
            return None, None
        
        # Extract metadata from rows BEFORE header
        for idx in range(header_row_idx):
            line = lines[idx]
            parts = [p.strip() for p in line.split(separator)]
            
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                
                # Store metadata if both key and value are valid
                if key and value and value.lower() not in ['nan', '']:
                    metadata[key] = value
        
        # Parse data section starting from header row
        if data_start_idx < len(lines):
            data_lines = '\n'.join(lines[data_start_idx:])
            df = pd.read_csv(StringIO(data_lines), sep=separator)
            df.columns = df.columns.str.strip()
            return df, metadata
        
        return None, metadata
    
    except Exception as e:
        st.error(f"Error parsing XRF CSV: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


# ==================== XRF ACQUISITION CREATION ====================

def create_xrf_acquisition(db, sample_number: str, treatment_sequence: str,
                          treatment_method: str, material_layer: str,
                          metadata: Dict, acq_time: str, df: pd.DataFrame,
                          filename: str) -> Tuple[bool, str, Optional[str]]:
    """Create or update acquisition document for XRF measurement"""
    try:
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        
        # Single acquisition ID without layer suffix
        acq_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}_acq_xrf"
        
        # Extract energy range
        energy_values = metadata.get('energy_values', [])
        energy_min = float(energy_values[0]) if energy_values else 0
        energy_max = float(energy_values[-1]) if energy_values else 0
        measurement_count = len(df)
        
        # Extract material composition columns
        material_columns = metadata.get('material_columns', [])
        elements_detected = [mat['name'] for mat in material_columns]
        
        # Extract instrument and configuration from metadata
        instrument_model = metadata.get('Spectrometer', 'XRF')
        
        # Check if acquisition already exists
        existing_acq = db['acquisitions'].find_one({"_id": acq_id})
        
        if existing_acq:
            # Update existing acquisition by adding new layer
            material_layers = existing_acq.get('material_layers', [])
            
            # Check if this layer already exists
            layer_exists = False
            for i, layer in enumerate(material_layers):
                if layer.get('layer_name') == material_layer:
                    # Update existing layer
                    material_layers[i] = {
                        'layer_name': material_layer,
                        'elements_detected': elements_detected,
                        'filename': filename
                    }
                    layer_exists = True
                    break
            
            # Add new layer if it doesn't exist
            if not layer_exists:
                material_layers.append({
                    'layer_name': material_layer,
                    'elements_detected': elements_detected,
                    'filename': filename
                })
            
            # Update measurement count to maximum across all layers
            max_measurement_count = max(measurement_count, existing_acq.get('measurement_count', 0))
            
            db['acquisitions'].update_one(
                {"_id": acq_id},
                {"$set": {
                    "material_layers": material_layers,
                    "measurement_count": max_measurement_count,
                    "updated_date_time": current_time
                }}
            )
            
            return True, f"XRF Acquisition updated with {material_layer} data", acq_id
        
        else:
            # Create new acquisition document
            acq_doc = {
                "_id": acq_id,
                "sample_number": sample_number,
                "treatment_id": f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}",
                "treatment_method": treatment_method,
                "treatment_sequence": treatment_sequence,
                "technique": "xrf",
                "measurement_type": metadata.get('Measurement Type', 'Mapping XRF'),
                "instrument": {
                    "model": instrument_model,
                    "type": "x-ray_fluorescence_spectrometer"
                },
                "config": {
                    "real_time_ms": safe_int_extract(metadata.get('real_time_ms', 0)),
                    "live_time_ms": safe_int_extract(metadata.get('live_time_ms', 0)),
                    "dead_time_percent": safe_float_extract(metadata.get('dead_time_percent', 0)),
                    "voltage_kv": safe_float_extract(metadata.get('voltage_kV', 0)),
                    "current_micro_a": safe_float_extract(metadata.get('current_micro_A', 0)),
                    "calib_abs_kev": safe_float_extract(metadata.get('calib_abs_kev', 0)),
                    "calib_lin_kev_per_channel": safe_float_extract(metadata.get('calib_lin_kev_per_channel', 0)),
                    "material_layer_number": safe_int_extract(metadata.get('material layer number', 0))
                },
                "energy_range_kev": [energy_min, energy_max],
                "material_layers": [
                    {
                        'layer_name': material_layer,
                        'elements_detected': elements_detected,
                        'filename': filename
                    }
                ],
                "measurement_count": measurement_count,
                "measured_date(y/m/d)_time": acq_time,
                "created_date_time": current_time,
                "updated_date_time": current_time
            }
            
            db['acquisitions'].insert_one(acq_doc)
            
            return True, f"XRF Acquisition created with {material_layer} data", acq_id
    
    except Exception as e:
        return False, f"Error creating XRF acquisition: {str(e)}", None


# ==================== XRF SPECTRA CREATION ====================

def create_xrf_spectra(db, acq_id: str, sample_number: str, treatment_sequence: str,
                      treatment_method: str, material_layer: str, df: pd.DataFrame,
                      metadata: Dict, acq_time: str) -> Tuple[int, List[str]]:
    """Create or update XRF spectrum documents for each position with SPX intensity vs energy"""
    try:
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        treatment_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}"
        
        energy_values = [float(e) for e in metadata.get('energy_values', []) if e is not None]
        
        if not energy_values:
            return 0, ["No energy values found in metadata"]
        
        spectra_count = 0
        errors = []
        
        # Identify columns to skip
        skip_columns = ['x position (mm)', 'y position (mm)', 'spectrum', 'date', 'time', 'thickness (nm)']
        material_columns = metadata.get('material_columns', [])
        for mat in material_columns:
            skip_columns.append(mat['column'])
        
        for idx, row in df.iterrows():
            try:
                x_pos = extract_position(row, df, idx, 'x')
                y_pos = extract_position(row, df, idx, 'y')
                
                # Parse measurement time
                measurement_time = acq_time
                if 'date' in df.columns and 'time' in df.columns:
                    date_str = str(row['date']).strip()
                    time_str = str(row['time']).strip()
                    measurement_time = parse_datetime_from_strings(date_str, time_str, acq_time)
                
                # Extract SPX intensities
                spx_intensities = []
                for col in df.columns:
                    if col not in skip_columns:
                        try:
                            val = row[col]
                            spx_intensities.append(float(val) if val and str(val).upper() != 'N/A' else 0)
                        except (ValueError, TypeError):
                            spx_intensities.append(0)
                
                if not spx_intensities:
                    errors.append(f"Row {idx}: No SPX intensity values found")
                    continue
                
                spx_intensities = spx_intensities[:len(energy_values)]
                
                # Unified spectrum ID without layer suffix
                spectrum_id = f"{acq_id}_xrf_{x_pos}_{y_pos}"
                
                # Check if spectrum already exists
                existing_spectrum = db['spectra'].find_one({"_id": spectrum_id})
                
                if existing_spectrum:
                    # Update existing spectrum with new layer data
                    db['spectra'].update_one(
                        {"_id": spectrum_id},
                        {"$set": {
                            f"layers.{material_layer}": {
                                "quantity": "spx_intensity",
                                "values": spx_intensities
                            },
                            "updated_date_time": current_time
                        }}
                    )
                else:
                    # Create new spectrum document
                    spectrum_doc = {
                        "_id": spectrum_id,
                        "acq_id": acq_id,
                        "treatment_id": treatment_id,
                        "technique": "xrf",
                        "measurement_type": "Mapping XRF",
                        "position": {"x": x_pos, "y": y_pos, "unit": "mm"},
                        "x_axis": {"quantity": "energy_kev", "values": energy_values},
                        "layers": {
                            material_layer: {
                                "quantity": "spx_intensity",
                                "values": spx_intensities
                            }
                        },
                        "measurement_date_time": measurement_time,
                        "measured_date(y/m/d)_time": acq_time,
                        "created_date_time": current_time,
                        "updated_date_time": current_time
                    }
                    
                    db['spectra'].insert_one(spectrum_doc)
                
                spectra_count += 1
            
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                continue
        
        return spectra_count, errors
    
    except Exception as e:
        return 0, [f"Error creating XRF spectra: {str(e)}"]


# ==================== XRF PIXELS CREATION ====================

def create_xrf_pixels(db, acq_id: str, sample_number: str, treatment_sequence: str,
                     treatment_method: str, material_layer: str, df: pd.DataFrame,
                     metadata: Dict, acq_time: str) -> Tuple[int, List[str]]:
    """Create or update XRF pixel documents with thickness and material composition percentages"""
    try:
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        treatment_method_normalized = normalize_treatment_method(treatment_method)
        treatment_id = f"{sample_number}_treat{treatment_sequence}_{treatment_method_normalized}"
        
        pixels_count = 0
        errors = []
        
        # Get material columns from metadata
        material_columns = metadata.get('material_columns', [])
        
        if not material_columns:
            return 0, ["No material composition columns found in metadata"]
        
        for idx, row in df.iterrows():
            try:
                x_pos = extract_position(row, df, idx, 'x')
                y_pos = extract_position(row, df, idx, 'y')
                
                # Parse measurement time
                measurement_time = acq_time
                if 'date' in df.columns and 'time' in df.columns:
                    date_str = str(row['date']).strip()
                    time_str = str(row['time']).strip()
                    measurement_time = parse_datetime_from_strings(date_str, time_str, acq_time)
                
                # Extract thickness
                thickness_nm = safe_float_extract(row.get('thickness (nm)', 0))
                
                # Extract material composition
                composition = {}
                for mat_info in material_columns:
                    mat_name = mat_info['name']
                    mat_column = mat_info['column']
                    
                    if mat_column in df.columns:
                        mat_value = safe_float_extract(row.get(mat_column, 0))
                        composition[mat_name.lower()] = mat_value
                
                if not composition:
                    errors.append(f"Row {idx}: No material composition values found")
                    continue
                
                # Unified pixel ID without layer suffix
                pixel_id = f"{acq_id}_xrf_{x_pos}_{y_pos}"
                
                # Check if pixel already exists
                existing_pixel = db['pixels'].find_one({"_id": pixel_id})
                
                if existing_pixel:
                    # Update existing pixel with new layer data
                    db['pixels'].update_one(
                        {"_id": pixel_id},
                        {"$set": {
                            f"layers.{material_layer}.thickness_nm": thickness_nm,
                            f"layers.{material_layer}.composition_percent": composition,
                            "updated_date_time": current_time
                        }}
                    )
                else:
                    # Create new pixel document
                    pixel_doc = {
                        "_id": pixel_id,
                        "acq_id": acq_id,
                        "treatment_id": treatment_id,
                        "technique": "xrf",
                        "measurement_type": "Mapping XRF",
                        "position": {"x": x_pos, "y": y_pos, "unit": "mm"},
                        "layers": {
                            material_layer: {
                                "thickness_nm": thickness_nm,
                                "composition_percent": composition
                            }
                        },
                        "measurement_date_time": measurement_time,
                        "measured_date(y/m/d)_time": acq_time,
                        "created_date_time": current_time,
                        "updated_date_time": current_time
                    }
                    
                    db['pixels'].insert_one(pixel_doc)
                
                pixels_count += 1
            
            except Exception as e:
                errors.append(f"Row {idx}: {str(e)}")
                continue
        
        return pixels_count, errors
    
    except Exception as e:
        return 0, [f"Error creating XRF pixels: {str(e)}"]


# ==================== STREAMLIT UI ====================

def page_upload_measurement_data(db):
    """Streamlit page for uploading measurement mapping files (PL, UV-Vis, Resistance, MPL, XRF)"""
    st.markdown("<h5>Measurement Files</h5>", unsafe_allow_html=True)
    
    if 'processed_measurement_files' not in st.session_state:
        st.session_state.processed_measurement_files = set()
    
    uploaded_files = st.file_uploader(
        "Upload Measurement mapping CSV files:",
        type=['csv'],
        accept_multiple_files=True,
        key="measurement_uploader"
    )
    
    if not uploaded_files:
        return
    
    # Filter valid measurement files
    measurement_keywords = ['mapping_pl', 'transmittance', 'reflectance', 'absorptance', 
                           'mapping_resistance', 'mapping_mpl', 'mapping_xrf']
    valid_measurement_files = [f for f in uploaded_files if any(kw in f.name.lower() for kw in measurement_keywords)]
    invalid_files = [f for f in uploaded_files if f not in valid_measurement_files]
    
    if invalid_files:
        st.warning(f"⚠️ {len(invalid_files)} file(s) skipped (not measurement files)")
    
    if not valid_measurement_files:
        st.error("❌ No valid measurement files found")
        return
    
    all_files_data = []
    validation_errors = []
    validation_warnings = []
    
    # Validate all files
    for uploaded_file in valid_measurement_files:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.file_id}"
        
        if file_id in st.session_state.processed_measurement_files:
            st.success(f"✓ Already processed: {uploaded_file.name}")
            continue
        
        try:
            # Validate filename format and check database
            is_valid, error_msg, warning_msg = validate_measurement_csv(uploaded_file.name, db)
            
            if not is_valid:
                validation_errors.append((uploaded_file.name, error_msg))
                continue
            
            filename_meta = parse_measurement_filename(uploaded_file.name)
            if not filename_meta:
                validation_errors.append((uploaded_file.name, "Failed to parse filename"))
                continue
            
            content = uploaded_file.read().decode('utf-8')
            
            if filename_meta['technique'] == 'resistance':
                df, csv_metadata = parse_resistance_csv_data(content)
            elif filename_meta['technique'] == 'mpl':
                df, csv_metadata = parse_mpl_csv_data(content)
            elif filename_meta['technique'] == 'xrf':
                df, csv_metadata = parse_xrf_csv_data(content)
            else:
                df, csv_metadata = parse_spectrum_csv_data(content)
            
            if df is None or df.empty:
                validation_errors.append((uploaded_file.name, "Failed to parse CSV data"))
                continue
            
            all_files_data.append({
                'file_id': file_id,
                'filename': uploaded_file.name,
                'filename_meta': filename_meta,
                'csv_metadata': csv_metadata,
                'df': df,
                'warning': warning_msg
            })
            
            if warning_msg:
                validation_warnings.append((uploaded_file.name, warning_msg))
        
        except Exception as e:
            validation_errors.append((uploaded_file.name, str(e)))
    
    # Display errors
    if validation_errors:
        st.error("❌ Validation Errors")
        for filename, error in validation_errors:
            st.write(f"**{filename}**: {error}")
    
    # Display warnings
    if validation_warnings:
        st.warning("⚠️ Validation Warnings")
        for filename, warning in validation_warnings:
            st.write(f"**{filename}**: {warning}")
    
    # Show summary and upload button
    if all_files_data:
        st.subheader(f"✓ Ready to upload: {len(all_files_data)} file(s)")
        
        for file_info in all_files_data:
            meta = file_info['filename_meta']
            
            technique_labels = {
                'resistance': 'Resistance',
                'pl': 'PL',
                'uv_vis': f"UV-Vis ({meta['measurement_type'].capitalize()})",
                'mpl': f"MPL ({meta['measurement_type'].capitalize()})",
                'xrf': f"XRF ({meta.get('material_layer', 'unknown')})"
            }
            
            technique_label = technique_labels.get(meta['technique'], 'Unknown')
            
            st.write(f"**{file_info['filename']}**")
            st.write(f"Sample: {meta['sample_number']} | Type: {technique_label} | Treatment: {meta['treatment_method']} Seq: {meta['treatment_sequence']}")
            st.write(f"Data points: {len(file_info['df'])}")
        
        if st.button("Upload All Measurement Files", type="primary", use_container_width=True):
            successful_uploads = 0
            failed_uploads = 0
            
            for file_info in all_files_data:
                try:
                    meta = file_info['filename_meta']
                    sample_number = meta['sample_number']
                    treatment_seq = meta['treatment_sequence']
                    treatment_method = meta['treatment_method']
                    technique = meta['technique']
                    measurement_type = meta['measurement_type']
                    acq_time = meta['acq_time_iso']
                    
                    if technique == 'resistance':
                        success, msg, acq_id = create_resistance_acquisition(
                            db, sample_number, treatment_seq, treatment_method,
                            file_info['csv_metadata'], acq_time, file_info['df'],
                            file_info['filename']
                        )
                        
                        if not success:
                            st.error(f"✗ {file_info['filename']}: {msg}")
                            failed_uploads += 1
                            continue
                        
                        pixels_count, errors = create_resistance_pixels(
                            db, acq_id, sample_number, treatment_seq, treatment_method,
                            file_info['df'], acq_time
                        )
                        
                        if errors:
                            st.warning(f"⚠️ {file_info['filename']}: {len(errors)} points had issues")
                            for error in errors[:5]:
                                st.write(f"  - {error}")
                        
                        st.session_state.processed_measurement_files.add(file_info['file_id'])
                        st.success(f"✓ {file_info['filename']}: {msg} + {pixels_count} pixels")
                        successful_uploads += 1
                    
                    elif technique == 'mpl':
                        success, msg, acq_id = create_mpl_acquisition(
                            db, sample_number, treatment_seq, treatment_method,
                            measurement_type, file_info['csv_metadata'], acq_time,
                            file_info['df'], file_info['filename']
                        )
                        
                        if not success:
                            st.error(f"✗ {file_info['filename']}: {msg}")
                            failed_uploads += 1
                            continue
                        
                        spectra_count, spec_errors = create_mpl_spectra(
                            db, acq_id, sample_number, treatment_seq, treatment_method,
                            measurement_type, file_info['df'], file_info['csv_metadata'], acq_time
                        )
                        
                        pixels_count, pix_errors = create_mpl_pixels(
                            db, acq_id, sample_number, treatment_seq, treatment_method,
                            measurement_type, file_info['df'], acq_time
                        )
                        
                        if spec_errors:
                            st.warning(f"⚠️ {file_info['filename']}: {len(spec_errors)} spectra issues")
                            for error in spec_errors[:5]:
                                st.write(f"  - {error}")
                        
                        if pix_errors:
                            st.warning(f"⚠️ {file_info['filename']}: {len(pix_errors)} pixel issues")
                            for error in pix_errors[:5]:
                                st.write(f"  - {error}")
                        
                        st.session_state.processed_measurement_files.add(file_info['file_id'])
                        st.success(f"✓ {file_info['filename']}: {msg} + {spectra_count} spectra + {pixels_count} pixels")
                        successful_uploads += 1
                    
                    elif technique == 'xrf':
                        material_layer = meta.get('material_layer', 'unknown')
                        
                        success, msg, acq_id = create_xrf_acquisition(
                            db, sample_number, treatment_seq, treatment_method, material_layer,
                            file_info['csv_metadata'], acq_time, file_info['df'],
                            file_info['filename']
                        )
                        
                        if not success:
                            st.error(f"✗ {file_info['filename']}: {msg}")
                            failed_uploads += 1
                            continue
                        
                        spectra_count, spec_errors = create_xrf_spectra(
                            db, acq_id, sample_number, treatment_seq, treatment_method,
                            material_layer, file_info['df'], file_info['csv_metadata'], acq_time
                        )
                        
                        pixels_count, pix_errors = create_xrf_pixels(
                            db, acq_id, sample_number, treatment_seq, treatment_method,
                            material_layer, file_info['df'], file_info['csv_metadata'], acq_time
                        )
                        
                        if spec_errors:
                            st.warning(f"⚠️ {file_info['filename']}: {len(spec_errors)} spectra issues")
                            for error in spec_errors[:5]:
                                st.write(f"  - {error}")
                        
                        if pix_errors:
                            st.warning(f"⚠️ {file_info['filename']}: {len(pix_errors)} pixel issues")
                            for error in pix_errors[:5]:
                                st.write(f"  - {error}")
                        
                        st.session_state.processed_measurement_files.add(file_info['file_id'])
                        st.success(f"✓ {file_info['filename']}: {msg} + {spectra_count} spectra + {pixels_count} pixels")
                        successful_uploads += 1
                    
                    else:  # PL or UV-Vis
                        success, msg, acq_id = create_spectrum_acquisition(
                            db, sample_number, treatment_seq, treatment_method, technique,
                            measurement_type, file_info['csv_metadata'], acq_time, 
                            file_info['df'], file_info['filename']
                        )
                        
                        if not success:
                            st.error(f"✗ {file_info['filename']}: {msg}")
                            failed_uploads += 1
                            continue
                        
                        spectra_count, errors = create_spectrum_spectra(
                            db, acq_id, sample_number, treatment_seq, treatment_method,
                            technique, measurement_type, file_info['df'], 
                            file_info['csv_metadata'], acq_time
                        )
                        
                        if errors:
                            st.warning(f"⚠️ {file_info['filename']}: {len(errors)} points had issues")
                            for error in errors[:5]:
                                st.write(f"  - {error}")
                        
                        st.session_state.processed_measurement_files.add(file_info['file_id'])
                        
                        technique_names = {'pl': 'PL', 'uv_vis': 'UV-Vis'}
                        tech_name = technique_names.get(technique, technique)
                        
                        st.success(f"✓ {file_info['filename']}: {msg} + {spectra_count} spectra")
                        successful_uploads += 1
                
                except Exception as e:
                    st.error(f"✗ {file_info['filename']}: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    failed_uploads += 1
            
            st.write(f"**Total: {len(all_files_data)} | Successful: {successful_uploads} | Failed: {failed_uploads}**")