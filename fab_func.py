import streamlit as st
from datetime import datetime, timezone
import re
import pandas as pd
import numpy as np
import io
from difflib import SequenceMatcher


def normalize_string(s):
    """
    Normalize string for flexible comparison:
    - Convert to lowercase
    - Remove extra spaces
    - Remove special characters except letters, numbers
    """
    s = str(s).lower().strip()
    # Remove extra whitespace
    s = ' '.join(s.split())
    # Remove underscores and hyphens, replace with spaces for word separation
    s = s.replace('_', ' ').replace('-', ' ')
    # Remove extra spaces again after replacements
    s = ' '.join(s.split())
    return s


def flexible_string_match(filename_value, csv_value, match_type='flexible'):
    """
    Compare strings with flexibility for spacing and capitalization.
    
    Args:
        filename_value: Value extracted from filename
        csv_value: Value from CSV content
        match_type: 'strict' for exact match, 'flexible' for fuzzy matching
    
    Returns:
        tuple: (is_match: bool, match_quality: float 0-1)
    """
    if match_type == 'strict':
        return filename_value == csv_value, 1.0 if filename_value == csv_value else 0.0
    
    # Flexible matching
    norm_filename = normalize_string(filename_value)
    norm_csv = normalize_string(csv_value)
    
    # Exact match after normalization
    if norm_filename == norm_csv:
        return True, 1.0
    
    # Check if one contains the other
    if norm_filename in norm_csv or norm_csv in norm_filename:
        return True, 0.95
    
    # Use sequence matcher for fuzzy matching
    similarity = SequenceMatcher(None, norm_filename, norm_csv).ratio()
    
    # Accept if similarity is above 85%
    if similarity >= 0.85:
        return True, similarity
    
    return False, similarity


def sort_fab_ids_by_sequence(db, fab_ids):
    """Sort fab_ids by their sequence number"""
    fab_docs = []
    
    for fab_id in fab_ids:
        fab_doc = db['fabrications'].find_one({"_id": fab_id})
        if fab_doc:
            fab_docs.append({
                "_id": fab_id,
                "sequence": int(fab_doc.get('fab_sequence', 0))
            })
    
    fab_docs_sorted = sorted(fab_docs, key=lambda x: x['sequence'])
    
    return [doc['_id'] for doc in fab_docs_sorted]


def load_pvd_dataframe(file_content):
    """Load PVD-P data into pandas DataFrame"""
    lines = file_content.split('\n')
    data_start_idx = 0
    
    for i, line in enumerate(lines):
        if 'Time\tProcess Time in seconds' in line:
            data_start_idx = i
            break
    
    if data_start_idx == 0:
        return None
    
    cleaned_lines = []
    for i in range(data_start_idx, len(lines)):
        line = lines[i].strip()
        line = line.strip('"')
        line = line.rstrip(',')
        line = line.rstrip('"')
        
        if line and not line.startswith('#'):
            cleaned_lines.append(line)
    
    if not cleaned_lines:
        return None
    
    data_content = '\n'.join(cleaned_lines)
    try:
        df = pd.read_csv(io.StringIO(data_content), sep='\t')
        return df
    except:
        return None


def select_best_qcm(df):
    """Select the QCM sensor with the largest thickness range"""
    if df is None:
        return None, None
    
    qcm_thickness_cols = [col for col in df.columns if 'C_THIK' in col or 'THIK' in col]
    
    if not qcm_thickness_cols:
        return None, None
    
    max_range = {}
    for col in qcm_thickness_cols:
        try:
            values = df[col].dropna()
            if len(values) > 0:
                col_range = values.max() - values.min()
                if col_range > 0:
                    max_range[col] = col_range
        except:
            continue
    
    if not max_range:
        return None, None
    
    best_qcm = max(max_range, key=max_range.get)
    return best_qcm, max_range[best_qcm]


def find_deposition_period(df, qcm_column):
    """Find start and end indices of deposition period"""
    if df is None or qcm_column is None:
        return None, None
    
    thickness = df[qcm_column].fillna(0)
    
    start_idx = None
    for i, val in enumerate(thickness):
        if val > 0:
            start_idx = i
            break
    
    if start_idx is None:
        return None, None
    
    max_thickness = thickness.max()
    stable_threshold = max_thickness * 0.95
    
    end_idx = None
    for i in range(start_idx, len(thickness)):
        if thickness.iloc[i] >= stable_threshold:
            if i + 3 < len(thickness):
                next_values = thickness.iloc[i:i+3]
                variation = next_values.std()
                if variation < max_thickness * 0.01:
                    end_idx = i
                    break
    
    if end_idx is None:
        end_idx = len(df) - 1
    
    return start_idx, end_idx


def calculate_deposition_statistics(df, start_idx, end_idx):
    """Calculate average parameters during deposition period"""
    if df is None or start_idx is None or end_idx is None:
        return {}
    
    deposition_data = df.iloc[start_idx:end_idx+1]
    stats = {}
    
    if 'Time' in df.columns:
        try:
            start_time = pd.to_datetime(deposition_data['Time'].iloc[0], format='%H:%M:%S')
            end_time = pd.to_datetime(deposition_data['Time'].iloc[-1], format='%H:%M:%S')
            duration_seconds = (end_time - start_time).total_seconds()
            stats['duration_minutes'] = round(duration_seconds / 60, 2)
        except:
            if 'Process Time in seconds' in df.columns:
                try:
                    duration_seconds = (deposition_data['Process Time in seconds'].iloc[-1] - 
                                      deposition_data['Process Time in seconds'].iloc[0])
                    stats['duration_minutes'] = round(duration_seconds / 60, 2)
                except:
                    stats['duration_minutes'] = None
            else:
                stats['duration_minutes'] = None
    
    materials = ['PbI2', 'CsBr', 'CsI', 'SnI2']
    for i, material in enumerate(materials, 1):
        prefix = f'{i} - {material}'
        
        if f'{prefix} Aout' in deposition_data.columns:
            try:
                stats[f'{material}_Aout_avg'] = round(deposition_data[f'{prefix} Aout'].mean(), 2)
            except:
                stats[f'{material}_Aout_avg'] = None
        else:
            stats[f'{material}_Aout_avg'] = None
        
        if f'{prefix} PV' in deposition_data.columns:
            try:
                stats[f'{material}_PV_avg'] = round(deposition_data[f'{prefix} PV'].mean(), 2)
            except:
                stats[f'{material}_PV_avg'] = None
        else:
            stats[f'{material}_PV_avg'] = None
        
        if f'{prefix} T' in deposition_data.columns:
            try:
                stats[f'{material}_T_avg'] = round(deposition_data[f'{prefix} T'].mean(), 2)
            except:
                stats[f'{material}_T_avg'] = None
        else:
            stats[f'{material}_T_avg'] = None
    
    if 'Vacuum Pressure2' in df.columns:
        try:
            vacuum_avg = deposition_data['Vacuum Pressure2'].mean()
            stats['vacuum_pressure2_avg'] = float(f'{vacuum_avg:.2e}')
        except:
            stats['vacuum_pressure2_avg'] = None
    else:
        stats['vacuum_pressure2_avg'] = None
    
    return stats


def parse_pvdp_csv_advanced(content, filename):
    """Parse PVD-P CSV file with header format and extract statistics from data"""
    try:
        lines = content.split('\n')
        fab_data = {}
        
        for line in lines:
            line_clean = line.strip()
            
            if '# Substrate Number:' in line_clean:
                value = line_clean.split('# Substrate Number:')[1].strip()
                value = value.replace(',', '').replace('\t', '').strip()
                fab_data['sample_number'] = value
                
            elif '# process ID:' in line_clean:
                value = line_clean.split('# process ID:')[1].strip()
                value = value.replace(',', '').replace('\t', '').strip()
                fab_data['fab_sequence'] = value
                
            elif '# operator:' in line_clean:
                value = line_clean.split('# operator:')[1].strip()
                value = value.replace(',', '').replace('\t', '').strip()
                fab_data['fab_operator'] = value
                
            elif '# Date:' in line_clean:
                value = line_clean.split('# Date:')[1].strip()
                value = value.replace(',', '').replace('\t', '').strip()
                fab_data['fab_date'] = value
                
            elif '# Time:' in line_clean:
                value = line_clean.split('# Time:')[1].strip()
                value = value.replace(',', '').replace('\t', '').strip()
                fab_data['fab_time'] = value
                
            elif 'Controller settings:' in line_clean and 'PVD-P' in line_clean:
                fab_data['fab_method'] = 'PVD-P'
        
        fab_data['fab_institution'] = 'HZB'
        
        df = load_pvd_dataframe(content)
        if df is not None:
            best_qcm, final_thickness = select_best_qcm(df)
            
            if best_qcm is not None:
                start_idx, end_idx = find_deposition_period(df, best_qcm)
                
                if start_idx is not None and end_idx is not None:
                    fab_stats = calculate_deposition_statistics(df, start_idx, end_idx)
                    fab_data.update(fab_stats)
        
        return fab_data if fab_data else None
        
    except Exception as e:
        return None


def parse_fab_csv_simple(content):
    """Parse simple key-value fabrication CSV file"""
    try:
        lines = content.strip().split('\n')
        fab_data = {}
        
        for line in lines:
            parts = line.split(',', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().strip('"').rstrip(',').strip()
                fab_data[key] = value
        
        return fab_data if fab_data else None
    except Exception as e:
        return None


def parse_fab_csv(content, filename):
    """Parse fabrication CSV file - detects format and parses accordingly"""
    
    filename_lower = filename.lower()
    
    if 'pvdp' in filename_lower or 'pvd-p' in filename_lower:
        return parse_pvdp_csv_advanced(content, filename)
    else:
        return parse_fab_csv_simple(content)


def validate_fab_csv(uploaded_file, fab_data, db):
    """
    Validate fabrication CSV file with flexible Institution and Operator matching.
    
    Requirements:
    - Sample Number: STRICT exact match
    - Institution: FLEXIBLE (case-insensitive, spacing variations allowed)
    - Operator: FLEXIBLE (case-insensitive, spacing variations allowed)
    - Sequence: Must be present in filename as fab{number}
    - Method: Validated against allowed methods
    """
    
    filename = uploaded_file.name
    
    if '_fab' not in filename.lower():
        return False, None, None, None, "Invalid filename: Must contain _fab for fabrication files"
    
    # Updated regex to handle flexible spacing and case
    fab_match = re.search(r'_fab(\d+)_([A-Za-z]+)_(\d{8})_(\d{6})', filename)
    
    if not fab_match:
        return False, None, None, None, "Invalid filename: Must contain _fab#_METHOD_YYYYMMDD_HHMMSS"
    
    extracted_sequence = fab_match.group(1)
    filename_method = fab_match.group(2)
    
    filename_no_ext = filename.replace('.csv', '')
    parts_before_fab = filename_no_ext.split('_fab')[0]
    
    # Split by underscore to extract components
    filename_parts = parts_before_fab.split('_')
    
    if len(filename_parts) < 3:
        return False, None, None, None, "Invalid filename format: Need SampleNumber_Institution_Operator"
    
    filename_sample_number = filename_parts[0]
    # Everything after sample number and before _fab is institution_operator
    # Institution is parts[1], Operator is parts[2:]
    filename_institution = filename_parts[1]
    filename_operator = '_'.join(filename_parts[2:])
    
    # Extract values from CSV
    csv_sample_number = fab_data.get('sample_number', '').strip()
    csv_institution = fab_data.get('fab_institution', '').strip()
    csv_operator = fab_data.get('fab_operator', '').strip()
    csv_method = fab_data.get('fab_method', '').strip()
    
    # STRICT validation: Sample Number must match exactly
    if filename_sample_number != csv_sample_number:
        return False, None, None, None, f"Sample Mismatch: Filename '{filename_sample_number}' vs CSV '{csv_sample_number}' (must match exactly)"
    
    # FLEXIBLE validation: Institution
    institution_match, institution_quality = flexible_string_match(
        filename_institution, 
        csv_institution, 
        match_type='flexible'
    )
    
    if not institution_match:
        return False, None, None, None, f"Institution Mismatch: Filename '{filename_institution}' vs CSV '{csv_institution}' (similarity: {institution_quality:.1%})"
    
    # FLEXIBLE validation: Operator
    operator_match, operator_quality = flexible_string_match(
        filename_operator,
        csv_operator,
        match_type='flexible'
    )
    
    if not operator_match:
        return False, None, None, None, f"Operator Mismatch: Filename '{filename_operator}' vs CSV '{csv_operator}' (similarity: {operator_quality:.1%})"
    
    # Validate method
    method_mapping = {
        'pvdj': ['pvd-j', 'pvdj', 'pvd'],
        'pvd': ['pvd-j', 'pvdj', 'pvd'],
        'pvdp': ['pvdp', 'pvd-p'],
        'pld': ['pld'],
        'rtp': ['rtp'],
        'tubefurnace': ['tube furnace', 'tubefurnace'],
        'sputtering': ['sputtering', 'spt'],
        'spt': ['sputtering', 'spt']
    }
    
    filename_method_lower = filename_method.lower()
    csv_method_lower = csv_method.lower()
    
    if filename_method_lower not in method_mapping:
        return False, None, None, None, f"Unknown method: {filename_method}"
    
    if csv_method_lower not in method_mapping[filename_method_lower]:
        return False, None, None, None, f"Method Mismatch: Filename '{filename_method}' vs CSV '{csv_method}'"
    
    # Check for existing fabrication
    existing_fab = db['fabrications'].find_one({
        "sample_number": csv_sample_number,
        "fab_sequence": extracted_sequence
    })
    
    warning_msg = None
    if existing_fab:
        warning_msg = f"Sequence {extracted_sequence} already exists (will be updated)"
    
    return True, csv_sample_number, csv_method, extracted_sequence, warning_msg




# ==================== PVD-J CREATION FOR FABRICATION COLLECTION ====================

def create_pvdj_fabrication(db, sample_number, fab_data, extracted_sequence):
    """Create or update PVDJ fabrication document"""
    try:
        if not fab_data:
            return False, "No fabrication data provided"
        
        fab_data_clean = {}
        for key, value in fab_data.items():
            if isinstance(value, str):
                fab_data_clean[key] = value.rstrip(',').strip().strip('"')
            else:
                fab_data_clean[key] = value
        
        sample_number = sample_number.rstrip(',').strip()
        
        if fab_data_clean.get('sample_number', '') != sample_number:
            return False, "Sample number mismatch"
        
        fab_date_str = fab_data_clean.get('fab_date', '')
        fab_time_str = fab_data_clean.get('fab_time', '')
        
        try:
            datetime_str = f"{fab_date_str} {fab_time_str}"
            dt = datetime.strptime(datetime_str, "%A, %B %d, %Y %I:%M:%S %p")
            dt_utc = dt.replace(tzinfo=timezone.utc)
            fab_date_time_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            now_utc = datetime.now(timezone.utc)
            fab_date_time_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H.%M.%SZ")
        
        fab_sequence = extracted_sequence
        
        fab_deposition_params = {
            "fab_rate_nmol_per_cm2_per_sec": fab_data_clean.get('fab_rate_nmol_per_cm2_per_sec', ''),
            "fab_power_W": fab_data_clean.get('fab_power_W', ''),
            "fab_tooling_factor": fab_data_clean.get('fab_tooling_factor', ''),
            "fab_xtal": fab_data_clean.get('fab_xtal', '')
        }
        
        try:
            mass_before = float(fab_data_clean.get('fab_sample_mass_before_mg', 0) or 0)
            mass_after = float(fab_data_clean.get('fab_sample_mass_after_mg', 0) or 0)
            mass_change = mass_after - mass_before
        except:
            mass_before = 0.0
            mass_after = 0.0
            mass_change = 0.0
        
        fab_doc = {
            "_id": f"{sample_number}_fab{extracted_sequence}_pvdj",
            "sample_number": sample_number,
            "sub_id": f"{sample_number}_sub",
            "fab_method": fab_data_clean.get('fab_method', ''),
            "fab_sequence": fab_sequence,
            "fab_operator": fab_data_clean.get('fab_operator', ''),
            "fab_institution": fab_data_clean.get('fab_institution', ''),
            "fab_process_number": fab_data_clean.get('fab_process_number', ''),
            "fab_recipe_name": fab_data_clean.get('fab_recipe_name', ''),
            "fab_box_type": fab_data_clean.get('fab_box_type', ''),
            "fab_duration_minutes": fab_data_clean.get('fab_duration_minutes', ''),
            "fab_substrate_temperature_celsius": fab_data_clean.get('fab_substrate_temperature_celsius', ''),
            "fab_cooling_temperature_celsius": fab_data_clean.get('fab_cooling_temperature_celsius', ''),
            "fab_holding_time_seconds": fab_data_clean.get('fab_holding_time_seconds', ''),
            "fab_deposition_params": fab_deposition_params,
            "fab_sample_orientation": fab_data_clean.get('fab_sample_orientation', ''),
            "fab_sample_mass_before_mg": mass_before,
            "fab_sample_mass_after_mg": mass_after,
            "fab_sample_mass_change_mg": mass_change,
            "fab_date(y/m/d)_time": fab_date_time_iso,
            "created_date(y/m/d)_time": current_time,
            "updated_date(y/m/d)_time": current_time
        }
        
        fab_id = fab_doc["_id"]
        existing_fab = db['fabrications'].find_one({"_id": fab_id})
        
        if existing_fab:
            fab_doc["created_date_time"] = existing_fab.get("created_date_time", current_time)
            db['fabrications'].replace_one({"_id": fab_id}, fab_doc)
        else:
            db['fabrications'].insert_one(fab_doc)
        
        lib_id = f"{sample_number}_lib"
        existing_library = db['libraries'].find_one({"_id": lib_id})
        
        if existing_library:
            fab_ids = existing_library.get('fab_ids', [])
            if fab_id not in fab_ids:
                fab_ids.append(fab_id)
            
            fab_ids_sorted = sort_fab_ids_by_sequence(db, fab_ids)
            
            db['libraries'].update_one(
                {"_id": lib_id},
                {"$set": {"fab_ids": fab_ids_sorted, "updated_date_time": current_time}}
            )
        else:
            library_doc = {
                "_id": lib_id,
                "sample_number": sample_number,
                "sub_id": "",
                "fab_ids": [fab_id],
                "treat_ids": [],
                "created_date_time": current_time,
                "updated_date_time": current_time
            }
            db['libraries'].insert_one(library_doc)
        
        return True, "PVDJ Fabrication created/updated successfully"
        
    except Exception as e:
        return False, f"Error with PVDJ fabrication: {str(e)}"


# ==================== SPUTTERING CREATION FOR FABRICATION COLLECTION ====================

def create_sputtering_fabrication(db, sample_number, fab_data, extracted_sequence):
    """Create or update Sputtering fabrication document"""
    try:
        if not fab_data:
            return False, "No fabrication data provided"
        
        fab_data_clean = {}
        for key, value in fab_data.items():
            if isinstance(value, str):
                fab_data_clean[key] = value.rstrip(',').strip().strip('"')
            else:
                fab_data_clean[key] = value
        
        sample_number = sample_number.rstrip(',').strip()
        
        if fab_data_clean.get('sample_number', '') != sample_number:
            return False, "Sample number mismatch"
        
        fab_date_str = fab_data_clean.get('fab_date', '')
        fab_time_str = fab_data_clean.get('fab_time', '')
        
        try:
            datetime_str = f"{fab_date_str} {fab_time_str}"
            dt = datetime.strptime(datetime_str, "%A, %B %d, %Y %I:%M:%S %p")
            dt_utc = dt.replace(tzinfo=timezone.utc)
            fab_date_time_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            now_utc = datetime.now(timezone.utc)
            fab_date_time_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        fab_sequence = extracted_sequence
        
        fab_sputtering_params = {
            "fab_power_W": fab_data_clean.get('fab_power_W', ''),
            "fab_current_A": fab_data_clean.get('fab_current_A', ''),
            "fab_voltage_V": fab_data_clean.get('fab_voltage_V', ''),
            "fab_gas_mix": fab_data_clean.get('fab_gas_mix', ''),
            "fab_process_pressure_mbar": fab_data_clean.get('fab_process_pressure_mbar', ''),
            "fab_pre_fab_pressure": fab_data_clean.get('fab_pre_fab_pressure', '')
        }
        
        fab_doc = {
            "_id": f"{sample_number}_fab{extracted_sequence}_sputtering",
            "sample_number": sample_number,
            "sub_id": f"{sample_number}_sub",
            "fab_method": fab_data_clean.get('fab_method', ''),
            "fab_sequence": fab_sequence,
            "fab_operator": fab_data_clean.get('fab_operator', ''),
            "fab_institution": fab_data_clean.get('fab_institution', ''),
            "fab_program": fab_data_clean.get('fab_program', ''),
            "fab_duration_minutes": fab_data_clean.get('fab_duration_minutes', ''),
            "fab_sputtering_params": fab_sputtering_params,
            "fab_note": fab_data_clean.get('fab_note', ''),
            "fab_date(y/m/d)_time": fab_date_time_iso,
            "created_date(y/m/d)_time": current_time,
            "updated_date(y/m/d)_time": current_time
        }
        
        fab_id = fab_doc["_id"]
        existing_fab = db['fabrications'].find_one({"_id": fab_id})
        
        if existing_fab:
            fab_doc["created_date_time"] = existing_fab.get("created_date_time", current_time)
            db['fabrications'].replace_one({"_id": fab_id}, fab_doc)
        else:
            db['fabrications'].insert_one(fab_doc)
        
        lib_id = f"{sample_number}_lib"
        existing_library = db['libraries'].find_one({"_id": lib_id})
        
        if existing_library:
            fab_ids = existing_library.get('fab_ids', [])
            if fab_id not in fab_ids:
                fab_ids.append(fab_id)
            
            fab_ids_sorted = sort_fab_ids_by_sequence(db, fab_ids)
            
            db['libraries'].update_one(
                {"_id": lib_id},
                {"$set": {"fab_ids": fab_ids_sorted, "updated_date_time": current_time}}
            )
        else:
            library_doc = {
                "_id": lib_id,
                "sample_number": sample_number,
                "sub_id": "",
                "fab_ids": [fab_id],
                "treat_ids": [],
                "created_date_time": current_time,
                "updated_date_time": current_time
            }
            db['libraries'].insert_one(library_doc)
        
        return True, "Sputtering Fabrication created/updated successfully"
        
    except Exception as e:
        return False, f"Error with Sputtering fabrication: {str(e)}"


# ==================== TUBE FURNACE CREATION FOR FABRICATION COLLECTION ====================

def create_tube_furnace_fabrication(db, sample_number, fab_data, extracted_sequence):
    """Create or update Tube Furnace fabrication document"""
    try:
        if not fab_data:
            return False, "No fabrication data provided"
        
        fab_data_clean = {}
        for key, value in fab_data.items():
            if isinstance(value, str):
                fab_data_clean[key] = value.rstrip(',').strip().strip('"')
            else:
                fab_data_clean[key] = value
        
        sample_number = sample_number.rstrip(',').strip()
        
        if fab_data_clean.get('sample_number', '') != sample_number:
            return False, "Sample number mismatch"
        
        fab_date_str = fab_data_clean.get('fab_date', '')
        fab_time_str = fab_data_clean.get('fab_time', '')
        
        try:
            datetime_str = f"{fab_date_str} {fab_time_str}"
            dt = datetime.strptime(datetime_str, "%A, %B %d, %Y %I:%M:%S %p")
            dt_utc = dt.replace(tzinfo=timezone.utc)
            fab_date_time_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            now_utc = datetime.now(timezone.utc)
            fab_date_time_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        fab_sequence = extracted_sequence
        
        try:
            weight_before = float(fab_data_clean.get('fab_sample_weight_before_mg', 0) or 0)
            weight_after = float(fab_data_clean.get('fab_sample_weight_after_mg', 0) or 0)
            weight_change = weight_after - weight_before
        except:
            weight_before = 0.0
            weight_after = 0.0
            weight_change = 0.0
        
        fab_tube_furnace_params = {
            "fab_temperature_celsius": fab_data_clean.get('fab_temperature_celsius', ''),
            "fab_rample_celsius_per_min": fab_data_clean.get('fab_rample_celsius_per_min', ''),
            "fab_amount_selenium_g": fab_data_clean.get('fab_amount_selenium_g', ''),
            "fab_amount_sulfur_g": fab_data_clean.get('fab_amount_sulfur_g', ''),
            "fab_pressure_mbar": fab_data_clean.get('fab_pressure_mbar', ''),
            "fab_humidity_percent": fab_data_clean.get('fab_humidity_percent', '')
        }
        
        fab_doc = {
            "_id": f"{sample_number}_fab{extracted_sequence}_tube_furnace",
            "sample_number": sample_number,
            "sub_id": f"{sample_number}_sub",
            "fab_method": fab_data_clean.get('fab_method', ''),
            "fab_sequence": fab_sequence,
            "fab_operator": fab_data_clean.get('fab_operator', ''),
            "fab_institution": fab_data_clean.get('fab_institution', ''),
            "fab_tube_furnace_params": fab_tube_furnace_params,
            "fab_duration_minutes": fab_data_clean.get('fab_duration_minutes', ''),
            "fab_cooling_time_minutes": fab_data_clean.get('fab_cooling_time_minutes', ''),
            "fab_storage_days": fab_data_clean.get('fab_storage_days', ''),
            "fab_sample_orientation_in_box": fab_data_clean.get('fab_sample_orientation_in_box', ''),
            "fab_position_in_oven": fab_data_clean.get('fab_position_in_oven', ''),
            "fab_sample_weight_before_mg": weight_before,
            "fab_sample_weight_after_mg": weight_after,
            "fab_sample_weight_change_mg": weight_change,
            "fab_date(y/m/d)_time": fab_date_time_iso,
            "created_date_time": current_time,
            "updated_date_time": current_time
        }
        
        fab_id = fab_doc["_id"]
        existing_fab = db['fabrications'].find_one({"_id": fab_id})
        
        if existing_fab:
            fab_doc["created_date_time"] = existing_fab.get("created_date_time", current_time)
            db['fabrications'].replace_one({"_id": fab_id}, fab_doc)
        else:
            db['fabrications'].insert_one(fab_doc)
        
        lib_id = f"{sample_number}_lib"
        existing_library = db['libraries'].find_one({"_id": lib_id})
        
        if existing_library:
            fab_ids = existing_library.get('fab_ids', [])
            if fab_id not in fab_ids:
                fab_ids.append(fab_id)
            
            fab_ids_sorted = sort_fab_ids_by_sequence(db, fab_ids)
            
            db['libraries'].update_one(
                {"_id": lib_id},
                {"$set": {"fab_ids": fab_ids_sorted, "updated_date_time": current_time}}
            )
        else:
            library_doc = {
                "_id": lib_id,
                "sample_number": sample_number,
                "sub_id": "",
                "fab_ids": [fab_id],
                "treat_ids": [],
                "created_date_time": current_time,
                "updated_date_time": current_time
            }
            db['libraries'].insert_one(library_doc)
        
        return True, "Tube Furnace Fabrication created/updated successfully"
        
    except Exception as e:
        return False, f"Error with Tube Furnace fabrication: {str(e)}"



# ==================== RAPID THERMAL PROCESSING (RTP) CREATION FOR FABRICATION COLLECTION ====================

def create_rtp_fabrication(db, sample_number, fab_data, extracted_sequence):
    """Create or update RTP fabrication document"""
    try:
        if not fab_data:
            return False, "No fabrication data provided"
        
        fab_data_clean = {}
        for key, value in fab_data.items():
            if isinstance(value, str):
                fab_data_clean[key] = value.rstrip(',').strip().strip('"')
            else:
                fab_data_clean[key] = value
        
        sample_number = sample_number.rstrip(',').strip()
        
        if fab_data_clean.get('sample_number', '') != sample_number:
            return False, "Sample number mismatch"
        
        fab_date_str = fab_data_clean.get('fab_date', '')
        fab_time_str = fab_data_clean.get('fab_time', '')
        
        try:
            datetime_str = f"{fab_date_str} {fab_time_str}"
            dt = datetime.strptime(datetime_str, "%A, %B %d, %Y %I:%M:%S %p")
            dt_utc = dt.replace(tzinfo=timezone.utc)
            fab_date_time_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            now_utc = datetime.now(timezone.utc)
            fab_date_time_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        fab_sequence = extracted_sequence
        
        try:
            weight_before = float(fab_data_clean.get('fab_sample_weight_before_mg', 0) or 0)
            weight_after = float(fab_data_clean.get('fab_sample_weight_after_mg', 0) or 0)
            weight_change = weight_after - weight_before
        except:
            weight_before = 0.0
            weight_after = 0.0
            weight_change = 0.0
        
        fab_rtp_params = {
            "fab_pressure_mbar": fab_data_clean.get('fab_pressure_mbar', ''),
            "fab_box_type": fab_data_clean.get('fab_box_type', ''),
            "fab_amount_selenium_g": fab_data_clean.get('fab_amount_selenium_g', ''),
            "fab_amount_sulfur_g": fab_data_clean.get('fab_amount_sulfur_g', ''),
            "fab_steps": fab_data_clean.get('fab_steps', ''),
            "fab_recipe": fab_data_clean.get('fab_recipe', ''),
            "fab_rampe_K_per_second": fab_data_clean.get('fab_rampe_K_per_second', ''),
            "fab_holding_time_minutes": fab_data_clean.get('fab_holding_time_minutes', '')
        }
        
        fab_doc = {
            "_id": f"{sample_number}_fab{extracted_sequence}_rtp",
            "sample_number": sample_number,
            "sub_id": f"{sample_number}_sub",
            "fab_method": fab_data_clean.get('fab_method', ''),
            "fab_sequence": fab_sequence,
            "fab_operator": fab_data_clean.get('fab_operator', ''),
            "fab_institution": fab_data_clean.get('fab_institution', ''),
            "fab_rtp_params": fab_rtp_params,
            "fab_sample_weight_before_mg": weight_before,
            "fab_sample_weight_after_mg": weight_after,
            "fab_sample_weight_change_mg": weight_change,
            "fab_orientation": fab_data_clean.get('fab_orientation', ''),
            "fab_date(y/m/d)_time": fab_date_time_iso,
            "created_date_time": current_time,
            "updated_date_time": current_time
        }
        
        fab_id = fab_doc["_id"]
        existing_fab = db['fabrications'].find_one({"_id": fab_id})
        
        if existing_fab:
            fab_doc["created_date_time"] = existing_fab.get("created_date_time", current_time)
            db['fabrications'].replace_one({"_id": fab_id}, fab_doc)
        else:
            db['fabrications'].insert_one(fab_doc)
        
        lib_id = f"{sample_number}_lib"
        existing_library = db['libraries'].find_one({"_id": lib_id})
        
        if existing_library:
            fab_ids = existing_library.get('fab_ids', [])
            if fab_id not in fab_ids:
                fab_ids.append(fab_id)
            
            fab_ids_sorted = sort_fab_ids_by_sequence(db, fab_ids)
            
            db['libraries'].update_one(
                {"_id": lib_id},
                {"$set": {"fab_ids": fab_ids_sorted, "updated_date_time": current_time}}
            )
        else:
            library_doc = {
                "_id": lib_id,
                "sample_number": sample_number,
                "sub_id": "",
                "fab_ids": [fab_id],
                "treat_ids": [],
                "created_date_time": current_time,
                "updated_date_time": current_time
            }
            db['libraries'].insert_one(library_doc)
        
        return True, "RTP Fabrication created/updated successfully"
        
    except Exception as e:
        return False, f"Error with RTP fabrication: {str(e)}"


# ==================== PULSED LASER DEPOSITION (PLD) CREATION FOR FABRICATION COLLECTION ====================

def create_pld_fabrication(db, sample_number, fab_data, extracted_sequence):
    """Create or update PLD fabrication document"""
    try:
        if not fab_data:
            return False, "No fabrication data provided"
        
        fab_data_clean = {}
        for key, value in fab_data.items():
            if isinstance(value, str):
                fab_data_clean[key] = value.rstrip(',').strip().strip('"')
            else:
                fab_data_clean[key] = value
        
        sample_number = sample_number.rstrip(',').strip()
        
        if fab_data_clean.get('sample_number', '') != sample_number:
            return False, "Sample number mismatch"
        
        fab_date_str = fab_data_clean.get('fab_date', '')
        fab_time_str = fab_data_clean.get('fab_time', '')
        
        try:
            datetime_str = f"{fab_date_str} {fab_time_str}"
            dt = datetime.strptime(datetime_str, "%A, %B %d, %Y %I:%M:%S %p")
            dt_utc = dt.replace(tzinfo=timezone.utc)
            fab_date_time_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            now_utc = datetime.now(timezone.utc)
            fab_date_time_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        fab_sequence = extracted_sequence
        
        fab_pre_ablation_params = {
            "shots": fab_data_clean.get('shots', ''),
            "laser_frequency_hz": fab_data_clean.get('fab_laser_frequency_hz', ''),
            "laser_fluence": fab_data_clean.get('laser_fluence', ''),
            "gas_pressure_mbar": fab_data_clean.get('gas_pressure_mbar', ''),
            "gas_type": fab_data_clean.get('gas_type', ''),
            "duration_minutes": fab_data_clean.get('duration_minutes', '')
        }
        
        fab_deposition_params = {
            "temperature_C": fab_data_clean.get('fab_temperature_C', ''),
            "shots": fab_data_clean.get('fab_shots', ''),
            "laser_frequency_hz": fab_data_clean.get('fab_laser_frequency_hz', ''),
            "laser_fluence": fab_data_clean.get('fab_laser_fluence', ''),
            "gas_pressure_mbar": fab_data_clean.get('fab_gas_pressure_mbar', ''),
            "gas_type": fab_data_clean.get('fab_gas_type', ''),
            "duration_minutes": fab_data_clean.get('fab_duration_minutes', '')
        }
        
        fab_doc = {
            "_id": f"{sample_number}_fab{extracted_sequence}_pld",
            "sample_number": sample_number,
            "sub_id": f"{sample_number}_sub",
            "fab_method": fab_data_clean.get('fab_method', ''),
            "fab_sequence": fab_sequence,
            "fab_operator": fab_data_clean.get('fab_operator', ''),
            "fab_institution": fab_data_clean.get('fab_institution', ''),
            "fab_pre_ablation_params": fab_pre_ablation_params,
            "fab_deposition_params": fab_deposition_params,
            "fab_date(y/m/d)_time": fab_date_time_iso,
            "created_date_time": current_time,
            "updated_date_time": current_time
        }
        
        fab_id = fab_doc["_id"]
        existing_fab = db['fabrications'].find_one({"_id": fab_id})
        
        if existing_fab:
            fab_doc["created_date_time"] = existing_fab.get("created_date_time", current_time)
            db['fabrications'].replace_one({"_id": fab_id}, fab_doc)
        else:
            db['fabrications'].insert_one(fab_doc)
        
        lib_id = f"{sample_number}_lib"
        existing_library = db['libraries'].find_one({"_id": lib_id})
        
        if existing_library:
            fab_ids = existing_library.get('fab_ids', [])
            if fab_id not in fab_ids:
                fab_ids.append(fab_id)
            
            fab_ids_sorted = sort_fab_ids_by_sequence(db, fab_ids)
            
            db['libraries'].update_one(
                {"_id": lib_id},
                {"$set": {"fab_ids": fab_ids_sorted, "updated_date_time": current_time}}
            )
        else:
            library_doc = {
                "_id": lib_id,
                "sample_number": sample_number,
                "sub_id": "",
                "fab_ids": [fab_id],
                "treat_ids": [],
                "created_date_time": current_time,
                "updated_date_time": current_time
            }
            db['libraries'].insert_one(library_doc)
        
        return True, "PLD Fabrication created/updated successfully"
        
    except Exception as e:
        return False, f"Error with PLD fabrication: {str(e)}"


# ==================== PHYSICAL VAPOR DEPOSITION - PULSED (PVD-P) CREATION FOR FABRICATION COLLECTION ====================

def create_pvdp_fabrication(db, sample_number, fab_data, extracted_sequence):
    """Create or update PVDP fabrication document with advanced statistics"""
    try:
        if not fab_data:
            return False, "No fabrication data provided"
        
        fab_data_clean = {}
        for key, value in fab_data.items():
            if isinstance(value, str):
                fab_data_clean[key] = value.rstrip(',').strip().strip('"')
            else:
                fab_data_clean[key] = value
        
        sample_number = sample_number.rstrip(',').strip()
        
        if fab_data_clean.get('sample_number', '') != sample_number:
            return False, "Sample number mismatch"
        
        fab_date_str = fab_data_clean.get('fab_date', '')
        fab_time_str = fab_data_clean.get('fab_time', '')
        
        try:
            datetime_str = f"{fab_date_str} {fab_time_str}"
            dt = datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S")
            dt_utc = dt.replace(tzinfo=timezone.utc)
            fab_date_time_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            now_utc = datetime.now(timezone.utc)
            fab_date_time_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        fab_sequence = extracted_sequence
        
        fab_description = {
            "fab_duration_minutes": fab_data_clean.get('duration_minutes'),
            "fab_PbI2_Aout_avg": fab_data_clean.get('PbI2_Aout_avg'),
            "fab_PbI2_PV_avg": fab_data_clean.get('PbI2_PV_avg'),
            "fab_PbI2_T_avg": fab_data_clean.get('PbI2_T_avg'),
            "fab_CsBr_Aout_avg": fab_data_clean.get('CsBr_Aout_avg'),
            "fab_CsBr_PV_avg": fab_data_clean.get('CsBr_PV_avg'),
            "fab_CsBr_T_avg": fab_data_clean.get('CsBr_T_avg'),
            "fab_CsI_Aout_avg": fab_data_clean.get('CsI_Aout_avg'),
            "fab_CsI_PV_avg": fab_data_clean.get('CsI_PV_avg'),
            "fab_CsI_T_avg": fab_data_clean.get('CsI_T_avg'),
            "fab_SnI2_Aout_avg": fab_data_clean.get('SnI2_Aout_avg'),
            "fab_SnI2_PV_avg": fab_data_clean.get('SnI2_PV_avg'),
            "fab_SnI2_T_avg": fab_data_clean.get('SnI2_T_avg'),
            "fab_vacuum_pressure_avg_mbar": fab_data_clean.get('vacuum_pressure2_avg')
        }
        
        fab_doc = {
            "_id": f"{sample_number}_fab{extracted_sequence}_pvdp",
            "sample_number": sample_number,
            "sub_id": f"{sample_number}_sub",
            "fab_method": "PVD-P",
            "fab_sequence": fab_sequence,
            "fab_operator": fab_data_clean.get('fab_operator', ''),
            "fab_institution": fab_data_clean.get('fab_institution', ''),
            "fab_description": fab_description,
            "fab_date(y/m/d)_time": fab_date_time_iso,
            "created_date_time": current_time,
            "updated_date_time": current_time
        }
        
        fab_id = fab_doc["_id"]
        existing_fab = db['fabrications'].find_one({"_id": fab_id})
        
        if existing_fab:
            fab_doc["created_date_time"] = existing_fab.get("created_date_time", current_time)
            db['fabrications'].replace_one({"_id": fab_id}, fab_doc)
        else:
            db['fabrications'].insert_one(fab_doc)
        
        lib_id = f"{sample_number}_lib"
        existing_library = db['libraries'].find_one({"_id": lib_id})
        
        if existing_library:
            fab_ids = existing_library.get('fab_ids', [])
            if fab_id not in fab_ids:
                fab_ids.append(fab_id)
            
            fab_ids_sorted = sort_fab_ids_by_sequence(db, fab_ids)
            
            db['libraries'].update_one(
                {"_id": lib_id},
                {"$set": {"fab_ids": fab_ids_sorted, "updated_date_time": current_time}}
            )
        else:
            library_doc = {
                "_id": lib_id,
                "sample_number": sample_number,
                "sub_id": "",
                "fab_ids": [fab_id],
                "treat_ids": [],
                "created_date_time": current_time,
                "updated_date_time": current_time
            }
            db['libraries'].insert_one(library_doc)
        
        return True, "PVDP Fabrication created/updated successfully"
        
    except Exception as e:
        return False, f"Error with PVDP fabrication: {str(e)}"




def create_fabrication(db, sample_number, fab_data, fab_method, extracted_sequence):
    """Route to correct fabrication creation function based on method"""
    fab_method_lower = fab_method.lower()
    
    st.write(f"fab_method={fab_method}, fab_method_lower={fab_method_lower}")
    
    if 'pvdp' in fab_method_lower or 'pvd-p' in fab_method_lower:
        st.write("Routing to PVDP")
        return create_pvdp_fabrication(db, sample_number, fab_data, extracted_sequence)
    elif 'pld' in fab_method_lower:
        st.write("Routing to PLD")
        return create_pld_fabrication(db, sample_number, fab_data, extracted_sequence)
    elif 'rtp' in fab_method_lower:
        st.write("Routing to RTP")
        return create_rtp_fabrication(db, sample_number, fab_data, extracted_sequence)
    elif 'tube furnace' in fab_method_lower or 'tube_furnace' in fab_method_lower:
        st.write("Routing to Tube Furnace")
        return create_tube_furnace_fabrication(db, sample_number, fab_data, extracted_sequence)
    elif 'pvdj' in fab_method_lower or ('pvd' in fab_method_lower and 'pvdp' not in fab_method_lower and 'pvd-p' not in fab_method_lower):
        st.write("Routing to PVDJ")
        return create_pvdj_fabrication(db, sample_number, fab_data, extracted_sequence)
    elif 'sputtering' in fab_method_lower or 'spt' in fab_method_lower:
        st.write("Routing to Sputtering")
        return create_sputtering_fabrication(db, sample_number, fab_data, extracted_sequence)
    else:
        st.write(f"Unknown method {fab_method}")
        return False, f"Unknown fabrication method: {fab_method}"




# ==================== PAGE: UPLOAD FABRICATION DATA ====================

def page_upload_fabrication_data(db):
    """Handle fabrication data upload supports multiple methods and batch upload"""
    st.markdown("<h5>Fabrication Files</h5>", unsafe_allow_html=True)
    
    if 'processed_fab_files' not in st.session_state:
        st.session_state.processed_fab_files = set()
    
    uploaded_files = st.file_uploader(
        "Upload fabrication CSV files:",
        type=['csv'],
        accept_multiple_files=True,
        key="fab_uploader"
    )
    
    if not uploaded_files:
        return
    
    valid_fab_files = [f for f in uploaded_files if '_fab' in f.name.lower()]
    invalid_files = [f for f in uploaded_files if '_fab' not in f.name.lower()]
    
    if invalid_files:
        st.warning(f"Warning {len(invalid_files)} file(s) skipped missing _fab in filename")
    
    if not valid_fab_files:
        st.error("Error No valid fabrication files found")
        return
    
    all_files_data = []
    validation_errors = []
    validation_warnings = []
    
    for uploaded_file in valid_fab_files:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.file_id}"
        
        if file_id in st.session_state.processed_fab_files:
            st.success(f"Already processed: {uploaded_file.name}")
            continue
        
        try:
            content = uploaded_file.read().decode('utf-8')
            fab_data = parse_fab_csv(content, uploaded_file.name)
            
            if not fab_data:
                validation_errors.append((uploaded_file.name, "Could not parse CSV"))
                continue
            
            is_valid, sample_num, fab_method, extracted_sequence, warning_msg = validate_fab_csv(uploaded_file, fab_data, db)
            
            st.write(f"is_valid={is_valid}, fab_method={fab_method}")
            
            if is_valid:
                all_files_data.append({
                    'file_id': file_id,
                    'filename': uploaded_file.name,
                    'sample_number': sample_num,
                    'fab_method': fab_method,
                    'fab_data': fab_data,
                    'extracted_sequence': extracted_sequence,
                    'warning': warning_msg
                })
                if warning_msg:
                    validation_warnings.append((uploaded_file.name, warning_msg))
            else:
                validation_errors.append((uploaded_file.name, fab_method or "Validation failed"))
        
        except Exception as e:
            validation_errors.append((uploaded_file.name, str(e)))
    
    if validation_errors:
        st.error("Error Validation Errors")
        for filename, error in validation_errors:
            st.write(f"{filename}: {error}")
    
    if validation_warnings:
        st.warning("Warning Validation Warnings")
        for filename, warning in validation_warnings:
            st.write(f"{filename}: {warning}")
    
    if all_files_data:
        st.subheader(f"Ready to upload: {len(all_files_data)} file(s)")
        
        for file_info in all_files_data:
            st.write(f"**{file_info['filename']}**")
            st.write(f"Sample: {file_info['sample_number']} | Method: {file_info['fab_method']} | Sequence: {file_info['extracted_sequence']}")
        
        if st.button("Upload All Files", type="primary", use_container_width=True):
            successful_uploads = 0
            failed_uploads = 0
            
            for file_info in all_files_data:
                success, message = create_fabrication(
                    db,
                    file_info['sample_number'],
                    file_info['fab_data'],
                    file_info['fab_method'],
                    file_info['extracted_sequence']
                )
                
                if success:
                    st.session_state.processed_fab_files.add(file_info['file_id'])
                    successful_uploads += 1
                    st.success(f"Success {file_info['filename']}: {message}")
                else:
                    failed_uploads += 1
                    st.error(f"Error {file_info['filename']}: {message}")
            
            st.write(f"Total: {len(all_files_data)} | Successful: {successful_uploads} | Failed: {failed_uploads}")