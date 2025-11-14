import streamlit as st
from datetime import datetime, timezone
import re
from difflib import SequenceMatcher


def normalize_string(s):
    """Normalize string for flexible comparison"""
    s = str(s).lower().strip()
    s = ' '.join(s.split())
    s = s.replace('_', ' ').replace('-', ' ')
    s = ' '.join(s.split())
    return s


def flexible_string_match(filename_value, csv_value, match_type='flexible'):
    """Compare strings with flexibility for spacing and capitalization"""
    if match_type == 'strict':
        return filename_value == csv_value, 1.0 if filename_value == csv_value else 0.0
    
    norm_filename = normalize_string(filename_value)
    norm_csv = normalize_string(csv_value)
    
    if norm_filename == norm_csv:
        return True, 1.0
    
    if norm_filename in norm_csv or norm_csv in norm_filename:
        return True, 0.95
    
    similarity = SequenceMatcher(None, norm_filename, norm_csv).ratio()
    if similarity >= 0.85:
        return True, similarity
    
    return False, similarity


def sort_treat_ids_by_sequence(db, treat_ids):
    """Sort treat_ids by their sequence number"""
    treat_docs = []
    for treat_id in treat_ids:
        treat_doc = db['treatments'].find_one({"_id": treat_id})
        if treat_doc:
            treat_docs.append({
                "_id": treat_id,
                "sequence": int(treat_doc.get('treat_sequence', 0))
            })
    
    treat_docs_sorted = sorted(treat_docs, key=lambda x: x['sequence'])
    return [doc['_id'] for doc in treat_docs_sorted]


def parse_treat_csv(content):
    """Parse simple key-value treatment CSV file"""
    try:
        lines = content.strip().split('\n')
        treat_data = {}
        
        for line in lines:
            parts = line.split(',', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().strip('"').rstrip(',').strip()
                treat_data[key] = value
        
        return treat_data if treat_data else None
    except Exception as e:
        return None


def validate_treat_csv(uploaded_file, treat_data, db):
    """
    Validate treatment CSV file.
    Filename format: {SampleNumber}_{Institution}_{Operator}_treat{Sequence}_{Method}_{YYYYMMDD}_{HHMMSS}.csv
    Supported methods: Annealing, As-deposited
    """
    filename = uploaded_file.name
    
    if '_treat' not in filename.lower():
        return False, None, None, None, "Invalid filename: Must contain _treat"
    
    treat_match = re.search(r'_treat(\d+)_([A-Za-z-]+)_(\d{8})_(\d{6})', filename)
    if not treat_match:
        return False, None, None, None, "Invalid filename: Must contain _treat#_METHOD_YYYYMMDD_HHMMSS"
    
    extracted_sequence = treat_match.group(1)
    filename_method = treat_match.group(2)
    
    filename_no_ext = filename.replace('.csv', '')
    parts_before_treat = filename_no_ext.split('_treat')[0]
    filename_parts = parts_before_treat.split('_')
    
    if len(filename_parts) < 3:
        return False, None, None, None, "Invalid filename format: Need SampleNumber_Institution_Operator"
    
    filename_sample_number = filename_parts[0]
    filename_institution = filename_parts[1]
    filename_operator = '_'.join(filename_parts[2:])
    
    csv_sample_number = treat_data.get('sample_number', '').strip()
    csv_institution = treat_data.get('treat_institution', '').strip()
    csv_operator = treat_data.get('treat_operator', '').strip()
    csv_method = treat_data.get('treat_method', '').strip()
    
    # STRICT: Sample Number must match exactly
    if filename_sample_number != csv_sample_number:
        return False, None, None, None, f"Sample Mismatch: '{filename_sample_number}' vs '{csv_sample_number}'"
    
    # FLEXIBLE: Institution
    institution_match, institution_quality = flexible_string_match(
        filename_institution, csv_institution, match_type='flexible'
    )
    if not institution_match:
        return False, None, None, None, f"Institution Mismatch: '{filename_institution}' vs '{csv_institution}' ({institution_quality:.1%})"
    
    # FLEXIBLE: Operator
    operator_match, operator_quality = flexible_string_match(
        filename_operator, csv_operator, match_type='flexible'
    )
    if not operator_match:
        return False, None, None, None, f"Operator Mismatch: '{filename_operator}' vs '{csv_operator}' ({operator_quality:.1%})"
    
    # Check method - support both Annealing and As-deposited
    allowed_methods = ['annealing', 'anneal', 'thermal', 'as-deposited', 'asdeposited']
    filename_method_lower = filename_method.lower()
    csv_method_lower = csv_method.lower()
    
    if filename_method_lower not in allowed_methods and csv_method_lower not in allowed_methods:
        return False, None, None, None, f"Method must be Annealing or As-deposited (found: {filename_method})"
    
    # Normalize the method for the database
    if filename_method_lower in ['as-deposited', 'asdeposited']:
        normalized_method = 'As-deposited'
    else:
        normalized_method = 'Annealing'
    
    # Check for existing treatment
    existing_treat = db['treatments'].find_one({
        "sample_number": csv_sample_number,
        "treat_sequence": extracted_sequence
    })
    
    warning_msg = None
    if existing_treat:
        warning_msg = f"Sequence {extracted_sequence} already exists (will be updated)"
    
    return True, csv_sample_number, normalized_method, extracted_sequence, warning_msg


def create_treatment(db, sample_number, treat_data, extracted_sequence, treat_method):
    """Create or update Treatment document (handles both Annealing and As-deposited)"""
    try:
        if not treat_data:
            return False, "No treatment data provided"
        
        treat_data_clean = {}
        for key, value in treat_data.items():
            if isinstance(value, str):
                treat_data_clean[key] = value.rstrip(',').strip().strip('"')
            else:
                treat_data_clean[key] = value
        
        sample_number = sample_number.rstrip(',').strip()
        
        if treat_data_clean.get('sample_number', '') != sample_number:
            return False, "Sample number mismatch"
        
        # Parse date and time
        treat_date_str = treat_data_clean.get('treat_date', '')
        treat_time_str = treat_data_clean.get('treat_time', '')
        
        try:
            datetime_str = f"{treat_date_str} {treat_time_str}"
            dt = datetime.strptime(datetime_str, "%A, %B %d, %Y %I:%M:%S %p")
            dt_utc = dt.replace(tzinfo=timezone.utc)
            treat_date_time_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            now_utc = datetime.now(timezone.utc)
            treat_date_time_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        treat_sequence = extracted_sequence
        
        # Treatment method-specific parameters
        treat_params = {}
        
        if treat_method == 'Annealing':
            treat_params = {
                "treat_temperature_celsius": treat_data_clean.get('treat_temperature_celsius', ''),
                "treat_duration_second": treat_data_clean.get('treat_duration_second', ''),
                "treat_humidity_ppm": treat_data_clean.get('treat_humidity_ppm', ''),
                "treat_oxygen_concentration_ppm": treat_data_clean.get('treat_oxygen_concentration_ppm', ''),
                "treat_gas": treat_data_clean.get('treat_gas', ''),
                "treat_pressure_mbar": treat_data_clean.get('treat_pressure_mbar', '')
            }
        elif treat_method == 'As-deposited':
            # As-deposited has minimal parameters (just the deposition info)
            treat_params = {
                "treat_temperature_celsius": treat_data_clean.get('treat_temperature_celsius', ''),
                "treat_duration_second": treat_data_clean.get('treat_duration_second', '')
            }
        
        # Create treatment document
        treat_doc = {
            "_id": f"{sample_number}_treat{extracted_sequence}_{treat_method.lower()}",
            "sample_number": sample_number,
            "sub_id": f"{sample_number}_sub",
            "treat_method": treat_method,
            "treat_sequence": treat_sequence,
            "treat_operator": treat_data_clean.get('treat_operator', ''),
            "treat_institution": treat_data_clean.get('treat_institution', ''),
            "treat_place": treat_data_clean.get('treat_place', ''),
            "treat_params": treat_params,
            "treat_date(y/m/d)_time": treat_date_time_iso,
            "created_date_time": current_time,
            "updated_date_time": current_time
        }
        
        treat_id = treat_doc["_id"]
        existing_treat = db['treatments'].find_one({"_id": treat_id})
        
        # Insert or update
        if existing_treat:
            treat_doc["created_date_time"] = existing_treat.get("created_date_time", current_time)
            db['treatments'].replace_one({"_id": treat_id}, treat_doc)
        else:
            db['treatments'].insert_one(treat_doc)
        
        # Update library document
        lib_id = f"{sample_number}_lib"
        existing_library = db['libraries'].find_one({"_id": lib_id})
        
        if existing_library:
            treat_ids = existing_library.get('treat_ids', [])
            if treat_id not in treat_ids:
                treat_ids.append(treat_id)
            
            treat_ids_sorted = sort_treat_ids_by_sequence(db, treat_ids)
            
            db['libraries'].update_one(
                {"_id": lib_id},
                {"$set": {"treat_ids": treat_ids_sorted, "updated_date_time": current_time}}
            )
        else:
            library_doc = {
                "_id": lib_id,
                "sample_number": sample_number,
                "sub_id": "",
                "fab_ids": [],
                "treat_ids": [treat_id],
                "created_date_time": current_time,
                "updated_date_time": current_time
            }
            db['libraries'].insert_one(library_doc)
        
        return True, f"{treat_method} treatment created/updated successfully"
        
    except Exception as e:
        return False, f"Error with {treat_method} treatment: {str(e)}"


# ==================== PAGE: UPLOAD TREATMENT DATA ====================

def page_upload_treatment_data(db):
    """Streamlit page for uploading Treatment files (Annealing and As-deposited)"""
    st.markdown("<h5>Treatment Files</h5>", unsafe_allow_html=True)
    
    if 'processed_treat_files' not in st.session_state:
        st.session_state.processed_treat_files = set()
    
    uploaded_files = st.file_uploader(
        "Upload treatment CSV files:",
        type=['csv'],
        accept_multiple_files=True,
        key="treat_uploader"
    )
    
    if not uploaded_files:
        return
    
    valid_treat_files = [f for f in uploaded_files if '_treat' in f.name.lower()]
    invalid_files = [f for f in uploaded_files if '_treat' not in f.name.lower()]
    
    if invalid_files:
        st.warning(f"⚠️ {len(invalid_files)} file(s) skipped (missing _treat in filename)")
    
    if not valid_treat_files:
        st.error("❌ No valid treatment files found")
        return
    
    all_files_data = []
    validation_errors = []
    validation_warnings = []
    
    # Validate all files
    for uploaded_file in valid_treat_files:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.file_id}"
        
        if file_id in st.session_state.processed_treat_files:
            st.success(f"✓ Already processed: {uploaded_file.name}")
            continue
        
        try:
            content = uploaded_file.read().decode('utf-8')
            treat_data = parse_treat_csv(content)
            
            if not treat_data:
                validation_errors.append((uploaded_file.name, "Could not parse CSV"))
                continue
            
            is_valid, sample_num, treat_method, extracted_sequence, warning_msg = validate_treat_csv(
                uploaded_file, treat_data, db
            )
            
            if is_valid:
                all_files_data.append({
                    'file_id': file_id,
                    'filename': uploaded_file.name,
                    'sample_number': sample_num,
                    'treat_method': treat_method,
                    'treat_data': treat_data,
                    'extracted_sequence': extracted_sequence,
                    'warning': warning_msg
                })
                if warning_msg:
                    validation_warnings.append((uploaded_file.name, warning_msg))
            else:
                validation_errors.append((uploaded_file.name, treat_method or "Validation failed"))
        
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
            st.write(f"**{file_info['filename']}**")
            st.write(f"Sample: {file_info['sample_number']} | Method: {file_info['treat_method']} | Sequence: {file_info['extracted_sequence']}")
        
        if st.button("Upload All Treatment Files", type="primary", use_container_width=True):
            successful_uploads = 0
            failed_uploads = 0
            
            for file_info in all_files_data:
                success, message = create_treatment(
                    db,
                    file_info['sample_number'],
                    file_info['treat_data'],
                    file_info['extracted_sequence'],
                    file_info['treat_method']
                )
                
                if success:
                    st.session_state.processed_treat_files.add(file_info['file_id'])
                    successful_uploads += 1
                    st.success(f"✓ {file_info['filename']}: {message}")
                else:
                    failed_uploads += 1
                    st.error(f"✗ {file_info['filename']}: {message}")
            
            st.write(f"**Total: {len(all_files_data)} | Successful: {successful_uploads} | Failed: {failed_uploads}**")