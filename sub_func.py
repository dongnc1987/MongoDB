import streamlit as st
from datetime import datetime, timezone
import re


# ==================== PARSE SUBSTRATE CSV ====================

def parse_substrate_csv(content):
    """Parse substrate CSV file"""
    try:
        lines = content.strip().split('\n')
        substrate_data = {}
        
        for line in lines:
            parts = line.split(',', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().strip('"').rstrip(',').strip()  # Remove trailing commas
                substrate_data[key] = value
        
        return substrate_data
    except Exception as e:
        st.error(f"❌ Error parsing CSV: {str(e)}")
        return None


def validate_substrate_csv(uploaded_file, substrate_data, db):
    """Validate substrate CSV file comprehensively"""
    
    # ==================== EXTRACT FILENAME INFO ====================
    filename = uploaded_file.name
    
    # Check if "substrate" is in filename
    if "substrate" not in filename.lower():
        return False, None, "Invalid filename: Must contain 'substrate' in the filename"
    
    # Extract info from filename: sample_number_institution_operator_substrate_substrate_type_date_time.csv
    # Example: 3716-15_HZB_Dong Nguyen_substrate_quartz_20250710_140115.csv
    
    # Replace spaces with underscores to normalize for splitting
    filename_normalized = filename.replace(' ', '_').replace('.csv', '')
    filename_parts = filename_normalized.split('_')
    
    if len(filename_parts) < 8:
        return False, None, "Invalid filename format. Expected: sample_number_institution_operator_substrate_substrate_type_date_time.csv"
    
    filename_sample_number = filename_parts[0]
    filename_institution = filename_parts[1]
    # Operator is between institution and "substrate" keyword
    substrate_index = filename_parts.index('substrate')
    filename_operator = '_'.join(filename_parts[2:substrate_index])
    filename_substrate_type = filename_parts[substrate_index + 1]
    
    # ==================== EXTRACT CSV DATA ====================
    csv_sample_number = substrate_data.get('sample_number', '').strip()
    csv_institution = substrate_data.get('sub_institution', '').strip()
    csv_operator = substrate_data.get('sub_operator', '').strip()
    csv_substrate_type = substrate_data.get('sub_substrate_type', '').strip()
    
    # ==================== VALIDATION CHECKS ====================
    
    # 1. Check sample number match
    if filename_sample_number != csv_sample_number:
        return False, None, f"Sample Number Mismatch!\n   - From Filename: {filename_sample_number}\n   - Inside CSV: {csv_sample_number}"
    
    # 2. Check institution match
    if filename_institution.lower() != csv_institution.lower():
        return False, None, f"Institution Mismatch!\n   - From Filename: {filename_institution}\n   - Inside CSV: {csv_institution}"
    
    # 3. Check operator match
    if filename_operator.lower() != csv_operator.lower().replace(' ', '_'):
        return False, None, f"Operator Mismatch!\n   - From Filename: {filename_operator.replace('_', ' ')}\n   - Inside CSV: {csv_operator}"
    
    # 4. Check substrate type match
    if filename_substrate_type.lower() != csv_substrate_type.lower():
        return False, None, f"Substrate Type Mismatch!\n   - From Filename: {filename_substrate_type}\n   - Inside CSV: {csv_substrate_type}"
    
    # 5. Check if library exists for this sample number
    lib_id = f"{csv_sample_number}_lib"
    existing_library = db['libraries'].find_one({"_id": lib_id})
    
    if not existing_library:
        return False, None, f"Sample Number '{csv_sample_number}' does not exist in the library. Please create the library first."
    
    return True, csv_sample_number, None


# ==================== CREATE SUBSTRATE ====================

def create_substrate(db, sample_number, substrate_data):
    """Create or update substrate document"""
    try:
        if not substrate_data:
            return False, "No substrate data provided"
        
        # CLEAN UP all substrate_data values
        substrate_data_clean = {}
        for key, value in substrate_data.items():
            if isinstance(value, str):
                substrate_data_clean[key] = value.rstrip(',').strip()
            else:
                substrate_data_clean[key] = value
        
        sample_number = sample_number.rstrip(',').strip()
        
        # CHECK IF SAMPLE NUMBER MATCHES
        if substrate_data_clean.get('sample_number', '') != sample_number:
            return False, "Sample number mismatch!"
        
        # ==================== PARSE DATE AND TIME ====================
        sub_clean_date_str = substrate_data_clean.get('sub_clean_date', '')
        sub_clean_time_str = substrate_data_clean.get('sub_clean_time', '')
        
        try:
            # Parse date: "Thursday, July 10, 2025"
            date_part = sub_clean_date_str.split('\t')[0] if '\t' in sub_clean_date_str else sub_clean_date_str
            date_part = date_part.strip()
            
            # Parse time: "2:01:15 PM"
            time_part = sub_clean_time_str.split('\t')[0] if '\t' in sub_clean_time_str else sub_clean_time_str
            time_part = time_part.strip()
            
            # Combine and parse
            datetime_str = f"{date_part} {time_part}"
            dt = datetime.strptime(datetime_str, "%A, %B %d, %Y %I:%M:%S %p")
            dt_utc = dt.replace(tzinfo=timezone.utc)
            sub_date_time_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception as e:
            # Fallback to current time if parsing fails
            now_utc = datetime.now(timezone.utc)
            sub_date_time_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H.%M.%SZ")
        
        # BUILD sub_type OBJECT
        sub_type = {
            "sub_substrate_type": substrate_data_clean.get('sub_substrate_type', ''),  # Added
            "sub_production_batch": substrate_data_clean.get('sub_production_batch', ''),
            "sub_vendor": substrate_data_clean.get('sub_vendor', ''),
            "sub_manufacture": substrate_data_clean.get('sub_manufacture', ''),
            "sub_softing_point_celsius": substrate_data_clean.get('sub_softing_point_celsius', ''),
            "sub_expansion_coefficient": substrate_data_clean.get('sub_expansion_coefficient', ''),
            "sub_temp_celsius": substrate_data_clean.get('sub_temp_celsius', '')
        }
        
        # BUILD SUBSTRATE DOCUMENT
        substrate_doc = {
            "_id": f"{sample_number}_sub",
            "sample_number": sample_number,
            "sub_type": sub_type,
            "sub_thickness_mm": substrate_data_clean.get('sub_thickness_mm', ''),
            "sub_size_mm_x_y": substrate_data_clean.get('sub_size_mm_x_y', ''),
            "sub_materials": substrate_data_clean.get('sub_materials', ''),
            "sub_program": substrate_data_clean.get('sub_program', ''),
            "sub_operator": substrate_data_clean.get('sub_operator', ''),
            "sub_institution": substrate_data_clean.get('sub_institution', ''),
            "sub_clean_method": substrate_data_clean.get('sub_clean_method', ''),
            "sub_clean_description": substrate_data_clean.get('sub_clean_description', ''),
            "sub_clean_duration_Min": substrate_data_clean.get('sub_clean_duration_Min', ''),
            "sub_clean_temperature_C": substrate_data_clean.get('sub_clean_temperature_C', ''),
            "sub_clean_pressure_mbar": substrate_data_clean.get('sub_clean_pressure_mbar', ''),
            "sub_date(y/m/d)_time": sub_date_time_iso,
            "created_date_time": current_time,
            "updated_date_time": current_time
        }
        
        sub_id = f"{sample_number}_sub"
        existing_substrate = db['substrates'].find_one({"_id": sub_id})
        
        if existing_substrate:
            substrate_doc["created_date_time"] = existing_substrate.get("created_date_time", current_time)
            db['substrates'].replace_one({"_id": sub_id}, substrate_doc)
        else:
            db['substrates'].insert_one(substrate_doc)
        
        # UPDATE LIBRARY
        lib_id = f"{sample_number}_lib"
        existing_library = db['libraries'].find_one({"_id": lib_id})
        
        if existing_library:
            db['libraries'].update_one(
                {"_id": lib_id},
                {
                    "$set": {
                        "sub_id": sub_id,
                        "updated_date_time": current_time
                    }
                }
            )
        else:
            library_doc = {
                "_id": lib_id,
                "sample_number": sample_number,
                "sub_id": sub_id,
                "fab_ids": [],
                "treat_ids": [],
                "created_date_time": current_time,
                "updated_date_time": current_time
            }
            db['libraries'].insert_one(library_doc)
        
        return True, "Substrate created/updated successfully!"
        
    except Exception as e:
        return False, f"Error with substrate: {str(e)}"

# ==================== PAGE: UPLOAD SUBSTRATE DATA ====================

def page_upload_substrate_data(db):
    """Handle substrate data upload from CSV file"""
    st.markdown("<h5>Substrate File</h5>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload substrate CSV file:",
        type=['csv']
    )
    
    if not uploaded_file:
        return
    
    try:
        content = uploaded_file.read().decode('utf-8')
        substrate_data = parse_substrate_csv(content)
        
        if not substrate_data:
            st.error("❌ Could not parse CSV file")
            return
        
        # VALIDATE CSV FILE
        is_valid, sample_num, error_msg = validate_substrate_csv(uploaded_file, substrate_data, db)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return
        
        st.success("✅ Validation passed!")
        # st.write("**Parsed Data:**")
        # with st.expander("Show parsed JSON"):
        #     st.json(substrate_data)
        
        if st.button("Create Substrate", type="primary", use_container_width=True):
            success, message = create_substrate(db, sample_num, substrate_data)
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")