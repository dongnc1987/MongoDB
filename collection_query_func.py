"""
Collection Query Functions for High-throughput Measurement System
Handles all MongoDB query operations and data serialization
"""

import json
from bson import ObjectId
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st


# ==================== SERIALIZATION FUNCTIONS ====================

def serialize_value(value):
    """Serialize a single value to JSON-compatible format"""
    if isinstance(value, ObjectId):
        return str(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, dict):
        return serialize_dict(value)
    elif isinstance(value, list):
        return [serialize_value(item) for item in value]
    return value


def serialize_dict(data: Dict) -> Dict:
    """Recursively serialize nested dictionaries"""
    return {key: serialize_value(value) for key, value in data.items()}


def serialize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert MongoDB document to JSON-serializable format"""
    return serialize_dict(doc) if doc else None


def serialize_documents(documents: List[Dict]) -> List[Dict]:
    """Serialize multiple documents"""
    return [serialize_document(doc) for doc in documents]


# ==================== QUERY FUNCTIONS ====================

def get_collection_count(db, collection_name: str) -> int:
    """Get total document count in collection"""
    try:
        return db[collection_name].count_documents({})
    except Exception as e:
        raise Exception(f"Error getting collection count: {str(e)}")


def query_collection(db, collection_name: str, skip: int = 0, limit: int = 10, 
                    filter_query: Dict = None) -> List[Dict]:
    """Query documents from collection with pagination"""
    try:
        filter_query = filter_query or {}
        documents = list(db[collection_name].find(filter_query).skip(skip).limit(limit))
        return serialize_documents(documents)
    except Exception as e:
        raise Exception(f"Error querying collection: {str(e)}")


def get_all_sample_numbers(db) -> List[str]:
    """Get all available sample numbers from libraries"""
    try:
        libraries = db['libraries'].find({}, {"sample_number": 1})
        sample_numbers = [lib['sample_number'] for lib in libraries]
        return sorted(sample_numbers)
    except Exception as e:
        st.warning(f"Error retrieving sample numbers: {str(e)}")
        return []


# ==================== WORKFLOW FUNCTIONS ====================

def filter_sample_number(doc_dict: Dict) -> Dict:
    """Remove sample_number from document for cleaner display"""
    return {k: v for k, v in doc_dict.items() if k != 'sample_number'}


def get_sample_workflow(db, sample_number: str) -> Optional[Dict]:
    """
    Get complete chronological workflow for a sample
    
    Returns workflow structure:
    - substrate
    - fabrications (all steps)
    - treatments_with_acquisitions (list of {treatment, acquisitions})
    """
    try:
        lib_id = f"{sample_number}_lib"
        library = db['libraries'].find_one({"_id": lib_id})
        
        if not library:
            return None
        
        workflow = {
            'sample_number': sample_number,
            'sample_overview': serialize_document({
                '_id': library.get('_id'),
                'created_date_time': library.get('created_date_time'),
                'updated_date_time': library.get('updated_date_time')
            })
        }
        
        # Get substrate
        sub_id = f"{sample_number}_sub"
        substrate = db['substrates'].find_one({"_id": sub_id})
        workflow['substrate'] = filter_sample_number(serialize_document(substrate)) if substrate else None
        
        # Get all fabrications
        fab_ids = library.get('fab_ids', [])
        fabrications = []
        
        if fab_ids:
            for fab_id in fab_ids:
                fab_doc = db['fabrications'].find_one({"_id": fab_id})
                if fab_doc:
                    fabrications.append(filter_sample_number(serialize_document(fab_doc)))
            
            fabrications.sort(key=lambda x: int(x.get('fab_sequence', 0)))
        
        workflow['fabrications'] = fabrications
        
        # Get all treatments
        treat_ids = library.get('treat_ids', [])
        treatments = []
        
        if treat_ids:
            for treat_id in treat_ids:
                treat_doc = db['treatments'].find_one({"_id": treat_id})
                if treat_doc:
                    treatments.append(filter_sample_number(serialize_document(treat_doc)))
            
            treatments.sort(key=lambda x: int(x.get('treat_sequence', 0)))
        
        workflow['treatments'] = treatments
        
        # Get all acquisitions
        acquisitions = serialize_documents(list(db['acquisitions'].find({"sample_number": sample_number})))
        
        # Group acquisitions by treatment
        treatments_with_acquisitions = []
        
        for treatment in treatments:
            treat_sequence = treatment.get('treat_sequence')
            
            # Find all acquisitions for this treatment
            treatment_acquisitions = [
                filter_sample_number(acq)
                for acq in acquisitions 
                if acq.get('treatment_sequence') == treat_sequence
            ]
            
            # Sort acquisitions by created_date_time
            treatment_acquisitions.sort(key=lambda x: x.get('created_date_time', ''))
            
            treatments_with_acquisitions.append({
                'treatment': treatment,
                'acquisitions': treatment_acquisitions
            })
        
        workflow['treatments_with_acquisitions'] = treatments_with_acquisitions
        
        # Get metadata counts
        acq_ids = [acq.get('_id') for acq in acquisitions]
        
        resistance_pixels = []
        mpl_pixels = []
        xrf_pixels = []
        pl_spectra = []
        uv_vis_spectra = []
        mpl_spectra = []
        xrf_spectra = []
        
        if acq_ids:
            # Get pixels
            pixels = serialize_documents(list(db['pixels'].find({"acq_id": {"$in": acq_ids}})))
            resistance_pixels = [p for p in pixels if p.get('technique') == 'resistance']
            mpl_pixels = [p for p in pixels if p.get('technique') == 'mpl']
            xrf_pixels = [p for p in pixels if p.get('technique') == 'xrf']
            
            # Get spectra
            spectra = serialize_documents(list(db['spectra'].find({"acq_id": {"$in": acq_ids}})))
            pl_spectra = [s for s in spectra if s.get('technique') == 'pl']
            uv_vis_spectra = [s for s in spectra if s.get('technique') == 'uv_vis']
            mpl_spectra = [s for s in spectra if s.get('technique') == 'mpl']
            xrf_spectra = [s for s in spectra if s.get('technique') == 'xrf']
        
        # Get analyses count
        analyses = list(db['analyses'].find({"sample_number": sample_number}))
        
        # Determine workflow status
        workflow_status = "In Progress"
        
        # Check if workflow is complete
        has_substrate = substrate is not None
        has_fabrications = len(fabrications) > 0
        has_treatments = len(treatments) > 0
        has_acquisitions = len(acquisitions) > 0
        has_analyses = len(analyses) > 0
        
        # Check if ALL treatments have ALL measurement types
        all_treatments_complete = True
        if has_treatments:
            # All required techniques for each treatment
            required_techniques = {'pl', 'uv_vis', 'resistance', 'mpl', 'xrf'}
            
            for treatment_data in treatments_with_acquisitions:
                treatment_acqs = treatment_data['acquisitions']
                
                # Check if this treatment has acquisitions
                if not treatment_acqs:
                    all_treatments_complete = False
                    break
                
                # Get technique types for this treatment
                treatment_techniques = set([acq.get('technique') for acq in treatment_acqs])
                
                # Check if ALL required techniques are present
                if not required_techniques.issubset(treatment_techniques):
                    all_treatments_complete = False
                    break
        else:
            all_treatments_complete = False
        
        # Workflow is complete if:
        # 1. Has substrate
        # 2. Has fabrications
        # 3. Has treatments
        # 4. Has acquisitions
        # 5. ALL treatments have ALL measurement types (pl, uv_vis, resistance, mpl, xrf)
        # 6. Has analyses
        if has_substrate and has_fabrications and has_treatments and has_acquisitions and all_treatments_complete and has_analyses:
            workflow_status = "Completed"
        
        workflow['metadata'] = {
            'total_fabrication_steps': len(fabrications),
            'total_treatments': len(treatments),
            'total_acquisitions': len(acquisitions),
            'total_resistance_pixels_measured': len(resistance_pixels),
            'total_mpl_pixels_measured': len(mpl_pixels),
            'total_xrf_pixels_measured': len(xrf_pixels),
            'total_pl_spectra_recorded': len(pl_spectra),
            'total_uv_vis_spectra_recorded': len(uv_vis_spectra),
            'total_mpl_spectra_recorded': len(mpl_spectra),
            'total_xrf_spectra_recorded': len(xrf_spectra),
            'total_analyses': len(analyses),
            'workflow_status': workflow_status,
            'last_updated': library.get('updated_date_time')
        }
        
        return workflow
    
    except Exception as e:
        st.error(f"Error retrieving sample workflow: {str(e)}")
        return None


# ==================== UI FUNCTIONS ====================
def display_collection_browser(db, selected_collection: str):
    """Display collection browser UI with collection-specific filtering"""
    try:
        total_docs = get_collection_count(db, selected_collection)
    except Exception as e:
        st.error(f"{str(e)}")
        return
    
    st.markdown(f"<h5>{selected_collection.capitalize()}</h5>", unsafe_allow_html=True)
    st.write(f"Total Documents: {total_docs}")
    
    if total_docs == 0:
        st.info("No documents in this collection yet")
        return
    
    # Add collection-specific filters
    filter_query = {}
    
    # Filter configuration for each collection
    if selected_collection == 'libraries':
        st.markdown("---")
        sample_numbers = get_all_sample_numbers(db)
        if sample_numbers:
            selected_sample = st.selectbox(
                "Filter by Sample Number",
                options=["All"] + sample_numbers,
                key=f"sample_filter_{selected_collection}"
            )
            if selected_sample != "All":
                filter_query["sample_number"] = selected_sample
    
    elif selected_collection == 'substrates':
        st.markdown("---")
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            sample_numbers = get_distinct_values(db, selected_collection, "sample_number")
            if sample_numbers:
                selected_sample = st.selectbox(
                    "Filter by Sample Number",
                    options=["All"] + sample_numbers,
                    key=f"sample_filter_{selected_collection}"
                )
                if selected_sample != "All":
                    filter_query["sample_number"] = selected_sample
        
        with col_filter2:
            institutions = get_distinct_values(db, selected_collection, "sub_institution")
            if institutions:
                selected_institution = st.selectbox(
                    "Filter by Institution",
                    options=["All"] + institutions,
                    key=f"institution_filter_{selected_collection}"
                )
                if selected_institution != "All":
                    filter_query["sub_institution"] = selected_institution
    
    elif selected_collection == 'fabrications':
        st.markdown("---")
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            sample_numbers = get_distinct_values(db, selected_collection, "sample_number")
            if sample_numbers:
                selected_sample = st.selectbox(
                    "Filter by Sample Number",
                    options=["All"] + sample_numbers,
                    key=f"sample_filter_{selected_collection}"
                )
                if selected_sample != "All":
                    filter_query["sample_number"] = selected_sample
        
        with col_filter2:
            fab_methods = get_distinct_values(db, selected_collection, "fab_method")
            if fab_methods:
                selected_method = st.selectbox(
                    "Filter by Fabrication Method",
                    options=["All"] + fab_methods,
                    key=f"method_filter_{selected_collection}"
                )
                if selected_method != "All":
                    filter_query["fab_method"] = selected_method
        
        with col_filter3:
            institutions = get_distinct_values(db, selected_collection, "fab_institution")
            if institutions:
                selected_institution = st.selectbox(
                    "Filter by Institution",
                    options=["All"] + institutions,
                    key=f"institution_filter_{selected_collection}"
                )
                if selected_institution != "All":
                    filter_query["fab_institution"] = selected_institution
    
    elif selected_collection == 'treatments':
        st.markdown("---")
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            sample_numbers = get_distinct_values(db, selected_collection, "sample_number")
            if sample_numbers:
                selected_sample = st.selectbox(
                    "Filter by Sample Number",
                    options=["All"] + sample_numbers,
                    key=f"sample_filter_{selected_collection}"
                )
                if selected_sample != "All":
                    filter_query["sample_number"] = selected_sample
        
        with col_filter2:
            treat_methods = get_distinct_values(db, selected_collection, "treat_method")
            if treat_methods:
                selected_method = st.selectbox(
                    "Filter by Treatment Method",
                    options=["All"] + treat_methods,
                    key=f"method_filter_{selected_collection}"
                )
                if selected_method != "All":
                    filter_query["treat_method"] = selected_method
        
        with col_filter3:
            institutions = get_distinct_values(db, selected_collection, "treat_institution")
            if institutions:
                selected_institution = st.selectbox(
                    "Filter by Institution",
                    options=["All"] + institutions,
                    key=f"institution_filter_{selected_collection}"
                )
                if selected_institution != "All":
                    filter_query["treat_institution"] = selected_institution
    
    elif selected_collection == 'acquisitions':
        st.markdown("---")
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            sample_numbers = get_distinct_values(db, selected_collection, "sample_number")
            if sample_numbers:
                selected_sample = st.selectbox(
                    "Filter by Sample Number",
                    options=["All"] + sample_numbers,
                    key=f"sample_filter_{selected_collection}"
                )
                if selected_sample != "All":
                    filter_query["sample_number"] = selected_sample
        
        with col_filter2:
            techniques = get_distinct_values(db, selected_collection, "technique")
            if techniques:
                selected_technique = st.selectbox(
                    "Filter by Technique",
                    options=["All"] + techniques,
                    key=f"technique_filter_{selected_collection}"
                )
                if selected_technique != "All":
                    filter_query["technique"] = selected_technique
        
        with col_filter3:
            treatment_ids = get_distinct_values(db, selected_collection, "treatment_id")
            if treatment_ids:
                selected_treatment = st.selectbox(
                    "Filter by Treatment",
                    options=["All"] + treatment_ids,
                    key=f"treatment_filter_{selected_collection}"
                )
                if selected_treatment != "All":
                    filter_query["treatment_id"] = selected_treatment
    
    elif selected_collection in ['spectra', 'pixels', 'analyses']:
        st.markdown("---")
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            techniques = get_distinct_values(db, selected_collection, "technique")
            if techniques:
                selected_technique = st.selectbox(
                    "Filter by Technique",
                    options=["All"] + techniques,
                    key=f"technique_filter_{selected_collection}"
                )
                if selected_technique != "All":
                    filter_query["technique"] = selected_technique
        
        with col_filter2:
            treatment_ids = get_distinct_values(db, selected_collection, "treatment_id")
            if treatment_ids:
                selected_treatment = st.selectbox(
                    "Filter by Treatment",
                    options=["All"] + treatment_ids,
                    key=f"treatment_filter_{selected_collection}"
                )
                if selected_treatment != "All":
                    filter_query["treatment_id"] = selected_treatment
    
    elif selected_collection == 'methods':
        st.markdown("---")
        method_types = get_distinct_values(db, selected_collection, "method_type")
        if method_types:
            selected_method_type = st.selectbox(
                "Filter by Method Type",
                options=["All"] + method_types,
                key=f"method_type_filter_{selected_collection}"
            )
            if selected_method_type != "All":
                filter_query["method_type"] = selected_method_type
    
    # Recalculate total after filter
    if filter_query:
        try:
            total_docs = get_collection_count_with_filter(db, selected_collection, filter_query)
            st.write(f"Filtered Documents: {total_docs}")
        except Exception as e:
            st.error(f"Error applying filter: {str(e)}")
            return
    
    col1, col2 = st.columns(2)
    
    with col1:
        limit_option = st.selectbox(
            "Maximum documents to display",
            options=[1, 10, 100, 1000, "All"],
            key=f"limit_option_{selected_collection}"
        )
        limit = total_docs if limit_option == "All" else min(limit_option, total_docs)
    
    with col2:
        skip = st.number_input(
            "Skip first documents",
            min_value=0,
            max_value=max(0, total_docs - 1),
            value=0,
            step=1,
            key=f"skip_{selected_collection}"
        )
    
    st.markdown("---")
    
    try:
        documents = query_collection(db, selected_collection, skip=skip, limit=limit, filter_query=filter_query)
        
        if not documents:
            st.warning("No documents found in this range")
            return
        
        st.markdown(f"<h5>Showing {len(documents)} documents</h5>", unsafe_allow_html=True)
        st.markdown("---")
        
        for idx, doc in enumerate(documents, 1):
            with st.expander(f"Document {skip + idx}", expanded=(idx == 1)):
                st.json(doc)
    
    except Exception as e:
        st.error(f"Error querying collection: {str(e)}")


def get_distinct_values(db, collection_name: str, field_name: str) -> List[str]:
    """Get unique values for a specific field from collection"""
    try:
        values = db[collection_name].distinct(field_name)
        return sorted([str(v) for v in values if v])
    except Exception as e:
        st.warning(f"Error retrieving {field_name}: {str(e)}")
        return []


def get_collection_count_with_filter(db, collection_name: str, filter_query: Dict) -> int:
    """Get document count with filter applied"""
    try:
        return db[collection_name].count_documents(filter_query)
    except Exception as e:
        raise Exception(f"Error getting filtered collection count: {str(e)}")



def display_workflow_metrics(workflow: Dict):
    """Display workflow metadata metrics"""
    meta = workflow['metadata']
    
    # First row: Fabrication, Treatments, Acquisitions, Status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fabrication Steps", meta.get('total_fabrication_steps', 0))
    with col2:
        st.metric("Treatment Steps", meta.get('total_treatments', 0))
    with col3:
        st.metric("Acquisitions", meta.get('total_acquisitions', 0))
    with col4:
        pass
    
    # Second row: Pixel measurements
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Resistance Pixels", meta.get('total_resistance_pixels_measured', 0))
    with col6:
        st.metric("mPL Pixels", meta.get('total_mpl_pixels_measured', 0))
    with col7:
        st.metric("XRF Pixels", meta.get('total_xrf_pixels_measured', 0))
    with col8:
        pass
    
    # Third row: Spectra measurements
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        st.metric("PL Spectra", meta.get('total_pl_spectra_recorded', 0))
    with col10:
        st.metric("UV-Vis Spectra", meta.get('total_uv_vis_spectra_recorded', 0))
    with col11:
        st.metric("mPL Spectra", meta.get('total_mpl_spectra_recorded', 0))
    with col12:
        st.metric("XRF Spectra", meta.get('total_xrf_spectra_recorded', 0))


def display_workflow_details(workflow: Dict, sample_name: str):
    """Display detailed workflow information"""
    st.markdown(f"<h4>Complete Workflow of Sample: {sample_name}</h4>", unsafe_allow_html=True)
    
    display_workflow_metrics(workflow)
    
    with st.expander("View Full JSON", expanded=True):
        st.json(workflow)
    
    json_str = json.dumps(workflow, indent=2)
    st.download_button(
        label=" Download JSON",
        data=json_str,
        file_name=f"{sample_name}_workflow.json",
        mime="application/json"
    )
    
    st.markdown("---")
    
    # 1. Substrate
    if workflow.get('substrate'):
        st.markdown("<h5>I. Substrate Cleaning</h5>", unsafe_allow_html=True)
        substrate = workflow['substrate']
        st.write(f"**Operator:** {substrate.get('sub_operator', 'Unknown')} | "
                 f"**Institution:** {substrate.get('sub_institution', 'Unknown')}")
        with st.expander("View Substrate Details", expanded=False):
            st.json(substrate)
        st.markdown("---")
    
    # 2. Fabrications
    fabrications = workflow.get('fabrications', [])
    if fabrications:
        st.markdown(
            f"<h5>II. Fabrications - "
            f"<span style='color:#d32f2f; font-weight:700;'>{len(fabrications)} steps</span>"
            f"</h5>",
            unsafe_allow_html=True
        )
        
        for idx, fab in enumerate(fabrications, 1):
            fab_sequence = fab.get('fab_sequence', idx)
            st.write(f"**Step {fab_sequence}: {fab.get('fab_method', 'Unknown')}** | "
                    f"Operator: {fab.get('fab_operator', 'Unknown')} | "
                    f"Institution: {fab.get('fab_institution', 'Unknown')}")
            with st.expander(f"View Fabrication Step {fab_sequence} Details", expanded=False):
                st.json(fab)
        
        st.markdown("---")
    
    # 3. Treatments with Acquisitions
    treatments_with_acq = workflow.get('treatments_with_acquisitions', [])
    if treatments_with_acq:
        st.markdown(
            f"<h5>III. Treatments and Measurements – "
            f"<span style='color:#d32f2f; font-weight:700;'>{len(treatments_with_acq)} treatment(s)</span>"
            f"</h5>",
            unsafe_allow_html=True
        )
        
        for treat_data in treatments_with_acq:
            treatment = treat_data['treatment']
            acquisitions = treat_data['acquisitions']
            
            treat_sequence = treatment.get('treat_sequence', '0')
            treat_method = treatment.get('treat_method', 'Unknown')
            treat_operator = treatment.get('treat_operator', 'Unknown')
            treat_institution = treatment.get('treat_institution', 'Unknown')
            
            st.markdown(f"<h6>Treatment {treat_sequence}: {treat_method}</h6>", unsafe_allow_html=True)
            st.write(f"**Operator:** {treat_operator} | **Institution:** {treat_institution}")
            
            with st.expander(f"View Treatment {treat_sequence} Details", expanded=False):
                st.json(treatment)
            
            # Show acquisitions
            if acquisitions:
                st.markdown(
                    f"<h6>→ Acquisitions after Treatment {treat_sequence}: "
                    f"<span style='color:#d32f2f; font-weight:700;'>{len(acquisitions)} measurement(s)</span>"
                    f"</h6>",
                    unsafe_allow_html=True
                )
                
                for acq_idx, acq in enumerate(acquisitions, 1):
                    technique = acq.get('technique', 'Unknown').upper()
                    measurement_type = acq.get('measurement_type', 'N/A')
                    acq_id = acq.get('_id', 'Unknown')
                    
                    st.write(f"   {acq_idx}. {technique} - {measurement_type}")
                    with st.expander(f"View Acquisition Details: {acq_id}", expanded=False):
                        st.json(acq)
            else:
                st.write(f"**→ No acquisitions recorded after Treatment {treat_sequence}**")
            
            st.markdown("---")
    else:
        st.info("No treatments or acquisitions recorded yet")

def display_sample_workflow_browser(db):
    """Display sample workflow browser UI"""
    sample_numbers = get_all_sample_numbers(db)
    
    if not sample_numbers:
        st.warning("No samples found in database")
        return
    
    st.markdown("<h4>Select Sample Number</h4>", unsafe_allow_html=True)
    
    input_method = st.radio(
        "Input method",
        options=["Choose from list", "Type sample number"],
        horizontal=True,
        key="sample_input_method"
    )
    
    if input_method == "Type sample number":
        sample_to_query = st.text_input(
            "Enter sample number",
            placeholder="xxxx-xx",
            help="Enter sample number (e.g., 3716-15)",
            key="sample_manual_input"
        ).strip()
    else:
        select_options = ["Choose sample number"] + sample_numbers
        selected = st.selectbox("Select sample number", options=select_options, key="sample_selector")
        sample_to_query = "" if selected == "Choose sample number" else selected
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Search", type="primary", use_container_width=True):
            if sample_to_query:
                try:
                    workflow = get_sample_workflow(db, sample_to_query)
                    if workflow:
                        st.session_state.workflow_data = workflow
                        st.session_state.selected_sample = sample_to_query
                    else:
                        st.error(f"Sample {sample_to_query} not found")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.error("Please select or enter a sample number")
    
    with col2:
        if st.button("Clear", type="secondary", use_container_width=True):
            st.session_state.workflow_data = None
            st.session_state.selected_sample = ""
            st.rerun()
    
    st.markdown("---")
    
    # Display stored workflow
    if st.session_state.workflow_data:
        display_workflow_details(
            st.session_state.workflow_data,
            st.session_state.selected_sample
        )


# ==================== MAIN PAGE ====================

def page_collection_query(db):
    """Main collection query page with sample workflow option"""
    st.markdown('<div class="main-header">Complete Workflow Query</div>', unsafe_allow_html=True)
    
    if db is None:
        st.error("Database not connected")
        return
    
    # Initialize session state
    if 'workflow_data' not in st.session_state:
        st.session_state.workflow_data = None
    if 'selected_sample' not in st.session_state:
        st.session_state.selected_sample = ""
    
    query_mode = st.radio(
        "Select Query Mode",
        options=["Browse Collections", "Browse Sample Workflow"],
        horizontal=True,
        key="query_mode_radio"
    )
    
    st.markdown("---")
    
    if query_mode == "Browse Collections":
        st.markdown("<h5>Browse Collections</h5>", unsafe_allow_html=True)
        
        collections = ['libraries', 'substrates', 'fabrications', 'treatments', 
                      'acquisitions', 'pixels', 'spectra', 'analyses', 'methods']
        
        selected_collection = st.selectbox(
            "Select Collection",
            collections,
            key="collection_selector"
        )
        
        st.markdown("---")
        display_collection_browser(db, selected_collection)
    
    else:  # Sample Workflow mode
        display_sample_workflow_browser(db)