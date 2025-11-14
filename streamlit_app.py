import streamlit as st
from pymongo import MongoClient
from datetime import datetime, timezone

from sub_func import page_upload_substrate_data
from collection_query_func import page_collection_query
from fab_func import page_upload_fabrication_data
from treat_func import page_upload_treatment_data
from meas_func import page_upload_measurement_data
from analysis_visual_func import page_analysis


# ==================== PAGE CONFIG & STYLING ====================

st.set_page_config(
    page_title="High-throughput Measurement System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== DATABASE FUNCTIONS ====================

def connect_to_mongodb(connection_string):
    """Connect to MongoDB and verify connection"""
    try:
        client = MongoClient(connection_string)
        client.server_info()
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None


def create_library(db, sample_number):
    """Create a new library document"""
    try:
        if not sample_number.strip():
            st.error("Please enter a sample number")
            return False
        
        lib_id = f"{sample_number}_lib"
        
        if db['libraries'].find_one({"_id": lib_id}):
            st.warning("Library already exists. Users cannot create library more.")
            return False
        
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H.%M.%SZ")
        
        library_doc = {
            "_id": lib_id,
            "sample_number": sample_number,
            "sub_id": "",
            "fab_ids": [],
            "treat_ids": [],
            "created_date_time": current_time,
            "updated_date_time": current_time
        }
        
        db['libraries'].insert_one(library_doc)
        st.success("Library created successfully!")
        st.json(library_doc)
        return True
        
    except Exception as e:
        st.error(f"Error creating library: {str(e)}")
        return False


def delete_database(client, database_name):
    """Delete entire database"""
    try:
        client.drop_database(database_name)
        st.success("Database deleted successfully!")
        return True
    except Exception as e:
        st.error(f"Error deleting database: {str(e)}")
        return False


def delete_multiple_collections(db, collection_names):
    """Delete multiple collections from database"""
    try:
        deleted_count = 0
        failed_collections = []
        
        for coll_name in collection_names:
            try:
                db[coll_name].drop()
                deleted_count += 1
            except Exception as e:
                failed_collections.append({'name': coll_name, 'error': str(e)})
        
        if deleted_count > 0:
            st.success(f"Successfully deleted {deleted_count} collection(s)")
        
        if failed_collections:
            st.warning(f"Failed to delete {len(failed_collections)} collection(s)")
        
        return deleted_count, failed_collections
        
    except Exception as e:
        st.error(f"Error deleting collections: {str(e)}")
        return 0, collection_names


@st.cache_data(ttl=0.5)
def get_collection_counts(_db):
    """Get document counts for all collections"""
    collections = ['libraries', 'substrates', 'fabrications', 'treatments', 
                   'acquisitions', 'pixels', 'spectra', 'analyses', 'methods']
    return {coll: _db[coll].count_documents({}) for coll in collections}


# ==================== PAGE FUNCTIONS ====================

def page_home(db):
    """Display database statistics and upload data interface"""
    st.markdown('<div class="main-header">MongoDB Database Information</div>', unsafe_allow_html=True)
    
    if db is None:
        st.error("Database not connected!")
        return
    
    # === DATABASE STATISTICS SECTION ===
    try:
        counts = get_collection_counts(db)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics_layout = [
            (col1, [("Libraries", 'libraries'), ("Acquisitions", 'acquisitions')]),
            (col2, [("Substrates", 'substrates'), ("Spectra", 'spectra')]),
            (col3, [("Fabrications", 'fabrications'), ("Pixels", 'pixels')]),
            (col4, [("Treatments", 'treatments'), ("Analyses", 'analyses')]),
            (col5, [("Methods", 'methods')])
        ]
        
        for col, metrics in metrics_layout:
            with col:
                for label, key in metrics:
                    st.metric(label, counts[key])
    
    except Exception as e:
        st.warning(f"Could not retrieve collection counts: {str(e)}")
    
    st.markdown("---")
    
    # === UPLOAD DATA SECTION ===
    st.markdown('<div class="main-header">Upload Data</div>', unsafe_allow_html=True)
    
    st.subheader("Create Library")
    
    sample_number = st.text_input(
        "Sample Number",
        placeholder="xxxx-xx",
        help="Enter sample number (e.g., 3716-15)",
        key="create_lib_sample"
    )
    
    library_exists = False
    if sample_number.strip():
        lib_id = f"{sample_number}_lib"
        library_exists = db['libraries'].find_one({"_id": lib_id}) is not None
    
    if st.button("Create Library", type="primary", use_container_width=True, disabled=library_exists):
        if sample_number.strip():
            create_library(db, sample_number)
        else:
            st.error("Please enter a sample number")
    
    if library_exists:
        st.warning(f"Sample Number {sample_number} already exists.")
    
    col = st.columns(4)
    upload_pages = [
        page_upload_substrate_data,
        page_upload_fabrication_data,
        page_upload_treatment_data,
        page_upload_measurement_data
    ]
    
    for i, page_func in enumerate(upload_pages):
        with col[i]:
            page_func(db)


def main():
    """Main application entry point"""   
    with st.expander("MongoDB Configuration", expanded=False):

        st.header("MongoDB Configuration")

        col = st.columns(2)

        with col[0]:
            connection_string = st.text_input(
                "MongoDB Connection String",
                value="mongodb://localhost:27017/",
                type="password"
            )

        with col[1]:
            database_name = st.text_input(
                "Database Name",
                value="High_throughput"
            )

        col = st.columns(6)

        with col[0]:
            if st.button("Connect", type="primary", use_container_width=True):
                client = connect_to_mongodb(connection_string)
                if client:
                    st.session_state['mongo_client'] = client
                    st.session_state['database_name'] = database_name
                    st.success("Connected to MongoDB")

        # Deletion Options
        if 'mongo_client' in st.session_state:
            st.markdown("### Deletion Options")
            
            col_del = st.columns([1, 3, 1])
            
            with col_del[0]:
                st.markdown("**Select to Delete:**")
            
            with col_del[1]:
                # Checkbox for entire database
                delete_database_checkbox = st.checkbox(
                    f" Database: '{st.session_state['database_name']}' ",
                    key="delete_db_checkbox"
                )
                
                st.markdown("**Collections:**")
                col_grid = st.columns(3)
                
                collection_options = [
                    'libraries', 'substrates', 'fabrications', 
                    'treatments', 'acquisitions', 'pixels', 
                    'spectra', 'analyses', 'methods'
                ]
                
                # Store selected collections
                selected_collections = []
                
                for idx, coll_name in enumerate(collection_options):
                    with col_grid[idx % 3]:
                        if st.checkbox(f" {coll_name}", key=f"delete_{coll_name}"):
                            selected_collections.append(coll_name)
            
            with col_del[2]:
                st.markdown("**Action:**")
                # Show delete button only if something is selected
                if delete_database_checkbox or selected_collections:
                    
                    # Show what will be deleted
                    if delete_database_checkbox:
                        st.error(f"⚠️ Will delete entire database")
                    elif selected_collections:
                        st.warning(f"⚠️ Will delete {len(selected_collections)} collection(s)")
                    
                    if st.button(" DELETE", type="secondary", use_container_width=True):
                        client = st.session_state['mongo_client']
                        
                        if delete_database_checkbox:
                            # Delete entire database
                            if delete_database(client, st.session_state['database_name']):
                                del st.session_state['mongo_client']
                                st.rerun()
                        
                        elif selected_collections:
                            # Delete only selected collections
                            db = st.session_state['mongo_client'][st.session_state['database_name']]
                            delete_multiple_collections(db, selected_collections)
                            st.rerun()

    
    if 'mongo_client' in st.session_state:
        db = st.session_state['mongo_client'][st.session_state['database_name']]
        
        st.markdown("""
            <style>
            .stTabs [data-baseweb="tab-list"] button {
                font-size: 24px;
                font-weight: bold;
                padding: 16px;
            }
            .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
                color: #ff4b4b;
                border-bottom: 6px solid #ff4b4b;
            }
            </style>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Home", " Query", " Analysis"])
        
        with tab1:
            page_home(db)
        with tab2:
            page_collection_query(db)
        with tab3:
            page_analysis(db)
    else:
        st.info("Please connect to MongoDB first")


if __name__ == "__main__":
    main()