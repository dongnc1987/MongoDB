import streamlit as st
from datetime import datetime, timezone
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots



# ==================== HELPER FUNCTIONS ====================

def gaussian(x, amplitude, center, sigma, offset):
    """Gaussian function for fitting PL peak"""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset


def calculate_bandgap_ev(wavelength_nm):
    """Calculate bandgap in eV from wavelength in nm"""
    if wavelength_nm <= 0:
        return 0.0
    return round(1239.84 / wavelength_nm, 2)


def filter_valid_wavelength_range(wavelength, intensities, min_wl=400, max_wl=1100):
    """Filter data to keep only valid wavelength range for PL measurements"""
    wavelength = np.array(wavelength)
    intensities = np.array(intensities)
    valid_mask = (wavelength >= min_wl) & (wavelength <= max_wl)
    return wavelength[valid_mask], intensities[valid_mask]


def validate_pl_data(wavelength, intensities, debug=False):
    """Common validation for PL data"""
    wavelength = np.array(wavelength)
    intensities = np.array(intensities)
    
    if len(wavelength) < 10 or len(intensities) < 10:
        if debug:
            st.write(f"❌ Not enough data points: {len(wavelength)}")
        return None, None
    
    if debug:
        st.write(f"Original wavelength range: {wavelength.min():.2f} - {wavelength.max():.2f} nm")
    
    wavelength, intensities = filter_valid_wavelength_range(wavelength, intensities, min_wl=400, max_wl=1100)
    
    if len(wavelength) < 10:
        if debug:
            st.write(f"❌ Not enough data points after filtering: {len(wavelength)}")
        return None, None
    
    if debug:
        st.write(f"Filtered range: {wavelength.min():.2f} - {wavelength.max():.2f} nm")
        st.write(f"Intensity range: {intensities.min():.2f} - {intensities.max():.2f}")
    
    return wavelength, intensities


def validate_peak_metrics(peak_wavelength, peak_intensity, fwhm_nm, debug=False):
    """Validate peak metrics against physical constraints"""
    if peak_wavelength < 300 or peak_wavelength > 2000:
        if debug:
            st.write(f"❌ Peak wavelength out of range: {peak_wavelength:.2f} nm")
        return False
    
    if peak_intensity <= 0:
        if debug:
            st.write(f"❌ Peak intensity <= 0")
        return False
    
    if fwhm_nm < 5 or fwhm_nm > 300:
        if debug:
            st.write(f"❌ FWHM out of range: {fwhm_nm:.2f} nm")
        return False
    
    bandgap_ev = calculate_bandgap_ev(peak_wavelength)
    if bandgap_ev < 0.5 or bandgap_ev > 5:
        if debug:
            st.write(f"❌ Bandgap out of range: {bandgap_ev:.2f} eV")
        return False
    
    if debug:
        st.write(f"✓ Bandgap: {bandgap_ev:.2f} eV, FWHM: {fwhm_nm:.2f} nm")
    
    return True


# ==================== METHOD MANAGEMENT ====================

def initialize_methods_collection(db):
    """Initialize methods collection with predefined analysis methods"""
    methods_coll = db['methods']
    
    methods = {
        "method_gaussian_fit_v1": {
            "_id": "method_gaussian_fit_v1",
            "name": "Gaussian Fitting",
            "version": "1.0",
            "technique": "pl",
            "description": "Single Gaussian peak fitting with baseline offset.",
            "algorithm": "scipy.optimize.curve_fit",
            "parameters": {
                "function": "gaussian",
                "equation": "A * exp(-((x - μ)²) / (2σ²)) + offset",
                "fitted_params": ["amplitude A", "center μ", "sigma σ", "offset"],
                "derived_metrics": ["peak_wavelength_nm", "fwhm_nm", "peak_intensity", "bandgap_ev"],
                "quality_metric": "r_squared"
            },
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    }
    
    count = 0
    for method_id, method_doc in methods.items():
        result = methods_coll.update_one(
            {"_id": method_id},
            {"$set": method_doc},
            upsert=True
        )
        if result.upserted_id or result.modified_count > 0:
            count += 1
    
    return count


def get_available_methods(db, technique='pl'):
    """Get all available methods for a technique, initialize if empty"""
    methods_coll = db['methods']
    methods = list(methods_coll.find({"technique": technique}))
    
    if not methods:
        initialize_methods_collection(db)
        methods = list(methods_coll.find({"technique": technique}))
    
    return methods


# ==================== ANALYSIS FUNCTIONS ====================

def analyze_pl_with_gaussian_fit(wavelength, intensities, debug=False):
    """Analyze PL spectrum using Gaussian fitting"""
    try:
        wavelength, intensities = validate_pl_data(wavelength, intensities, debug)
        if wavelength is None:
            return False, None, None, None
        
        baseline = np.percentile(intensities, 10)
        intensities_corrected = np.maximum(intensities - baseline, 0)
        
        max_idx = np.argmax(intensities_corrected)
        initial_peak_wavelength = wavelength[max_idx]
        initial_peak_intensity = intensities_corrected[max_idx]
        
        if debug:
            st.write(f"Initial peak: λ={initial_peak_wavelength:.2f} nm, I={initial_peak_intensity:.4f}")
        
        if initial_peak_intensity <= 0 or initial_peak_intensity < np.max(intensities_corrected) * 0.1:
            if debug:
                st.write(f"❌ Peak intensity too low")
            return False, None, None, None
        
        # Estimate sigma from FWHM
        half_max = initial_peak_intensity / 2
        above_half = intensities_corrected > half_max
        if np.sum(above_half) > 2:
            indices_above = np.where(above_half)[0]
            width_estimate = wavelength[indices_above[-1]] - wavelength[indices_above[0]]
            initial_sigma = np.clip(width_estimate / 2.355, 5, 100)
        else:
            initial_sigma = 20
        
        initial_guess = [initial_peak_intensity, initial_peak_wavelength, initial_sigma, 0]
        lower_bounds = [0, wavelength.min(), 1, -initial_peak_intensity * 0.1]
        upper_bounds = [initial_peak_intensity * 3, wavelength.max(), 150, initial_peak_intensity * 0.1]
        
        popt, pcov = curve_fit(
            gaussian, wavelength, intensities_corrected,
            p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=5000
        )
        
        amplitude_fit, center_fit, sigma_fit, offset_fit = popt
        
        # Calculate R-squared
        fitted_curve = gaussian(wavelength, *popt)
        ss_res = np.sum((intensities_corrected - fitted_curve) ** 2)
        ss_tot = np.sum((intensities_corrected - np.mean(intensities_corrected)) ** 2)
        r_squared = 0 if ss_tot == 0 else 1 - (ss_res / ss_tot)
        
        fwhm_nm = 2.355 * abs(sigma_fit)
        
        if debug:
            st.write(f"Fitted: λ={center_fit:.2f} nm, A={amplitude_fit:.4f}, σ={sigma_fit:.2f}, R²={r_squared:.4f}")
        
        if not validate_peak_metrics(center_fit, amplitude_fit, fwhm_nm, debug):
            return False, None, None, None
        
        fit_params = {
            "amplitude": round(float(amplitude_fit), 4),
            "center": round(float(center_fit), 2),
            "sigma": round(float(abs(sigma_fit)), 2),
            "offset": round(float(offset_fit), 4),
            "r_squared": round(float(r_squared), 4)
        }
        
        derived_metrics = {
            "peak_wavelength_nm": round(float(center_fit), 2),
            "bandgap_ev": calculate_bandgap_ev(center_fit),
            "peak_intensity": round(float(amplitude_fit), 4),
            "fwhm_nm": round(float(fwhm_nm), 2)
        }
        
        return True, fit_params, derived_metrics, "method_gaussian_fit_v1"
        
    except Exception as e:
        if debug:
            st.write(f"❌ Exception: {str(e)}")
        return False, None, None, None


def analyze_pl_spectrum(wavelength, intensities, method_id="method_gaussian_fit_v1", debug=False):
    """Analyze PL spectrum using specified method"""
    result = {
        "success": False,
        "method_id": None,
        "derived": {},
        "fit_params": None
    }
    
    if method_id == "method_gaussian_fit_v1":
        success, fit_params, derived_metrics, method_id = analyze_pl_with_gaussian_fit(wavelength, intensities, debug)
        if success:
            result.update({"success": True, "method_id": method_id, "derived": derived_metrics, "fit_params": fit_params})
    
    return result


# ==================== DATABASE FUNCTIONS ====================

def create_pl_analysis(db, spectrum_id, acq_id, treatment_id, sample_number,
                      wavelength, intensities, method_id="method_gaussian_fit_v1"):
    """Create analysis document for PL spectrum"""
    try:
        analysis_result = analyze_pl_spectrum(wavelength, intensities, method_id)
        
        if not analysis_result["success"]:
            return False, f"Analysis failed for spectrum {spectrum_id}", None
        
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        analysis_id = f"{spectrum_id}_analysis"
        
        spectrum = db['spectra'].find_one({"_id": spectrum_id})
        position = {}
        if spectrum and 'position' in spectrum:
            position = {
                "x": spectrum['position'].get('x'),
                "y": spectrum['position'].get('y'),
                "unit": "mm"
            }
        
        analysis_doc = {
            "_id": analysis_id,
            "spectrum_id": spectrum_id,
            "acq_id": acq_id,
            "treatment_id": treatment_id,
            "sample_number": sample_number,
            "technique": "pl",
            "position": position,
            "method_id": analysis_result["method_id"],
            "derived_metrics": analysis_result["derived"],
            "created_date_time": current_time,
            "updated_date_time": current_time
        }
        
        if analysis_result["fit_params"]:
            analysis_doc["fit_params"] = analysis_result["fit_params"]
        
        existing_analysis = db['analyses'].find_one({"_id": analysis_id})
        if existing_analysis:
            analysis_doc["created_date_time"] = existing_analysis.get("created_date_time", current_time)
            db['analyses'].replace_one({"_id": analysis_id}, analysis_doc)
        else:
            db['analyses'].insert_one(analysis_doc)
        
        return True, "Analysis created successfully", analysis_id
    
    except Exception as e:
        return False, f"Error creating analysis: {str(e)}", None


def batch_analyze_pl_acquisition(db, acq_id, method_id="method_gaussian_fit_v1"):
    """Batch analyze all PL spectra for an acquisition"""
    try:
        spectra = list(db['spectra'].find({"acq_id": acq_id, "technique": "pl"}))
        
        if not spectra:
            return {
                "success": False,
                "message": f"No PL spectra found for acquisition {acq_id}",
                "total": 0,
                "analyzed": 0,
                "failed": 0
            }
        
        total = len(spectra)
        analyzed = 0
        failed = 0
        failed_spectra = []
        
        for spectrum in spectra:
            spectrum_id = spectrum["_id"]
            treatment_id = spectrum["treatment_id"]
            sample_number = spectrum_id.split("_")[0]
            
            wavelength = spectrum["x_axis"]["values"]
            intensities = spectrum["y"]["values"]
            
            success, message, analysis_id = create_pl_analysis(
                db, spectrum_id, acq_id, treatment_id, sample_number,
                wavelength, intensities, method_id
            )
            
            if success:
                analyzed += 1
            else:
                failed += 1
                failed_spectra.append(spectrum_id)
        
        return {
            "success": True,
            "message": f"Analyzed {analyzed}/{total} spectra",
            "total": total,
            "analyzed": analyzed,
            "failed": failed,
            "failed_spectra": failed_spectra
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Error in batch analysis: {str(e)}",
            "total": 0,
            "analyzed": 0,
            "failed": 0
        }


# ==================== VISUALIZATION FUNCTIONS ====================

def format_resistance_value(resistance_ohm):
    """Convert resistance to appropriate unit (Ω, MΩ, GΩ)"""
    if resistance_ohm is None or pd.isna(resistance_ohm):
        return None, None
    
    if resistance_ohm >= 1e9:
        return resistance_ohm / 1e9, "GΩ"
    elif resistance_ohm >= 1e6:
        return resistance_ohm / 1e6, "MΩ"
    else:
        return resistance_ohm, "Ω"


def determine_resistance_unit(resistance_values):
    """Determine the best unit for a series of resistance values"""
    # Remove None and NaN values
    valid_values = [r for r in resistance_values if r is not None and not pd.isna(r)]
    
    if not valid_values:
        return "Ω"
    
    median_value = np.median(valid_values)
    
    if median_value >= 1e9:
        return "GΩ"
    elif median_value >= 1e6:
        return "MΩ"
    else:
        return "Ω"


def convert_resistance_to_unit(resistance_ohm, target_unit):
    """Convert resistance from Ohm to target unit"""
    if resistance_ohm is None or pd.isna(resistance_ohm):
        return None
    
    if target_unit == "GΩ":
        return resistance_ohm / 1e9
    elif target_unit == "MΩ":
        return resistance_ohm / 1e6
    else:  # Ω
        return resistance_ohm


def create_heatmap(df, x_col, y_col, z_col, title, colorscale='Viridis', zmin=None, zmax=None, z_unit=None,
                   x_font_size=20, y_font_size=20):
    """Create a single heatmap from dataframe with square aspect ratio and consistent display"""
    if z_col not in df.columns or df[z_col].isna().all():
        # Return empty heatmap if no data
        fig = go.Figure()
        fig.add_annotation(
            text=f"No {title} data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            height=600
        )
        return fig
    
    # Pivot data for heatmap
    pivot_df = df.pivot(index=y_col, columns=x_col, values=z_col)
    
    # Sort indices for proper display
    pivot_df = pivot_df.sort_index(ascending=False)  # y-axis descending
    pivot_df = pivot_df.sort_index(axis=1)  # x-axis ascending
    
    # Prepare title with unit if provided
    display_title = f"{title} ({z_unit})" if z_unit else title
    hover_label = f"{title} ({z_unit})" if z_unit else title
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(
            thickness=15, 
            len=0.7,  # Full height of plot area
            yanchor="middle",
            y=0.5
        ),
        hovertemplate=f'x: %{{x}} mm<br>y: %{{y}} mm<br>{hover_label}: %{{z}}<extra></extra>',
        zsmooth=False, 
        xgap=0,  # Remove grid
        ygap=0   # Remove grid
    ))
    
    fig.update_layout(
        title=dict(text=display_title, font=dict(size=20)),
        xaxis_title="x (mm)",
        yaxis_title="y (mm)",
        height=600,  # Fixed height for square aspect
        width=600,   # Fixed width to ensure square
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(size=20)
    )
    
    # Set aspect ratio to be equal (square) and remove grid
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        constrain='domain',
        showgrid=False,  # Remove grid lines
        title_font=dict(size=y_font_size),
        tickfont=dict(size=y_font_size)
    )
    
    fig.update_xaxes(
        constrain='domain',
        showgrid=False,  # Remove grid lines
        title_font=dict(size=x_font_size),
        tickfont=dict(size=x_font_size)
    )
    
    return fig


# ====================================================================
#                       STREAMLIT USER INTERFACE
# ====================================================================

# ==================== PAGE OF ANALYSING DATABASE ====================

def page_pl_analysis(db):
    """Streamlit page for PL spectrum analysis"""
    st.markdown("<h5>PL Analysis</h5>", unsafe_allow_html=True)
    
    methods = get_available_methods(db, technique='pl')
    if not methods:
        st.error("❌ Failed to initialize analysis methods")
        return
    
    acquisitions = list(db['acquisitions'].find({"technique": "pl"}))
    if not acquisitions:
        st.warning("⚠️ No PL acquisitions found in database")
        return
    
    sample_numbers = sorted(list(set([acq['sample_number'] for acq in acquisitions])))
    
    selected_sample = st.selectbox("Select Sample Number", sample_numbers, index=0, key="analysis_sample_select")
    
    # Method selection
    method_options = {m['name']: m['_id'] for m in methods}
    selected_method_name = st.selectbox(
        "Select method for estimating bandgap",
        options=list(method_options.keys()),
        index=0,
        key="analysis_method_select"
    )
    selected_method_id = method_options[selected_method_name]
    
    if selected_sample:
        sample_acquisitions = [acq for acq in acquisitions if acq['sample_number'] == selected_sample]
        
        st.write("**Select Acquisitions to Analyze:**")
        
        selected_acq_ids = []
        for acq in sample_acquisitions:
            acq_label = f"{acq['_id']} - {acq['treatment_method']} (Seq {acq['treatment_sequence']}) - {acq['measurement_count']} spectra"
            if st.checkbox(acq_label, key=f"acq_checkbox_{acq['_id']}"):
                selected_acq_ids.append(acq['_id'])
        
        if selected_acq_ids:
            st.write(f"**{len(selected_acq_ids)} acquisition(s) selected**")
            
            if st.button("Analyze All Selected Acquisitions", type="primary", use_container_width=True):
                total_analyzed = 0
                total_failed = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, acq_id in enumerate(selected_acq_ids):
                    status_text.text(f"Analyzing {acq_id}...")
                    
                    results = batch_analyze_pl_acquisition(db, acq_id, selected_method_id)
                    
                    if results["success"]:
                        total_analyzed += results['analyzed']
                        total_failed += results['failed']
                    
                    progress_bar.progress((idx + 1) / len(selected_acq_ids))
                
                status_text.empty()
                progress_bar.empty()
                
                if total_analyzed > 0:
                    st.success(f"✓ Analysis Complete!")
                else:
                    st.error(f"❌ All analyses failed!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Analyzed", total_analyzed)
                with col2:
                    st.metric("Total Failed", total_failed)
                
                if total_failed > 0:
                    st.warning(f"⚠️ {total_failed} spectra failed. Enable Debug Mode to investigate.")


# ==================== COMBINED PAGE: VIEW RESULTS & VISUALIZATION ====================

def page_view_and_visualize_analyses(db):
    """Combined page for viewing analysis results in table and heatmaps"""
    st.markdown("<h5>View & Visualize Analysis Results</h5>", unsafe_allow_html=True)
    
    analyses = list(db['analyses'].find({"technique": "pl"}))
    
    if not analyses:
        st.warning("⚠️ No analyses found in database. Please run analysis first.")
        return
    
    st.write(f"**Total analyses:** {len(analyses)}")
    
    # Get unique sample numbers
    sample_numbers = sorted(list(set([a['sample_number'] for a in analyses])))
    selected_sample = st.selectbox("Select Sample Number", sample_numbers, index=0, key="view_sample_select")
    
    if not selected_sample:
        return
    
    # Filter analyses by sample
    sample_analyses = [a for a in analyses if a['sample_number'] == selected_sample]
    
    # Get treatment types from acquisitions
    acq_ids = list(set([a['acq_id'] for a in sample_analyses]))
    acquisitions = list(db['acquisitions'].find({"_id": {"$in": acq_ids}}))
    
    # Create treatment options (treatment_method + sequence)
    treatment_options = {}
    for acq in acquisitions:
        treatment = acq.get('treatment_method', 'Unknown')
        sequence = acq.get('treatment_sequence', 0)
        key = f"{treatment} (Seq {sequence})"
        treatment_options[key] = {"method": treatment, "sequence": sequence, "acq_id": acq['_id']}
    
    if not treatment_options:
        st.warning("⚠️ No treatment data found for this sample.")
        return
    
    selected_treatment = st.selectbox(
        "Select Treatment Type (Sequence)", 
        options=sorted(treatment_options.keys()), 
        index=0, 
        key="viz_treatment_select"
    )
    
    if not selected_treatment:
        return
    
    # Get the selected treatment details
    treatment_info = treatment_options[selected_treatment]
    selected_acq_id = treatment_info['acq_id']
    treatment_sequence = treatment_info['sequence']
    
    # Get method_id for the selected acquisition
    acq_analyses = [a for a in sample_analyses if a['acq_id'] == selected_acq_id]
    if not acq_analyses:
        st.warning(f"⚠️ No analyses found for {selected_treatment}")
        return
    
    method_id = acq_analyses[0]['method_id']
    
    st.write(f"**Visualizing {len(acq_analyses)} data points**")
    
    # Build dataframe with all required data
    df_data = []
    for a in acq_analyses:
        x_pos = y_pos = resistance = tau_attenuation = tau_phaseshift = None
        xrf_thickness = xrf_composition = None
        
        if 'position' in a:
            x_pos = a['position'].get('x')
            y_pos = a['position'].get('y')
            
            if x_pos is not None and y_pos is not None:
                # Get resistance FOR THE SAME TREATMENT SEQUENCE
                pixel_resistance = db['pixels'].find_one({
                    "position.x": x_pos,
                    "position.y": y_pos,
                    "technique": "resistance",
                    "treatment_id": a['treatment_id']
                })
                
                if pixel_resistance and 'resistance_ohm' in pixel_resistance:
                    resistance = pixel_resistance['resistance_ohm']
                
                # Get lifetime parameters FOR THE SAME TREATMENT SEQUENCE
                pixel_lifetime = db['pixels'].find_one({
                    "position.x": x_pos,
                    "position.y": y_pos,
                    "technique": "mpl",
                    "treatment_id": a['treatment_id']
                })
                
                if pixel_lifetime and 'lifetime_parameters' in pixel_lifetime:
                    tau_amp_s = pixel_lifetime['lifetime_parameters'].get('tau_amplitude_s')
                    tau_phase_s = pixel_lifetime['lifetime_parameters'].get('tau_phase_s')
                    
                    if tau_amp_s is not None:
                        tau_attenuation = round(tau_amp_s * 1e6, 2)  # s to μs
                    if tau_phase_s is not None:
                        tau_phaseshift = round(tau_phase_s * 1e6, 2)  # s to μs
                
                # Get XRF data FOR THE SAME TREATMENT SEQUENCE
                pixel_xrf = db['pixels'].find_one({
                    "position.x": x_pos,
                    "position.y": y_pos,
                    "technique": "xrf",
                    "treatment_id": a['treatment_id']
                })
                
                if pixel_xrf and 'layers' in pixel_xrf:
                    # Get first layer data (or you can iterate through all layers)
                    layers = pixel_xrf['layers']
                    if layers:
                        first_layer = list(layers.values())[0]
                        xrf_thickness = first_layer.get('thickness_nm')
                        xrf_composition = first_layer.get('composition_percent', {})
        
        # Get treatment information for this analysis
        acq = next((acq for acq in acquisitions if acq['_id'] == a['acq_id']), None)
        treatment_method = acq['treatment_method'] if acq else 'Unknown'
        treatment_sequence_val = acq['treatment_sequence'] if acq else 0
        
        row = {
            "Treatment": treatment_method,
            "Sequence": treatment_sequence_val,
            "x (mm)": x_pos,
            "y (mm)": y_pos,
            "Bandgap (eV)": a['derived_metrics'].get('bandgap_ev'),
            "Peak λ (nm)": a['derived_metrics'].get('peak_wavelength_nm'),
            "FWHM (nm)": a['derived_metrics'].get('fwhm_nm'),
            "Peak Intensity": a['derived_metrics'].get('peak_intensity'),
            "Resistance (Ω)": resistance,
            "τ_attenuation (μs)": tau_attenuation,
            "τ_phaseshift (μs)": tau_phaseshift,
            "XRF Thickness (nm)": xrf_thickness
        }
        
        # Add composition data dynamically
        if xrf_composition:
            for element, percent in xrf_composition.items():
                row[f"{element.capitalize()} (%)"] = percent
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Sort by Sequence (treatment order)
    df = df.sort_values('Sequence')
    
    # Determine best resistance unit and convert
    resistance_unit = determine_resistance_unit(df["Resistance (Ω)"].tolist())
    df["Resistance_converted"] = df["Resistance (Ω)"].apply(
        lambda x: convert_resistance_to_unit(x, resistance_unit)
    )
    
    st.markdown("---")
    
    # === VISUALIZATION SECTION ===
    st.subheader("Heatmap Visualizations")
   
    # Check if we have position data for visualization
    df_viz = df.dropna(subset=["x (mm)", "y (mm)"])
    
    if not df_viz.empty:
        # Row 1: PL metrics
        st.markdown("<h4>Analyses from PL</h4>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_bandgap = create_heatmap(
                df_viz, "x (mm)", "y (mm)", "Bandgap (eV)", 
                "Bandgap",
                colorscale='Plasma',
                z_unit="eV"
            )
            st.plotly_chart(fig_bandgap, use_container_width=True)
        
        with col2:
            fig_peak_wl = create_heatmap(
                df_viz, "x (mm)", "y (mm)", "Peak λ (nm)", 
                "Peak Wavelength",
                colorscale='Viridis',
                z_unit="nm"
            )
            st.plotly_chart(fig_peak_wl, use_container_width=True)
        
        with col3:
            fig_peak_int = create_heatmap(
                df_viz, "x (mm)", "y (mm)", "Peak Intensity", 
                "Peak Intensity",
                colorscale='YlOrRd',
                z_unit="a.u."
            )
            st.plotly_chart(fig_peak_int, use_container_width=True)
        
        # Row 2: Lifetime and Resistance - ONLY SHOW IF DATA EXISTS
        has_lifetime = ("τ_attenuation (μs)" in df_viz.columns and not df_viz["τ_attenuation (μs)"].isna().all()) or \
                       ("τ_phaseshift (μs)" in df_viz.columns and not df_viz["τ_phaseshift (μs)"].isna().all())
        has_resistance = "Resistance_converted" in df_viz.columns and not df_viz["Resistance_converted"].isna().all()
        
        if has_lifetime or has_resistance:
            st.markdown("<h4>Lifetime (estimated from mPL) and Resistance</h4>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            if has_lifetime and "τ_attenuation (μs)" in df_viz.columns and not df_viz["τ_attenuation (μs)"].isna().all():
                with col1:
                    fig_tau_att = create_heatmap(
                        df_viz, "x (mm)", "y (mm)", "τ_attenuation (μs)", 
                        "τ_attenuation",
                        colorscale='Greens',
                        z_unit="μs"
                    )
                    st.plotly_chart(fig_tau_att, use_container_width=True)
            
            if has_lifetime and "τ_phaseshift (μs)" in df_viz.columns and not df_viz["τ_phaseshift (μs)"].isna().all():
                with col2:
                    fig_tau_phase = create_heatmap(
                        df_viz, "x (mm)", "y (mm)", "τ_phaseshift (μs)", 
                        "τ_phaseshift",
                        colorscale='Purples',
                        z_unit="μs"
                    )
                    st.plotly_chart(fig_tau_phase, use_container_width=True)

            if has_resistance:
                with col3:
                    fig_resistance = create_heatmap(
                        df_viz, "x (mm)", "y (mm)", "Resistance_converted", 
                        "Resistance",
                        colorscale='Blues',
                        z_unit=resistance_unit
                    )
                    st.plotly_chart(fig_resistance, use_container_width=True)
        
        # Row 3: XRF data - ONLY SHOW IF DATA EXISTS
        has_xrf = "XRF Thickness (nm)" in df_viz.columns and not df_viz["XRF Thickness (nm)"].isna().all()
        
        if has_xrf:
            st.markdown("<h4>XRF Measurements</h4>", unsafe_allow_html=True)
            
            # Get composition columns dynamically
            composition_cols = [col for col in df_viz.columns if col.endswith(" (%)")]
            
            # Plot composition elements FIRST
            colorscales = ['Reds', 'Greens', 'Blues', 'Purples', 'YlOrBr', 'PuRd']
            
            if composition_cols:
                # Create rows of 3 columns for composition
                for row_start in range(0, len(composition_cols), 3):
                    row_elements = composition_cols[row_start:row_start + 3]
                    col1, col2, col3 = st.columns(3)
                    
                    # First element
                    if len(row_elements) > 0:
                        element_name = row_elements[0].replace(" (%)", "")
                        with col1:
                            fig_comp = create_heatmap(
                                df_viz, "x (mm)", "y (mm)", row_elements[0],
                                element_name,
                                colorscale=colorscales[row_start % len(colorscales)],
                                z_unit="%"
                            )
                            st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Second element
                    if len(row_elements) > 1:
                        element_name = row_elements[1].replace(" (%)", "")
                        with col2:
                            fig_comp = create_heatmap(
                                df_viz, "x (mm)", "y (mm)", row_elements[1],
                                element_name,
                                colorscale=colorscales[(row_start + 1) % len(colorscales)],
                                z_unit="%"
                            )
                            st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Third element
                    if len(row_elements) > 2:
                        element_name = row_elements[2].replace(" (%)", "")
                        with col3:
                            fig_comp = create_heatmap(
                                df_viz, "x (mm)", "y (mm)", row_elements[2],
                                element_name,
                                colorscale=colorscales[(row_start + 2) % len(colorscales)],
                                z_unit="%"
                            )
                            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Plot thickness LAST in its own row
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_thickness = create_heatmap(
                    df_viz, "x (mm)", "y (mm)", "XRF Thickness (nm)",
                    "Thickness",
                    colorscale='Oranges',
                    z_unit="nm"
                )
                st.plotly_chart(fig_thickness, use_container_width=True)

    else:
        st.warning("⚠️ No position data available for heatmap visualization")
    
    st.markdown("---")
    
    # === DETAILED RESULTS TABLE SECTION - SHOWING ALL TREATMENTS ===
    st.subheader("Detailed Results Table")

    # Get all treatments for this sample (not just selected one)
    all_acq_analyses = sample_analyses  # Use all sample analyses

    # Build comprehensive dataframe with all treatments
    all_df_data = []
    for a in all_acq_analyses:
        x_pos = y_pos = resistance = tau_attenuation = tau_phaseshift = None
        xrf_thickness = xrf_composition = None
        
        if 'position' in a:
            x_pos = a['position'].get('x')
            y_pos = a['position'].get('y')
            
            if x_pos is not None and y_pos is not None:
                # Get resistance FOR THE SAME TREATMENT SEQUENCE
                pixel_resistance = db['pixels'].find_one({
                    "position.x": x_pos,
                    "position.y": y_pos,
                    "technique": "resistance",
                    "treatment_id": a['treatment_id']
                })
                
                if pixel_resistance and 'resistance_ohm' in pixel_resistance:
                    resistance = pixel_resistance['resistance_ohm']
                
                # Get lifetime parameters FOR THE SAME TREATMENT SEQUENCE
                pixel_lifetime = db['pixels'].find_one({
                    "position.x": x_pos,
                    "position.y": y_pos,
                    "technique": "mpl",
                    "treatment_id": a['treatment_id']
                })
                
                if pixel_lifetime and 'lifetime_parameters' in pixel_lifetime:
                    tau_amp_s = pixel_lifetime['lifetime_parameters'].get('tau_amplitude_s')
                    tau_phase_s = pixel_lifetime['lifetime_parameters'].get('tau_phase_s')
                    
                    if tau_amp_s is not None:
                        tau_attenuation = round(tau_amp_s * 1e6, 2)
                    if tau_phase_s is not None:
                        tau_phaseshift = round(tau_phase_s * 1e6, 2)
                
                # Get XRF data FOR THE SAME TREATMENT SEQUENCE
                pixel_xrf = db['pixels'].find_one({
                    "position.x": x_pos,
                    "position.y": y_pos,
                    "technique": "xrf",
                    "treatment_id": a['treatment_id']
                })
                
                if pixel_xrf and 'layers' in pixel_xrf:
                    layers = pixel_xrf['layers']
                    if layers:
                        first_layer = list(layers.values())[0]
                        xrf_thickness = first_layer.get('thickness_nm')
                        xrf_composition = first_layer.get('composition_percent', {})
        
        # Get treatment information
        acq = next((acq for acq in acquisitions if acq['_id'] == a['acq_id']), None)
        treatment_method = acq['treatment_method'] if acq else 'Unknown'
        treatment_sequence_val = acq['treatment_sequence'] if acq else 0
        
        row = {
            "Treatment": treatment_method,
            "Sequence": treatment_sequence_val,
            "x (mm)": x_pos,
            "y (mm)": y_pos,
            "Bandgap (eV)": a['derived_metrics'].get('bandgap_ev'),
            "Peak λ (nm)": a['derived_metrics'].get('peak_wavelength_nm'),
            "FWHM (nm)": a['derived_metrics'].get('fwhm_nm'),
            "Peak Intensity": a['derived_metrics'].get('peak_intensity'),
            "Resistance (Ω)": resistance,
            "τ_attenuation (μs)": tau_attenuation,
            "τ_phaseshift (μs)": tau_phaseshift,
            "XRF Thickness (nm)": xrf_thickness
        }
        
        # Add composition data dynamically
        if xrf_composition:
            for element, percent in xrf_composition.items():
                row[f"{element.capitalize()} (%)"] = percent
        
        all_df_data.append(row)

    all_df = pd.DataFrame(all_df_data)

    # Sort by Sequence (treatment order) first, then by position
    all_df = all_df.sort_values(['Sequence', 'x (mm)', 'y (mm)'])

    # Convert resistance for display with appropriate unit
    all_df["Resistance_converted"] = all_df["Resistance (Ω)"].apply(
        lambda x: convert_resistance_to_unit(x, resistance_unit)
    )

    # Create resistance column with unit label
    all_df[f"Resistance ({resistance_unit})"] = all_df["Resistance_converted"]

    # Prepare display columns (dynamically include composition columns)
    base_columns = [
        "Treatment", "Sequence", "x (mm)", "y (mm)", 
        "Bandgap (eV)", "Peak λ (nm)", "FWHM (nm)", "Peak Intensity",
        f"Resistance ({resistance_unit})", "τ_attenuation (μs)", "τ_phaseshift (μs)",
        "XRF Thickness (nm)"
    ]
    
    # Add composition columns that exist in dataframe
    composition_cols = [col for col in all_df.columns if col.endswith(" (%)")]
    display_columns = base_columns + composition_cols
    
    # Filter to only existing columns
    display_columns = [col for col in display_columns if col in all_df.columns]
    
    display_df = all_df[display_columns].copy()

    # Display with proper handling of None values
    st.dataframe(display_df, use_container_width=True)

    # CSV download
    csv = display_df.fillna('').to_csv(index=False)
    st.download_button(
        " Download Detailed Results as CSV",
        csv,
        f"analysis_{selected_sample}_all_treatments.csv",
        "text/csv",
        use_container_width=True
    )


    # === SUMMARY STATISTICS SECTION - BY TREATMENT ===
    st.subheader("Summary Statistics")
    
    # Group by treatment sequence and calculate statistics
    summary_by_treatment = []
    
    for seq in sorted(all_df['Sequence'].unique()):
        treatment_df = all_df[all_df['Sequence'] == seq]
        treatment_name = treatment_df['Treatment'].iloc[0]
        
        # Build summary columns including XRF data
        summary_cols = [
            ("Bandgap (eV)", "eV"),
            ("Peak λ (nm)", "nm"),
            ("FWHM (nm)", "nm"),
            ("Peak Intensity", "a.u."),
            ("Resistance_converted", resistance_unit),
            ("τ_attenuation (μs)", "μs"),
            ("τ_phaseshift (μs)", "μs"),
            ("XRF Thickness (nm)", "nm")
        ]
        
        # Add composition columns
        for col in composition_cols:
            summary_cols.append((col, "%"))
        
        for col, unit in summary_cols:
            if col in treatment_df.columns:
                valid_data = treatment_df[col].dropna()
                if not valid_data.empty:
                    # Get metric name for display
                    metric_name = col.replace("_converted", "").replace(" (eV)", "").replace(" (nm)", "").replace(" (μs)", "").replace(" (%)", "")
                    if col == "Resistance_converted":
                        metric_name = "Resistance"
                    elif col == "Peak Intensity":
                        metric_name = "Peak Intensity"
                    elif col == "XRF Thickness (nm)":
                        metric_name = "XRF Thickness"
                    
                    summary_by_treatment.append({
                        "Treatment": treatment_name,
                        "Sequence": seq,
                        "Metric": f"{metric_name} ({unit})",
                        "Mean": round(valid_data.mean(), 3),
                        "Std Dev": round(valid_data.std(), 3),
                        "Min": round(valid_data.min(), 3),
                        "Max": round(valid_data.max(), 3),
                        "Count": len(valid_data)
                    })
    
    if summary_by_treatment:
        summary_df = pd.DataFrame(summary_by_treatment)
        st.dataframe(summary_df, use_container_width=True)
        
        # CSV download for summary statistics
        summary_csv = summary_df.to_csv(index=False)
        st.download_button(
            " Download Summary Statistics as CSV",
            summary_csv,
            f"summary_{selected_sample}_all_treatments.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("No data available for summary statistics")





# ==================== UPDATED MAIN PAGE ====================

def page_analysis(db):
    """Main analysis page with radio button selection - UPDATED"""
    st.markdown('<div class="main-header">Analysis</div>', unsafe_allow_html=True)
    
    if db is None:
        st.error("Database not connected!")
        return
    
    analysis_mode = st.radio(
        "Choose Action",
        options=["Run Analysis", "View Results"],
        horizontal=True,
        key="analysis_mode_radio"
    )
    
    st.markdown("---")
    
    if analysis_mode == "Run Analysis":
        page_pl_analysis(db)
    else:  # View Results
        page_view_and_visualize_analyses(db)
