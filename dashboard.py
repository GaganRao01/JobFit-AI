# dashboard.py (Chunk 1)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

def show_dashboard():
    st.title("Indian Job Market Insights Dashboard")

    st.markdown("""
        This dashboard provides insights into the Indian job market.
        Use the filters on the sidebar to explore the data. The visualizations below will update dynamically.

        **Key Features (Job Postings):**

        *   **Job Posting Trends:** See how job postings change over time.
        *   **Top Job Titles:** Discover the most in-demand roles.
        *   **Location Analysis:** Identify the cities with the most job opportunities.
        *   **Company Insights:** Find out which companies are hiring the most.
        *   **Work Type Analysis:** Explore the prevalence of On-site, Remote, and Hybrid roles.
        *   **Sector Breakdown:** Analyze job postings across different industries.
        *   **Application Trends:** Analyze application trends.
        *   **Job Freshness:** Analyze job freshness.
    """)

    # Removed Back button from here

    @st.cache_data
    def load_data():
        try:
            st.info("Loading data from local CSV file...")
            script_dir = os.path.dirname(__file__)
            csv_file_path = os.path.join(script_dir, "LinkedIn_Jobs_Data_India.csv")
            data = pd.read_csv(csv_file_path)

            # --- Data Cleaning (Case-Sensitive) ---
            data.columns = data.columns.str.strip()  # Keep whitespace removal
            # NO lowercasing or renaming needed.  Use original column names.

            # Convert 'publishedAt' to datetime 
            try:
                data['publishedAt'] = pd.to_datetime(data['publishedAt'], errors='coerce')
            except Exception as e:
                st.warning("Date conversion issue. Check 'publishedAt' column in data.")
                st.write(f"Error details: {e}")

                data['publishedAt'] = pd.to_datetime(data['publishedAt'], errors='coerce')

            
            data.dropna(subset=['publishedAt'], inplace=True)
            data['applicationsCount'] = data['applicationsCount'].fillna(0)
            data['workType'] = data['workType'].fillna("Unspecified")
            data.dropna(subset=['sector'], inplace=True)
            # --- End Data Cleaning ---

            st.success("Data loaded successfully!")
            return data

        except FileNotFoundError:
            st.error(f"Error: CSV file not found at {csv_file_path}. Make sure the file is in the same directory as your script.")
            return None
        except pd.errors.ParserError:
            st.error("Error: Could not parse the CSV file.  Check for formatting issues (e.g., incorrect delimiters).")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None

    data = load_data()

    if data is None:
        st.error("Failed to load job data.")

        def generate_sample_data():
            st.warning("Using generated sample data since real data couldn't be loaded.")
            np.random.seed(42)
            job_titles = ['Data Scientist', 'Software Engineer', 'Product Manager', 'Data Analyst', 'UX Designer']
            locations = ['Bangalore', 'Hyderabad', 'Mumbai', 'Delhi', 'Chennai', 'Pune']
            companies = ['Tech Co', 'Data Corp', 'Startup Inc', 'BigTech', 'AI Solutions']
            work_types = ['On-site', 'Remote', 'Hybrid']
            sectors = ['Technology, Information and Internet', 'IT Services and IT Consulting', 'Financial Services']

            
            df = pd.DataFrame({
                'title': np.random.choice(job_titles, 1000),
                'companyName': np.random.choice(companies, 1000),  # Corrected
                'city': np.random.choice(locations, 1000),
                'publishedAt': pd.date_range(start='2023-01-01', end='2024-03-01', periods=1000), # Corrected
                'workType': np.random.choice(work_types, 1000), # Corrected
                'applicationsCount': np.random.randint(0, 500, 1000), # Corrected
                'sector': np.random.choice(sectors, 1000)
            })
            return df

        data = generate_sample_data()

        with st.expander("View Data Sample"):
            st.dataframe(data.head(10))

    # --- Sidebar Filters (Case-Sensitive) ---
    st.sidebar.header("Filters")

    # Date Range filter
    if 'publishedAt' in data.columns and pd.api.types.is_datetime64_any_dtype(data['publishedAt']):
        min_date = data['publishedAt'].min().date()
        max_date = data['publishedAt'].max().date()
        date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = date_range
            data = data[(data['publishedAt'].dt.date >= start_date) & (data['publishedAt'].dt.date <= end_date)]
    else:
        st.sidebar.warning("Date filter not available due to data issues.")

    # Job title filter
    if 'title' in data.columns:
        titles = sorted(data['title'].dropna().unique())
        if titles:
            selected_titles = st.sidebar.multiselect("Job Titles", options=titles, default=[])
            if selected_titles:
                data = data[data['title'].isin(selected_titles)]
        else:
            st.sidebar.write("No job titles available to filter.")
    else:
        st.sidebar.warning("Job Title filter not available due to data issues.")

    # Location filter
    if 'city' in data.columns:
        locations = sorted(data['city'].dropna().unique())
        if locations:
            selected_locations = st.sidebar.multiselect("Locations", options=locations, default=[])
            if selected_locations:
                data = data[data['city'].isin(selected_locations)]
        else:
            st.sidebar.write("No locations available to filter.")
    else:
        st.sidebar.warning("Location filter not available due to data issues.")

    # Work type filter
    if 'workType' in data.columns:
        work_types = sorted(data['workType'].dropna().unique())
        if work_types:
            selected_work_types = st.sidebar.multiselect("Work Types", options=work_types, default=[])
            if selected_work_types:
                data = data[data['workType'].isin(selected_work_types)]
        else:
            st.sidebar.write("No work types available to filter")
    else:
        st.sidebar.warning("Work Type filter not available due to data issues.")

    # Sector filter
    if 'sector' in data.columns:
        sectors = sorted(data['sector'].dropna().unique())
        if sectors:
            selected_sectors = st.sidebar.multiselect("Sectors", options=sectors, default=[])
            if selected_sectors:
                data = data[data['sector'].isin(selected_sectors)]
        else:
            st.sidebar.write("No Sectors available to filter.")
    else:
        st.sidebar.warning("Sector filter not available due to data issues.")

   

    # --- Visualizations (Case-Sensitive) ---
    st.header("Job Posting Insights")

    # --- Row 1: Basic metrics ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Job Postings", len(data))

    with col2:
        if not data.empty:
            top_company = data['companyName'].value_counts().index[0]
            st.metric("Top Hiring Company", top_company)
        else:
            st.metric("Top Hiring Company", "N/A")

    with col3:
        top_location = data['city'].value_counts().index[0] if not data['city'].empty else "N/A"
        st.metric("Top Location", top_location)

    # --- Row 2: Job posting trends ---
    st.subheader("Job Posting Trends")
    if 'publishedAt' in data.columns:
        data['month_year'] = data['publishedAt'].dt.to_period('M')
        monthly_posts = data.groupby('month_year').size().reset_index(name='count')
        monthly_posts['month_year'] = monthly_posts['month_year'].astype(str)
        fig = px.line(monthly_posts, x='month_year', y='count', title='Monthly Job Postings', labels={'month_year': 'Month', 'count': 'Number of Postings'}, color_discrete_sequence=px.colors.qualitative.Prism)  # Added color
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Job posting trends not available.")

    # --- Row 3: Application Trends ---
    st.subheader("Application Trends")
    avg_applications = data['applicationsCount'].mean()
    st.metric("Average Applications per Posting", f"{avg_applications:.2f}")

    if 'publishedAt' in data.columns:
        application_trends = data.groupby('month_year')['applicationsCount'].sum().reset_index()
        application_trends['month_year'] = application_trends['month_year'].astype(str)
        fig_apps = px.line(application_trends, x='month_year', y='applicationsCount', title='Total Applications Over Time', labels={'month_year': 'Month', 'applicationsCount': 'Total Applications'}, color_discrete_sequence=px.colors.qualitative.Vivid) # Added color
        st.plotly_chart(fig_apps, use_container_width=True)
    else:
        st.write("Application trend over time not available")

    # --- Row 4: Peak Posting Times ---
    st.subheader("Peak Posting Times")
    if 'publishedAt' in data.columns:
        data['day_of_week'] = data['publishedAt'].dt.day_name()
        #data['hour_of_day'] = data['publishedAt'].dt.hour 
        day_of_week_counts = data['day_of_week'].value_counts().reset_index()
        day_of_week_counts.columns = ['Day of Week', 'Count']
        
        fig_day = px.bar(day_of_week_counts, x='Day of Week', y='Count', title='Job Postings by Day of the Week', labels={'Count': 'Number of Postings'}, color='Day of Week', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_day, use_container_width=True)


        #hourly_counts = data['hour_of_day'].value_counts().sort_index().reset_index()
        #hourly_counts.columns = ['Hour of Day', 'Count'] 
        # Use a different color scale for hours of the day
        #fig_hour = px.bar(hourly_counts, x='Hour of Day', y='Count', title='Job Postings by Hour of the Day', labels={'Count': 'Number of Postings'}, color='Hour of Day', color_continuous_scale=px.colors.sequential.Viridis) 
        #st.plotly_chart(fig_hour, use_container_width=True) 
    else:
        st.write("Peak posting times are not available.")

    # --- Row 5: Job Freshness ---
    st.subheader("Job Freshness")
    if 'publishedAt' in data.columns:
        data['days_since_posted'] = (pd.Timestamp.now().normalize() - data['publishedAt'].dt.normalize()).dt.days
        average_days_old = data['days_since_posted'].mean()
        st.metric("Average Days Since Posted", f"{average_days_old:.1f} days")
        # Use a different color scale for the histogram
        #fig_freshness = px.histogram(data, x='days_since_posted', nbins=30, title='Distribution of Days Since Posted', labels={'days_since_posted': 'Days Since Posted'}, color_discrete_sequence=px.colors.sequential.RdBu) 
        #st.plotly_chart(fig_freshness, use_container_width=True) 
    else:
        st.write("Job freshness information not available.")

    # --- Row 6: Job distribution visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        if 'title' in data.columns:
            title_counts = data['title'].value_counts().head(10).reset_index()
            title_counts.columns = ['Title', 'Count']
            # Use color and a different orientation
            fig = px.bar(title_counts, x='Count', y='Title', title='Top 10 Job Titles', orientation='h', color='Title', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Job title distribution not available.")

    with col2:
        if 'city' in data.columns:
            loc_counts = data['city'].value_counts().head(10).reset_index()
            loc_counts.columns = ['Location', 'Count']
            fig = px.pie(loc_counts, values='Count', names='Location', title='Top Locations', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2) # Added Color
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Location distribution not available.")

    # Row 7: Company hiring the most
    st.subheader("Company Hiring The Most")
    if 'companyName' in data.columns:
        company_counts = data['companyName'].value_counts().head(10).reset_index()
        company_counts.columns = ['Company Name', 'Count']
        
        fig = px.bar(company_counts, x='Company Name', y='Count', title='Top 10 Hiring Companies', color='Company Name', color_discrete_sequence=px.colors.qualitative.Alphabet)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Top hiring companies information not available.")

    # --- Row 8: Work type distribution ---
    st.subheader("Work Type Distribution")
    if 'workType' in data.columns:
        work_type_counts = data['workType'].value_counts().reset_index()
        work_type_counts.columns = ['Work Type', 'Count']
        fig = px.pie(work_type_counts, values='Count', names='Work Type', title='Distribution of Work Types', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set1) #Added color
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Work type distribution is not available.")

    # Row 9: Sector-wise job distribution
    st.subheader("Sector-wise Job Distribution")
    if 'sector' in data.columns:
        sector_counts = data['sector'].value_counts().head(10).reset_index()
        sector_counts.columns = ['Sector', 'Count']
        fig = px.bar(sector_counts, x='Sector', y='Count', title='Top 10 Sectors with Job Postings', labels={'Sector': 'Sector', 'Count': 'Number of Postings'}, color='Sector', color_discrete_sequence=px.colors.qualitative.Dark24) # Added Color
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Sector-wise distribution is not available.")



def dashboard_page():
    show_dashboard()
