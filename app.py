import streamlit as st
import pandas as pd
import base64
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
import io
import os.path
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
st.set_page_config(page_title="ModelMasterMind", page_icon="&#xF28C;")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def set_theme(theme):
    if theme == "Light":
        st.set_theme("light")
    elif theme == "Dark":
        st.set_theme("dark")

# Add a selectbox widget to the sidebar that allows the user to choose a theme
theme = st.sidebar.selectbox("Choose a theme", ["Light", "Dark"])

# Add a button to the sidebar that allows the user to apply the selected theme
if st.sidebar.button("Apply Theme"):
    set_theme(theme)
def download_report(df,filename, suffix=""):
    pr = ProfileReport(df, explorative=True)
    filename, extension = os.path.splitext(filename)
    new_filename = f"{filename}{suffix}{extension}"
    pr.to_file("report.html")
    with open("report.html", "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'data:file/html;base64,{b64}'
        return href, new_filename
    return href
def download_file(df, filename, suffix=""):
    filename, extension = os.path.splitext(filename)
    new_filename = f"{filename}{suffix}{extension}"
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'data:file/csv;base64,{b64}'
    return href, new_filename

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def main():
    

    selected_option1 = st.sidebar.selectbox("Select an Option", ["DashBoard","DP Tool", "AutoML Tool", "HPO Tool", "EDA Tool"])
    if selected_option1 == "DP Tool":
        st.title("Data Preprocessing Tool")
        st.write("This tool is clean, transform, and prepare your data for analysis.")
        st.write("**Data Cleaning:** Data preprocessing tools provide you options like remove missing values, Handle outliers, remove duplicates for data cleaning")
        st.write("Data Transformation:** Data preprocessing tools provide you options like Changing Data Types,Encode Categorical Variables, Scale Features for data transformation")
        # Data input
        st.write("**Data Input:**")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(df)
            except pd.errors.EmptyDataError:
                st.error("The selected file does not contain any data.")
                st.stop()
            if df.empty:
                st.error("The selected file does not contain any columns.")
                st.stop()
            # Preprocessing functions
            st.sidebar.header("Preprocessing Options")
            selected_option2 = st.sidebar.selectbox("Select an Process:", ["Select the Process","Data Cleaning","Data Transformation"])
            if selected_option2 == "Select an Process":
                st.warning("Please select the Process.")
            elif selected_option2 == "Data Cleaning":
                selected_option = st.sidebar.selectbox("Select an Method", ["Select an Method", "Missing Values", "Duplicate Rows", "Handle Outliers"])
                if selected_option == "Select an Method":
                    st.warning("Please select the Method.")
                elif selected_option == "Missing Values":
                    st.header("Handling Missing Values")
                    null_cols = df.columns[df.isnull().any()]
                    st.write("Columns with Missing Values:")
                    st.write(null_cols)
                    impute_options = ["KNN Imputer","Mean", "Median", "Mode"]
                    impute_method = st.selectbox("Select an Imputation Method", impute_options)
                    if st.button("Impute Missing Values"):
                        if impute_method == "KNN Imputer":
                            k = st.slider("Select Number of Neighbors (k)", min_value=1, max_value=10)
                            imputer = KNNImputer(n_neighbors=k)
                            df[null_cols] = imputer.fit_transform(df[null_cols])
                            st.write(df)
                            href, new_filename = download_file(df, uploaded_file.name, "_with_imputed_missing_value")
                            st.download_button(
                                label="Download File",
                                data=href,
                                file_name=new_filename,
                                mime="text/csv"
                                )
                        elif impute_method == "Mean":
                            df[null_cols] = df[null_cols].fillna(df.mean())
                            href, new_filename = download_file(df, uploaded_file.name, "_with_imputed_missing_value")
                            st.write(df)
                            href, new_filename = download_file(df, uploaded_file.name, "_with_imputed_missing_value")
                            st.download_button(
                                label="Download File",
                                data=href,
                                file_name=new_filename,
                                mime="text/csv"
                                )
                        elif impute_method == "Median":
                            df[null_cols] = df[null_cols].fillna(df.median())
                            st.write(df)
                            href, new_filename = download_file(df, uploaded_file.name, "_with_imputed_missing_value")
                            st.download_button(
                                label="Download File",
                                data=href,
                                file_name=new_filename,
                                mime="text/csv"
                                )
                        elif impute_method == "Mode":
                            df[null_cols] = df[null_cols].fillna(df.mode().iloc[0])
                            st.write(df)
                        href, new_filename = download_file(df, uploaded_file.name, "_with_imputed_missing_value")
                        st.download_button(
                            label="Download File",
                            data=href,
                            file_name=new_filename,
                            mime="text/csv"
                
                            )
                elif selected_option == "Duplicate Rows":
                    st.header("Removing Duplicate Rows")

                    num_duplicates = df.duplicated().sum()
                    st.write("Number of Duplicate Rows:", num_duplicates)
                    if st.button("Remove Duplicate Rows"):
                        df.drop_duplicates(inplace=True)
                        st.write(df)
                        href, new_filename = download_file(df, uploaded_file.name, "_without_Dup_row")
                        st.download_button(
                            label="Download File",
                            data=href,
                            file_name=new_filename,
                            mime="text/csv"
                
                            )
                elif selected_option == "Handle Outliers":
                    st.header("Handling Outliers")
    
                    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
                    selected_num_col = st.selectbox("Select a Numeric Column", num_cols)
                    outlier_options = ["Remove Outliers", "Cap Outliers"]
                    outlier_method = st.selectbox("Select an Outlier Handling Method", outlier_options)
    
                    if st.button("Handle Outliers"):
                        if outlier_method == "Remove Outliers":
                            q1 = df[selected_num_col].quantile(0.25)
                            q3 = df[selected_num_col].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - (1.5 * iqr)
                            upper_bound = q3 + (1.5 * iqr)
                            st.write(df)
                            df = df[(df[selected_num_col] > lower_bound) & (df[selected_num_col] < upper_bound)]
                            href, new_filename = download_file(df, uploaded_file.name, "_Outlierremoved")
                            st.download_button(
                                label="Download File",
                                data=href,
                                file_name=new_filename,
                                mime="text/csv"
                                )
                        elif outlier_method == "Cap Outliers":
                            q1 = df[selected_num_col].quantile(0.25)
                            q3 = df[selected_num_col].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - (1.5 * iqr)
                            upper_bound = q3 + (1.5 * iqr)
                            df[selected_num_col] = df[selected_num_col].clip(lower=lower_bound, upper=upper_bound)
                            st.write(df)
                            href, new_filename = download_file(df, uploaded_file.name, "_Cap_outliers")
                            st.download_button(
                                label="Download File",
                                data=href,
                                file_name=new_filename,
                                mime="text/csv"
                                )        
            

            elif selected_option2 == "Data Transformation":
                selected_option = st.sidebar.selectbox("Select an Method", ["Select an Method", "Changing Data Types","Encode Categorical Variables", "Scale Features"])
                if selected_option == "Select an Method":
                    st.warning("Please select the Method.")
                elif selected_option == "Changing Data Types":
                    st.header("Changing Data Types")
                    if df.select_dtypes(include=['int64', 'float64']).columns.tolist():
                        dtypes = df.select_dtypes(include=['int', 'float']).dtypes
                        st.write("Column Data Types:")
                        st.write(dtypes)
                        col_options = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        selected_col = st.selectbox("Select a Column", col_options)
                        new_dtype = st.selectbox("Select a New Data Type", ["int64", "float64"])
                        if st.button("Change Data Type"):
                            df[selected_col] = df[selected_col].astype(new_dtype)
                            st.write(df)
                            href, new_filename = download_file(df, uploaded_file.name, "_Changed_Datatype")
                            st.download_button(
                                label="Download File",
                                data=href,
                                file_name=new_filename,
                                mime="text/csv"
                                )
                        else:
                            st.write("The CSV file does not contain any columns with Integer and Float data type.")
                            st.write("Changing of Dataypes is not possible.")
                elif selected_option == "Encode Categorical Variables":
                    st.header("Encoding Categorical Variables")
                    if df.select_dtypes(include=['object']).columns.tolist():
                        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
                        selected_cat_col = st.selectbox("Select a Categorical Column", cat_cols)
                        encoding_options = ["Label Encoding", "One-Hot Encoding"]
                        encoding_method = st.selectbox("Select an Encoding Method", encoding_options)
                        if st.button("Encode"):
                            if encoding_method == "Label Encoding":
                                df[selected_cat_col] = df[selected_cat_col].astype("category").cat.codes
                                st.write(df)
                                href, new_filename = download_file(df, uploaded_file.name, "_Changed_Datatype")
                                st.download_button(
                                    label="Download File",
                                    data=href,
                                    file_name=new_filename,
                                    mime="text/csv"
                                    )
                            elif encoding_method == "One-Hot Encoding":
                                df = pd.get_dummies(df, columns=[selected_cat_col])
                                st.write(df)
                                st.download_button(
                                    label="Download File",
                                    data=href,
                                    file_name=new_filename,
                                    mime="text/csv"
                                    )
                            else:
                                st.write("The CSV file does not contain any columns with object data type.")
                                st.write("Encoding of Categorical Variables is not possible.")   
                elif selected_option == "Scale Features":
                    st.header("Scaling Features")
                    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
                    selected_num_col = st.selectbox("Select a Numeric Column", num_cols)
                    scaling_options = ["Standard Scaler", "Min-Max Scaler"]
                    scaling_method = st.selectbox("Select a Scaling Method", scaling_options)
                    if st.button("Scale"):
                        if scaling_method == "Standard Scaler":
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            df[selected_num_col] = scaler.fit_transform(df[[selected_num_col]])
                            st.write(df)
                            href, new_filename = download_file(df, uploaded_file.name, "_Standard Scaling")
                            st.download_button(
                                label="Download File",
                                data=href,
                                file_name=new_filename,
                                mime="text/csv"
                                )
                        elif scaling_method == "Min-Max Scaler":
                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            df[selected_num_col] = scaler.fit_transform(df[[selected_num_col]])
                            st.write(df)
                            href, new_filename = download_file(df, uploaded_file.name, "_MinMaxScaling")
                            st.download_button(
                                label="Download File",
                                data=href,
                                file_name=new_filename,
                                mime="text/csv"
                                )
        
            
                    
    elif selected_option1 ==  "HPO Tool":
        st.title("Hyperparameter Optimization Tools")
        st.write("""
        The RandomForestRegressor() function is used in this tool for build a regression model using the Random Forest algorithm.
        """)    
        st.write("**Data Input:**")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        st.sidebar.header('Set Parameters')
        split_size =  st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        st.sidebar.subheader('Learning Parameters')
        parameter_n_estimators =st.sidebar.slider('Number of estimators (n_estimators)', 0, 500, (10,50), 50)
        parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
        st.sidebar.write('---')
        parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1,3), 1)
        st.sidebar.number_input('Step size for max_features', 1)
        st.sidebar.write('---')        
        parameter_min_samples_split = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
        parameter_min_samples_leaf =st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)
        st.sidebar.subheader('General Parameters')
        parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
        parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
        parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
        parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
        parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])
        n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1]+parameter_n_estimators_step, parameter_n_estimators_step)
        max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1]+1, 1)
        param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)   
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(df)
            except pd.errors.EmptyDataError:
                st.error("The selected file does not contain any data.")
                st.stop()
            if df.empty:
                st.error("The selected file does not contain any columns.")
                st.stop()
            X = df.iloc[:,:-1] # Using all column except for the last column as X
            Y = df.iloc[:,-1] # Selecting the last column as Y
            st.markdown('A model is being built to predict the following **Y** variable:')
            st.info(Y.name)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_size)
            rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
                                   random_state=parameter_random_state,
                                   max_features=parameter_max_features,
                                   criterion=parameter_criterion,
                                   min_samples_split=parameter_min_samples_split,
                                   min_samples_leaf=parameter_min_samples_leaf,
                                   bootstrap=parameter_bootstrap,
                                   oob_score=parameter_oob_score,
                                   n_jobs=parameter_n_jobs)
            grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
            grid.fit(X_train, Y_train)
            st.subheader('Model Performance')
            Y_pred_test = grid.predict(X_test)
            st.write('Coefficient of determination ($R^2$):')
            st.info( r2_score(Y_test, Y_pred_test) )
            st.write('Error (MSE or MAE):')
            st.info( mean_squared_error(Y_test, Y_pred_test) )
            st.write("The best parameters are %s with a score of %0.2f"
                     % (grid.best_params_, grid.best_score_))
            st.subheader('Model Parameters')
            st.write(grid.get_params())
            grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
            grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
            grid_reset = grid_contour.reset_index()
            grid_reset.columns = ['max_features', 'n_estimators', 'R2']
            grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
            x = grid_pivot.columns.levels[1].values
            y = grid_pivot.index.values
            z = grid_pivot.values
            layout = go.Layout(
                xaxis=go.layout.XAxis(
                title=go.layout.xaxis.Title(
                text='n_estimators')
                ),
                yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(
                text='max_features')
                ) )
            fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
            fig.update_layout(title='Hyperparameter tuning',
                              scene = dict(
                xaxis_title='n_estimators',
                yaxis_title='max_features',
                zaxis_title='R2'),
                autosize=False,
                width=800, height=800,
                margin=dict(l=65, r=50, b=65, t=90))
            st.plotly_chart(fig)
            x = pd.DataFrame(x)
            y = pd.DataFrame(y)
            z = pd.DataFrame(z)
            df = pd.concat([x,y,z], axis=1)
            href, new_filename = download_file(df, uploaded_file.name, "Hy_param_tuning")
            st.download_button(
                label="Download File",
                data=href,
                file_name=new_filename,
                mime="text/csv"
                )
    elif selected_option1=="AutoML Tool":
        st.title("AutoML")
        st.markdown("This is The Machine Learning Algorithm Comparison App")
        st.markdown("The lazypredict library is used for building several machine learning models at once.")
        st.write("**Data Input:**")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])       
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(df)
            except pd.errors.EmptyDataError:
                st.error("The selected file does not contain any data.")
                st.stop()
            if df.empty:
                st.error("The selected file does not contain any columns.")
                st.stop()
            df = df.loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            X = df.iloc[:,:-1] # Using all column except for the last column as X
            Y = df.iloc[:,-1] # Selecting the last column as Y
            st.markdown('**1.2. Dataset dimension**')
            st.write('X')
            st.info(X.shape)
            st.write('Y')
            st.info(Y.shape)
            st.markdown('**1.3. Variable details**:')
            st.write('X variable (first 20 are shown)')
            st.info(list(X.columns[:20]))
            st.write('Y variable')
            st.info(Y.name)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
            reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
            models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
            models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)
            st.subheader('2. Table of Model Performance')
            st.write('Training set')
            st.write(predictions_train)
            href, new_filename = download_file(df, uploaded_file.name, "_training_set")
            st.download_button(
                label="Download Training Set File",
                data=href,
                file_name=new_filename,
                mime="text/csv"
                )
            st.write('Test set')
            st.write(predictions_test)
            href, new_filename = download_file(df, uploaded_file.name, "_Test_set")
            st.download_button(
                label="Download Test Set File",
                data=href,
                file_name=new_filename,
                mime="text/csv"
                )
            st.header('3. Plot of Model Performance (Test set)')
            with st.subheader('3.1 R-squared'):
                predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"] ]
                plt.figure(figsize=(3, 9))
                sns.set_theme(style="whitegrid")
                ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
                ax1.set(xlim=(0, 1))
            st.markdown(imagedownload(plt,'plot-r2-tall.pdf'), unsafe_allow_html=True)
            plt.figure(figsize=(9, 3))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
            ax1.set(ylim=(0, 1))
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.markdown(imagedownload(plt,'plot-r2-wide.pdf'), unsafe_allow_html=True)
            with st.subheader('3.2 RMSE (capped at 50)'):
                predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"] ]
                plt.figure(figsize=(3, 9))
                sns.set_theme(style="whitegrid")
                ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
            st.markdown(imagedownload(plt,'plot-rmse-tall.pdf'), unsafe_allow_html=True)
            plt.figure(figsize=(9, 3))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
            plt.xticks(rotation=90)
            st.pyplot(plt)
            
            st.markdown(imagedownload(plt,'plot-rmse-wide.pdf'), unsafe_allow_html=True)
            with st.header('3.3 Calculation time'):
                predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
                plt.figure(figsize=(3, 9))
                sns.set_theme(style="whitegrid")
                ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
            st.markdown(imagedownload(plt,'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
            plt.figure(figsize=(9, 3))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
            plt.xticks(rotation=90)
            st.pyplot(plt)
            st.markdown(imagedownload(plt,'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)      
    elif selected_option1 == "EDA Tool":
        st.header("Exploratory Data Analysis Tool")
        st.subheader("For EDA ,Pandas Profiling is used ")
        st.subheader("Data Input")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(df)
            except pd.errors.EmptyDataError:
                st.error("The selected file does not contain any data.")
                st.stop()
            if df.empty:
                st.error("The selected file does not contain any columns.")
                st.stop()
            pr = ProfileReport(df, explorative=True)
            st.header('**Input DataFrame**')
            st.write(df)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)
            href, new_filename = download_report(df, uploaded_file.name, "_Report")
            st.download_button(
                label="Download Test Set File",
                data=href,
                file_name=new_filename,
                mime="text/pdf"
                )
    elif selected_option1=="DashBoard":
        columns = st.columns((2, 2, 2, 2))
        with columns[0]:
            st.subheader("DP Tool")
            st.write("""
            - Data Preprocessing Tool.\n- This tool helps to prepare raw data for analysis and modeling.
            """)
            st.markdown("#### Features provided:")
            st.markdown("- Missing Values\n- Duplicate Rows\n- Handle Outliers\n- Data Types\n- Encode Categorical Variables\n- Scale Features")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
    
        with columns[1]:
            st.subheader("AutoML Tool")
            st.write("""
            - Auto Machine Learning  Tools.\n- It compare more than 40 machine learning algorithms at one place!
            """)
            st.markdown("#### Features provided:")
            st.markdown("- Dataset dimension\n- Variable details\n- Divide data into training and test dataset\n- R-squared plot\n- RMSE (capped at 50) plot\n- Calculation time plot")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
    
        with columns[2]:
            st.subheader("HPO Tool")
            st.write("""
            - HyperParameter Optimization Tool.\n- It uses RandomForestRegressor() function for build a regression model .
            """)
            st.markdown("#### Features provided:")
            st.markdown("- Set Parameters\n- Coefficient of determination\n- Error (MSE or MAE)\n- Best parameters\n- Model parameters\n- Hyperparameter tuning graph")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
    
        with columns[3]:
            st.subheader("EDA Tool")
            st.write("""
            - Exploratory Data Analysis Tool.\n- It is uses pandas-profiling for analyzing and visualizing the data 
            """)
            st.markdown("#### Features provided:")
            st.markdown("- Dataset statistics\n- Data interactions\n- Correlations\n- Missing values\n- Alerts\n- Reproduction")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")     

if __name__ == "__main__":
    main()
