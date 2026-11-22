import streamlit as st, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
st.set_page_config(page_title="Air Quality PCA Analysis",page_icon="🌫️",layout="wide")
st.title("🌫️ Air Quality Analysis using PCA")
st.caption("Clean pollutant measurements, explore correlations, standardize features, and reduce dimensions using PCA.")
up=st.file_uploader("Upload air-quality CSV",type="csv")
if up:
    df=pd.read_csv(up)
else:
    try: df=pd.read_csv("air_quality_pca_results.csv")
    except: df=None
if df is not None:
    st.subheader("Dataset preview")
    st.dataframe(df.head(12),use_container_width=True)
    candidates=["PM2.5","PM10","NO2","SO2","CO","O3"]
    matched=[c for c in candidates if c in df.columns]
    if len(matched)<2:
        normalized={str(c).lower().replace(" ",""):c for c in df.columns}
        matched=[normalized[k] for k in ["pm25","pm10","no2","so2","co","o3"] if k in normalized]
    if len(matched)<2:
        st.warning("Could not auto-detect at least two pollutant columns. Select numeric features manually.")
        matched=df.select_dtypes(include=np.number).columns.tolist()
    features=st.multiselect("Features for PCA",df.columns,default=matched)
    threshold=st.slider("Variance retention target",0.50,0.99,0.90,0.01)
    if st.button("Run PCA",type="primary"):
        x=df[features].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
        x=x.dropna(axis=1,how="all").fillna(x.median())
        if x.shape[1]<2 or len(x)<2: st.error("Select at least two usable numeric columns.")
        else:
            scaler=StandardScaler();scaled=scaler.fit_transform(x)
            pca=PCA().fit(scaled)
            exp=pca.explained_variance_ratio_;cum=exp.cumsum()
            n=int(np.argmax(cum>=threshold)+1)
            reduced=PCA(n_components=n).fit_transform(scaled)
            reduced_df=pd.DataFrame(reduced,columns=[f"PC{i+1}" for i in range(n)])
            loadings=pd.DataFrame(pca.components_.T,index=x.columns,columns=[f"PC{i+1}" for i in range(len(exp))])
            st.session_state["air_exp"]=exp
            st.session_state["air_cum"]=cum
            st.session_state["air_reduced"]=reduced_df
            st.session_state["air_loadings"]=loadings
            st.session_state["air_corr"]=x.corr()
            st.session_state["air_n"]=n
    if "air_exp" in st.session_state:
        exp=st.session_state["air_exp"];cum=st.session_state["air_cum"];r=st.session_state["air_reduced"];loads=st.session_state["air_loadings"]
        st.subheader("Variance explained")
        st.metric("Components needed",st.session_state["air_n"])
        st.bar_chart(pd.Series(exp*100,index=[f"PC{i+1}" for i in range(len(exp))],name="Explained variance (%)"))
        st.line_chart(pd.Series(cum*100,index=[f"PC{i+1}" for i in range(len(cum))],name="Cumulative variance (%)"))
        st.subheader("Correlation matrix")
        st.dataframe(st.session_state["air_corr"].round(2),use_container_width=True)
        if "PC2" in r.columns: st.subheader("PCA projection"); st.scatter_chart(r,x="PC1",y="PC2")
        st.subheader("Loadings")
        st.dataframe(loads.round(3),use_container_width=True)
        st.bar_chart(loads["PC1"].sort_values())
        st.subheader("Reduced dataset")
        st.dataframe(r.head(20),use_container_width=True)
        st.download_button("Download reduced PCA data",r.to_csv(index=False),"air_quality_pca_results.csv","text/csv")
else:
    st.info("Upload your raw air-quality CSV to begin.")
