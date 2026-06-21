import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn as nn
from scipy.stats import pearsonr

class ExpressionMLP(nn.Module):
    def __init__(self, input_dim=5313, output_dim=68):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.network(x)

def one_hot_encode(seq):
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    seq_arr = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
    arr[seq_arr == 65, 0] = 1.0
    arr[seq_arr == 67, 1] = 1.0
    arr[seq_arr == 71, 2] = 1.0
    arr[seq_arr == 84, 3] = 1.0
    return arr

def pad_or_crop_seq(seq, target_len=393216):
    seq = "".join(seq.split()).upper()
    if len(seq) == target_len:
        return seq
    elif len(seq) > target_len:
        start = (len(seq) - target_len) // 2
        return seq[start:start+target_len]
    else:
        pad_total = target_len - len(seq)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return "N" * pad_left + seq + "N" * pad_right

@st.cache_resource
def load_models():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    enformer = hub.load("https://www.kaggle.com/models/deepmind/enformer/TensorFlow2/enformer/1").model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = ExpressionMLP(input_dim=5313, output_dim=68)
    if os.path.exists("model.pt"):
        mlp.load_state_dict(torch.load("model.pt", map_location=device))
    mlp.to(device)
    mlp.eval()
    return enformer, mlp, device

@st.cache_data
def get_tissues():
    txt_file = "data_embeddings/tissue_names.txt"
    if os.path.exists(txt_file):
        with open(txt_file, "r") as f:
            return [line.strip() for line in f if line.strip()]
    gtex_file = "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz"
    if os.path.exists(gtex_file):
        df = pd.read_csv(gtex_file, sep='\t', skiprows=2, nrows=1)
        return list(df.columns[2:])
    raise FileNotFoundError("Could not find tissue names list. Please ensure 'data_embeddings/tissue_names.txt' is present.")

@st.cache_data
def load_eval_data():
    X = np.load("data_embeddings/embeddings.npy")
    y = np.load("data_embeddings/targets.npy")
    meta_df = pd.read_csv("data_embeddings/metadata.csv")
    tissue_cols = get_tissues()
    return X, y, meta_df, tissue_cols

def main():
    st.set_page_config(page_title="Enformer-GTEx Predictor", layout="wide")
    
    st.markdown("""
        <style>
        .reportview-container {
            background: #0e1117;
        }
        .main h1 {
            color: #f0f2f6;
            font-family: 'Inter', sans-serif;
            font-weight: 700;
        }
        .academic-text {
            font-family: 'Georgia', serif;
            font-size: 1.1rem;
            color: #e0e0e0;
            line-height: 1.6;
            text-align: justify;
        }
        </style>
        """, unsafe_allow_html=True)
        
    st.title("Enformer-to-GTEx Tissue Expression Predictor")
    
    st.markdown("""
        <div class='academic-text'>
        This system implements a hierarchical transfer learning architecture to predict tissue-specific gene expression levels directly from primary genomic sequences. 
        A pretrained, frozen Enformer model acts as a deep feature extractor, converting a 393,216 base-pair genomic sequence centered on a Transcription Start Site (TSS) into a 5,313-dimensional regulatory representation. 
        A downstream Multi-Layer Perceptron (MLP) mapping head then translates these high-dimensional representations into quantitative tissue-specific expression levels (log2(TPM + 1)) across 68 distinct tissues defined in the GTEx annotation.
        </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    
    if not os.path.exists("model.pt"):
        st.error("Error: Trained model weights ('model.pt') not detected in the workspace. Please make sure the model weights are renamed to 'model.pt' before running predictions.")
        return
        
    try:
        enformer, mlp, device = load_models()
        tissue_cols = get_tissues()
    except Exception as e:
        st.error(f"Failed to load resources: {str(e)}")
        return

    # Tabs for navigation
    tab1, tab2 = st.tabs(["DNA Sequence Predictor", "Model Performance and Variance Analysis"])

    with tab1:
        st.subheader("Input Genomic Sequence")
        
        # Example GC-rich sequence snippet
        example_seq = "GGCGGGCGCGGGCCGGGCGCGGGGCGGGGGCGGGGCGCGGGCGCGGGCGCGGGGCGGGGGCGGGGCGCGGGCGCGGGCGCGGGGCGGGGGCGGGGCGCGGGCGCGGGCGCGGGGCGGGGGCGGGGCGCGGGCGCGGGCGCGGGGCGGGGGCGGGGCGCGGGCGCGGGCGCGGGGCGGGGGCGGGGCGCGGGCGCGGGCGCGGGGCGGGGGCGGGGCGCGGGCGCGGGCGC"
        
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            if st.button("Load Example DNA Sequence"):
                st.session_state["seq_input_area"] = example_seq
        
        raw_input = st.text_area(
            label="Enter raw DNA sequence (A, C, G, T, N). The input will be centered and padded or cropped to exactly 393,216 base pairs.",
            value=st.session_state.get("seq_input_area", ""),
            height=200,
            key="seq_input_area"
        )
        
        if st.button("Execute Prediction"):
            if not raw_input.strip():
                st.warning("Please enter a valid sequence.")
            else:
                with st.spinner("Processing sequence and running deep inference..."):
                    processed_seq = pad_or_crop_seq(raw_input)
                    x = one_hot_encode(processed_seq)
                    x_batch = np.expand_dims(x, axis=0)
                    
                    try:
                        with tf.device('/GPU:0'):
                            preds = enformer.predict_on_batch(x_batch)
                            human_preds = preds['human'].numpy()
                    except Exception:
                        with tf.device('/CPU:0'):
                            preds = enformer.predict_on_batch(x_batch)
                            human_preds = preds['human'].numpy()
                            
                    emb = human_preds.mean(axis=1).squeeze()
                    emb_t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        pred_expr = mlp(emb_t).cpu().numpy().squeeze()
                        
                    results_df = pd.DataFrame({
                        "Tissue": tissue_cols,
                        "Predicted Expression (log2(TPM + 1))": pred_expr
                    }).sort_values(by="Predicted Expression (log2(TPM + 1))", ascending=False)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Predicted Expression Levels")
                        st.dataframe(results_df, use_container_width=True, height=500)
                        
                    with col2:
                        st.subheader("Tissue-Specific Expression Profile")
                        st.bar_chart(data=results_df, x="Tissue", y="Predicted Expression (log2(TPM + 1))", use_container_width=True)

    with tab2:
        st.subheader("Interactive Variance Filtering Analysis")
        st.markdown("""
            Sequence-to-expression models often face a correlation ceiling because a large portion of genes 
            are **housekeeping genes** that are expressed at near-identical levels across all tissues. 
            Calculating Pearson correlation on flat lines results in scores near `0.0` or `NaN`, dragging down the median.
            
            By adjusting the slider below, you can filter out low-variance genes and analyze how the model's 
            prediction accuracy (Pearson correlation) scales when focused on genes that show actual tissue-variable expression.
        """)
        
        threshold = st.slider("Minimum Tissue Expression Variance Threshold", 0.0, 0.5, 0.2, 0.1)
        
        with st.spinner("Filtering test set and calculating performance..."):
            # Load eval data
            X_eval, y_eval, meta_df, eval_tissues = load_eval_data()
            
            # Split test set (chr21, chr22)
            test_mask = meta_df["chrom"].isin(["chr21", "chr22"]).values
            
            # Compute gene-wise expression variance
            variances = np.var(y_eval, axis=1)
            var_filter = variances > threshold
            test_sel = test_mask & var_filter
            
            num_genes = int(test_sel.sum())
            
            if num_genes == 0:
                st.warning("No genes in the test split match this variance threshold. Please select a lower value.")
            else:
                # Predict test set using baseline MLP
                test_emb = torch.tensor(X_eval[test_sel], dtype=torch.float32).to(device)
                with torch.no_grad():
                    test_preds = mlp(test_emb).cpu().numpy()
                test_targets = y_eval[test_sel]
                
                # Compute performance metrics
                mse = np.mean((test_targets - test_preds) ** 2)
                mae = np.mean(np.abs(test_targets - test_preds))
                
                pearsons = []
                for i in range(test_targets.shape[1]):
                    t_col = test_targets[:, i]
                    p_col = test_preds[:, i]
                    if np.std(t_col) > 0 and np.std(p_col) > 0:
                        r, _ = pearsonr(t_col, p_col)
                        if not np.isnan(r):
                            pearsons.append(r)
                median_r = np.median(pearsons) if pearsons else 0.0
                
                # Display metrics columns
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Genes in Evaluation Set", f"{num_genes}")
                col_m2.metric("Median Pearson Correlation (r)", f"{median_r:.4f}")
                col_m3.metric("Overall MSE", f"{mse:.4f}")
                col_m4.metric("Overall MAE", f"{mae:.4f}")
                
                # Reshape for scatter plot
                flat_true = test_targets.flatten()
                flat_pred = test_preds.flatten()
                
                # Downsample plot points to keep UI smooth
                if len(flat_true) > 5000:
                    idx = np.random.choice(len(flat_true), 5000, replace=False)
                    plot_true = flat_true[idx]
                    plot_pred = flat_pred[idx]
                else:
                    plot_true = flat_true
                    plot_pred = flat_pred
                    
                plot_df = pd.DataFrame({
                    "Actual Expression": plot_true,
                    "Predicted Expression": plot_pred
                })
                
                st.write("---")
                st.subheader(f"Predicted vs. Actual Expression (Threshold = {threshold})")
                st.scatter_chart(plot_df, x="Actual Expression", y="Predicted Expression", height=450)

if __name__ == "__main__":
    main()
