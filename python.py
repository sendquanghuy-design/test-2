import streamlit as st
import pandas as pd
import numpy_financial as npf
from google import genai
from google.genai.errors import APIError
from docx import Document
import json
import io
import time # For exponential backoff simulation (best practice)

# --- Thư viện cần thiết: streamlit, pandas, numpy, python-docx, google-genai ---

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh giá Phương án Kinh doanh AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Khởi tạo session state ---
if "parameters" not in st.session_state:
    st.session_state.parameters = None
if "cash_flow_df" not in st.session_state:
    st.session_state.cash_flow_df = None
if "metrics" not in st.session_state: # SỬA LỖI: Bỏ đi từ 'not' thừa
    st.session_state.metrics = None
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = None

st.title("💡 Đánh giá Phương án Kinh doanh với AI")
st.markdown("---")

# ----------------------------------------------------
#               CÁC HÀM XỬ LÝ VÀ TÍNH TOÁN
# ----------------------------------------------------

def extract_text_from_docx(file):
    """Đọc và trích xuất toàn bộ văn bản từ file Word (.docx)."""
    try:
        doc = Document(file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except Exception as e:
        st.error(f"Lỗi khi đọc file Word: {e}")
        return None

def call_gemini_api(prompt, schema, api_key):
    """Hàm gọi Gemini API với cấu hình schema JSON để trích xuất dữ liệu."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash-preview-05-20'

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema
            },
        }
        
        # Thực hiện API call với exponential backoff
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=payload['contents'],
                    config=payload['generationConfig']
                )
                
                # Kiểm tra và parse JSON
                json_text = response.text.strip()
                if json_text.startswith('```json'):
                    json_text = json_text.replace('```json', '').replace('```', '').strip()
                
                return json.loads(json_text)
            except (APIError, json.JSONDecodeError) as e:
                if attempt < 2:
                    st.warning(f"Lỗi API hoặc JSON. Đang thử lại lần {attempt + 2}...")
                    time.sleep(2 ** attempt) # Exponential backoff
                    continue
                else:
                    raise e

        return None

    except APIError as e:
        return {"error": f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"}
    except Exception as e:
        return {"error": f"Đã xảy ra lỗi không xác định: {e}"}

def extract_financial_parameters(raw_text, api_key):
    """
    Nhiệm vụ 1: Lọc các thông tin tài chính bằng AI và trả về JSON.
    """
    system_prompt = (
        "Bạn là một AI chuyên gia tài chính. Hãy đọc văn bản sau và trích xuất "
        "chính xác 6 thông số tài chính được yêu cầu. Các giá trị phải là SỐ (không có đơn vị tiền tệ) "
        "và Thuế/WACC phải ở dạng thập phân (ví dụ: 10% là 0.1)."
    )
    
    # Định nghĩa Schema JSON cho đầu ra mong muốn
    schema = {
        "type": "OBJECT",
        "properties": {
            "vốn_đầu_tư": {"type": "NUMBER", "description": "Tổng vốn đầu tư ban đầu (VND, chỉ là số)."},
            "dòng_đời_dự_án": {"type": "INTEGER", "description": "Dòng đời dự án theo năm."},
            "doanh_thu_hàng_năm": {"type": "NUMBER", "description": "Doanh thu hoạt động trung bình hàng năm (VND, chỉ là số)."},
            "chi_phí_hàng_năm": {"type": "NUMBER", "description": "Tổng chi phí hoạt động trung bình hàng năm (VND, không bao gồm khấu hao/lãi vay)."},
            "wacc": {"type": "NUMBER", "description": "Chi phí vốn bình quân (WACC) dưới dạng thập phân (ví dụ: 10% là 0.1)."},
            "thuế": {"type": "NUMBER", "description": "Thuế suất thu nhập doanh nghiệp dưới dạng thập phân (ví dụ: 20% là 0.2)."}
        },
        "required": ["vốn_đầu_tư", "dòng_đời_dự_án", "doanh_thu_hàng_năm", "chi_phí_hàng_năm", "wacc", "thuế"]
    }
    
    prompt = f"{system_prompt}\n\nVăn bản: {raw_text}"
    
    return call_gemini_api(prompt, schema, api_key)

def calculate_cash_flow(params):
    """
    Nhiệm vụ 2: Xây dựng bảng dòng tiền từ các thông số đã lọc.
    Giả định: Dòng tiền hoạt động ổn định từ năm 1 đến năm N.
    """
    try:
        I = params['vốn_đầu_tư']
        N = params['dòng_đời_dự_án']
        R = params['doanh_thu_hàng_năm']
        C = params['chi_phí_hàng_năm']
        T = params['thuế']
        
        # Tính toán Dòng tiền hoạt động (Operating Cash Flow - OCF)
        # OCF = (Doanh thu - Chi phí) * (1 - Thuế suất)
        OCF = (R - C) * (1 - T)

        # Xây dựng bảng Dòng tiền
        years = list(range(N + 1)) # Năm 0 đến Năm N
        cash_flows = [-I] + [OCF] * N
        
        df = pd.DataFrame({
            'Năm': years,
            'Dòng tiền Hoạt động (OCF)': [0] + [OCF] * N,
            'Đầu tư (I)': [-I] + [0] * N,
            'Dòng tiền Thuần (CF)': cash_flows
        })
        
        return df
    except Exception as e:
        st.error(f"Lỗi trong quá trình tính toán Dòng tiền: {e}")
        return None

def calculate_metrics(df, wacc):
    """
    Nhiệm vụ 3: Tính toán các chỉ số đánh giá hiệu quả dự án.
    """
    cash_flows = df['Dòng tiền Thuần (CF)'].values
    
    # 1. NPV (Net Present Value)
    npv_value = np.npv(wacc, cash_flows)
    
    # 2. IRR (Internal Rate of Return)
    try:
        irr_value = np.irr(cash_flows)
    except:
        irr_value = np.nan # Có thể không tính được nếu dòng tiền không đổi dấu
    
    # 3. Payback Period (PP) và Discounted Payback Period (DPP)
    
    # Tính dòng tiền chiết khấu
    discounted_cf = []
    for t, cf in enumerate(cash_flows):
        discounted_cf.append(cf / ((1 + wacc) ** t))
        
    df['Dòng tiền Chiết khấu'] = discounted_cf
    
    # Tính dòng tiền tích lũy và dòng tiền chiết khấu tích lũy
    cumulative_cf = np.cumsum(cash_flows)
    cumulative_dcf = np.cumsum(discounted_cf)
    
    # 4. PP (Payback Period)
    try:
        investment = abs(cash_flows[0])
        payback_year = np.where(cumulative_cf >= 0)[0][0]
        # Thời gian hoàn vốn = Năm trước + (Khoản còn thiếu / Dòng tiền năm đó)
        if payback_year == 0:
             pp_value = 0
        else:
             cf_prior = cumulative_cf[payback_year - 1]
             cf_at_payback = cash_flows[payback_year]
             pp_value = (payback_year - 1) + (investment + cf_prior) / cf_at_payback
    except:
        pp_value = np.inf # Không bao giờ hoàn vốn

    # 5. DPP (Discounted Payback Period) - Logic tương tự PP, dùng DCF
    try:
        discounted_investment = abs(discounted_cf[0])
        dpp_year = np.where(cumulative_dcf >= 0)[0][0]
        if dpp_year == 0:
             dpp_value = 0
        else:
             dcf_prior = cumulative_dcf[dpp_year - 1]
             dcf_at_payback = discounted_cf[dpp_year]
             dpp_value = (dpp_year - 1) + (discounted_investment + dcf_prior) / dcf_at_payback
    except:
        dpp_value = np.inf

    return {
        'NPV (Giá trị hiện tại ròng)': npv_value,
        'IRR (Tỷ suất sinh lời nội bộ)': irr_value,
        'PP (Thời gian hoàn vốn)': pp_value,
        'DPP (Thời gian hoàn vốn chiết khấu)': dpp_value
    }

def get_ai_business_analysis(metrics, wacc, api_key):
    """
    Nhiệm vụ 4: Phân tích các chỉ số hiệu quả dự án bằng AI.
    """
    system_prompt = (
        "Bạn là một chuyên gia thẩm định dự án đầu tư. "
        "Dựa trên các chỉ số hiệu quả dự án sau, hãy đưa ra một đánh giá khách quan, "
        "ngắn gọn (khoảng 3 đoạn) về khả năng chấp nhận đầu tư của dự án. "
        "Lưu ý: WACC (Chi phí vốn) là {wacc:.2%}. Dùng ngôn ngữ chuyên nghiệp và dễ hiểu."
    ).format(wacc=wacc)

    metrics_text = (
        f"NPV (Giá trị hiện tại ròng): {metrics['NPV (Giá trị hiện tại ròng)']:.0f} VND\n"
        f"IRR (Tỷ suất sinh lời nội bộ): {metrics['IRR (Tỷ suất sinh lời nội bộ)']:.2%}\n"
        f"PP (Thời gian hoàn vốn): {metrics['PP (Thời gian hoàn vốn)']:.2f} năm\n"
        f"DPP (Thời gian hoàn vốn chiết khấu): {metrics['DPP (Thời gian hoàn vốn chiết khấu)']:.2f} năm"
    )

    prompt = f"{system_prompt}\n\nCác Chỉ số Hiệu quả Dự án:\n{metrics_text}"

    # Sử dụng hàm gọi API chung, không cần schema vì đây là output dạng text
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Lỗi gọi AI để phân tích: {e}"

# ----------------------------------------------------
#                    GIAO DIỆN STREAMLIT
# ----------------------------------------------------

# Lấy API Key từ Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Lỗi cấu hình: Vui lòng thêm Khóa API Gemini ('GEMINI_API_KEY') vào Streamlit Secrets.")

st.markdown("---")
st.markdown("### 1. Tải lên và Trích xuất Dữ liệu")

uploaded_file = st.file_uploader(
    "📤 Tải file Word (.docx) chứa Phương án Kinh doanh:",
    type=['docx'],
    help="File Word của bạn sẽ được AI đọc để trích xuất các thông số tài chính cơ bản."
)

if uploaded_file and GEMINI_API_KEY:
    
    # Nút bấm kích hoạt AI Extraction
    if st.button("🤖 Kích hoạt AI Lọc Dữ liệu", use_container_width=True):
        st.session_state.cash_flow_df = None
        st.session_state.metrics = None
        st.session_state.ai_analysis = None

        with st.spinner("⏳ Đang trích xuất dữ liệu tài chính từ file Word..."):
            
            # Đọc nội dung file Word
            raw_text = extract_text_from_docx(uploaded_file)
            if raw_text:
                
                # Gọi AI để trích xuất JSON
                params = extract_financial_parameters(raw_text, GEMINI_API_KEY)
                
                if params and "error" not in params:
                    st.session_state.parameters = params
                    st.session_state.ai_analysis = None
                    st.success("✅ Trích xuất thành công! Dữ liệu đã được chuẩn hóa.")
                else:
                    st.session_state.parameters = None
                    st.error(f"❌ Lỗi trích xuất: {params.get('error', 'AI không thể tìm thấy đủ 6 thông số tài chính được yêu cầu trong file. Vui lòng kiểm tra lại nội dung.')}")


# --- Hiển thị Thông số đã Trích xuất ---
if st.session_state.parameters:
    
    st.markdown("### 📝 Các Thông số Tài chính đã Trích xuất")
    col1, col2, col3 = st.columns(3)
    
    P = st.session_state.parameters
    
    with col1:
        st.metric("Vốn Đầu tư (I)", f"{P.get('vốn_đầu_tư', 0):,.0f} VND")
        st.metric("Doanh thu Hàng năm (R)", f"{P.get('doanh_thu_hàng_năm', 0):,.0f} VND")
    with col2:
        st.metric("Chi phí Hàng năm (C)", f"{P.get('chi_phí_hàng_năm', 0):,.0f} VND")
        st.metric("Dòng đời Dự án (N)", f"{P.get('dòng_đời_dự_án', 0):.0f} năm")
    with col3:
        st.metric("WACC", f"{P.get('wacc', 0):.2%}")
        st.metric("Thuế suất (T)", f"{P.get('thuế', 0):.2%}")
        
    st.markdown("---")
    st.markdown("### 2. Xây dựng Bảng Dòng tiền")
    
    # Kích hoạt tính toán Dòng tiền
    st.session_state.cash_flow_df = calculate_cash_flow(P)
    
    if st.session_state.cash_flow_df is not None:
        st.dataframe(st.session_state.cash_flow_df.style.format({
            'Dòng tiền Hoạt động (OCF)': '{:,.0f}',
            'Đầu tư (I)': '{:,.0f}',
            'Dòng tiền Thuần (CF)': '{:,.0f}',
            'Dòng tiền Chiết khấu': '{:,.0f}' # Thêm cột này để dễ nhìn
        }), use_container_width=True)

        st.markdown("---")
        st.markdown("### 3. Tính toán Các Chỉ số Hiệu quả")
        
        # Kích hoạt tính toán Chỉ số
        st.session_state.metrics = calculate_metrics(st.session_state.cash_flow_df.copy(), P['wacc'])
        
        M = st.session_state.metrics
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            # NPV
            npv_color = 'green' if M['NPV (Giá trị hiện tại ròng)'] > 0 else 'red'
            st.markdown(f"""
            <div style='background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid {npv_color};'>
                <small>NPV (VND)</small>
                <h3 style='margin:0; color:{npv_color};'>{M['NPV (Giá trị hiện tại ròng)']:.0f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
        with metric_col2:
            # IRR
            irr_color = 'green' if M['IRR (Tỷ suất sinh lời nội bộ)'] > P['wacc'] else 'red'
            st.markdown(f"""
            <div style='background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid {irr_color};'>
                <small>IRR</small>
                <h3 style='margin:0; color:{irr_color};'>{M['IRR (Tỷ suất sinh lời nội bộ)']:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)

        with metric_col3:
            # PP
            st.metric("Thời gian Hoàn vốn (PP)", f"{M['PP (Thời gian hoàn vốn)']:.2f} năm")

        with metric_col4:
            # DPP
            st.metric("Hoàn vốn Chiết khấu (DPP)", f"{M['DPP (Thời gian hoàn vốn chiết khấu)']:.2f} năm")

        st.markdown("---")
        st.markdown("### 4. Phân tích Chuyên sâu bởi AI")
        
        # Nút bấm kích hoạt AI Analysis
        if st.button("🚀 Yêu cầu AI Phân tích Hiệu quả Dự án", use_container_width=True):
            with st.spinner("🧠 Gemini đang đánh giá và đưa ra nhận xét..."):
                analysis = get_ai_business_analysis(M, P['wacc'], GEMINI_API_KEY)
                st.session_state.ai_analysis = analysis

        if st.session_state.ai_analysis:
            st.markdown("**Kết quả Phân tích từ Chuyên gia AI:**")
            st.info(st.session_state.ai_analysis)
