import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage
from PyPDF2 import PdfReader
import pandas as pd
from io import StringIO
import docx
import os
import matplotlib.pyplot as plt

# API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="LHP AI Chatbot",
    page_icon="new_icon.png",  # File icon tùy chỉnh
    layout="wide"
)

# Giao diện chính
st.title("LHP AI Chatbot")
st.write("Ask questions directly, or upload files for additional context.")

# Hàm đọc nội dung file
def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_txt(file):
    stringio = StringIO(file.getvalue().decode("utf-8"))
    return stringio.read()

def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_excel(file):
    """Đọc nội dung từ file Excel và trả về DataFrame."""
    return pd.read_excel(file)

def plot_chart(data, chart_type, x_column, y_column):
    """Hàm vẽ biểu đồ."""
    plt.figure(figsize=(10, 6))
    if chart_type == "Bar Chart":
        plt.bar(data[x_column], data[y_column])
    elif chart_type == "Line Chart":
        plt.plot(data[x_column], data[y_column], marker="o")
    elif chart_type == "Scatter Plot":
        plt.scatter(data[x_column], data[y_column])
    
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{chart_type}: {y_column} vs {x_column}")
    st.pyplot(plt)

# Tạo mô hình nhúng
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Lưu trạng thái phản hồi
if "response" not in st.session_state:
    st.session_state["response"] = ""

# đặt câu hỏi
user_question = st.text_input("Type your question here:")

# Tải lên file
uploaded_files = st.file_uploader(
    "Optional: Upload files (PDF, TXT, DOCX, Excel supported) for context", 
    type=["pdf", "txt", "docx", "xlsx", "xls"], 
    accept_multiple_files=True
)

# Đọc nội dung từ file và lưu trữ
excel_tables = []
file_contents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            file_contents.append(read_pdf(uploaded_file))
        elif uploaded_file.name.endswith(".txt"):
            file_contents.append(read_txt(uploaded_file))
        elif uploaded_file.name.endswith(".docx"):
            file_contents.append(read_docx(uploaded_file))
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = read_excel(uploaded_file)  # Lưu DataFrame
            excel_tables.append((uploaded_file.name, df))

# Xử lý file Excel nếu có
if excel_tables:
    st.subheader("Excel Processing:")
    selected_file = st.selectbox("Select an Excel file to process:", [name for name, _ in excel_tables])
    
    if selected_file:
        selected_df = next(df for name, df in excel_tables if name == selected_file)

        # Lựa chọn cột và loại biểu đồ
        columns = selected_df.columns.tolist()
        x_column = st.selectbox("Select X-axis:", columns, key=f"x_{selected_file}")
        y_column = st.selectbox("Select Y-axis:", columns, key=f"y_{selected_file}")
        chart_type = st.selectbox(
            "Select Chart Type:", 
            ["Bar Chart", "Line Chart", "Scatter Plot"], 
            key=f"chart_{selected_file}"
        )

        # Vẽ biểu đồ khi nhấn nút
        if st.button("Generate Chart"):
            plot_chart(selected_df, chart_type, x_column, y_column)

# Kết hợp nội dung file vào câu hỏi nếu có
if st.button("Submit"):
    if user_question.strip():
        if file_contents:
            combined_text = "\n".join(file_contents)
            user_question = f"Based on the uploaded documents:\n{combined_text}\n\n{user_question}"

        # Tạo mô hình GPT
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=1000,
            model_name="gpt-3.5-turbo",
        )

        # Gửi câu hỏi tới GPT
        input_message = HumanMessage(content=user_question)
        response = llm([input_message])
        st.session_state["response"] = response.content

# Hiển thị kết quả trả lời
if st.session_state["response"]:
    st.subheader("Answer:")
    st.write(st.session_state["response"])
