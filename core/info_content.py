import streamlit as st

def show_info_page():
    st.markdown("## 📋 Giải mã Thuật toán & Khoa học")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        <div class='info-card'>
        <h4>1. Tại sao sử dụng Machine Learning?</h4>
        Các công thức truyền thống (như US Navy hoặc BMI) thường chỉ sử dụng 
        2–3 chỉ số cơ thể. Điều này dẫn đến sai số cao đối với người tập Gym, 
        vì cơ bắp phát triển có thể bị nhầm thành mỡ.
        <br><br>
        <b>Mô hình của hệ thống:</b>
        <ul>
        <li>Sử dụng thuật toán <b>XGBoost Gradient Boosting</b>.</li>
        <li>Phân tích đồng thời <b>8 chỉ số cơ thể</b>.</li>
        <li>Học mối tương quan phi tuyến giữa các nhóm cơ và phân bố mỡ.</li>
        </ul>
        Điều này giúp mô hình thích nghi tốt hơn với nhiều kiểu hình thể khác nhau.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
        <h4>2. Cơ chế học của XGBoost</h4>
        XGBoost hoạt động bằng cách kết hợp nhiều <b>Decision Tree</b> nhỏ 
        để xây dựng một mô hình mạnh hơn.
        <br><br>
        Quy trình:
        <ol>
        <li>Mỗi cây học một phần sai số của cây trước.</li>
        <li>Các cây được cộng lại để tạo thành mô hình cuối.</li>
        <li>Thuật toán tối ưu để giảm overfitting và tăng độ chính xác.</li>
        </ol>
        Phương pháp này thường đạt độ chính xác cao trong các bài toán 
        dự đoán sinh học và y học.
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown("""
        <div class='info-card'>
        <h4>3. Ý nghĩa các chỉ số đo</h4>
        <ul>
        <li><b>Weight & Height:</b> xác định BMI và khối lượng cơ thể tổng.</li>
        <li><b>Chest:</b> thể hiện phát triển cơ ngực và lưng trên.</li>
        <li><b>Abdomen:</b> chỉ số quan trọng nhất liên quan đến mỡ nội tạng.</li>
        <li><b>Hip:</b> vùng tích mỡ chính của cơ thể.</li>
        <li><b>Thigh:</b> phản ánh sự phát triển của cơ đùi.</li>
        <li><b>Biceps:</b> chỉ báo mức độ phát triển cơ bắp tay.</li>
        </ul>
        Các thông số này kết hợp giúp AI phân tích phân bố mỡ toàn cơ thể.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
        <h4>4. Lưu ý để đo chính xác</h4>
        <ol>
        <li><b>Đo vào buổi sáng:</b> khi cơ thể chưa ăn và chưa tập luyện.</li>
        <li><b>Dùng thước dây mềm:</b> đo sát da nhưng không siết quá chặt.</li>
        <li><b>Giữ thước ngang:</b> luôn song song với mặt đất.</li>
        </ol>
        Sai số đo lường có thể ảnh hưởng trực tiếp đến kết quả dự đoán.
        </div>
        """, unsafe_allow_html=True)

    if st.button("⬅️ QUAY LẠI MÁY TÍNH"):
        st.session_state.page = 'home'
        st.rerun()

    st.markdown(
        "<br><p style='text-align: center; color: #4B5563; font-size: 12px;'>ThongThien Fitness AI © 2026 | Machine Learning for Body Composition Analysis</p>",
        unsafe_allow_html=True
    )