import uuid

def get_human_svg(bf, color="#3B82F6"): # Thêm tham số color
    unique_id = str(uuid.uuid4())[:8]
    # Giới hạn bf từ 0-45 để tính tỷ lệ lấp đầy
    fill_h = max(0, min(100, (bf / 45) * 100))
    y_pos = 210 - (fill_h * 2.1)

    svg = f"""
    <div style="display:flex;justify-content:center;align-items:center;flex-direction:column;font-family:sans-serif;">
        <svg width="160" height="320" viewBox="0 0 100 220">
            <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30 C45,30 40,25 40,20 C40,15 45,10 50,10 M40,32 L60,32 L65,80 L75,130 L70,210 L55,210 L50,140 L45,210 L30,210 L25,130 L35,80 Z" fill="#2D3748"/>
            
            <defs>
                <clipPath id="cp_{unique_id}">
                    <rect x="0" y="{y_pos}" width="100" height="210"/>
                </clipPath>
            </defs>
            
            <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30 C45,30 40,25 40,20 C40,15 45,10 50,10 M40,32 L60,32 L65,80 L75,130 L70,210 L55,210 L50,140 L45,210 L30,210 L25,130 L35,80 Z" 
                  fill="{color}" clip-path="url(#cp_{unique_id})"/>
            
            <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30 C45,30 40,25 40,20 C40,15 45,10 50,10 M40,32 L60,32 L65,80 L75,130 L70,210 L55,210 L50,140 L45,210 L30,210 L25,130 L35,80 Z" 
                  fill="none" stroke="#4A5568" stroke-width="2"/>
        </svg>
        <p style="color:{color};font-weight:bold;margin-top:10px;font-size:18px;">{bf:.1f}% Body Fat</p>
    </div>
    """
    return svg

def get_custom_css():
    return """
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    div[data-testid="stNumberInput"] { margin-bottom: -15px; }
    div[data-testid="stNumberInput"] label { font-size: 13px !important; color: #9CA3AF !important; }
    .result-box { background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); padding: 25px; border-radius: 15px; border: 1px solid #3B82F6; text-align: center; margin-bottom: 20px; }
    .big-value { font-size: 60px !important; font-weight: 900; color: #00FF00; }
    .metric-item { background: rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; margin-bottom: 8px; border: 1px solid #334155; }
    .expert-note { background-color: rgba(59,130,246,0.1); border-left: 4px solid #3B82F6; padding: 15px; border-radius: 4px; margin-top: 10px; }
    .info-card { background: #1E293B; padding: 20px; border-radius: 10px; border-left: 5px solid #3B82F6; margin-bottom: 20px; }
    </style>
    """