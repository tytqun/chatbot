from fastapi import FastAPI, HTTPException, status
from loguru import logger

from utils import load_churn_customers_data, get_google_gemini_llm, generate_promo_message_for_customer # Đã thêm lại get_google_gemini_llm

app = FastAPI()

llm_model = None 
churn_customers_df = None 

@app.on_event("startup")
async def startup_event():
    """
    Sự kiện khởi động ứng dụng FastAPI: tải dữ liệu khách hàng và khởi tạo LLM.
    """
    global llm_model, churn_customers_df 
    logger.info("Đang khởi tạo ứng dụng: tải dữ liệu và mô hình LLM...")
    
    churn_customers_df = load_churn_customers_data("churn_prediction.csv") 
    
    
    try:
        llm_model = get_google_gemini_llm()
        logger.info("Mô hình LLM đã được khởi tạo thành công.")
    except ValueError as e:
        logger.error(f"Lỗi khởi tạo LLM: {e}. Vui lòng kiểm tra GOOGLE_API_KEY của bạn.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Lỗi cấu hình LLM: {e}")
    
    logger.info("Khởi tạo ứng dụng hoàn tất.")

@app.post("/chat")
def chat(user_id: str):
    """
    Endpoint để tạo tin nhắn khuyến mãi đa dạng cho một khách hàng được dự đoán churn.
    """
    logger.info(f"Yêu cầu tạo tin nhắn khuyến mãi cho user ID: {user_id}")

    if churn_customers_df is None or churn_customers_df.empty:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dữ liệu khách hàng churn chưa được tải hoặc trống. Vui lòng kiểm tra file dữ liệu và kết quả dự đoán."
        )
    
    if llm_model is None: 
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình LLM chưa được khởi tạo. Vui lòng kiểm tra cấu hình."
        )

    customer_row = churn_customers_df[churn_customers_df['user_id'].astype(str) == user_id]

    if customer_row.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy user ID: {user_id} trong danh sách khách hàng được dự đoán là CHURN."
        )

    customer_data = customer_row.iloc[0].to_dict()

    try:
    
        promo_message = generate_promo_message_for_customer(customer_data, llm_model) 
        logger.info(f"Đã tạo tin nhắn cho user {user_id}")
        return {"user_id": user_id, "promo_message": promo_message}
    except Exception as e:
        logger.error(f"Lỗi khi tạo tin nhắn cho user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không thể tạo tin nhắn khuyến mãi cho user ID: {user_id}. Lỗi: {e}"
        )



