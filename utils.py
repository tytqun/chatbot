import os
import pandas as pd
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI # Đã bật lại
from langchain_core.prompts import PromptTemplate # Đã bật lại

# Đảm bảo bạn đã đặt biến môi trường GOOGLE_API_KEY
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

@lru_cache(maxsize=1)
def load_churn_customers_data(file_name: str = "churn_prediction.csv"):
    """
    Tải dữ liệu khách hàng churn từ file CSV vào DataFrame.
    Sử dụng lru_cache để đảm bảo dữ liệu chỉ được tải một lần khi ứng dụng khởi động.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "data", file_name)

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Định dạng file không được hỗ trợ. Vui lòng sử dụng .csv hoặc .xlsx")
        
        required_columns = ['user_id', 'xgb_predicted_churn']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"File dữ liệu phải có cột '{col}'.")
            
        # Lọc những khách hàng churn với giá trị 1
        df_churn_only = df[df['xgb_predicted_churn'].astype(int) == 1].copy() 

        print(f"Đã tải dữ liệu khách hàng churn từ '{file_path}' thành công.")
        print(f"Tổng số khách hàng trong file: {len(df)}")
        print(f"Số lượng khách hàng được dự đoán churn: {len(df_churn_only)}")
        
        return df_churn_only
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại đường dẫn: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return pd.DataFrame()


def get_google_gemini_llm():
    """
    Khởi tạo và trả về một thể hiện của ChatGoogleGenerativeAI.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("Biến môi trường GOOGLE_API_KEY chưa được đặt.")
        
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",
        convert_system_message_to_human=True,  
        google_api_key=GOOGLE_API_KEY,
    )
    return llm

def generate_promo_message_for_customer(customer_data: dict, llm): 
     
    sample_message = (
        "Cocoon chào bạn, Cocoon biết rằng ngoài kia còn nhiều những bộn bề và áp lực bạn phải chịu đựng và vượt qua. "
        "Nhưng Cocoon tin rằng mỗi hành trình luôn có ý nghĩa của riêng nó. \n\n"
        "Vì thế, Cocoon xin dành tặng cho bạn chương trình \"Nhấn một nút thương để xích gần nhau hơn\". "
        "Chương trình của chúng mình sẽ có ưu đãi 50% cho tất cả những sản phẩm fullsize, đồng thời với hóa đơn trị giá trên 2 triệu đồng "
        "sẽ được tặng kèm combo sửa rửa mặt và tẩy trang bí đao từ nhà Cocoon. Đây là chương trình đặc biệt Cocoon mong muốn dành riêng cho bạn, "
        "để bạn có thể có thêm nhiều cơ hội trải nghiệm sản phẩm cũng như chăm sóc cho chính mình sau những ngày dài mệt mỏi. "
        "Đừng chần chừ hãy cùng Cocoon khám phá chương trình thú vị này nhé!"
    )

    # Prompt mới để hướng dẫn LLM tạo sinh tin nhắn đa dạng từ sample message
    promo_template = """Bạn là một chuyên gia marketing và chăm sóc khách hàng của Cocoon.
    Hãy tạo một tin nhắn khuyến mãi ngắn gọn (tối đa 4-5 câu) dành cho khách hàng.
    Tin nhắn này cần:
    - Bắt đầu với lời chào từ Cocoon.
    - Duy trì giọng điệu thấu cảm, quan tâm và tích cực như tin nhắn mẫu.
    - Luôn gọi khách hàng là "bạn" (không dùng tên riêng hay các thông tin cá nhân khác).
    - Giới thiệu ưu đãi đặc biệt của chương trình "Nhấn một nút thương để xích gần nhau hơn":
        - Ưu đãi 50% cho tất cả sản phẩm fullsize.
        - Tặng kèm combo sữa rửa mặt và tẩy trang bí đao với hóa đơn trên 2 triệu đồng.
    - Kết thúc bằng lời kêu gọi hành động thân thiện.
    
    Đây là ví dụ về tin nhắn bạn có thể lấy cảm hứng về nội dung và phong cách, nhưng hãy tạo ra một phiên bản khác đa dạng hơn về từ ngữ và cách diễn đạt:
    ---
    {sample_message}
    ---
    Tin nhắn của bạn:
    """

    prompt = PromptTemplate.from_template(promo_template)
    
    chain = prompt | llm

    response = chain.invoke({
        "sample_message": sample_message # Truyền tin nhắn mẫu vào prompt
    })
    
    return response.content