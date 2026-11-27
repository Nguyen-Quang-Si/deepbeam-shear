# Hướng dẫn Deploy ứng dụng OpenBeam-ML lên Streamlit Community Cloud (Miễn phí 100%)

Thư mục này đã được chuẩn bị đầy đủ 100% các tệp cần thiết để bạn đưa ứng dụng lên mạng Internet qua **Streamlit Community Cloud** (tương tự như link `https://flatslabswithopening-3cv4z4eknhg6afxk27ee5z.streamlit.app/`).

---

## Cấu trúc các tệp đã chuẩn bị sẵn:
```text
Streamlit_Deploy/
├── app.py                      # Mã nguồn chính của ứng dụng web
├── requirements.txt            # Danh sách thư viện Python cần cài đặt
├── .gitignore                  # Bỏ qua các tệp tạm không cần thiết
├── assets/
│   └── beam_opening.png        # Sơ đồ hình học dầm cao có lỗ mở
└── models/
    ├── best_model.joblib       # Mô hình học máy tối ưu (XGBoost ~1.0 MB)
    └── app_meta.json           # Dữ liệu miền biên (Min-Max) và thông số kỹ thuật
```

---

## 3 Bước đơn giản để đưa ứng dụng lên Streamlit Cloud:

### Bước 1: Tạo Repository trên GitHub
1. Đăng nhập vào tài khoản [GitHub](https://github.com/) của bạn (nếu chưa có thì đăng ký miễn phí).
2. Nhấn nút **New** (Tạo repository mới):
   * Đặt tên Repository, ví dụ: `openbeam-ml` hoặc `deepbeam-opening-shear`.
   * Chọn chế độ **Public**.
   * Nhấn **Create repository**.

### Bước 2: Tải các tệp trong thư mục `Streamlit_Deploy` lên GitHub
Có 2 cách rất dễ dàng:
* **Cách 1 (Kéo thả trực tiếp trên trình duyệt - Dễ nhất):**
  1. Trên trang repository vừa tạo ở GitHub, nhấn **uploading an existing file**.
  2. Kéo toàn bộ các tệp và thư mục trong `Streamlit_Deploy` thả vào cửa sổ trình duyệt:
     * Kéo file `app.py`
     * Kéo file `requirements.txt`
     * Kéo thư mục `assets` (chứa `beam_opening.png`)
     * Kéo thư mục `models` (chứa `best_model.joblib` và `app_meta.json`)
  3. Nhấn nút **Commit changes** ở phía dưới để lưu lại.

* **Cách 2 (Dùng lệnh Git nếu bạn quen dùng terminal):**
  ```bash
  cd "C:\Users\OWNER\Desktop\Theory\san lo mo - a Khuong\Streamlit_Deploy"
  git init
  git add .
  git commit -m "Deploy OpenBeam-ML app"
  git branch -M main
  git remote add origin https://github.com/<tai-khoan-github-cua-ban>/<ten-repo>.git
  git push -u origin main
  ```

---

### Bước 3: Deploy lên Streamlit Cloud
1. Truy cập vào [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấn nút **New app** (hoặc **Create app**):
   * **Repository**: Chọn repository bạn vừa tạo ở Bước 1 (ví dụ: `your-username/openbeam-ml`).
   * **Branch**: Chọn `main`.
   * **Main file path**: Điền `app.py`.
   * **App URL (tùy chọn)**: Bạn có thể tự đặt tên miền phụ đẹp mắt, ví dụ: `openbeam-ml.streamlit.app`.
3. Nhấn nút **Deploy!**.
4. Chờ khoảng 1-2 phút để Streamlit tự động cài đặt các thư viện trong `requirements.txt` và khởi chạy. 
5. Sau khi hoàn tất, ứng dụng của bạn sẽ có đường link công khai trên toàn thế giới giống như link `flatslabswithopening`!
