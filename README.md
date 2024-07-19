## BÀI TẬP KẾT THÚC LÝ THUYẾT ##
# BUTTERFLIES & MOTHS IMAGE CLASSIFICATION #
### 1. Mô tả project ###
* Mục tiêu:
  - Sử dụng thư viện Pytorch để training mô hình CNN một cách hiệu quả.
  - Giải quyết bài toán phân loại 50 loài bướm từ bộ dữ liệu Phân loại hình ảnh 50 loài bướm từ kaggle.
  - Convert Pytorch model to ONNX.
  - Triển khai PyTorch bằng Python thông qua API với Flask.
### 2. Butterflies & Moths Image Classification dataset"
  - Bộ dữ liệu Phân loại hình ảnh 50 loài bướm từ Kaggle chứa 4955 hình ảnh cho training, 250 hình ảnh cho validation và 250 hình ảnh để test. Và tất cả hình ảnh đều là hình ảnh RGB 224×224 chiều (có 3 kênh màu).
  - Mỗi folder train, validation và test đều có 50 thư mục con đóng vai trò là labels cho hình ảnh.
  - Cấu trúc thư mục input:
    ```
    ├── input
    │   ├── test
    │   │   ├── adonis
    │   │   │   ├── 1.jpg  
    │   │   ...
    │   │   └── zebra long wing
    │   │   	├── 1.jpg
    │   │   	...
    │   ├── train
    │   │   ├── adonis [96 entries exceeds filelimit, not opening dir]
    │   │   ...
    │   │   └── zebra long wing [108 entries exceeds filelimit, not opening dir]
    │   ├── valid
    │   │   ├── adonis
    │   │   │   ├── 1.jpg
    │   │   ...
    │   │   └── zebra long wing
    │   │   	├── 1.jpg
    │   │   	...
    ```
### 3. Xây dựng model CNN 
#### 3.1. Xác nhận các phép biến đổi để tăng cường dữ liệu.
* Cần thực hiện các phép biến đổi để đảm bảo tất cả hình ảnh được đưa vào mô hình AI đều có cùng kích thước.
  - `transforms.ToTensor()`: chuyển đổi hình ảnh sang tensor (là kiểu dữ liệu đa chiều được sử dụng trong pytorch). Về cơ bản, kỹ thuật này chuyển đổi các pixel của mỗi hình ảnh có màu thành độ sáng của màu, từ 0 đến 255. Các giá trị này được chia cho 255, vì vậy chúng có thể nằm trong khoảng từ 0 đến 1.
  - `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`:  trừ đi giá trị trung bình rồi chia cho độ lệch chuẩn. Với mô hình được đào tạo trước, ta cần sử dụng các giá trị trung bình và độ lệch chuẩn mà Pytorch chỉ định. Có ba giá trị trong giá trị trung bình và độ lệch chuẩn để khớp với từng ảnh RGB.
* Các biến đổi tăng cường dữ liệu được thực hiện bằng thư viện torchvision.transforms.
  - `transforms.Resize((224))`: thay đổi kích thước hình ảnh sao cho cạnh ngắn nhất có chiều dài 224 pixel. Cạnh còn lại được chia tỷ lệ để duy trì tỷ lệ khung hình của hình ảnh.
  - `transforms.RandomHorizontalFlip(p=0.5)`: lật ngẫu nhiên hình ảnh theo chiều ngang.
  - `transforms.RandomVerticalFlip(p=0.5)`: lật ngẫu nhiên hình ảnh theo chiều dọc.
  - `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`: thay đổi ngẫu nhiên độ sáng, độ tương phản và độ bão hòa của hình ảnh.
  - `transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))`: thực hiện các phép biến đổi affine ngẫu nhiên như dịch chuyển, phóng to/thu nhỏ.
  - ...
#### 3.2. Nhập dữ liệu và đưa vào DataLoader.
* Ta input ảnh của mình vào chương trình bằng cách sử dụng thư viện `torchvision.datasets`.
* Ở đây cần chỉ định hai bộ dữ liệu khác nhau, một bộ dữ liệu để đào tạo mô hình (bộ đào tạo – training set) và bộ còn lại để kiểm tra mô hình (bộ xác thực – validation set).
```
# training dataset
train_dataset = datasets.ImageFolder(
    root='../butterflies/input/train',
    transform=train_transform
)
# validation dataset
valid_dataset = datasets.ImageFolder(
    root='../butterflies/input/valid',
    transform=valid_transform
)
```
* Sau đó, cần đưa hình ảnh đã nhập vào Dataloader. Dataloader có thể lấy ra các mẫu dữ liệu ngẫu nhiên, vì vậy, mô hình sẽ không phải xử lý toàn bộ tập dữ liệu một lần. Điều này làm cho việc đào tạo trở nên hiệu quả hơn.
* Ta có thể chỉ định số lượng hình ảnh xử lý cùng một lúc làm batch_size (ví dụ 32 có nghĩa là Dataloader sẽ trả về 32 mẫu cùng một lúc). Ta cũng có thể xáo trộn hình ảnh để nó được huấn luyện ngẫu nhiên vào mô hình với tham số shuffle=True.
```
# training data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True
)
# validation data loaders
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True
)
```
### 3.3. Xây dựng mô hình.
* Ta sử dụng thư viện `torch.nn` để tạo bộ phân loại.
  - `nn.Linear` chỉ định sự tương tác giữa hai lớp. Ta cung cấp cho nó 2 số, chỉ định số lượng nút trong hai lớp. Ví dụ: trong nn.Linear đầu tiên, lớp đầu tiên là lớp đầu vào và ta có thể chọn bao nhiêu số ta muốn trong lớp thứ hai.
  - `nn.ReLU` là chức năng kích hoạt cho các lớp ẩn. Các chức năng kích hoạt giúp mô hình tìm hiểu các mối quan hệ phức tạp giữa đầu vào và đầu ra. Ta sử dụng ReLU trên tất cả các lớp ngoại trừ đầu ra.
  - `nn.Conv2d` là lớp tích chập 2D (2-dimensional convolutional layer). Nó giúp trích xuất các đặc trưng từ hình ảnh đầu vào bằng cách áp dụng các kernel (hoặc filter) qua hình ảnh để phát hiện các đặc trưng như cạnh, góc, và các kết cấu phức tạp. 
  - `nn.MaxPool2d` là lớp pooling tối đa 2D. Nó giúp giảm kích thước không gian của các đặc trưng (feature maps) và làm cho mô hình bớt phức tạp và dễ xử lý hơn.
* Model bốn lớp tích chập. Và mỗi lớp tiếp theo có số lượng out_channels gấp đôi so với lớp trước đó.
* Mỗi lớp tích chập đều được theo sau bởi hàm kích hoạt ReLU và max-pool 2D. Chúng ta chỉ có một lớp tuyến tính với 50 out_features tương ứng với số lượng lớp trong tập dữ liệu. 
* Điều duy nhất cần lưu ý trong mạng nơ-ron này là kích thước kernel. Hai lớp tích chập đầu tiên có kernel kích thước 5×5, sau đó là lớp có kernel 3×3, và lớp cuối cùng lại có kernel kích thước 5×5.




