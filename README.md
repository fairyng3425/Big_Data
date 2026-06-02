# Critical Reproduction and Modernized Solar Forecasting Project

## 1. Tổng quan đồ án

Đồ án này thực hiện tái lập có phê bình và cải tiến một bài báo khoa học về bài toán dự báo bức xạ/năng lượng mặt trời tại khu vực Errachidia, Morocco.

Hướng triển khai của đồ án gồm 4 phần chính:

1. Tái lập lại hướng tiếp cận của bài báo gốc để hiểu pipeline ban đầu.
2. Phân tích hạn chế của phương pháp trong bài báo gốc.
3. Xây dựng pipeline forecasting nghiêm ngặt hơn theo chuỗi thời gian.
4. Đề xuất và đánh giá các mô hình cải tiến: boosting, deep learning sequence models, ensemble, daily total forecasting và multi-horizon forecasting.

Các nguồn dữ liệu chính được sử dụng là dữ liệu NASA POWER Hourly cho khu vực Errachidia, Morocco. Do dữ liệu gốc của bài báo không được công khai đầy đủ, kết quả reproduction không nhằm so sánh 1-1 tuyệt đối với bài báo gốc, mà nhằm tái lập hướng phương pháp và đề xuất pipeline cải tiến.

## 2. Cấu trúc notebook

| Notebook | Vai trò chính | Ghi chú |
|---|---|---|
| `00_reproduction_baseline_paper_simulation.ipynb` | Tái lập hướng tiếp cận của bài báo gốc | Dùng LR, SVR, RF, ANN/MLP và Pearson correlation theo tinh thần bài báo |
| `01_data_eda_feature_engineering.ipynb` | Tiền xử lý, EDA và feature engineering cho dữ liệu 3 năm | Tạo target `target_1h`, `target_24h`, cyclic features, lag/rolling features, clear-sky proxy, simulated PV features |
| `02_ml_baselines_boosting_ablation.ipynb` | Machine learning / boosting cho `target_1h` và `target_24h` trên dữ liệu 3 năm | Có ablation study, feature importance và error analysis |
| `03_deep_learning_ensemble_interpretability.ipynb` | Deep learning sequence và ensemble cho point forecasting trên dữ liệu 3 năm | Huấn luyện LSTM, GRU, TSMixer-lite, PatchTST-lite, iTransformer-mini và ensemble với boosting |
| `04_daily_total_forecasting_final_comparison.ipynb` | Dự báo tổng năng lượng ngày kế tiếp trên dữ liệu 3 năm | Thử nghiệm ban đầu cho `daily_total`; dữ liệu daily 3 năm còn ít |
| `05_extended_daily_total_forecasting_20y.ipynb` | Mở rộng bài toán daily total với dữ liệu 20 năm | Kết quả daily-total chính; kiểm tra tác động của việc tăng quy mô dữ liệu |
| `06_multihorizon_24h_sequence_forecasting.ipynb` | Multi-horizon 24h forecasting với full feature engineering | Thí nghiệm phụ/đối chứng khi tabular models được hỗ trợ nhiều feature |
| `07_compact_multihorizon_24h_sequence_forecasting.ipynb` | Multi-horizon 24h forecasting với compact feature setting | Kết quả chính cho phần multi-horizon và vai trò của deep sequence models |
| `08_standard_deep_sequence_models_all_tasks.ipynb` | Chạy deep sequence models bản standard trên nhiều task | Kiểm chứng lại TSMixer-standard, PatchTST-standard, iTransformer-standard trên 3 năm, 20 năm, daily total và multi-horizon |
| `09_results_synthesis_all_tasks.ipynb` | Tổng hợp kết quả cuối cùng từ các notebook | Xếp hạng mô hình theo từng task/dataset |

## 3. Thứ tự chạy khuyến nghị

Nên chạy notebook theo thứ tự:

```text
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

Trong đó:

- Notebook `00` là phần reproduction baseline.
- Notebook `01` chuẩn bị dữ liệu và feature cho các notebook sau.
- Notebook `02–03` xử lý point forecasting.
- Notebook `04–05` xử lý next-day daily total forecasting.
- Notebook `06–07` xử lý multi-horizon 24h forecasting.
- Notebook `08` kiểm chứng deep standard models.
- Notebook `09` tổng hợp kết quả cuối cùng cho báo cáo.

Nếu chỉ cần xem kết quả chính, ưu tiên đọc:

```text
00, 01, 02, 05, 07, 08, 09
```

## 4. Dữ liệu cần chuẩn bị

Các file dữ liệu nên đặt trong thư mục project hoặc thư mục dữ liệu tương ứng theo notebook:

```text
morocco_data.csv
nasa_power_errachidia_hourly_2004_2008_LST.csv
nasa_power_errachidia_hourly_2009_2013_LST.csv
nasa_power_errachidia_hourly_2014_2018_LST.csv
nasa_power_errachidia_hourly_2019_2023_LST.csv
```

Trong đó:

- `morocco_data.csv`: dữ liệu 3 năm 2016–2018.
- 4 file NASA POWER 5 năm: dùng để ghép thành bộ dữ liệu 20 năm 2004–2023.

## 5. Các task chính trong đồ án

| Task | Ý nghĩa |
|---|---|
| `target_1h` | Dự báo năng lượng/bức xạ mặt trời sau 1 giờ |
| `target_24h` | Dự báo cùng giờ ngày hôm sau |
| `daily_total` | Dự báo tổng năng lượng mặt trời của ngày kế tiếp |
| `multi-horizon 24h` | Dùng 168 giờ quá khứ để dự báo toàn bộ 24 giờ tiếp theo |

## 6. Nhóm mô hình được sử dụng

| Nhóm mô hình | Mô hình |
|---|---|
| Linear / classical ML | Linear Regression, Ridge, SVR |
| Tree / boosting | Random Forest, HistGradientBoosting, XGBoost, CatBoost |
| Neural network cơ bản | MLP / ANN |
| Deep sequence models | LSTM, GRU |
| Modern deep sequence backbones | TSMixer, PatchTST, iTransformer |
| Ensemble | Weighted ensemble giữa boosting và deep sequence models |

## 7. Quy tắc chọn mô hình và báo cáo kết quả

Đồ án sử dụng quy tắc lựa chọn mô hình theo validation set:

```text
Train set      : dùng để huấn luyện mô hình.
Validation set : dùng để chọn mô hình / cấu hình / ensemble tốt nhất.
Test set       : dùng để báo cáo hiệu năng cuối cùng.
```

Metric ranking theo từng task:

| Task | Metric dùng để chọn / sort |
|---|---|
| `target_1h` | `val_daylight_RMSE` |
| `target_24h` | `val_daylight_RMSE` |
| `daily_total` | `val_RMSE` |
| `multi-horizon 24h` | `val_daylight_RMSE` |
| Derived h1/h24 từ multi-horizon | Chỉ dùng tham khảo, không trộn trực tiếp với direct point forecasting |

Trong báo cáo nên hiển thị cả validation metric và test metric. Validation metric dùng để giải thích vì sao chọn mô hình, còn test metric dùng để đánh giá khả năng tổng quát hóa cuối cùng.

## 8. Các thư mục output chính

Các notebook lưu kết quả vào những thư mục sau:

```text
outputs/
outputs_02/
outputs_03/
outputs_04/
outputs_05/
outputs_06/
outputs_07/
outputs_08/
outputs_09/

figures/
figures_02/
figures_03/
figures_04/
figures_05/
figures_06/
figures_07/
figures_08/

```

Các file quan trọng nhất để viết báo cáo thường nằm trong:

```text
outputs_02/02_summary_for_report.csv
outputs_03/03_results_all_models.csv
outputs_05/05_final_daily_comparison.csv
outputs_07/07_final_summary_for_report.csv
outputs_08/08_results_all_models.csv
outputs_09/09_corrected_final_results_for_report.csv
outputs_09/09_corrected_best_by_task_dataset.csv
```

## 9. Kết quả và nhận xét chính

Một số kết luận thực nghiệm chính:

1. Các mô hình boosting như XGBoost, CatBoost và HistGradientBoosting rất mạnh khi dữ liệu được feature-engineer tốt.
2. Với `target_1h`, ensemble giữa boosting và deep models đạt kết quả rất cạnh tranh trên dữ liệu 3 năm.
3. Với `target_24h`, dữ liệu 20 năm giúp deep standard models cải thiện rõ rệt, đặc biệt là TSMixer-standard và PatchTST-standard.
4. Với `daily_total`, mở rộng dữ liệu từ 3 năm lên 20 năm giúp kết quả tốt hơn đáng kể.
5. Với multi-horizon 24h, ensemble giữa boosting và TSMixer-lite trong compact setup đạt kết quả tốt nhất.
6. Các kết quả derived từ multi-horizon như h1/h24/daily total chỉ nên xem là phân tích phụ, không trộn trực tiếp với direct point forecasting.

## 10. Lưu ý phương pháp luận

- Không nên so sánh trực tiếp 1-1 kết quả với bài báo gốc vì dữ liệu gốc của tác giả không được công khai đầy đủ.
- Notebook `00` là reproduction-style baseline, không phải pipeline chính cuối cùng.
- Notebook `07` được xem là kết quả chính cho multi-horizon compact forecasting.
- Notebook `08` kiểm chứng bổ sung rằng deep standard models có thể cải thiện khi dùng dữ liệu dài hơn hoặc task phù hợp hơn.
- Notebook `09` dùng để tổng hợp kết quả, nhưng cần đảm bảo phần mô tả cuối notebook thống nhất với logic ranking theo validation metric.
- Simulated PV output là đặc trưng mô phỏng, không phải sản lượng đo thực tế từ nhà máy PV.

## 11. Cài đặt môi trường

Các thư viện chính:

```bash
pip install numpy pandas matplotlib scikit-learn joblib xgboost catboost torch torchvision torchaudio
```

Nếu dùng GPU NVIDIA, nên cài PyTorch bản CUDA phù hợp với máy. Ví dụ:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Kiểm tra GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```
