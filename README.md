# webcam_depth
webカメラの画像からリアルタイムで深度解析をします。

## 使い方
```
pip install -r requirements.txt
python main.py
```
webカメラの種類を変えたい場合は `main.py` の `61` 行目 `cap = cv2.VideoCapture(0)` の部分を変更してください。
