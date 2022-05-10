## YOLOv5로 자동차 detection 후, 총 개수 계산하여 교통량 파악하기


### 1. 환경 세팅하기
- [yolov5 학습 튜토리얼](https://lynnshin.tistory.com/47) 참고하여 **"1. 환경 세팅"**까지 완료할 것

<br>

### 2. 모델 학습시키기
- [울고넘는 딥러닝](https://minding-deep-learning.tistory.com/19) 참고하여 **모델 학습 후 사용**까지 진행하였음.

- 나의 경우, kaggle - car dataset을 이용하여 학습시켰다.
  - 학습시킨 데이터 셋 : [kaggle - Car_detection](https://www.kaggle.com/datasets/ahmedhaytham/car-detection)


- dir 구조는 다음과 같다. (datset의 train, valid 폴더를 ./yolov5/에도 넣어줘야 코드가 작동했다.)

./yolov5/ | ./yolov5/dataset/
--|--
<img src="https://user-images.githubusercontent.com/53934639/167644914-32e65f66-926b-4179-9207-35f39fab2be1.png" style="width:200px">|<img src="https://user-images.githubusercontent.com/53934639/167645334-7bf2d672-e8aa-46fd-a704-c81eeb4df8fa.png" style="width:200px">



- data.yaml 파일의 내용은 다음과 같다.
<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/167643816-f417389d-e19b-4a56-9e6d-daa2729ee0ad.png" style="width:200px"></p>

- preprocessing.ipynb 파일을 통해 이미지의 주소들을 txt파일로 모아준 뒤 경로를 재설정 해준다.

- 다음 코드를 통해 학습을 시작한다.
  - parameter 종류
    - --img : 이미지 크기
    - --batch : 배치 크기
    - --epochs : epoch 크기
    - --data : 데이터파일 (data.yaml 파일 경로 지정)
    - --cfg : 위에서 정한 모델 크기 (yolov5/models 폴더에 yaml파일로 저장되어 있음)
    - --weights : 미리 학습된 모델로 학습할 경우 (yolov5s.pt 등의 형식으로 다운로드 가능)
    - --name : 학습된 모델의 이름

```
python train.py --img 640 --batch 16 --epochs 20 --data ./dataset/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name yolov5_cars 
```




<br>

## 3. 학습된 모델 사용해보기

- 이때, 차량의 총 개수를 파악하기 위해 detect.py를 수정했다.

- detect 결과 파일은 runs/detect/exp 에서 확인할 수 있다.

```
python detect.py --source cctv.mp4 
```

<br>

### 🙂 결과물

![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/53934639/167647089-5540d88a-6eef-465d-a139-65a0ddd4d967.gif)
