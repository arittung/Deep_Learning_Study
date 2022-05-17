## YOLOv5로 motorcycle detection 후, 총 개수 계산하여 시간대별 교통량 파악하기


### 1. 환경 세팅하기
- [yolov5 학습 튜토리얼](https://lynnshin.tistory.com/47) 참고하여 **1. 환경 세팅**까지 완료할 것

<br>

### 2. 모델 학습시키기
- [울고넘는 딥러닝](https://minding-deep-learning.tistory.com/19) 참고하여 **모델 학습 후 사용**까지 진행하였음.

#### 1. **kaggle - car dataset**을 이용하여 학습시켰다.
  - 학습시킨 데이터 셋 : [kaggle - Car_detection](https://www.kaggle.com/datasets/ahmedhaytham/car-detection)

- data.yaml 파일의 내용은 다음과 같다.
<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/167643816-f417389d-e19b-4a56-9e6d-daa2729ee0ad.png" style="width:200px"></p>

#### 2. **Microsoft COCO 2020 Dataset**을 이용하여 학습시켰다.
  - 학습시킨 데이터 셋 : [roboflow - Microsoft COCO 2017 Dataset](https://public.roboflow.com/object-detection/microsoft-coco-subset)

- data.yaml 파일의 내용은 다음과 같다.
<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/168790504-8918c7da-6867-4451-bf6c-7c451c86ad59.png" style="width:300px"></p>

<br>

- dir 구조는 다음과 같다. (datset의 train, valid 폴더를 ./yolov5/에도 넣어줘야 코드가 작동했다.)

./yolov5/ | ./yolov5/dataset/
--|--
<img src="https://user-images.githubusercontent.com/53934639/167644914-32e65f66-926b-4179-9207-35f39fab2be1.png" style="width:200px">|<img src="https://user-images.githubusercontent.com/53934639/167645334-7bf2d672-e8aa-46fd-a704-c81eeb4df8fa.png" style="width:200px">





- **preprocessing.ipynb** 파일을 통해 이미지의 주소들을 txt파일로 모아준 뒤 경로를 재설정 해준다.

- 학습을 시키기 전 **YOLOv5의 모델 중 어떤 것을 사용할지 결정**한다.(나는 YOLOv5s 사용했다)
  - 크기가 크면 클수록 복잡해지고 정확성이 높아지는 대신, 시간이 오래걸리고 GPU의 메모리를 많이 차지한다.
<p align="center">
<img src="https://user-images.githubusercontent.com/53934639/167777596-aa4a2a1a-83dd-44a3-a9e2-68348f01ef8c.png" style="width:500px"></p>



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

- 이때, **오토바이의 총 개수를 파악**하기 위해 **detect.py**를 수정했다.

- line 112에 motorcycle list를 정의한다.
```
    motorcycle=[]
```

- line 155에 total, tot_motorcycle 변수 초기화하고, line 166에서 클래스가 motorcycle인 것만 골라 숫자를 더해준다.

```
            total = 0
            tot_motorcycle =0
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    
                    if(names[int(c)] =="bus" or names[int(c)] =="car" or names[int(c)] =="truck" or names[int(c)] == "motorcycle"):
                        total += int(f"{n}")
                        
                        if(names[int(c)] == "motorcycle"):
                            tot_motorcycle += int(f"{n}")
                            
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                                        
                print("------------------------", total)
```
  - line 195에 한 프레임의 오토바이 총 개수를 적어주고, line 197에서 화면에 나타내어 준다.
  - 이때 글자 크기, 모양, 색, 위치 등을 조절할 수 있다.

```
# Stream results
            im0 = annotator.result()
            im0 = cv2.putText(im0, str(total)+" Cars ", (10, 200), 0, 2, (0, 0, 255), 2, 8);
            im0 = cv2.putText(im0, str(tot_motorcycle)+" Motorcycles ", (10, 100), 0, 2, (0, 0, 255), 2, 8);
            #if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(20)  # 1 millisecond
```

- 최종 terminal 화면에 1분 길이 영상의 각 프레임에 나타난 오토바이 개수를 평균으로 보여준다.
```
# Print results
    print(np.mean(motorcycle), " motorcycles")

```

<br>

---

- detect 결과 파일은 **runs/detect/exp** 에서 확인할 수 있다.


### custom dataset으로 훈련시킨 경우

```
python detect.py --source ./video/cctv.mp4 --weights ./runs/train/yolov5_cars/weights/best.pt --img 640 --conf 0.5 
```

### pre-trained yolov5 쓰는 경우 

- [Pre-trained 모델 다운로드 경로](https://github.com/ultralytics/yolov5/releases)
- 실험 결과 yolov5x가 제일 좋았다.

```
python detect.py --source ./video/cctv.mp4 --weights ./runs/yolov5x.pt
```

<br>

### 🙂 역삼역 yolov5X

![image](https://user-images.githubusercontent.com/53934639/168793988-5593850b-fad3-411b-af0b-36122116fa5e.png)


