# 딥러닝에 필요한 케라스 함수 호출
from keras.models import load_model
from keras.utils import np_utils

# 필요 라이브러리 호출(PIL은 이미지파일 처리위함)
from PIL import Image
import numpy as np
import sys

def main(argv):

    # test.png 파일 열어서 L(256단계 흑백이미지)로 변환
    img = Image.open(argv).convert("L")
    #img.show()

    # 이미지를 784개 흑백 픽셀로 사이즈 변환
    img = np.resize(img, (1, 784))

    # 데이터를 모델에 적용할 수 있도록 가공
    test_data = ((np.array(img) / 255) - 1) * -1

    # 모델 불러오기
    model = load_model('Predict_Model.h5')
    model.summary()
    # 클래스 예측 함수에 가공된 테스트 데이터 넣어 결과 도출
    res = model.predict_classes(test_data)
    print("예측 숫자:", res)

if __name__ == "__main__":
    main(sys.argv[1])