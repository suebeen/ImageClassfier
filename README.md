# 실내, 실외, 음식 사진에 대한 ImageClassfier

<img width="344" alt="스크린샷 2021-06-19 오전 12 47 12" src="https://user-images.githubusercontent.com/56287836/122586646-e412ab80-d097-11eb-97b8-d63cd3dc1d6d.png">
음식, 실내, 실외사진 폴더를 각각 만들어서 분류하고 모델 학습 후 예측할 샘플 사진도 폴더를 만들어 저장한 후 numpy array형식으로 변환합니다.

총 45000개의 data가 들어있고 shuffle한 후 학습과 테스트모델에 저장합니다.

모델 구성은 다음과 같습니다.

<img width="546" alt="스크린샷 2021-06-19 오전 12 46 38" src="https://user-images.githubusercontent.com/56287836/122586581-d0ffdb80-d097-11eb-9661-225d07470f40.png">

layer을 추가하면 파라미터 개수가 너무 많아져 2개만 사용했고 사진을 전체적으로 보기 위해 max pooling을 사용했습니다.

모델의 layer구성과 epoch, batch사이즈를 변경해가며 테스트했습니다.

<img width="891" alt="스크린샷 2021-06-19 오전 12 54 44" src="https://user-images.githubusercontent.com/56287836/122587575-f214fc00-d098-11eb-9e58-2e7b885736a3.png">
