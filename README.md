# DACON-Basic_Purchase-Prediction
# Preprocessing

target값 Categorize

```python
train = pd.read_csv('train.csv')

data_train_T=train['target']
target=[]
for i in range(len(data_train_T)):
    if data_train_T.iloc[i] < 250:
        target.append(1)
    elif data_train_T.iloc[i] < 1000:
        target.append(2)
    elif data_train_T.iloc[i] < 1800:
        target.append(3)
    else:
        target.append(4)
        
train["Range"] = target
```
<img width="535" alt="image" src="https://user-images.githubusercontent.com/24906028/173172880-c28b3d9a-2431-4cd3-bd27-e5f581c4884e.png">
![image](https://user-images.githubusercontent.com/24906028/173172895-88842093-5ae3-45a4-8ca8-0479ccee0f4b.png)

Marital_Status에서 'Alone', 'YOLO', 'Absurd'는 데이터가 극히 작고 Single에 범주에 포함되므로 single로 통합

Kidhome와 Teenhome에서 자녀 및 청소년을 2명둔 사람은 소수 → Kidhome, Teenhome 합쳐서 Binary 변환

Education 중 빈도가 가장 낮은 Basic은 2n Cycle와 합쳐서 새로운 카테고리로

소득은 주로 25,000~75,000 사이에 분포하며, 150,000 이상의 값을 이상치로서 제거할 수 있으나 이상치와 target의 상관관계가 유의미하다고 판단하여 제거하지 않음
![image](https://user-images.githubusercontent.com/24906028/173172909-ee8d895f-a9e5-44ae-9807-84cc1cb0a4b9.png)

매장, 웹사이트, 카탈로그 구매 수의 총합을 `total`로 나타내고, 각 구매 경로별 비율(0~1 사이)로 나타냈습니다.구매 경로는 매장 구매의 비중이 가장 높은 편이며, 다음으로 웹사이트, 카탈로그 순으로 나타납니다.
총 구매 횟수가 낮을수록 카탈로그의 비중이 0.2 이하로 매우 낮은 편. 총 구매 횟수가 높아질수록 구매경로별 비중의 차이가 좁혀짐 → 카탈로그 비중 피쳐 추가

![image](https://user-images.githubusercontent.com/24906028/173172918-be033046-d70f-4c8f-a28a-c6f570baa29a.png)

평균 `target`값은 연령대가 높을수록 증가하는 추세
연령대별 `target`의 분포를 살펴보면, 80대를 제외한 연령대는 0 가까이에 치우친 분포를 보입니다. 80대의 경우, 평균이 1000인 normal distribution과 비슷한 형태
80대처럼 연령대가 `target`값을 구분하는 키가 될 수 있으므로, 연령대를 학습 모델의 feature로 추가

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3634bf21-d3c4-4b99-9d43-3dc441847fbd/Untitled.png)

자녀수가 1명~2명인 경우, `target`값이 0에 몰리는 비슷한 분포. 이에 비해 자녀 없음(0)은 `target`이 0부터 1,500 사이에 고르게 분포

청소년수`Teenhome`는 모든 케이스(0,1,2)가 0 가까이에 모이는 분포

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1b3e7026-1cf8-4332-8834-7e77658685b4/Untitled.png)

binary 타입으로 변환했을 때 자녀수`Kidhome`은 자녀있음(1)과 자녀없음(0)의 `target` 분포 형태에 크게 차이가 있음

청소년수`Teenhome`은 청소년있음(1)과 청소년없음(0) 모두 0에 쏠리는 분포

청소년없음(0)에서 `target`값이 1000 이상인 구간의 밀도가 비교적 높게 나타나는 것이 특징

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d8b85589-a071-4546-bfd6-0775a74cf1c7/Untitled.png)

`TotalCmp`값에 따라 `target`의 분포가 달라질 수 있음 → 피쳐 추가

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b00193db-70b1-4d6a-a6f7-797b1537a6ef/Untitled.png)

총 구매 횟수`TotalPurchases`가 커질수록 `target`값도 증가하는 추세 → 피쳐 추가

Pandas profiling High Correlation

`Income`-`Kidhome`-`NumWebPurchases`-`NumStorePurchases`-`NumStorePurchases`-`NumWebVisitsMonth`-`AcceptedCmp1`-`AcceptedCmp5`-`target`

`NumDealsPurchases`-`Teenhome`

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/10bd4f0b-dd66-4765-b1b3-d4cea0a8b9df/Untitled.png)

상관관계 낮은 Recency, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4 피쳐 drop

왜곡된 분포는 모델 학습에 안좋은 영향을 줄 수 있다. 높은 skewness를 가지고 있는 `NumDealsPurchases`에 Yeo-Johnson transformation

```python
# 필요없는 feature drop

train = train.drop('id', axis = 1)
train = train.drop('Recency', axis = 1)

# 가장 과거 시점의 회사 등록일로부터 며칠이 지났는지를 뜻하는 Pass_Customer변수

train["Dt_Customer"] = pd.to_datetime(train["Dt_Customer"], format='%d-%m-%Y')
train_diff_date = train["Dt_Customer"] - train["Dt_Customer"].min()
train["Pass_Customer"] = [i.days for i in train_diff_date]
train = train.drop('Dt_Customer', axis = 1)

# 이상치 제거
# 100세 이상 제거

train = train.loc[(2022-train['Year_Birth']+1)<100, :].reset_index(drop=True)

# 나이 연령대로 나타내기

train['age'] = 2022 - train['Year_Birth'] + 1
train['age_group'] = train['age'] // 10 * 10
train = train.drop('age', axis = 1)
train = train.drop('Year_Birth', axis = 1)

# 총 구매횟수 낮을수록 카탈로그 비중 매우 낮음

train['TotalPurchases'] = train['NumCatalogPurchases']+train['NumStorePurchases']+train['NumWebPurchases']
train['RCatalogPurchases'] = np.where(train['TotalPurchases']==0, 0, train['NumCatalogPurchases'] / train['TotalPurchases'])

# 캠페인 제안 수락 횟수

train['TotalCmp'] = train['AcceptedCmp1']+train['AcceptedCmp2']+train['AcceptedCmp3']+train['AcceptedCmp4']+train['AcceptedCmp5']+train['Response']

train = train.drop('AcceptedCmp2', axis = 1)
train = train.drop('AcceptedCmp3', axis = 1)
train = train.drop('AcceptedCmp4', axis = 1)

# 학력 Basic, 2ncycle 합치기

train['Education'] = np.where(train['Education'] == 'Basic', 'Under', train['Education'])
train['Education'] = np.where(train['Education'] == '2n Cycle', 'Under', train['Education'])

# 결혼 'Alone', 'YOLO', 'Absurd'을 'Single'에 합치기

train['Marital_Status'] = np.where(train['Marital_Status'] == 'Alone', 'Single', train['Marital_Status'])
train['Marital_Status'] = np.where(train['Marital_Status'] == 'YOLO', 'Single', train['Marital_Status'])
train['Marital_Status'] = np.where(train['Marital_Status'] == 'Absurd', 'Single', train['Marital_Status'])

# 자녀 수, 청소년 수 합쳐서 Binary 변환

train['Dependents'] = train['Kidhome'] + train['Teenhome']
train['Dependents'] = np.where(train['Dependents'] >= 1, 1, train['Dependents'])

# 높은 skewness를 가지고 있는 NumDealsPurchases 변수
# Yeo-Johnson transformation
jy = PowerTransformer(method = 'yeo-johnson')
jy.fit(train['NumDealsPurchases'].values.reshape(-1, 1))
x_yj = jy.transform(train['NumDealsPurchases'].values.reshape(-1, 1))

train['NumDealsPurchases'] = x_yj

#  Education와 Marital_Status 피쳐 Label Encoding

def make_label_map(dataframe):
    label_maps = {}
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            label_map = {'unknown' : 0}
            for i, key in enumerate(dataframe[col].unique()):
                label_map[key] = i
            label_maps[col] = label_map
    return label_maps

def label_encoder(dataframe, label_map):
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            dataframe[col] = dataframe[col].map(label_map[col])
    return dataframe

	train = label_encoder(train, make_label_map(train))
```

# Train

NMAE로 평가

```python
import numpy as np
from sklearn.metrics import mean_squared_error

def NMAE(true, pred):

    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    
    return score
```

```python
data_train_X = train.drop('target', axis = 1) #training 데이터에서 피쳐 추출
data_train_y = train['target'] #training 데이터에서 소비량 추출
```

```python
skf = StratifiedKFold(n_splits = 10, random_state = 42, shuffle = True) #총 10번의 fold 진행
n = 0 #x번째 fold인지 기록

fold_target_pred = []
fold_score = []

for train_index, valid_index in skf.split(data_train_X, data_train_X['Range']): #range 기준으로 stratified k fold 진행
    n += 1
    
    val_pred_name = [] #validation pred model 이름 저장
    val_pred = []      #validation set pred 결과 저장
    target_pred = []   #test set pred 결과 저장
    
    train_X = np.array(data_train_X.drop("Range", axis = 1)) #분배된 학습을 위해 생성한 Range feature 제거
    train_Y = np.array(data_train_y)
    
    X_train, X_valid = train_X[train_index], train_X[valid_index]
    y_train, y_valid = train_Y[train_index], train_Y[valid_index]
    
    X_test = np.array(test)

    ### Create Model ###
    
    ###모델을 생성하고 집어넣으면 됩니다.
    
    ### LGBMRegressor ###
#     model = LGBMRegressor(n_estimators=134,max_depth=16,random_state = 42) #추가적으로 하이퍼파라미터 튜닝 필요
#     model.fit(X_train, y_train) # 모델 학습
    
#     val_pred_name.append("LGBMRegressor")      # 모델 이름 저장
#     val_pred.append(model.predict(X_valid))   # validation set pred 결과 저장
#     target_pred.append(model.predict(X_test)) # test set pred 결과 저장
    
    ### XGBRegressor ###
    model = XGBRegressor(n_estimators=194, max_depth=7, random_state = 42) #추가적으로 하이퍼파라미터 튜닝 필요
    model.fit(X_train, y_train)
    
    val_pred_name.append("XGBRegressor")      # 모델 이름 저장
    val_pred.append(model.predict(X_valid))   # validation set pred 결과 저장
    target_pred.append(model.predict(X_test)) # test set pred 결과 저장
    
    ### CatBoostRegressor ###
    model = CatBoostRegressor(n_estimators=1200,max_depth=8,random_state = 42, silent=True) #추가적으로 하이퍼파라미터 튜닝 필요
    model.fit(X_train, y_train)
    
    val_pred_name.append("CatBoostRegressor")      # 모델 이름 저장
    val_pred.append(model.predict(X_valid))   # validation set pred 결과 저장
    target_pred.append(model.predict(X_test)) # test set pred 결과 저장

    ### RandomForestRegressor ###
    model = RandomForestRegressor(n_estimators=177,max_depth=16,random_state = 42, criterion="mae") #추가적으로 하이퍼파라미터 튜닝 필요
    model.fit(X_train, y_train)
    
    val_pred_name.append("RandomForestRegressor")      # 모델 이름 저장
    val_pred.append(model.predict(X_valid))   # validation set pred 결과 저장
    target_pred.append(model.predict(X_test)) # test set pred 결과 저장    
    
    ### voting ###
    
    ### average validation pred ###
    preds = np.array(val_pred[0])
    for i in range(1, len(val_pred)):
        preds += val_pred[i]
    
    preds = preds/len(val_pred)
    
    ### average target pred ###
    target_preds = np.array(target_pred[0])
    for i in range(1, len(target_pred)):
        target_preds += target_pred[i]
    
    target_preds = target_preds/len(target_pred)
    
    fold_target_pred.append(target_preds) # append final target pred
    
    print("========== fold %d ==========" %(n))
    for i in range(len(val_pred)):
        print("%s model NMAE : %0.4f" %(val_pred_name[i], NMAE(y_valid, val_pred[i].astype(int))))
        
    print("==============================")
    print("Average NMAE %0.4f" %(NMAE(y_valid, preds.astype(int))))
    print("")
    
    fold_score.append(NMAE(y_valid, preds.astype(int)))

total_score = fold_score[0]
for i in range(2, len(fold_score), 1):
    total_score += fold_score[i]
total_score = total_score/(len(fold_score))    

print("==============================")
print("Total Average NMAE %0.4f" %(total_score)) #최종 average score 출력
```

# 회고

전처리에 대한 실험 충분히 하지 못함

- age 카테고리화 하지 않고 해볼걸
- 예측해야하는 label인 `targe` 변수는 정수 타입으로 고객의 제품 총 소비량을 나타내며, 최소 6부터 최대 2525까지 분포 
`target` 값은 대부분 500 이하에 분포되며, 약 50% 정도의 `target` 값이 412 이하로 분포
나머지 50%의 경우, 412 이상부터 최대 2525까지 분포
400이하의 값 중 100 이하에 분포가 집중됨 
→ `target`값이 100 이하인 케이스와 `target`값이 400 이하인 케이스, `target`값이 400 이상인 케이스로 나누어 학습하면 어떨까
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b4edcc9c-03ef-4ea4-9c22-c1436819c755/Untitled.png)
    
- 마지막 캠페인을 포함하여 총 6번의 캠페인 중 제안을 수락한 횟수는 0부터 최대 5회까지 발생
6번의 캠페인 제안 중 어느 하나도 수락하지 않은 0이 가장 높은 비중을 차지
다음으로 캠페인 제안을 1번 수락한 케이스가 높은 비중을 차지하지만, `TotalCmp` 값이 0인 케이스의 약 25%에 불과
→ `TotalCmp` 변수도 어떠한 캠페인 제안도 수락하지 않음(0)과 하나 이상의 제안에 수락함(1)을 나타내는 binary 타입의 변수 추가해볼걸
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4f8abbac-31bf-4b17-a7cd-a4303c9784c3/Untitled.png)
    
- Basic은 다른 학력에 비해 `target`의 분포가 0에 집중된 형태. 데이터 수가 가장 적었던 클래스인만큼 분포도 더 극단적인 형태를 띈 것이 아닐까.
- Basic을 제외한 4가지 학력 기준의 `target`의 분포는 연령대별 `target`분포와 비슷한 형태. → Basic을 합치지 말고 drop?

연령과 소득, 학력과 소득, 연령과 자녀수 같이 변수간 상관관계는 왜 분석하는거?

독립변수 간의 높은 상관관계는 다중공선성을 유발하기 때문에 좋지 않다. 이는 변수 선택, 차원 축소, 규제 등의 방법으로 해결할 수 있고, 모델에 규제를 적용하거나 다중공선성의 영향을 적게 받는다고 생각되는 Decision Tree 베이스의 모델을 사용 할 수 있다.

Lasso, Ridge regression은 Linear regression에 규제를 적용하는 방법. 이 두 모델의 규제를 모두 적용할 수 있는 **Elastic-Net**
이 있다.

`NumDealsPurchases`의 이상치 개수가 46개로 많이 발생했음에도 처리를 하지 않았는데 이상치는 학습에 어떻게 영향을 주고 어떻게 제거하면 좋을지 공부하기

깃헙에 코드 기록하면서 하기. 예전 코드 날리지 말기
