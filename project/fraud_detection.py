import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.decomposition import PCA
pd.options.display.max_rows= 500
pd.options.display.max_columns= 500


############## 1. 데이터

#데이터 불러오기
train_identity = pd.read_csv("./data/project1/train_identity.csv", index_col='TransactionID')
train_transaction = pd.read_csv("./data/project1/train_transaction.csv", index_col='TransactionID')
test_identity = pd.read_csv("./data/project1/test_identity.csv", index_col='TransactionID')
test_transaction = pd.read_csv("./data/project1/test_transaction.csv", index_col='TransactionID')

# test_identity 컬럼 이름 변경
test_identity.columns = ['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08',
                         'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16',
                         'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
                         'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
                         'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
                         'DeviceInfo']

# print("before merging:")
# print("train_transaction.shape: ", train_transaction.shape) #(590540, 392)
# print("test_transaction.shape: ", test_transaction.shape) #(506691, 392)
# print("train_identity.shape: ", train_identity.shape) #(144233, 40)
# print("test_identity.shape: ", test_identity.shape, "\n") #(141907, 40)

# index 기준으로 merge
train = pd.merge(train_transaction, train_identity, how='left', left_index=True, right_index=True)
test = pd.merge(test_transaction, test_identity, how='left', left_index=True, right_index=True)
# print("train : ",train.shape) #(590540, 433)
# print("test: ",test.shape) #(506691, 432)

# y값 분리
y_train = train_transaction['isFraud']
x_train = train.drop('isFraud', axis=1)

# 전처리 위해 train, test 값 합쳐줌
all_data = pd.concat([x_train, test])
# print(all_data.shape) #(1097231, 432)


# 컬럼 추가 (애플 제품이면 1 아니면 0)
apple = ['iOS Device','iPhone','MacOS']
all_data["DeviceInfo"] = all_data["DeviceInfo"].astype('str')
all_data['IsApple'] = all_data["DeviceInfo"].map(lambda x: 1 if any(a in x for a in apple) else 0)
# print(all_data['IsApple'].value_counts())

# 컬럼 추가 (삼성 제품이면 1 아니면 0)
samsung = ['SAMSUNG', 'SM', "GT-"]
all_data["DeviceInfo"] = all_data["DeviceInfo"].astype('str')
all_data['IsSamsung'] = all_data["DeviceInfo"].map(lambda x: 1 if any(s in x for s in samsung) else 0)
# print(all_data['IsSamsung'].value_counts())

# 컬럼 추가 (거래가 일어난 요일)
all_data['ts_day'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)
# print(all_data['ts_day'].value_counts().sort_index())

# 컬럼 추가 (거래가 일어난 시간)
all_data['ts_hour'] = np.floor(test['TransactionDT'] / 3600) % 24
# print(all_data['ts_hour'].value_counts().sort_index())

# 컬럼 추가 (이메일 도메인 카테고리 & 서비스 국가 카테고리)
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 
          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 
          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 
          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 
          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']

for c in ['P_emaildomain', 'R_emaildomain']:
    all_data[c[0]+'_emaildomain_c'] = all_data[c].map(emails)
    all_data[c[0]+'_email_country'] = all_data[c].map(lambda x: str(x).split('.')[-1])
    all_data[c[0]+'_email_country'] = all_data[c[0]+'_email_country'].map(lambda x: x if str(x) not in us_emails else 'us')

# print(all_data['P_emaildomain_c'].value_counts())
# print(all_data['P_email_country'].value_counts())
# print(all_data['R_emaildomain_c'].value_counts())
# print(all_data['R_email_country'].value_counts())


# 컬럼 추가 (브라우저 카테고리)
all_data['Browser'] = np.NaN

all_data.loc[all_data['id_31'].str.contains('chrome', na=False), 'Browser'] = 'Chrome'
all_data.loc[all_data['id_31'].str.contains('firefox', na=False), 'Browser'] = 'Firefox'
all_data.loc[all_data['id_31'].str.contains('safari', na=False), 'Browser'] = 'Safari'
all_data.loc[all_data['id_31'].str.contains('edge', na=False), 'Browser'] = 'Edge'
all_data.loc[all_data['id_31'].str.contains('ie', na=False), 'Browser'] = 'IE'
all_data.loc[all_data['id_31'].str.contains('samsung', na=False), 'Browser'] = 'Samsung'
all_data.loc[all_data['id_31'].str.contains('opera', na=False), 'Browser'] = 'Opera'
all_data['Browser'].fillna("others", inplace=True)
# print(all_data['Browser'].value_counts())


############## 2. 전처리
# 결측치 수 확인 + 80%이상 결측치면 열을 데이터에서 빼줌 (완전 제거법 -> 결측치가 80%에 가까운 경우 그 변수 자체를 제거하는 방식) 
# 결측치를 지우면서 데이터 자체의 편향(bias)이 생길 수 있어서 조심히 접근해야함
col_drop_list = []
for i in all_data.columns:
    # print(i, "missing values: ", all_data[i].isnull().sum())
    missing_percent = all_data[i].isnull().sum() / len(all_data[i]) * 100
    if missing_percent > 80:
        col_drop_list.append(i)

# print("data drop list : ", col_drop_list)
# print("data drop col # : ", len(col_drop_list)) #74열
# print("[before drop] all_data.shape: ", all_data.shape) #(1097231, 441)
all_data = all_data.drop(columns = col_drop_list)
# print("[after drop] all_data.shape: ", all_data.shape) #드랍 후 (1097231, 367)

# 컬럼 타입 확인 (범주형, 수치형)
c = (all_data.dtypes == 'object') #True/Fase 반환
n = (all_data.dtypes != 'object')
cat_cols = list(c[c].index) #True의 index만 리스트로 변환
num_cols = list(n[n].index) 
# print(cat_cols)
# print("number categorical features: ", len(cat_cols), "\n") #31열
# print(num_cols)
# print("number numerical features: ", len(num_cols)) #336열

# 범주형 데이터 label encoding
# 사이킷런은 문자열 값을 입력 값으로 처리 하지 않기 때문에 숫자 형으로 변환
# 원핫인코딩 / 라벨인코딩 설명
le = LabelEncoder()
for i in cat_cols:
    all_data[i] = le.fit_transform(list(all_data[i]))
# print(all_data[['ProductCD','DeviceType']])

# MinMaxScaler 수치형 칼럼만
# StandardScaler => 평균과 표준편차 사용하기 때문에 부적합
# 2차원 -> 수치형 칼럼만 뽑아서 numpy array로 바꿔줘야되는 불편함
for i in num_cols:
    all_data[i] = (minmax_scale(all_data[i], feature_range=(0,1)))

# 결측치 처리 수치형만 -1
for i in num_cols:
    # print(col, "missing values: ", all_data[col].isnull().sum())
    all_data[i].fillna(-1, inplace=True)

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# np.set_printoptions(threshold=np.inf)
# print(all_data.columns.tolist())

train2_x = all_data[:len(x_train)]
train2_y = y_train
test2 = all_data[len(x_train):]
print("train : ",train2_x.shape) #(590540, 367)
print("test: ",test2.shape) #(506691, 367)
print("train_y: ",train2_y.shape) #(590540,)

#npy 저장
# x_train = train2_x.to_numpy()
# y_train = train2_y.to_numpy()
# x_test = test2.to_numpy()
# np.save('./data/project1/x_train.npy', arr=x_train)
# np.save('./data/project1/y_train.npy', arr=y_train)
# np.save('./data/project1/x_test.npy', arr=x_test)

# Vxxx 컬럼 PCA 적용하기! 292개의 컬럼 -> pca 0.9나 0.99 (누적된 분산의 비율이 95%, 99%가 되는 주성분 축, 차원을 선택)
def PCA_change(df, cols, n_components, prefix='PCA_'):
    pca = PCA(n_components=n_components, random_state=77)
    pc = pca.fit_transform(df[cols])
    print('원래 차원(픽셀) 수 :', len(cols)) #292 컬럼
    print('선택한 차원(픽셀) 수 :', pca.n_components_) #0.95: 4 컬럼 / 0.99: 8컬럼

    pc_df = pd.DataFrame(pc, index=df.index) #pca 된 컬럼들 df
    pc_df.rename(columns=lambda x: str(prefix)+str(x), inplace=True) #pca 된 컬럼들 이름 바꾸기

    df = df.drop(cols, axis=1) #pca할 컬럼들 원래 df에서 드롭
    print("dropped df shape : ", df.shape)
    # print(df)
    print("pc_df shape : ", pc_df.shape)
    # print(pc_df)
    new_df = pd.concat([df,pc_df], axis=1) #원래 df에 pca된 컬럼들 합치기
    # print("concated df shape : ",new_df.shape)
    return new_df

# print(all_data.columns.get_loc("V1")) #45
# print(all_data.columns.get_loc("V321")) #336
v_cols = all_data.columns[45:337]
# print("V columns : ", v_cols)

# PCA 95%
all_data2 = PCA_change(all_data, v_cols, n_components=0.95, prefix='PCA_V_')
print("원래 데이터 shape : ", all_data.shape) #원래 데이터 shape :  (1097231, 367)
print("PCA(0.95) 데이터 shape : ", all_data2.shape) #PCA(0.95) 데이터 shape :  (1097231, 79)

# PCA 99%
all_data3 = PCA_change(all_data, v_cols, n_components=0.99, prefix='PCA_V_')
print("원래 데이터 shape : ", all_data.shape) #원래 데이터 shape :  (1097231, 367)
print("PCA(0.99) 데이터 shape : ", all_data3.shape) #PCA(0.99) 데이터 shape :  (1097231, 83)

# train test 다시 나눠주기
# PCA 95%
train3_x = all_data2[:len(x_train)]
train3_y = y_train
test3 = all_data2[len(x_train):]
print("train : ",train3_x.shape) #(590540, 79)
print("test: ",test3.shape) #(506691, 79)
print("train_y: ",train3_y.shape) #(590540,)

# PCA 99%
train4_x = all_data3[:len(x_train)]
train4_y = y_train
test4 = all_data3[len(x_train):]
print("train : ",train4_x.shape) #(590540, 83)
print("test: ",test4.shape) #(506691, 83)
print("train_y: ",train4_y.shape) #(590540,)

#npy 저장
x_train_95 = train3_x.to_numpy()
y_train_95 = train3_y.to_numpy()
x_test_95 = test3.to_numpy()
np.save('./data/project1/x_train_95.npy', arr=x_train_95)
np.save('./data/project1/y_train_95.npy', arr=y_train_95)
np.save('./data/project1/x_test_95.npy', arr=x_test_95)

x_train_99 = train4_x.to_numpy()
y_train_99 = train4_y.to_numpy()
x_test_99 = test4.to_numpy()
np.save('./data/project1/x_train_99.npy', arr=x_train_99)
np.save('./data/project1/y_train_99.npy', arr=y_train_99)
np.save('./data/project1/x_test_99.npy', arr=x_test_99)

