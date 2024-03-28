# 画像のパターンマッチング
テンプレートとして選んだ画像と類似度が高い部分を抽出する。  
- google colab
https://colab.research.google.com/drive/15K3r2psWsI6RGNC1bY5NYo77hybdf1O9?usp=sharing
- 使用画像の引用元
https://ninchi.life/print12684/

## 具体的な手順について  
### テンプレートの作成  
今回は、「友」の漢字を抽出したいので、「友」の漢字1文字が含まれるように画像を切り取り、これをテンプレートとして使用する。
~~~
#画像を切り取る座標を指定する
ini_x = 451
ini_y = 411
size = (29,29)

template = img[ini_y:ini_y+size[1], ini_x:ini_x+size[0]]

%matplotlib inline

plt.imshow(template)
plt.show()
~~~
### 解像度を落とす・グレースケール化
あとで用いる画像類似度の計算で、1ピクセルずらして元画像とテンプレートの類似度を計算するという操作を繰り返すため、ここで解像度を落とした方が計算が早くなります。
~~~
def degrade_gray(self, img, template, de_res):

    #画像の解像度を下げる→処理を軽くするため
    img_d = cv2.resize(img, dsize=None, fx=de_res, fy=de_res)
    template_d = cv2.resize(template, dsize=None, fx=de_res, fy=de_res)

    #グレースケール化
    img_gray = cv2.cvtColor(img_d, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template_d, cv2.COLOR_RGB2GRAY)

    return img_gray, template_gray
~~~
### 少し回転させたテンプレート画像を作成
半導体チップの場合は顕微鏡の画像上で角度が最大10°程度回転している場合があったため、この操作によってマッチング率が大幅に向上した。
~~~
 #ちょっとずつ角度を変えたtemplate画像を作る
  def rotate_img(self, template, match_angles):

    rot_image_list = []

    for angle in match_angles:

      if len(template.shape) == 3:
        height, width, dim = template.shape
      elif len(template.shape) == 2:
        height, width = template.shape
      #回転の中心を指定する
      center = (int(width/2) , int(height/2))
      #回転行列
      M = cv2.getRotationMatrix2D(center, angle, 1.0)
      rot_image_list.append(cv2.warpAffine(template, M, (width,height)))

    #回転した画像の表示など
    print('-------------解像度を下げて回転させたテンプレートの画像テスト----------------')
    # サブプロットの数に応じて処理を分ける
    if len(rot_image_list) == 1:
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(rot_image_list[0], cmap='gray')
    else:
        fig, axs = plt.subplots(len(rot_image_list))
        # axsが配列ではない場合（サブプロットが1つのみの場合）、配列に変換
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        for i, img in enumerate(rot_image_list):
            axs[i].axis('off')
            axs[i].imshow(img, cmap='gray')
    plt.show()
    return rot_image_list
~~~
### 類似度の計算
実際に、テンプレートと画像の類似度(マッチング度)を計算していく。
~~~
  #元画像に対して、テンプレート画像の類似度を計算する
  #類似度が高い範囲の左下の点を取得する(cv2.matchtemplate)
  def do_matching(self, ori_img_gray, template_list, match_threshold):
    res_list = []

    #マッチング度合いを計算する
    #resは各ピクセルにテンプレート画像を重ねた際の類似度のマップ
    for tmp in template_list:
      res = cv2.matchTemplate(ori_img_gray, tmp, cv2.TM_CCOEFF_NORMED)
      res_list.append(res)
    point_list = []

    for res in res_list:
      loc = np.where(res >= match_threshold)
      for pt in zip(*loc[::-1]):
        point_list.append(pt)

    print('マッチング数→', len(point_list))
    return point_list
~~~
### 重複してマッチングしている点をまとめる
一つの文字に対して、複数回マッチングすることが考えられるため、近くにあるマッチング点をグループ化して一つにまとめる。  
具体的には、左下から走査して最初のマッチング点を基準とし、近くの点を同じグループとする。グループ化した点の重心点を、そのグループの代表の点とする
~~~
def cal_distance(self, a, b):
    a, b = list(a), list(b)
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

#距離が近い点同士をまとめる
def group_neighbors(self, pt_list, distance):
    result = []
    for i in pt_list:
      temp_list = []
      for j in pt_list:
        if self.cal_distance(i,j) < distance:
          temp_list.append(j)
      for k in temp_list:
        pt_list.remove(k)
      result.append(temp_list)

    print('距離が近い点のグループ数→', len(result))
    return result
    
#まとめたグループ同士の平均の位置を求める
def group_mean(self, pt_group):
    mean_pt_list = []
    for pts in pt_group:
      x_sum = 0
      y_sum = 0

      pt_num = len(pts)

      for pt in pts:
        x_sum += pt[0]
        y_sum += pt[1]

      mean_pt_list.append((int(x_sum/pt_num), int(y_sum/pt_num)))

    return mean_pt_list
~~~

このようにして算出された座標が、マッチングした四角形の左下の位置となる。