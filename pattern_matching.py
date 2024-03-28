import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

class chip_extraction():
  def __init__(self, ori_img, ori_template, match_angles, match_threshold, de_res, distance):
    #self.angles = match_angles
    self.ori_img = ori_img
    self.ori_temp = ori_template
    self.w = ori_template.shape[0]
    self.h = ori_template.shape[1]
    self.de_res = de_res
    self.match_threshold = match_threshold
   
    # 解像度の低いグレー画像を作る
    img_gray, template_gray = self.degrade_gray(ori_img, ori_template, de_res)

    #ちょっとずつ角度を変えたtemplate画像を作る
    template_list = self.rotate_img(template_gray, match_angles)

    #元画像に対して、テンプレート画像との類似度を算出
    #取得するのは左下の点
    point_list_all = self.do_matching(img_gray,template_list, match_threshold)

    #距離が近い点同士をまとめる
    #解像度が下がった
    result_point = self.group_neighbors(point_list_all,distance)
    #print('取得した点→',result_point)

    #まとめたグループ同士の平均の位置を求める
    chip_point_list = self.group_mean(result_point)
    self.chip_point_list = chip_point_list
  
  # 解像度の低いグレー画像を作る
  def degrade_gray(self, img, template, de_res):

    #画像の解像度を下げる→処理を軽くするため
    img_d = cv2.resize(img, dsize=None, fx=de_res, fy=de_res)
    template_d = cv2.resize(template, dsize=None, fx=de_res, fy=de_res)

    #グレースケール化
    img_gray = cv2.cvtColor(img_d, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template_d, cv2.COLOR_RGB2GRAY)

    return img_gray, template_gray
  
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
    print('-------------解像度を下げて回転させたテンプレートの画像の表示----------------')
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
    print('----------------------------------------------------------------------')
    return rot_image_list

  #元画像に対して、テンプレート画像の類似度を計算する
  #類似度が高い範囲の左下の点を取得する(cv2.matchtemplate)
  def do_matching(self, ori_img_gray, template_list, match_threshold):
    res_list = []

    #マッチング度合いを計算する
    for tmp in template_list:
      res = cv2.matchTemplate(ori_img_gray, tmp, cv2.TM_CCOEFF_NORMED)
      res_list.append(res)
    point_list = []

    for res in res_list:
      loc = np.where(res >= match_threshold)
      for pt in zip(*loc[::-1]):
        point_list.append(pt)

    print('マッチング数(閾値を超えてマッチングしたと判断された点の数)→', len(point_list))
    return point_list

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

    print('距離が近い点(distanceで指定した距離以内の点)のグループ数→', len(result))
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

  def check_result(self):
    img_red = copy.copy(self.ori_img)

    #解像度をもどして四角形を書く
    for pt in self.chip_point_list:
      cv2.rectangle(img=img_red, 
                    pt1=(int(pt[0]/self.de_res), int(pt[1]/self.de_res)), pt2=(int(pt[0]/self.de_res)+self.h, int(pt[1]/self.de_res)+self.w), 
                    color=(255,0,0), thickness=2)

    plt.imshow(img_red)
    plt.show()

    #self.img_red = img_red
    #return img_red

  def one_show(self,num):

    pt = self.chip_point_list[num]

    pt_ori = (pt[0]/self.de_res, pt[1]/self.de_res)
    right_upper = (pt[0]/self.de_res+self.h, pt[1]/self.de_res+self.w)

    plt.imshow(self.ori_img[pt_ori[1]:right_upper[1], pt_ori[0]:right_upper[0]])
    plt.show()