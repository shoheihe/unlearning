# ファイル構造
### Programs
  * weight <br>
  事前学習済みモデルの重みと特徴量可視化結果
  * result <br>
  unlearningによる精度と特徴量可視化結果
  * data <br>
  cifar10とcifar100のデータファイル
  * proposed <br>
    * SCRUB-main <br>
      参考にしたアンラーニング手法SCRUBのscript
    * pro_SCRUB
      * sh <br>
        shell scriptファイルがいくつか含まれてる
      * main.py　<br>
        事前学習用ファイル
      * unlearning.py <br>
        unlearningを行うファイル

# 実行方法
### 事前学習
~~~py:main.py
python main.py --dataset cifar10 --model preactresnet18 --noise_rate 0.5 --noise_mode sym --save True --tsne True
~~~
### scrub
~~~py:unlearning.py
python unlearning.py --dataset cifar100 --model preactresnet18 --noise_rate 0.5 --noise_mode sym --method scrub
~~~
### 提案手法
~~~py:unlearning.py
python unlearning.py --dataset cifar10 --model preactresnet18 --noise_rate 0.5 --noise_mode sym --method pro --delta 500 --zeta 20 --eta 5
~~~

##### データセット
* cifar10
* cifar100

##### モデル
* preactresnet18
* allcnn

##### noise_mode
* sym
* asym
* SDN (cifar100のみ)

##### delta, zeta, eta
損失項のハイパーパラメータ
