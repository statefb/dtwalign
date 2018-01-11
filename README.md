# Multivariate Time-series Shape Analysis for .NET
## 概要
[mtsaパッケージ](https://isq-github.yamatake.local/t-suzuki-ug/mtsa)の
C#向けラッパーパッケージ  
[HA異常検知DLL](https://isq-github.yamatake.local/t-suzuki-ug/MtsaLogic)
のロジック部から呼び出される

## Installation
### 1. mtsaパッケージのインストール
[こちら](https://isq-github.yamatake.local/t-suzuki-ug/mtsa/tree/master#installation)を参照  


### 2. mtsa-netパッケージ本体のインストール
1. [本体](https://isq-github.yamatake.local/t-suzuki-ug/mtsa-net/archive/master.zip)のダウンロード
2. 解凍後，以下をターミナル(コマンドプロンプト)で実行

```
python setup.py install
```

### 3. プロジェクト参照設定
※開発者向け  

* 以下にあるPython.Runtime.dllをVSの参照設定に追加  
ドライブ:\Users\ユーザー名\AppData\Local\Continuum\Anaconda3\pkgs\pythonnet-2.3.0-py35_0\Lib\site-packages

* ターゲットビルドをPython実行環境に合わせる(64bit or 32bit)


## Usage
本パッケージは単体で利用する物ではないので、特になし
