# Numerical experimation of "Deep Learning is Not So Mysterious or Different"

https://arxiv.org/abs/2503.02113


# contents 
* Benign overfitting of linear model(Ridge regression)
+ double descent curve in kernel regression
+ generization of GBDT(or ramdom forest)
+ deviation of PAC-bayes inequiality
+ representaiton learing as DNN specific feature


https://zenn.dev/link/comments/156ec4609d55b2 in Japanese

https://docs.google.com/presentation/d/1TLnCElJZAcqm3ZK8ANj3iHNT8-OXl1MHJysPhEto1MI/edit?usp=sharing

# code

> python3 allexperiments.py

## Benign overfitting of linear model(Ridge regression)

## deviation of PAC-bayes inequiality

##

# Critics

総じて既存の研究事実の追認である

- DNN以外のDDの存在、挙動は既に論文化されているものである。
- パラメータを増大させた場合の挙動に関するステートメントは検証されていないし、特異学習理論で想定される挙動と同一かどうか、それを含むのかどうかも主張できていない。
- PAC-Bayes boundの計算では明示的に事前分布を使っていてコルモゴロフ複雑性を使った形式にはなっていない
- コルモゴロフ複雑性の実例である圧縮アルゴリズムは例示されていない。
- 表現学習ではそもそもDNNは最終層を取り替えることができるで高性能になるというのは当たり前のようで主張が弱い

## Altanative ideas
- Benign overfittingの再現　「構造化されたデータに対しては帰納的バイアスが低いモデルでも汎化する」
- Solomonoff priorを使ったモデルのPAC-Bayes boundの実証(link)
- 次数に応じた正則化項(事前分布)の類似したより自然な概念の提唱
- [Surprises in High-Dimensional Ridgeless Least Squares Interpolation]()における　γ=n/p(データ数/パラメータ数)一定、あるいは変化させた場合のDNNでの対応事例
- 学習データを増やすと性能が悪化する（Effective Model Complexity）現象の再現
- SLT(Singular Learning Theory)における描像との対応付け
- 表現学習でのDNN１層分の表現能力の高さ(微分不可能関数などが近似できる)ことを主張する(既存研究あり)

