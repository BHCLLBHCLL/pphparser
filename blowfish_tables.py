# -*- coding: utf-8 -*-
"""标准 Blowfish P/S 表（π 十六进制数字，Schneier 参考实现同值）。

数据来源: SCTprime_Bx64.dll .data (P @ RVA 0xD0E990, S @ 0xD0E9E0,
文件态与运行时一致)。布局: P[18] (0x48) + 8 字节间隙丢弃 + S[4][256] (0x1000)。
"""
import base64

_B85 = """
h-yD1(+H!5E{YiwM00QhA~*@8&@rDWnEEaViEK{n<{>CWcN5S!&t_D<Yz*l%w<)Z^-
B5hVwbg__7YRqT+0_=38~b^QrVBCAthL{m+j9Lcx8EAj?XTgumVPE`MUZ^DnSU(tN134{_iT}~;_?9q7W|Hd*dS?RX--
#hqyAW=em#@)k93s{ShkLGSj~%Z?n)Jc9i&uywOLxiIn^+n6JYWxBe2oM@P%L~7<oa;?>O7Iu-ti$4j4LMiw>S_K8hZ&!FLtcCrd
HC+AptoU|e8T^Cea0l&e*$Vwgx*Kosa>YB|bPwh*f-
G+fNI&gelAuZC2ilj(AN5ftvTD!gxFUB#&@_AwZA7Cv0g9+QWgGrHEVT+bwIfm1SicZQWHm`IO1uenQW8|c5plOZT(%n8_Gk*Oj3
V61#dGJxz|U0sClv2}&%0wX5eh#O_=fj)^N#jKXo^KWerIYRV<f<y~01R#W@O7O^OUYj4DLTJVznr!LzVVn$S@QBl^(x6dlXfIUQ
D5npWqcc&fYzyyh<U4v3Q1Cmtm@4~z9cA&Mb^)(BK3U3Uhz>)78HVnRw4ZN9!=;3MzFvzv*mZB<b0EQmpF|o!rfR`fVye?l26sO;
a^L<NJp!9TB-hY4ND>Or)9Me~o51l&$#PQ%dmEX6*m=|?_ulCC8c_b@J4|`Dz2Iz@x(2`mwoj?S#9*I5!k%1SV<c)BuW$QkwNq?9
>#{i$Z|qV#A5r{mER`^uL`8wj3B6vg1kmHY{Yo?52Ph9_vr8@+SE$Rt4|K>yIbRFX+xyeGy})@@3Nm0C#sHz#d2%d3{w1Gh%%dOs
`01sN_#!gf-xhT|6>DMr$R1EHt5OB6`n5eWV2309NP96;g5Ll>yH{MEpp0+IE>?#c+i4fy_NYSD!~cFL#xiHDb5&1_uqRt-
$XLp|U8Dcupb_x6m_7OsxPu}6Y_<GKThq}kdE`@?WkvY5yh)DIkeB;Q+Vb7uGk(jYK@<C8=*HyE+R7mB0Zex`{+@o&v>)*+P1@U)
n30gKagMF(pw*LW(9w?2;3dbeFIzm0x0H2``{MSGWGfQ#61a$n9Pka0pkA#{9K(-jk@3&bt-
+|X7$Pq(cNY%6{w;Otp&tT^56qzD=yY$j*7K|wnc|7g;7_Qux8M`~fjh)}*{H43W};sf1$UKz6m!gzcN7{SWgxBQhWfR4^+Hs~&o
!O<46n`Wpou<v(;Gq7Nq!!#Ee<6BUbAr^y8vh+ui&^>n>J)59=Qqd9g$;Is;1vri9^A7e^Xl7qFa3+#ku7Ib^|7Zsn3;SXc@>6K}
vH@$}LB;N~sikQUFmK6)97iKUWXd<i?x4cBEh{0CeVowQsrzAL$!cYwVWt71<3cA!TE>w)wf|E(J9IWLAa6T`jOvp^u-
bnMb+^Y6pdG>2P{WL@BeiE(vwpBPJQdu%>RUecz`^xM1#_X0q*%ag3_${}-
BRY*J)a;hwR=r2;k?DNG4<KoeP@J~|lWnwV5SWt~D>*5r?C)<5@|2b|KO^)Tq~<~S`)!CfWrh9KQcChKs7#_57%Ud$c>KWk_S$?v
*86d0F0p>S$zgf)L?hT@=71yh{2HwRFv9E1lxT&~`K>_mNzxbhZPH`+R|4Gd851Rosm|FZxB8ubjWvUIgPSbBpcy&(zQ`4f@R_Aj
Y?b4N0i0Y~*Bf#u~o+}gr7G<LPfr`_|KMqxy$4g=5+KF7%HK^}Fanaw!jFAnNEp}T-
GF|$4xIEz$H2)S)f0}VoNzXS^QkP^5od3={7a<F+;uZgd`A9tGG5D1gm60VE0E<exQA94~^Bynpl=G_`lhs}ha7)Mxo7upDPyqcf
9jD1V#I_!Gv`W=PaW<v>M(!*rLM;ILM2-y`fH#<Y!7P`VDLrtL)QN(2`0sxdy-
Q+qGp7`H!RZcMw)_1Idn;8*b@m4hiYsaJ4I~Wl>38f@i?dFg2`up*hEWf&*Zaftpa7E*Wv2W?>3SJI|vnoDX9OoZ*`UX8tx!h$b4
;|;3)`>oVCCFw{c*#sIYO@fWx(*e_c<SPmQ#|~o^eqY=r%w0sJu4kH4<<PoV0pqB2&W@bwh|NfZvL(7W*@`ql|{VcgUEZP(KmmwD
F2KB@7*%PrCMyhg&|mF0+_3452eoTJC!fnt$pt*gfDJeD7GOLaA5~2bw}Oo5EY+cFsRHEmc3!?9{w}~&tn6skX&=MIihe(3!a|U6
yB=KyoSu~rz~P%t6Zy`ZiMr)uO5qJ%J97zxoIPyP`g@XGFoUrv$QHa)#;uw_qZXz84FaKnV?^Xewp``sC_+Rnus^}cP-<WUz6<-
ff8sIDTp-
=)*t4_q2Jz?nYvhbrG)iYV{#%J|HFfumPW!F>k6~YR4^`6<Ve|%C^5Wk@AB(d>i@<xVeKgXb3A<R*%V4$<F{n!6kQNP;1fO|w&Lz
Z>Z_`w6;Ev2&`-<yLi9q%wYzHX9Xn5D1tHBro_QYF$4!NXYDY`tP-
1~S@}XkSMkbA0po56~qqfJv!z2}dl61>y3yOqCg|d=X0KZ!aok$t2bg^O-
01kp9Dvd%|>h)C&KJ={}VQ@bqlJIgfL4K2s@$6sL+dCp`SvTH%V07-
wr}BkBZZdbygn$R8o>2H1RoL(7HJ908s;6nG!Uhb*{Hp|7+{z1pE_y_sgfvCN1!vX$$(|k#)7x{~&4^VHdD>rRKtpHaWi-Ua*f>6
Mp7<y|ApdRg=OI27N<FZTE1%=$_pO85Xj(n#_dtP^941WmG%0D6_aGHP_tXM*E^G9>XaJ(raU=-
TYV;s8x74>ouVDaCF7`PdMnxo-bWb4^K!}UK9sHF|uaUKu)7|uNMK7Rc><PTTg_pep&~2=L1cl2nvnT78K{@?Z=0_#knhMIRC3sL
M^eF^W+J-
C&`)#)AVieqH0BK0qq`(en?v0@*qW(WmjID?02H=dfw$}7|d>+oK>|Zz2nWK2lLMm%OHJ<(;xrOuDtJgX1i%t?d_xi~lRv2wdF=n
GCvX|rPboy>OGDBP4_vb-
q`yhDAPW1}=m$UxZtX4opCzVLLIyzHThmC_Ex2bFXOO=+;yl1FYSeg}~V=1Z3Guz>rR!S<S`6V$w9Dej(d@(7I0_gbta4#nWTot>
3<18ozNEN}AB5lUxKNG=7+=dUw?#cNF4<7_Uq<Kd`7jB4M>rr1a(ZHM4kHNh1WHk{%G<bU@V4Nyoqv-hG8*F1A!n6_&o-
*MA(NAWt6@k&p;FTk5;gUWxVhbcYBDucq4x+M!nGL$;j0|$#DEFc*MR*d>{gk(rVhDX?@b%2+Z=*?6`bd3;C;gqn9*sWpL1RY>bp
I}yt8Q;PH~skZVB8X@_}%NFOyL`u4Qp=O5LJ7|H!NpuJJe+-
1nAJ*#|<g*qyGTRk`Fnx3+)eS`=5JeoPFEQ3(t|Dqh1x+h%XbnB&}Ofzj>5<>()DVvo1L|d07$6m*OUXEio?ir>#P0J1c6&%uIC>
9PuuAH$oC>=TVZk=DVQ=P-
FW`7!Ye3`t8aQ*u5n^$;08|Sr$S>h7%GBZtM)%D(b7%PG_%VU#N!Ki0Qw+{=?*CSAe{ohQRmm_<ML@O<)6HMuYv)upc<~1g=GQ%=
|XjGiySFajPEihe3d_UtRz_zE_}kB(CVanL<WYVJ=v|kMvkhqW#_SIPY}y!o7)y!}(@YbdIygRq}PTx!L?iVJ7Q(gx?*Dc@AoX;+
0>Kj#+L-a96Y-kyX`<Ox^;?;j9a}&;^3KNMfiFo~d+owi$o73EZjRp$RQ!Ge$DR0v}rFjK1&*C7_u~5dLl*9X+`k-
=w7r5AuegY4K<%gW9*;{suW$o6h2)Qh&`)0bUVM`hy0i#I*vTC(z~%Cye--
L54qf225bXwFanRD0&z0;D+_Us#stEVtw2&*PiAPW9m2~l-<HpG#0|bR_@EY-nP7up?&=8b{$#H3FHNDhyi>{JqmI;Bz%&5UvlGy
xt&dNv|GU%{J5UecvV#GwWa(B(|kSH#1E}aUhhyI_~v1;*%XDMJQGoD=f~A=#7^N>&c7-)H^|1_G%}jk5`trrj`|K~-
~eDiH_kaa&-
MD#H+RCT8!g3Jo@cOJLN}vNKqrFJzMH(6osJRJ6>|^2ejF{<d&B_8Ya5KWMUbH(zOinyv~D(PFRMsbd2W=v(srZ9$ihiw?)VQ=ec
p{m9SU>R#!b*n+q*d_P)53`=q8n{1mj-
5@YVXEno%uk;*9TOBJPHoxWb9R_AVqtsskRSq|ox5x?#YBO={_zP!;8m)>~w{`JyH$;W|2}hLx#H>0(vy)9)|G+V@iTX>SBSSql1
h6{+Na0fw-Lt>y`vljS};TK$mem)A6&*|+d4QHv`AI;_=}eWrZ@);`bCC@p<_&n2H9o4E3=a<tZ3O!bIbajYre=A{|n{j9J@oBEU
Ojl|RF%vU=oDQ4C9C@vFukpW+LRbX}PL=Kksj9$~@)NKV7x^47`C1Ifhz3^lO6`t$EqF0bS>>4MRDhE2Ln{7WE^&w*(`(~W=8S~s
GDA_Z0_5D?`f;3hGyF9vz5qD82_}L1=XHm?nUy`lNQ5Wb<j@&RfVpyFwk@+B#!jO9E&U-
%lWX>T`GQLlDe&e>NMm;IRX;a=vfaViq5D2cKB(iPY{Vj!QW+4X&3Pzf<-M~d;&)#fUuE-!9-
S@s*K#f=%e*x0WyW_NLerlrKMgLhkL<%)N)y=fLsLtwgyM$!=t`cUAM{hj8W8|CCo?S0g8+XF4aAQvPjSUWTR})+3aTap;U42up2
+Ke~%;L08YDUsDgs&9=C<L(K9Xgnmw4b;JNT6=cf;&HNf~z1kOC1^k_$P8Fv0xQp+&`1&D|tG*y(L97;W>z)OL@*wx5+Xu$+{n)e
#ji)_R+_@!w~_`$ExU|Nr#Xcn!Qid%ih}1INA!;D#JMgXEu?>d@=crPb;zDw^^R|I=l5l{~6WtoJH9uEFzb`D*WaZar}`FB^8vLV
Uy+h>zuaES!Aff(Wtr-UI)Rp3~B}AWl+)(LZk!B4sPt=JKLBRzMz;)Wa)S^GL;{n-
;&edD>R_f9`bRCL39cmj5MQ5AaTCM*fMs*jh{8&E-
#szM{f(}9r5E2RN74Ck=WWE&v|0bZ+<?_X0aF+1syE?#nO+2nIil4SMwwFBX%;TF*T@@0?o`;VuA3iwd-
|Smo^q|%yZI+l46$L&`G(08<0>;6jsJ@y~oDp3KV)W2GHR}ntSrY{Zp#R01v2Q;=d)k_R_sI1!)p-
A_4@me9z0aE1Y)Cz&;UE)8jxEV7;q2@U2HBoFF*5cFsok#i6fvU|@A1PX5b<*p29lu=%QbeyaIQT*62hjEegL<Z8nK`Qhu<Y53HT
px#`jEhPy*pXLa{GGR`)TjF=l<KK^0=5oV2
"""

TABLES = base64.b85decode("".join(_B85.split()))
assert len(TABLES) == 0x48 + 0x1000
