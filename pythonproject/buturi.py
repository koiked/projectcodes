# -*- coding: utf-8 -*-
import vpython as vs
import ode
import numpy as np 
import math
import colorsys
# 衝突判定の計算
#def near_callback(args, geom1, geom2):
#    e = 0.9 # 反発係数
#    contacts = ode.collide(geom1, geom2)
#    world, contactgroup = args
#    for c in contacts:
#        c.setBounce(e) # 反発係数の値をセット
#        c.setMu(5000)  # 静止摩擦係数
#        j = ode.ContactJoint(world, contactgroup, c)
#        j.attach(geom1.getBody(), geom2.getBody())


def main():
    g = -9.81 # 重力加速度
    r = 0.1 # ボールの半径
    dt = 0.001 # 1フレームの時間
    t = 0.0    # 経過時間
    nuc=np.loadtxt('nuclei255.txt',usecols=(0,1,2,3))
    txts=np.loadtxt('nuclei255.txt',usecols=4,dtype=str)

    
    # ODEワールドの作成
    world = ode.World()
    world.setGravity((0, g, 0))

    # 剛体の作成
    #ball_body = ode.Body(world) 
    #m = ode.Mass()              
    #m.setSphere(2500.0, 0.5)  
    #ball_body.setMass(m)        
    #ball_body.setPosition((0, 4, 0))

    # 衝突判定の設定
    space = ode.Space()
    floor_geom = ode.GeomPlane(space, (0, 1, 0), 0)
    #ball_geom = ode.GeomSphere(space, radius=r)
    #ball_geom.setBody(ball_body)
    #contactgroup = ode.JointGroup()

    # フィールドの生成
    i=0
    for(a,b,c,d) in nuc:
        #print(a,b,c,d)
        hu2=(math.log10(d)+62)/106
        hu=math.log2(c+1)/8
        #print(hu)
        rgbc=colorsys.hsv_to_rgb(0.8*hu,1,1)
        rgbc2=colorsys.hsv_to_rgb(0.8*hu2,1,1)
        #print(int(255*rgbc[0]),int(255*rgbc[1]),int(255*rgbc[2]))
        field = vs.box(size=vs.vector(4, c, 4),
                   pos=vs.vector(4*a,c/2, 4*b),
                   color=vs.vector((rgbc2[0]),(rgbc2[1]),(rgbc2[2])))
        #field2 = vs.box(size=vs.vector(4, 1, 4),
        #           pos=vs.vector(4*a,-1, 4*b),
        #           color=vs.vector((rgbc2[0]),(rgbc2[1]),(rgbc2[2])))
        tex=vs.text(text=txts[i],pos=vs.vector(4*a,c,4*b),color=vs.vector(0,0,250),billboard=True,align='center')    
        i+=1
        print(i)
    
    #Lab=vs.label(pos=vs.vector(0,0,0),text='H')
    #tex=vs.text(text='H-2\n 0.0015',pos=vs.vector(4,0.1,0),color=vs.vector(0,0,250),billboard=True,align='center')
    # ボールの生成
    #tex2=vs.wtext(pos=vs.vector(0,0,0),text="1")
    #ball = vs.sphere(pos=vs.vector(0, 4, 0),
                     #radius=r,
                     #color=vs.vector(255,0,255))
    
    #while t < 10.0:
        # 衝突判定
        #space.collide((world, contactgroup), near_callback)

        # ODEの計算結果に応じてボールの位置を変化
        #(x, y, z) =  ball_body.getPosition()
        #ball.pos = vs.vector(x, y, z)
        #vs.rate(1/dt)  #フレームレート
        #world.step(dt) # 時間をdtだけ進める
        #t += dt        # 経過時間の更新
        #contactgroup.empty()

if __name__ == '__main__':
    main()