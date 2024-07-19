"""
import torch
from fast_kinematics import FastKinematics
import numpy as np

N = 1024  # number of parallel calculations

# need the urdf file, number of parallel calculations and end effector link
model = FastKinematics("test.urdf", N, "ee_link")

# this robot has 7 joints, so the joint configuration has size (N*7,) (1d array)
# the first 7 values are for the first robot... Note we need to use float32!
joint_config = np.random.rand(N*6).astype(np.float32)

# get the forward kinematics size (N*7,) (1d array)
# the first 7 values are the first robot's pose [x,y,z,qw,qx,qy,qz]
ee_pose = model.forward_kinematics(joint_config)

# get the jacobian size (N*6*7,) (1d array) The 7 here is the number of joints
jac = model.jacobian_mixed_frame(joint_config)
# we can reshape the jacobian to (N,7,6) and then transpose to (N,6,7)
jac = jac.reshape(-1, 6, 6)#.transpose(0, 2, 1)
"""
import numpy as np
import torch
from fast_kinematics import FastKinematics
import time
import pandas as pd
#torch.backends.cuda.preferred_linalg_library("cusolver")
#torch.cuda.set_device(0)



def getQs(EELoc, q0, model, step=0.001, eps=0.0005, maxSteps = 1000000):
    
    ee = model.forward_kinematics(q0, 1)[:6].reshape((6,1))
    ee[3:] *= 0
    print(ee, ee.shape)
    print(EELoc, EELoc.shape)
    #eps2 = eps * eps
    q = q0.copy()
    iter = 0
    print(np.linalg.norm((EELoc - ee)))
    while np.linalg.norm((EELoc - ee)) > eps and iter < maxSteps:
        #print(np.linalg.norm((EELoc - ee)) > eps, iter < maxSteps, np.linalg.norm((EELoc - ee)))
        #print(q.shape)
        J = model.jacobian_world_frame(q, 1).reshape((6,6))
        J = J#[:3,:]
        #print(J.shape)
        #print("Diff:", (EELoc - ee).shape)
        dq = np.matmul(J.T,  (EELoc - ee))
        #print(dq.shape)
        q += dq * step / np.linalg.norm((EELoc - ee))
        ee = model.forward_kinematics(q, 1)[:6].reshape((6,1))
        ee[3:] *= 0
        #print("new ee:", ee.T)
        iter += 1
    
    print("Final Error:", np.linalg.norm((EELoc - ee)))
    #if iter >= maxSteps:
    #    return None
    return q
    
def getSteps(mx, mn, step):
    return int((mx - mn)/step)

def EEPose(step, xMin, xMax, yMin, yMax, zMin, zMax):
    nx = getSteps(xMax, xMin, step)
    ny = getSteps(yMax, yMin, step)
    nz = getSteps(zMax, zMin, step)

    nConfigs = nx * ny * nz
    print("Configurations:", nConfigs)
    N = 1024

    nSwaps = nConfigs // N + (0 if nConfigs%N == 0 else 1)
    print("GPU swaps", nSwaps)
    model = FastKinematics("test.urdf", 1, "ee_link")

    q = np.array([[0,0,0,0,0,0]], dtype=np.float32).T
    configMax = 1
    conCount = 0
    start = time.time()
    for xStep in range(0, nx):
        x = xMin + xStep * step
        for yStep in range(0, ny):
            y = yMin + yStep * step
            for zStep in range(0, nz):
                z = zMin + zStep * step
                q = getQs(np.array([[x,y,z]]).T, q, model)
                if not q is None:
                    print(q)
                else:
                    print("Failed to find config")
                conCount += 1
                if conCount >= configMax:
                    break
            if conCount >= configMax:
                break
        if conCount >= configMax:
            break
    end = time.time()
    print(f"Took {end-start}s to IK {nConfigs}")

def main():
    N = 8192000 
    #N = 10240
    model = FastKinematics("test.urdf", N, "ee_link")

    jnt_lims = [[0, 3.14],
                [0, 3.14],
                [0, 3.14],
                [0, 3.14],
                [0, 3.14],
                [0, 3.14]
                ]

    deg_step = 5
    step = deg_step * 3.14 / 180

    tot_configs = 1
    for minj, maxj in jnt_lims:
        tot_configs *= (maxj - minj) / step

    print("total configs:", tot_configs)
    print("epochs:", tot_configs/N)
    joint_config = torch.rand(N*6, dtype=torch.float32, device="cpu")
    #jc_cuda = torch.empty(N*6, dtype=torch.float32, device="cuda:0")
    idx = 0
    vis = 0
    #vis_max = int(0.01 * tot_configs/ N)
    vis_max = 1
    print("Starting:", vis_max)
    start = time.time()
    #U = torch.empty((N, 6,6), device="cuda")
    #print(U.device)
    #S = torch.empty((N,6), device="cuda")
    #Vt = torch.empty( (N,6,6), device="cuda")
    for q1 in range(0, 314, int(100 * step) ):
        for q2 in range(0, 314, int(100 * step) ):
            for q3 in range(0, 314, int(100 * step) ):
                for q4 in range(0, 314, int(100 * step) ):
                    for q5 in range(0, 314, int(100 * step) ):
                        for q6 in range(0, 314, int(100 * step) ):
                            joint_config[idx:idx + 6] = torch.tensor([q1, q2, q3, q4, q5, q6], 
                                                                     dtype=torch.float32)/100.0#, 
                                                                     #device="cpu") / 100.0
                            idx += 1
                            if idx%(N/1000) == 0:
                                print(idx/N)
                            if idx % N == 0:
                                #torch.cuda.empty_cache()
                                #print(joint_config.device)
                                #joint_config.to(device="cuda:0",copy=True)
                                #joint_config.cuda()
                                #print("JC final:", joint_config.device)
                                #print("Starting GPU transfer")
                                #jc_cuda = jc_cuda.copy_(joint_config)
                                #print("jc_cuda:", jc_cuda.device)
                                jac = model.jacobian_world_frame_pytorch(joint_config, N)
                                jac = jac.view(-1,6,6)
                                #print(jac.device)
                                U, S, Vt = torch.linalg.svd(jac)
                                #print(U.shape,  S.shape, Vt.shape)
                                ee_pose = model.forward_kinematics_pytorch(joint_config, N)
                                idx = 0
                                vis += 1
                                #joint_config.cpu()
                                #print(joint_config.device)
                                #print(vis/vis_max)
                            if vis >= vis_max:
                                print("+1")
                                vis = 0
                                end = time.time()
                                print(f"Took {(end-start)} to do {vis_max * N} states or {vis_max * N / tot_configs}\%")
                                return
                                
    end = time.time()
    print(f"Took {(end-start)} to do {vis_max * N} states or {vis_max * N / tot_configs}\%")
                                
import matplotlib.pyplot as plt

def createPD(x, y, z, v, rob_name="Bravo"):
    data = []
    for i in range(len(x)):
        data.append([x[i], y[i], z[i], v[i], rob_name])

    return pd.DataFrame(data, columns=['x','y','z','V', 'Robot'])


def qSample(model, N, fp='results/test.png', num_jnts=6, rob_name="Bravo", df=None):

    # setup panda's dataframe
    start = time.time()
    joint_config = torch.rand(N*num_jnts, dtype=torch.float32, device="cuda:0") * 3.14159 * 2 # on range 0-2pi
    maxs = [
        2.7437,
        1.7837,
        2.9007,
        -0.1518,
        2.8065,
        4.5169,
        3.0159
    ]

    mins = [
        -2.7437,
        -1.7837,
        -2.9007,
        -3.0421,
        -2.8065,
        0.5445,
        -3.0159
    ]
    for i in range(N):
        for k in range(num_jnts):
            joint_config[i * num_jnts + k] * (maxs[k] - mins[k]) + mins[k]
    #print("jc:", torch.min(joint_config), torch.max(joint_config))
    jac = model.jacobian_world_frame_pytorch(joint_config)
    print("jac:", jac.shape, jac.shape[0]/36)
    jac = jac.view(-1,6,num_jnts)
    #print(jac.device)
    U, S, Vt = torch.linalg.svd(jac)
    #print(U.shape,  S.shape, Vt.shape)
    ee_pose = model.forward_kinematics_pytorch(joint_config).view(-1,7)
    print(f"Took {time.time() - start}s to do {N} samples")

    print(ee_pose.shape)
    x = ee_pose.cpu().numpy()[:,0]
    print("x:", np.min(x), np.max(x))
    y = ee_pose.cpu().numpy()[:,1]
    print("y:", np.min(y), np.max(y))
    z = ee_pose.cpu().numpy()[:,2]
    print("z:", np.min(z), np.max(z))
    Vs = torch.prod(S, 1).reshape(N,1).cpu().numpy()
    print("Volume Dist:", Vs.shape, np.max(Vs), np.min(Vs))
    sVs = Vs/np.max(Vs) - np.min(Vs)# scale 0-1
    #sVs = Vs/2.25
    sVs = np.clip(sVs, 0.0,1.0)
    print("Volume Scaled:", sVs.shape, np.max(sVs), np.min(sVs))
    colors = np.array([0.0, 1.0, 0.0, 0.0]) * sVs + np.array([1.0,0.0,0.0, 0.0]) * (1-sVs)
    colors[:,3] = 0.5
    print("Color data:", colors.shape, type(colors), colors.dtype)

    if df is None:
        df = createPD(x,y,z,Vs, rob_name)
    else:
        df = pd.concat([df, createPD(x,y,z,Vs, rob_name)], ignore_index=True)

    return df
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(rob_name + " Manipulability Volume")

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    
    ax.scatter(x,y,z, c=colors, s=0.75)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    fig2, axi = plt.subplots(1,3)
    fig2.suptitle(rob_name + " MEP projections")
    fig2.set_figwidth(30)
    fig2.set_figheight(10)
    labels = [
        ['x', 'y', 'Projected XY'],
        ['x', 'z', 'Projected XZ'],
        ['y', 'z', 'Projected YZ']
        ]
    for i, axises in enumerate([[x,y],[x,z],[y,z]]):
        a,b = axises
        #fig_tmp = plt.figure()
        #axi[i] = fig.add_subplot(1,i)
        axi[i].scatter(a,b,c=colors)#, s=0.75)
        axi[i].set_xlabel(labels[i][0])
        axi[i].set_ylabel(labels[i][1])
        axi[i].set_aspect(1)
        #axi[i].set_xlim([-1,1])
        #axi[i].set_ylim([-1,1])
        axi[i].set_title(labels[i][2])
    
    #plt.show()
    fig.savefig(fp + '_3D.png')
    fig2.savefig(fp + '_projections.png')

    mean = np.mean(Vs)
    std = np.std(Vs)
    print(f"{rob_name} mean:{mean}, std:{std}")
    return df


def distGraph(df, N, fileName=""):
    fig, ax = plt.subplots()

    robs = df['Robot'].unique()
    v = np.zeros( (N, len(robs) ) )
    print(v.shape, robs)
                
    for i in range(len(robs)):
        a = ( df['V'][df['Robot'] == robs[i]]).to_numpy(dtype=np.float64)#.reshape(N,1)
        v[:,i] = a
        print(a.shape, v[:,i].shape)


    """
    ax.violinplot(v,
                  showmeans=False,
                  showmedians=True)
    
    ax.set_title("Distribution of MEP-V by Robot")
    ax.set_ylabel("MEP-Volume") 
    ax.set_xticks(np.arange(1, len(robs) + 1), labels=robs)
    ax.set_xlim(0.25, len(robs) + 0.75)
    ax.set_xlabel("Robot")
    fig.savefig(f"data_analysis/results/manip_dist_{N}.png")
    """

    rob = robs[0]
    df = df[df['Robot']==rob]
    x = df['x'].to_numpy(dtype=np.float64).reshape((N,1))
    y = df['y'].to_numpy(dtype=np.float64).reshape((N,1))
    z = df['z'].to_numpy(dtype=np.float64).reshape((N,1))
    V = df['V'].to_numpy(dtype=np.float64).reshape((N,1))

    """
    print(V.shape, np.max(V), np.min(V))
    sVs = V/np.max(V) - np.min(V)# scale 0-1
    #sVs = Vs/2.25
    sVs = np.clip(sVs, 0.0, 1.0)
    print("Volume Scaled:", sVs.shape, np.max(sVs), np.min(sVs))
    colors = np.array([0.0, 1.0, 0.0, 0.0]) * sVs + np.array([1.0,0.0,0.0, 0.0]) * (1-sVs)
    colors[:,2] = 0.5
    print("Color data:", colors.shape, type(colors), colors.dtype)
    
    fig2, ax2 = plt.subplots()
    ax2.scatter(x, V, c=(1-sVs), cmap="jet")
    fig3, ax3 = plt.subplots()
    ax3.scatter(y, V, c=(1-sVs), cmap="jet")
    fig4, ax4 = plt.subplots()
    ax4.scatter(z, V, c=(1-sVs), cmap="jet")
    """

    
    zs = [0.0,0.05, 0.1,0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5]
    for i in range(len(zs)-1):
        z_min = zs[i]
        z_max = zs[i+1]
        print(z_min, z_max)
        dfZ = df[ (df['z'] >= z_min) & (df['z'] <= z_max)]# & (df['x'] > 0) ]
        #Vz = (dfZ['V'] > 0.133).to_numpy()
        Vz = dfZ['V'].to_numpy(dtype=np.float64)
        #sVz = Vz/np.max(V) - np.min(V)# scale 0-1
        #sVz = np.clip(sVz, 0.0, 1.0)
        Xz = dfZ['x'].to_numpy(dtype=np.float64)
        Yz = dfZ['y'].to_numpy(dtype=np.float64)

        figz, axz = plt.subplots()
        pm = axz.scatter(Xz, Yz, c=Vz, cmap="jet", alpha=0.25, vmin=0.0, vmax=2.3)
        axz.set_xlabel("X (m)")
        axz.set_ylabel("Y (m)")
        axz.set_title(f"XY Crosscut from {z_min}-{z_max}")
        figz.colorbar(pm)
        figz.savefig(f"data_analysis/results/{rob}_ws/scatters/{z_min}_{fileName}_scatters.png")
    
        #axz.legend()
    
    step = 0.025
    zzStep = 0.05
    zMin = 0.0
    zMax = 0.5#zzStep #np.max(z)
    xMin = np.min(x)
    xMax =np.max(x)
    yMin = np.min(y)
    yMax = np.max(y)
    #step = 0.01
    xSteps = int(np.ceil((xMax - xMin) / step))
    ySteps = int(np.ceil((yMax - yMin) / step))
    zSteps = int(np.ceil((zMax - zMin) / zzStep))
    print(f"x:{xMin}-{xMax} w/{xSteps}")
    print(f"y:{yMin}-{yMax} w/{ySteps}")
    print(f"z:{zMin}-{zMax} w/{zSteps}")
    print(f"total:{xSteps * ySteps * zSteps}")

    low_man_value = 0.133
    Fgrid = np.zeros((xSteps, ySteps, zSteps), dtype=np.float64)
    Bgrid = np.zeros((xSteps, ySteps, zSteps), dtype=bool)
    fil_box = 0
    fnd_pts = 0
    for ix in range(xSteps):
        blockXMin = xMin + ix * step
        xdfBlock = df[(df['x'] >= blockXMin) & (df['x'] < blockXMin + step)]
        for iy in range(ySteps):
            blockYMin = yMin + iy * step
            ydfBlock = xdfBlock[(xdfBlock['y'] >= blockYMin) & (xdfBlock['y'] < blockYMin + step)]
            for iz in range(zSteps):
                blockZMin = zMin + iz * zzStep
                blockVs = ydfBlock['V'][(ydfBlock['z'] >= blockZMin) & 
                                  (ydfBlock['z'] < blockZMin + step) 
                                  ].to_numpy()
                if len(blockVs) == 0:
                    Fgrid[ix,iy,iz] = 0.0
                    Bgrid[ix,iy,iz] = False
                    continue
                #print(blockVs.shape)
                fnd_pts += len(blockVs)
                fil_box += 1
                Vavg = np.average(blockVs)
                #Vavg = np.median(blockVs)
                #print(Vavg)
                Fgrid[ix,iy,iz] = Vavg
                Bgrid[ix,iy,iz] = Vavg > low_man_value
    
    print(f"Found:{fnd_pts}/{len(df['V'])}; Filled:{fil_box}/{xSteps*ySteps*zSteps}")

    sx = np.zeros((xSteps,ySteps))
    sy = np.zeros(sx.shape)
    sV = np.zeros(sx.shape)
    print("sV:", np.min(sV), np.max(sV))
    for iz in range(zSteps):
        for ix in range(xSteps):
            for iy in range(ySteps):
                sx[ix, iy] = xMin + ix * step
                sy[ix, iy] = yMin + iy * step
                sV[ix, iy] = Fgrid[ix, iy, iz]
        
        lfig, lax = plt.subplots()
        pm = lax.pcolormesh(sx,sy,sV, cmap="jet", vmin=0.0, vmax=1.85)
        lfig.colorbar(pm)
        lax.set_ylim(yMin, yMax)
        lax.set_xlim(xMin, xMax)
        lax.set_xlabel("X (m)")
        lax.set_ylabel("Y (m)")

        lax.set_title(f"{rob} workspace z=[{zMin+ zzStep * iz}, {zMin+zzStep*iz + zzStep}]")
        lfig.savefig(f"data_analysis/results/{rob}_ws/voxels/{iz}_{fileName}_h={zMin + zzStep * iz}.png")
    #plt.show()
    print("Workspace Described")
    


if __name__ == "__main__":
    #main()
    #qSample(2000)
    #N = 20000

    N = 1000000
    #N=2000
    #model = FastKinematics("test.urdf", N, "ee_link")
    #df = qSample(model, N, 'data_analysis/results/bravo_1M')

    print("\n\nStarting Panda Analysis")
    
    model = FastKinematics("panda.urdf", N, "panda_hand")
    df = qSample(model, N, "data_analysis/results/panda_1M", num_jnts=7, rob_name="Franka-Panda")

    distGraph(df, N, "median")
    df.to_csv(f"data_analysis/results/raw_data_{N}.csv", index=False)

    """
    #EEPose(0.001, 0.0, 0.5, -0.25, .25, 0, 0.5)
    EELoc = np.array([[0.5587, 0.0, 0.262,0,0,0]]).T
    model = FastKinematics("test.urdf", 1, "ee_link")
    qGoal = np.array([[180, 132, 17, 180, 31, 62]], dtype=np.float32).T * 3.14/180.0 
    print(qGoal.T)
    q0 = qGoal.copy() #* 0.95
    q0 *=0.95
    print("q0:", q0.T)
    q = getQs(EELoc, q0, model, step=0.001, eps=0.0005, maxSteps = 100000)
    print("q",q.T)
    print("qg",qGoal.T)
    print("q0",q0.T)
    print(np.linalg.norm(q-qGoal))
    print("completed")
    """
