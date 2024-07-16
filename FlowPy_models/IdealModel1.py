def Idealmodel1(wel):
    import os
    import numpy as np
    import pandas as pd
    import flopy
    from tempfile import TemporaryDirectory

    # filename
    temp_dir = TemporaryDirectory()
    sim_ws = os.path.join(temp_dir.name)
    sim_name = "IdealModel1"
    gwfname = "gwf_" + sim_name
    gwtname = "gwt_" + sim_name

    # units
    length_units = "meters"
    time_units = "days"

    # Discretization
    nper = 1  # 稳定流，周期数为 1
    nlay = 1  # 层数
    nrow = 17  # 行数
    ncol = 23  # 列数
    delr = 150  # 单位行长
    delc = 150  # 单位列长
    # delz = 10.0  # 单位层高
    top = 30  # 顶部高程
    botm = -10  # 底部高程
    # idomain = 1
    idomain = np.ones((nlay, nrow, ncol), dtype=int)  # 参与模拟的区域，1 为参与

    # GWF
    icelltype = 1  # 0 = 承压含水层； 1 = 潜水含水层；

    k11 = np.full([11, 11], 90)  # 渗透系数
    k11[:4, :6] = 150
    k11[8:, :] = 60
    k11[5:8, 9:] = 60

    # Temporal discretization
    perlen = 1000.0
    nstp = 1
    tsmult = 1.0
    tdis_ds = []
    tdis_ds.append((perlen, nstp, tsmult))

    # Solver parameters
    nouter, ninner = 100, 300
    hclose, rclose, relax = 1e-6, 1e-6, 1.0

    # Initial conditions
    # Starting Heads
    strt = np.full([11, 11], 30, dtype=float)

    # Boundary conditions
    # wel_spd

    wel_spd = [[0, 2, 3, -wel[0]], [0, 5, 4, -wel[1]], [0, 6, 10, -wel[2]]]
    wel_spd = {0: wel_spd}

    # chd_spd
    chd_spd = []
    for i in np.arange(ncol):
        chd_spd.append([0, 10, i, 30.0])

    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name="mf6")

    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)

    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwfname,
        save_flows=True,
        model_nam_file="{}.nam".format(gwfname),
    )

    ims = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=hclose,
        outer_maximum=nouter,
        under_relaxation="NONE",
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=rclose,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=relax,
        filename="{}.ims".format(gwfname),
    )
    sim.register_ims_package(ims, [gwf.name])

    flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idomain,
        filename="{}.dis".format(gwfname),
    )

    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=False,
        icelltype=icelltype,
        k=k11,
        k33=k11,
        save_specific_discharge=True,
        filename="{}.npf".format(gwfname),
    )

    flopy.mf6.ModflowGwfic(gwf, strt=strt, filename="{}.ic".format(gwfname))

    flopy.mf6.ModflowGwfchd(
        gwf,
        maxbound=len(chd_spd),
        stress_period_data=chd_spd,
        save_flows=False,
        pname="CHD-1",
        filename="{}.chd".format(gwfname),
    )

    flopy.mf6.ModflowGwfwel(
        gwf,
        print_input=True,
        print_flows=True,
        stress_period_data=wel_spd,
        save_flows=True,
        pname="WEL-1",
        filename="{}.wel".format(gwfname),
    )

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="{}.hds".format(gwfname),
        budget_filerecord="{}.bud".format(gwfname),
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    if not success:
        raise Exception("MODFLOW 6 did not terminate normally.")

    head = gwf.oc.output.head().get_alldata()[0, 0]

    return head


import flopy

# head = flopy.utils.binaryfile.HeadFile("trans" + ".hed")
# budg = flopy.utils.binaryfile.CellBudgetFile("trans" + ".ccf")


import matplotlib.pyplot as plt

test = [10000, 10000, 10000]
res = Idealmodel1(test)
print(res)
plt.imshow(res)
plt.colorbar()
plt.show()
