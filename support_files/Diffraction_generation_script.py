"""
This McStasScript file was generated from a
McStas instrument file. It is advised to check
the content to ensure it is as expected.
"""
import mcstasscript as ms
from mcstasscript.interface import instr, plotter, functions
import matplotlib.pyplot as plt
"""
Detector: monitor_ndt_ch11_I=7.21929e+09 monitor_ndt_ch11_ERR=4.41871e+06 monitor_ndt_ch11_N=3.75053e+06 "monitor_ndt_ch11_1723626561.t"
Detector: Mon_source_lambda_I=1.30704e+07 Mon_source_lambda_ERR=150046 Mon_source_lambda_N=53226 "Mon_source_lambda.dat"
Detector: Detector_I=7.34182e+06 Detector_ERR=220657 Detector_N=16379 "Detector_1723626561.th_L"

Detector: monitor_ndt_ch11_I=8.6598e+08 monitor_ndt_ch11_ERR=1.53079e+06 monitor_ndt_ch11_N=449661 "monitor_ndt_ch11_1723626639.t"
Detector: Mon_source_lambda_I=1.28271e+07 Mon_source_lambda_ERR=147657 Mon_source_lambda_N=53367 "Mon_source_lambda.dat"
Detector: Detector_I=7.12581e+06 Detector_ERR=214371 Detector_N=16239 "Detector_1723626639.th_L"
"""
class RenderPredicts:

    def __init__(self, mcrun_path, mcstas_path):

        # predict_1 - длинна между чопперами, prdict_2 - фаза второго чоппера, prdict_3 - щель второго чоппера

        self.mcrun_path = mcrun_path
        self.mcstas_path = mcstas_path

    def get_diffraction(self, cristal, stats, pulce_duration):

        my_configurator = ms.Configurator()

        my_configurator.set_mcrun_path(self.mcrun_path)
        my_configurator.set_mcstas_path(self.mcstas_path)

        #dif60 = instr.McStas_instr("dif60_generated")

        dif60 = ms.McStas_instr("Diffraction_imshow")

        dif60.add_declare_var("double", "T1", value=98.3)
        dif60.add_declare_var("double", "I1", value=264000000000.0)
        dif60.add_declare_var("double", "T2", value=303.1)
        dif60.add_declare_var("double", "I2", value=119000000000.0)
        dif60.add_declare_var("double", "T3", value=29.9)
        dif60.add_declare_var("double", "I3", value=310000000000.0)
        dif60.add_declare_var("double", "T")
        dif60.add_declare_var("double", "t")
        dif60.add_declare_var("double", "source_freq", value=60.0)
        dif60.add_declare_var("double", "source_pulse_number", value=2.0)
        dif60.add_declare_var("double", "pulse_length", value=pulce_duration) #value=2.0
        dif60.append_initialize("I1 = I1 * pulse_length * 1e-6 * source_freq; ")
        dif60.append_initialize("I2 = I2 * pulse_length * 1e-6 * source_freq; ")
        dif60.append_initialize("I3 = I3 * pulse_length * 1e-6 * source_freq; ")

        origin = dif60.add_component("origin", "Progress_bar")
        origin.set_AT(['0', ' 0', ' 0'], RELATIVE="ABSOLUTE")

        source = dif60.add_component("source", "Source_gen")
        source.dist = 0.1
        source.focus_xw = 0.03
        source.focus_yh = 0.03
        source.lambda0 = 5.0
        source.dlambda = 4.9
        source.I1 = "I1"
        source.yheight = 0.04
        source.xwidth = 0.04
        source.T1 = "T1"
        source.T2 = "T2"
        source.I2 = "I2"
        source.T3 = "T3"
        source.I3 = "I3"
        source.append_EXTEND("T = floor(rand01()*source_pulse_number);")
        source.append_EXTEND("t = rand01()*pulse_length*1e-6 + T*1/source_freq;")
        source.set_AT(['0', '0', '0'], RELATIVE="origin")

        """
        mon_sou_xy = dif60.add_component("mon_sou_xy", "Monitor_nD")
        mon_sou_xy.xwidth = 0.1
        mon_sou_xy.yheight = 0.1
        mon_sou_xy.restore_neutron = 1
        mon_sou_xy.options = "\"x limits =[-0.05 0.05] bins = 100 y limits =[-0.05 0.05] bins = 100\""
        mon_sou_xy.set_AT(['0', ' 0', '0.10001'], RELATIVE="source")

        mon_sou_lam = dif60.add_component("mon_sou_lam", "Monitor_nD")
        mon_sou_lam.xwidth = 0.1
        mon_sou_lam.yheight = 0.1
        mon_sou_lam.restore_neutron = 1
        mon_sou_lam.options = "\"lambda limits =[0 10] bins = 100\""
        mon_sou_lam.set_AT(['0', ' 0', '0.10001'], RELATIVE="source")

        sou_t_2p = dif60.add_component("sou_t_2p", "Monitor_nD")
        sou_t_2p.xwidth = 0.1
        sou_t_2p.yheight = 0.1
        sou_t_2p.restore_neutron = 1
        sou_t_2p.options = "\"t limits =[-0.00001 0.02] bins = 600\""
        sou_t_2p.set_AT(['0', ' 0', '1e-6'], RELATIVE="source")

        sou_t_1p = dif60.add_component("sou_t_1p", "Monitor_nD")
        sou_t_1p.xwidth = 0.1
        sou_t_1p.yheight = 0.1
        sou_t_1p.restore_neutron = 1
        sou_t_1p.options = "\"t limits =[-0.00001 0.00039] bins = 80\""
        sou_t_1p.set_AT(['0', ' 0', '0'], RELATIVE="PREVIOUS")

        before_ch1_t_2p = dif60.add_component("before_ch1_t_2p", "Monitor_nD")
        before_ch1_t_2p.xwidth = 0.1
        before_ch1_t_2p.yheight = 0.1
        before_ch1_t_2p.restore_neutron = 1
        before_ch1_t_2p.options = "\"t limits =[-0.00001 0.02] bins = 600\""
        before_ch1_t_2p.set_AT(['0', ' 0', '0.1001'], RELATIVE="source")

        before_ch1_t_1p = dif60.add_component("before_ch1_t_1p", "Monitor_nD")
        before_ch1_t_1p.xwidth = 0.1
        before_ch1_t_1p.yheight = 0.1
        before_ch1_t_1p.restore_neutron = 1
        before_ch1_t_1p.options = "\"t limits =[-0.00001 0.00039] bins = 80\""
        before_ch1_t_1p.set_AT(['0', ' 0', '0'], RELATIVE="PREVIOUS")
        """

        Ch1 = dif60.add_component("Ch1", "DiskChopper")
        Ch1.theta_0 = 8
        Ch1.radius = 0.75
        Ch1.yheight = 0
        Ch1.nu = 60
        Ch1.nslit = 1
        Ch1.delay = "0.000175/2"
        Ch1.isfirst = 0
        Ch1.set_AT(['0', '0', '0.101'], RELATIVE="source")

        Ch11 = dif60.add_component("Ch11", "DiskChopper")
        Ch11.theta_0 = 8
        Ch11.radius = 0.75
        Ch11.yheight = 0
        Ch11.nu = -60
        Ch11.nslit = 1
        Ch11.delay = "0.000175/2"
        Ch11.isfirst = 0
        Ch11.set_AT(['0', '0', '1e-6'], RELATIVE="PREVIOUS")

        monitor_ndt_ch11 = dif60.add_component("monitor_ndt_ch11", "Monitor_nD")
        monitor_ndt_ch11.xwidth = 0.1
        monitor_ndt_ch11.yheight = 0.1
        monitor_ndt_ch11.restore_neutron = 1
        monitor_ndt_ch11.options = "\"t limits =[-0.00001 0.2] bins = 600\""
        monitor_ndt_ch11.set_AT(['0', ' 0', '1e-5'], RELATIVE="Ch1")

        """
        mon_afterch1_xy = dif60.add_component("mon_afterch1_xy", "Monitor_nD")
        mon_afterch1_xy.xwidth = 0.1
        mon_afterch1_xy.yheight = 0.1
        mon_afterch1_xy.restore_neutron = 1
        mon_afterch1_xy.options = "\"x limits =[-0.02 0.02] bins = 200 y limits =[-0.02 0.02] bins = 200\""
        mon_afterch1_xy.set_AT(['0', ' 0', '1e-5'], RELATIVE="Ch1")

        monitor_ndt_ch11 = dif60.add_component("monitor_ndt_ch11", "Monitor_nD")
        monitor_ndt_ch11.xwidth = 0.1
        monitor_ndt_ch11.yheight = 0.1
        monitor_ndt_ch11.restore_neutron = 1
        monitor_ndt_ch11.options = "\"t limits =[-0.00001 0.02] bins = 600\""
        monitor_ndt_ch11.set_AT(['0', ' 0', '1e-5'], RELATIVE="Ch1")

        monitor_ndt_ch12 = dif60.add_component("monitor_ndt_ch12", "Monitor_nD")
        monitor_ndt_ch12.xwidth = 0.1
        monitor_ndt_ch12.yheight = 0.1
        monitor_ndt_ch12.restore_neutron = 1
        monitor_ndt_ch12.options = "\"t limits =[-0.00001+ 0.00039] bins = 500\""
        monitor_ndt_ch12.set_AT(['0', ' 0', '0'], RELATIVE="PREVIOUS")

        monitor_dxy_ch1 = dif60.add_component("monitor_dxy_ch1", "Monitor_nD")
        monitor_dxy_ch1.xwidth = 0.03
        monitor_dxy_ch1.yheight = 0.03
        monitor_dxy_ch1.options = "\"dx limits=[-5 5] bins=200 dy limits=[-5 5] bins=200\""
        monitor_dxy_ch1.set_AT(['0', ' 0', '0'], RELATIVE="PREVIOUS")
        """

        guide = dif60.add_component("guide", "Guide_gravity")
        guide.w1 = 0.03
        guide.h1 = 0.03
        guide.w2 = 0.03
        guide.h2 = 0.03
        guide.l = 5.5
        guide.m = 2.5
        guide.set_AT(['0', '0', '0.01'], RELATIVE="PREVIOUS")

        """
        Ch2 = dif60.add_component("Ch2", "DiskChopper")
        Ch2.theta_0 = 90.06
        Ch2.radius = 0.5
        Ch2.yheight = 0
        Ch2.nu = 60
        Ch2.nslit = 1
        Ch2.isfirst = 0
        Ch2.n_pulse = 0
        Ch2.phase = 86
        Ch2.set_AT(['0', '0', '5.51'], RELATIVE="PREVIOUS")
        """

        bender = dif60.add_component("bender", "Bender")
        bender.w = 0.03
        bender.h = 0.03
        bender.r = 870
        bender.k = 2
        bender.l = 12
        bender.ma = 2.5
        bender.mi = 2.5
        bender.ms = 2.5
        bender.set_AT(['0', '0', '5.51'], RELATIVE="PREVIOUS")

        Mon_source_lambda = dif60.add_component("Mon_source_lambda", "L_monitor")
        Mon_source_lambda.nL = 250
        Mon_source_lambda.xwidth = 0.1
        Mon_source_lambda.yheight = 0.1
        Mon_source_lambda.Lmin = 0
        Mon_source_lambda.Lmax = 15
        Mon_source_lambda.restore_neutron = 1
        Mon_source_lambda.set_AT(['0', '0', '12.01'], RELATIVE="PREVIOUS")

        """
        monitor_ndt = dif60.add_component("monitor_ndt", "Monitor_nD")
        monitor_ndt.xwidth = 0.1
        monitor_ndt.yheight = 0.1
        monitor_ndt.restore_neutron = 1
        monitor_ndt.options = "\"t limits =[0 0.05] bins = 100\""
        monitor_ndt.set_AT(['0', ' 0', '0'], RELATIVE="PREVIOUS")

        mon_det_xy = dif60.add_component("mon_det_xy", "Monitor_nD")
        mon_det_xy.xwidth = 0.1
        mon_det_xy.yheight = 0.1
        mon_det_xy.restore_neutron = 1
        mon_det_xy.options = "\"x limits =[-0.02 0.02] bins = 100 y limits =[-0.02 0.02] bins = 100\""
        mon_det_xy.set_AT(['0', ' 0', '0'], RELATIVE="PREVIOUS")

        monitor_dxy_bender = dif60.add_component("monitor_dxy_bender", "Monitor_nD")
        monitor_dxy_bender.xwidth = 0.03
        monitor_dxy_bender.yheight = 0.03
        monitor_dxy_bender.options = "\"dx limits=[-0.7 0.7] bins=100 dy limits=[-0.7 0.7] bins=100\""
        monitor_dxy_bender.set_AT(['0', ' 0', '1e-5'], RELATIVE="PREVIOUS")
        """

        Arm_sample = dif60.add_component("Arm_sample", "Arm")
        Arm_sample.set_AT(['0', ' 0', ' 1.05'], RELATIVE="PREVIOUS")

        powdern = dif60.add_component("powdern", "PowderN")
        powdern.reflections = cristal   #"\"Al.laz\""
        powdern.radius = 0.05
        powdern.yheight = 0.1
        powdern.set_AT(['0', ' 0', ' 0'], RELATIVE="Arm_sample")
        powdern.set_ROTATED(['0', ' 0', ' 0'], RELATIVE="PREVIOUS")

        Detector = dif60.add_component("Detector", "Monitor_nD")
        Detector.yheight = 0.5
        Detector.radius = 0.5
        Detector.options = "\"banana theta limits = [-170 170] bins = 480 lambda limits = [0.1 10]] bins 250\""
        Detector.set_AT(['0', ' 0', ' 0'], RELATIVE="Arm_sample")

        dif60.settings(ncount=stats) #2E7
        dif60.set_parameters()

        data = dif60.backengine()

        Diffraction = ms.name_search("Detector", data)
        Diffraction_imshow = Diffraction.Intensity

        #ms.make_plot(data)

        return Diffraction_imshow


#mcrun_path = "/Applications/McStas-3.3.app/Contents/Resources/mcstas/3.3/bin/"
#mcstas_path = "/Applications/McStas-3.3.app/Contents/Resources/mcstas/3.3/"

#render = RenderPredicts(mcrun_path, mcstas_path)

#cristals = ["\"Ag.laz\"", "\"Al.laz\"", "\"Al2O3_sapphire.laz\"", "\"Au.laz\"", "\"B4C.laz\"", "\"Ba.laz\"", "\"Be.laz\"", "\"BeO.laz\"", "\"C_diamond.laz\"", "\"C_graphite.laz\"", "\"Cr.laz\"", "\"Cs.laz\"", "\"Cu.laz\"", "\"Cu2MnAl.laz\"", "\"Fe.laz\"", "\"Ga.laz\"", "\"Gd.laz\"", "\"Ge.laz\"", "\"H2O_ice_1h.laz\"", "\"He4_hcp.laz\"", "\"Hg.laz\"", "\"I2.laz\"", "\"K.laz\"", "\"Li.laz\"", "\"LiF.laz\"", "\"Mo.laz\"", "\"Na2Ca3Al2F14.laz\"", "\"Nb.laz\"", "\"Ni.laz\"", "\"Pb.laz\"", "\"Pt.laz\"", "\"Rb.laz\"", "\"Si.laz\"", "\"Ti.laz\"", "\"Tl.laz\"", "\"UO2.laz\"", "\"Zn.laz\"", "\"Y2O3.laz\""]

#cristal = "\"Al.laz\""

#stats = 2E7

#pulce_duration = 300.0

#DI = render.get_diffraction(cristal, stats, pulce_duration)

#print(DI)

#plt.imshow(DI,norm='log')
#plt.show()

