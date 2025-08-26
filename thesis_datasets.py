from raw_data.core import Core
from pathlib import Path
from raw_data.dataset import Dataset
from preprocessing.preprocess import Preprocessor, apply_preprocessing_step, apply_correlation_pruning
from raw_data.dataset import Dataset, exclude_indicators_from_dataset, split_dataset
import numpy as np


def download_raw_data():
    folders = {
        "config": Path("raw_data/config"),
        "data": Path("data"),
    }
    core = Core(folders=folders,multiprocessing=False)

    countries = [
        'AUS',  # Australia
        'AUT',  # Austria
        'BEL',  # Belgium
        'CAN',  # Canada
        'CHE',  # Switzerland
        'CHL',  # Chile
        'COL',  # Colombia
        'CRI',  # Costa Rica
        'CZE',  # Czech Republic
        'DEU',  # Germany
        'DNK',  # Denmark
        'ESP',  # Spain
        'EST',  # Estonia
        'FIN',  # Finland
        'FRA',  # France
        'GBR',  # United Kingdom
        'GRC',  # Greece
        'HUN',  # Hungary
        'IRL',  # Ireland
        'ISL',  # Iceland
        'ISR',  # Israel
        'ITA',  # Italy
        'JPN',  # Japan
        'KOR',  # Korea, Rep.
        'LTU',  # Lithuania
        'LUX',  # Luxembourg
        'LVA',  # Latvia
        'MEX',  # Mexico
        'NLD',  # Netherlands
        'NOR',  # Norway
        'NZL',  # New Zealand
        'POL',  # Poland
        'PRT',  # Portugal
        'SVK',  # Slovak Republic
        'SVN',  # Slovenia
        'SWE',  # Sweden
        'TUR',  # Turkey
        'USA'   # United States
    ]
    indicators = ["SP.POP.GROW","SP.DYN.CBRT.IN","SP.DYN.CDRT.IN","SP.POP.0004.FE.5Y","SP.POP.0004.MA.5Y","SP.POP.0509.FE.5Y","SP.POP.0509.MA.5Y","SP.POP.1014.FE.5Y","SP.POP.1014.MA.5Y","SP.POP.1519.FE.5Y","SP.POP.1519.MA.5Y","SP.POP.2024.FE.5Y","SP.POP.2024.MA.5Y","SP.POP.2529.FE.5Y","SP.POP.2529.MA.5Y","SP.POP.3034.FE.5Y","SP.POP.3034.MA.5Y","SP.POP.3539.FE.5Y","SP.POP.3539.MA.5Y","SP.POP.4044.FE.5Y","SP.POP.4044.MA.5Y","SP.POP.4549.FE.5Y","SP.POP.4549.MA.5Y","SP.POP.5054.FE.5Y","SP.POP.5054.MA.5Y","SP.POP.5559.FE.5Y","SP.POP.5559.MA.5Y","SP.POP.6064.FE.5Y","SP.POP.6064.MA.5Y","SP.POP.6569.FE.5Y","SP.POP.6569.MA.5Y","SP.POP.7074.FE.5Y","SP.POP.7074.MA.5Y","SP.POP.7579.FE.5Y","SP.POP.7579.MA.5Y","SP.POP.0014.TO.ZS","SP.POP.0014.FE.ZS","SP.POP.0014.MA.ZS","SP.POP.1564.TO.ZS","SP.POP.1564.FE.ZS","SP.POP.1564.MA.ZS","SP.POP.65UP.TO.ZS","SP.POP.65UP.FE.ZS","SP.POP.65UP.MA.ZS","SP.POP.80UP.FE.5Y","SP.POP.80UP.MA.5Y","SP.POP.DPND","SP.POP.DPND.YG","SP.POP.DPND.OL","SP.POP.BRTH.MF","SP.POP.TOTL.FE.ZS","SP.POP.TOTL.MA.ZS","EN.POP.DNST","SP.URB.TOTL.IN.ZS","SP.URB.GROW","EN.URB.LCTY.UR.ZS","EN.URB.MCTY.TL.ZS","SP.RUR.TOTL.ZS","SP.RUR.TOTL.ZG","SE.ADT.LITR.ZS","SE.ADT.LITR.FE.ZS","SE.ADT.LITR.MA.ZS","SE.ADT.1524.LT.FM.ZS","SE.ADT.1524.LT.ZS","SE.ADT.1524.LT.FE.ZS","SE.ADT.1524.LT.MA.ZS","SE.ENR.PRIM.FM.ZS","SE.ENR.PRSC.FM.ZS","SE.ENR.SECO.FM.ZS","SE.ENR.TERT.FM.ZS","SE.PRM.ENRL.FE.ZS","SE.SEC.ENRL.GC.FE.ZS","SE.SEC.ENRL.FE.ZS","SE.SEC.ENRL.VO.FE.ZS","SE.PRM.TCHR.FE.ZS","SE.SEC.TCHR.FE.ZS","SE.TER.TCHR.FE.ZS","SE.PRM.UNER.ZS","SE.PRM.UNER.FE.ZS","SE.PRM.UNER.MA.ZS","SE.SEC.UNER.LO.ZS","SE.SEC.UNER.LO.FE.ZS","SE.SEC.UNER.LO.MA.ZS","SE.PRE.ENRR","SE.PRE.ENRR.FE","SE.PRE.ENRR.MA","SE.PRM.ENRR","SE.PRM.ENRR.FE","SE.PRM.ENRR.MA","SE.PRM.NENR","SE.PRM.NENR.FE","SE.PRM.NENR.MA","SE.PRM.TENR","SE.PRM.TENR.FE","SE.PRM.TENR.MA","SE.PRM.GINT.ZS","SE.PRM.GINT.FE.ZS","SE.PRM.GINT.MA.ZS","SE.PRM.NINT.ZS","SE.PRM.NINT.FE.ZS","SE.PRM.NINT.MA.ZS","SE.PRM.PRIV.ZS","SE.SEC.ENRR","SE.SEC.ENRR.FE","SE.SEC.ENRR.MA","SE.SEC.NENR","SE.SEC.NENR.FE","SE.SEC.NENR.MA","SE.SEC.PRIV.ZS","SE.TER.ENRR","SE.TER.ENRR.FE","SE.TER.ENRR.MA","SE.PRM.OENR.ZS","SE.PRM.OENR.FE.ZS","SE.PRM.OENR.MA.ZS","SE.PRM.PRS5.ZS","SE.PRM.PRS5.FE.ZS","SE.PRM.PRS5.MA.ZS","SE.PRM.PRSL.ZS","SE.PRM.PRSL.FE.ZS","SE.PRM.PRSL.MA.ZS","SE.SEC.PROG.ZS","SE.SEC.PROG.FE.ZS","SE.SEC.PROG.MA.ZS","SE.PRM.REPT.ZS","SE.PRM.REPT.FE.ZS","SE.PRM.REPT.MA.ZS","SE.PRM.CMPT.ZS","SE.PRM.CMPT.FE.ZS","SE.PRM.CMPT.MA.ZS","SE.SEC.CMPT.LO.ZS","SE.SEC.CMPT.LO.FE.ZS","SE.SEC.CMPT.LO.MA.ZS","SE.PRM.CUAT.ZS","SE.PRM.CUAT.FE.ZS","SE.PRM.CUAT.MA.ZS","SE.SEC.CUAT.LO.ZS","SE.SEC.CUAT.LO.FE.ZS","SE.SEC.CUAT.LO.MA.ZS","SE.SEC.CUAT.UP.ZS","SE.SEC.CUAT.UP.FE.ZS","SE.SEC.CUAT.UP.MA.ZS","SE.SEC.CUAT.PO.ZS","SE.SEC.CUAT.PO.FE.ZS","SE.SEC.CUAT.PO.MA.ZS","SE.TER.CUAT.ST.ZS","SE.TER.CUAT.ST.FE.ZS","SE.TER.CUAT.ST.MA.ZS","SE.TER.CUAT.BA.ZS","SE.TER.CUAT.BA.FE.ZS","SE.TER.CUAT.BA.MA.ZS","SE.TER.CUAT.MS.ZS","SE.TER.CUAT.MS.FE.ZS","SE.TER.CUAT.MS.MA.ZS","SE.TER.CUAT.DO.ZS","SE.TER.CUAT.DO.FE.ZS","SE.TER.CUAT.DO.MA.ZS","SE.PRM.AGES","SE.SEC.AGES","SE.PRE.DURS","SE.COM.DURS","SE.PRM.DURS","SE.SEC.DURS","SE.PRE.ENRL.TC.ZS","SE.PRM.ENRL.TC.ZS","SE.SEC.ENRL.TC.ZS","SE.SEC.ENRL.LO.TC.ZS","SE.SEC.ENRL.UP.TC.ZS","SE.TER.ENRL.TC.ZS","SE.PRE.TCAQ.ZS","SE.PRE.TCAQ.FE.ZS","SE.PRE.TCAQ.MA.ZS","SE.PRM.TCAQ.ZS","SE.PRM.TCAQ.FE.ZS","SE.PRM.TCAQ.MA.ZS","SE.SEC.TCAQ.ZS","SE.SEC.TCAQ.FE.ZS","SE.SEC.TCAQ.MA.ZS","SE.SEC.TCAQ.LO.ZS","SE.SEC.TCAQ.LO.FE.ZS","SE.SEC.TCAQ.LO.MA.ZS","SE.SEC.TCAQ.UP.ZS","SE.SEC.TCAQ.UP.FE.ZS","SE.SEC.TCAQ.UP.MA.ZS","SE.XPD.CPRM.ZS","SE.XPD.CSEC.ZS","SE.XPD.CTER.ZS","SE.XPD.CTOT.ZS","SE.XPD.PRIM.ZS","SE.XPD.SECO.ZS","SE.XPD.TERT.ZS","SE.XPD.TOTL.GD.ZS","SE.XPD.TOTL.GB.ZS","SE.XPD.PRIM.PC.ZS","SE.XPD.SECO.PC.ZS","SE.XPD.TERT.PC.ZS","SP.DYN.LE00.MA.IN","SP.DYN.LE00.FE.IN","SP.DYN.LE00.IN","SH.STA.BRTC.ZS","SH.STA.ANVC.ZS","SH.STA.ARIC.ZS","SH.HIV.ARTC.ZS","SH.HIV.PMTC.ZS","SH.MLR.TRET.ZS","SP.REG.BRTH.ZS","SP.REG.BRTH.FE.ZS","SP.REG.BRTH.MA.ZS","SP.REG.BRTH.RU.ZS","SP.REG.BRTH.UR.ZS","SP.REG.DTHS.ZS","SH.STA.ORCF.ZS","SH.STA.ORTH","SH.IMM.IDPT","SH.IMM.HEPB","SH.IMM.MEAS","SH.VAC.TTNS.ZS","SH.STA.BRTW.ZS","SH.TBS.DTEC.ZS","SH.UHC.SRVS.CV.XD","SH.MED.CMHW.P3","SH.MED.BEDS.ZS","SH.MED.NUMW.P3","SH.MED.PHYS.ZS","SH.SGR.PROC.P5","SH.MED.SAOP.P5","SH.TBS.CURE.ZS","SH.HIV.INCD.YG.P3","SH.HIV.INCD.ZS","SH.HIV.INCD.TL.P3","SH.DYN.AIDS.ZS","SH.HIV.1524.FE.ZS","SH.HIV.1524.MA.ZS","SH.STA.DIAB.ZS","SH.MLR.INCD.P3","SH.TBS.INCD","SH.ANM.CHLD.ZS","SH.ANM.NPRG.ZS","SH.PRG.ANEM","SH.ANM.ALLW.ZS","SH.H2O.BASW.ZS","SH.H2O.BASW.RU.ZS","SH.H2O.BASW.UR.ZS","SH.H2O.SMDW.ZS","SH.H2O.SMDW.RU.ZS","SH.H2O.SMDW.UR.ZS","SH.STA.BASS.ZS","SH.STA.BASS.RU.ZS","SH.STA.BASS.UR.ZS","SH.STA.SMSS.ZS","SH.STA.SMSS.RU.ZS","SH.STA.SMSS.UR.ZS","SH.STA.HYGN.ZS","SH.STA.HYGN.RU.ZS","SH.STA.HYGN.UR.ZS","SH.STA.ODFC.ZS","SH.STA.ODFC.RU.ZS","SH.STA.ODFC.UR.ZS","SH.XPD.OOPC.CH.ZS","SH.XPD.OOPC.PP.CD","SH.UHC.NOP1.ZS","SH.UHC.NOP2.ZS","SH.UHC.FBP1.ZS","SH.UHC.FBP2.ZS","SH.UHC.TOT1.ZS","SH.UHC.TOT2.ZS","SH.UHC.NOPR.ZS","SH.UHC.FBPR.ZS","SH.UHC.TOTR.ZS","SH.UHC.OOPC.10.ZS","SH.UHC.OOPC.25.ZS","SH.SGR.CRSK.ZS","SH.SGR.IRSK.ZS","SH.XPD.CHEX.GD.ZS","SH.XPD.CHEX.PP.CD","SH.XPD.GHED.GD.ZS","SH.XPD.GHED.CH.ZS","SH.XPD.GHED.GE.ZS","SH.XPD.GHED.PP.CD","SH.XPD.PVTD.CH.ZS","SH.XPD.PVTD.PP.CD","SH.XPD.EHEX.CH.ZS","SH.XPD.EHEX.PP.CD","SP.DYN.TFRT.IN","SP.DYN.WFRT","SP.ADO.TFRT","SP.MTR.1519.ZS","SH.CON.1524.FE.ZS","SH.CON.1524.MA.ZS","SP.DYN.CONU.ZS","SP.DYN.CONM.ZS","SH.FPL.SATM.ZS","SP.UWT.TFRT","SH.DTH.INJR.ZS","SH.DTH.NCOM.ZS","SH.DTH.COMM.ZS","SH.STA.TRAF.P5","SH.DYN.NCOM.ZS","SH.DYN.NCOM.FE.ZS","SH.DYN.NCOM.MA.ZS","SH.STA.AIRP.P5","SH.STA.AIRP.FE.P5","SH.STA.AIRP.MA.P5","SH.STA.POIS.P5","SH.STA.POIS.P5.FE","SH.STA.POIS.P5.MA","SH.STA.WASH.P5","SH.STA.SUIC.P5","SH.STA.SUIC.FE.P5","SH.STA.SUIC.MA.P5","SH.DYN.MORT","SH.DYN.MORT.FE","SH.DYN.MORT.MA","SH.DYN.0509","SH.DYN.1014","SH.DYN.1519","SH.DYN.2024","SP.DYN.AMRT.FE","SP.DYN.AMRT.MA","SP.DYN.TO65.FE.ZS","SP.DYN.TO65.MA.ZS","SH.DYN.NMRT","SP.DYN.IMRT.IN","SP.DYN.IMRT.MA.IN","SP.DYN.IMRT.FE.IN","SH.STA.MMRT","SH.MMR.RISK.ZS","SH.MMR.RISK","SN.ITK.SALT.ZS","SN.ITK.VITA.ZS","SN.ITK.MSFI.ZS","SN.ITK.SVFI.ZS","SN.ITK.DEFC.ZS","SH.STA.STNT.ZS","SH.STA.STNT.FE.ZS","SH.STA.STNT.MA.ZS","SH.STA.MALN.ZS","SH.STA.MALN.FE.ZS","SH.STA.MALN.MA.ZS","SH.STA.OWGH.ZS","SH.STA.OWGH.FE.ZS","SH.STA.OWGH.MA.ZS","SH.STA.WAST.ZS","SH.STA.WAST.FE.ZS","SH.STA.WAST.MA.ZS","SH.SVR.WAST.ZS","SH.SVR.WAST.FE.ZS","SH.SVR.WAST.MA.ZS","SM.POP.TOTL.ZS","BX.TRF.PWKR.DT.GD.ZS","SM.POP.REFG","SM.POP.REFG.OR","SI.POV.GINI","SI.DST.50MD","SI.SPR.PCAP","SI.SPR.PC40","SI.DST.10TH.10","SI.DST.05TH.20","SI.DST.04TH.20","SI.DST.03RD.20","SI.DST.02ND.20","SI.DST.FRST.20","SI.DST.FRST.10","SI.POV.GAPS","SI.POV.LMIC.GP","SI.POV.UMIC.GP","SI.POV.DDAY","SI.POV.LMIC","SI.POV.UMIC","SI.POV.NAHC","EN.POP.SLUM.UR.ZS","SI.SPR.PC40.ZG","SI.SPR.PCAP.ZG","SP.HOU.FEMA.ZS","SH.STA.BFED.ZS","SH.PRV.SMOK","SH.PRV.SMOK.FE","SH.PRV.SMOK.MA","SH.ALC.PCAP.LI","SH.ALC.PCAP.FE.LI","SH.ALC.PCAP.MA.LI","AG.PRD.FOOD.XD","AG.PRD.CROP.XD","AG.YLD.CREL.KG","AG.PRD.LVSK.XD","AG.CON.FERT.ZS","AG.CON.FERT.PT.ZS","AG.LND.IRIG.AG.ZS","NV.MNF.CHEM.ZS.UN","NV.MNF.FBTO.ZS.UN","NV.MNF.MTRN.ZS.UN","NV.MNF.TECH.ZS.UN","NV.MNF.TXTL.ZS.UN","NV.MNF.OTHR.ZS.UN","EG.USE.PCAP.KG.OE","EG.USE.COMM.GD.PP.KD","EG.USE.ELEC.KH.PC","EG.ELC.COAL.ZS","EG.ELC.PETR.ZS","EG.ELC.NGAS.ZS","EG.ELC.FOSL.ZS","EG.ELC.HYRO.ZS","EG.ELC.NUCL.ZS","EG.USE.COMM.CL.ZS","EG.ELC.RNEW.ZS","EG.ELC.RNWX.ZS","EG.FEC.RNEW.ZS","EG.ELC.LOSS.ZS","EG.IMP.CONS.ZS","EG.USE.COMM.FO.ZS","EG.USE.CRNW.ZS","EG.EGY.PRIM.PP.KD","EG.GDP.PUSE.KO.PP.KD","ER.GDP.FWTL.M3.KD","FX.OWN.TOTL.ZS","FX.OWN.TOTL.FE.ZS","FX.OWN.TOTL.MA.ZS","FX.OWN.TOTL.YG.ZS","FX.OWN.TOTL.OL.ZS","FX.OWN.TOTL.PL.ZS","FX.OWN.TOTL.SO.ZS","FX.OWN.TOTL.40.ZS","FX.OWN.TOTL.60.ZS","FB.ATM.TOTL.P5","FB.CBK.BRCH.P5","SI.RMT.COST.OB.ZS","SI.RMT.COST.IB.ZS","FB.BNK.CAPA.ZS","FD.RES.LIQU.AS.ZS","FB.AST.NPER.ZS","FB.CBK.BRWR.P3","FM.LBL.BMNY.GD.ZS","FM.LBL.BMNY.ZG","FM.LBL.BMNY.IR.ZS","FM.AST.CGOV.ZG.M3","FS.AST.CGOV.GD.ZS","FM.AST.PRVT.ZG.M3","FS.AST.DOMO.GD.ZS","FM.AST.DOMO.ZG.M3","FB.CBK.DPTR.P3","IC.CRD.INFO.XQ","FS.AST.DOMS.GD.ZS","FS.AST.PRVT.GD.ZS","FD.AST.PRVT.GD.ZS","FR.INR.LNDP","FR.INR.LEND","CM.MKT.LCAP.GD.ZS","FM.AST.PRVT.GD.ZS","IC.CRD.PRVT.ZS","IC.CRD.PUBL.ZS","FR.INR.RISK","CM.MKT.INDX.ZG","CM.MKT.TRAD.GD.ZS","CM.MKT.TRNR","SL.UEM.TOTL.ZS","SL.UEM.TOTL.FE.ZS","SL.UEM.TOTL.MA.ZS","SL.TLF.CACT.FE.ZS","SL.TLF.CACT.MA.ZS","SL.TLF.CACT.ZS","SL.AGR.EMPL.FE.ZS","SL.AGR.EMPL.MA.ZS","SL.AGR.EMPL.ZS","SL.TLF.0714.FE.ZS","SL.TLF.0714.MA.ZS","SL.TLF.0714.ZS","SL.TLF.TOTL.FE.ZS","NV.AGR.TOTL.ZS","NV.AGR.TOTL.KD.ZG","NV.AGR.EMPL.KD","NV.IND.TOTL.ZS","NV.IND.TOTL.KD.ZG","NV.IND.EMPL.KD","NV.SRV.TOTL.ZS","NV.SRV.TOTL.KD.ZG","NV.SRV.EMPL.KD","NV.IND.MANF.ZS","NV.IND.MANF.KD.ZG","NY.GDP.PCAP.KD","NY.GDP.PCAP.PP.KD","NY.GNP.PCAP.PP.KD","NY.GDP.MKTP.KD.ZG","NY.GNP.MKTP.KD.ZG","NY.GDP.PCAP.KD.ZG","NY.GNP.PCAP.KD.ZG","NY.GDP.PETR.RT.ZS","NY.GDP.MINR.RT.ZS","NY.GDP.NGAS.RT.ZS","NY.GDP.COAL.RT.ZS","NY.GDP.FRST.RT.ZS","NY.GDP.TOTL.RT.ZS","NE.CON.TOTL.ZS","NE.CON.TOTL.KD.ZG","NE.CON.PRVT.ZS","NE.CON.PRVT.KD.ZG","NE.CON.PRVT.PC.KD","NE.CON.PRVT.PC.KD.ZG","NE.DAB.TOTL.ZS","NY.GNS.ICTR.ZS","NY.GNS.ICTR.GN.ZS","NE.GDI.TOTL.ZS","NE.GDI.TOTL.KD.ZG","NE.GDI.FTOT.ZS","NE.GDI.FTOT.KD.ZG","NY.GDS.TOTL.ZS","NE.GDI.FPRV.ZS","NY.GDP.DEFL.KD.ZG","FP.CPI.TOTL.ZG","FR.INR.DPST","FR.INR.RINR","FP.WPI.TOTL","NE.TRD.GNFS.ZS","BG.GSR.NFSV.GD.ZS","NE.EXP.GNFS.ZS","NE.EXP.GNFS.KD.ZG","TX.VAL.AGRI.ZS.UN","TX.VAL.FUEL.ZS.UN","TX.VAL.MMTL.ZS.UN","TX.VAL.ICTG.ZS.UN","BX.GSR.CCIS.ZS","TX.VAL.TECH.MF.ZS","BX.GSR.CMCP.ZS","BX.GSR.INSF.ZS","BX.GSR.TRAN.ZS","TX.VAL.TRAN.ZS.WT","BX.GSR.TRVL.ZS","TX.VAL.TRVL.ZS.WT","NE.IMP.GNFS.ZS","NE.IMP.GNFS.KD.ZG","TM.VAL.AGRI.ZS.UN","TM.VAL.FUEL.ZS.UN","TM.VAL.MMTL.ZS.UN","TM.VAL.ICTG.ZS.UN","BM.GSR.CMCP.ZS","BM.GSR.INSF.ZS","BM.GSR.TRAN.ZS","TM.VAL.TRAN.ZS.WT","BM.GSR.TRVL.ZS","TM.VAL.TRVL.ZS.WT","BN.CAB.XOKA.GD.ZS","NE.RSB.GNFS.ZS","FI.RES.TOTL.DT.ZS","FI.RES.TOTL.MO","DT.DOD.DECT.GN.ZS","DT.DOD.PVLX.GN.ZS","DT.DOD.PVLX.EX.ZS","DT.DOD.DSTC.XP.ZS","DT.DOD.DSTC.IR.ZS","DT.DOD.DSTC.ZS","DT.TDS.DECT.GN.ZS","DT.TDS.DECT.EX.ZS","DT.TDS.DPPF.XP.ZS","DT.TDS.MLAT.PG.ZS","DT.TDS.DPPG.GN.ZS","DT.TDS.DPPG.XP.ZS","BX.KLT.DINV.WD.GD.ZS","BM.KLT.DINV.WD.GD.ZS","DT.ODA.ODAT.GN.ZS","DT.ODA.ODAT.XP.ZS","DT.ODA.ODAT.GI.ZS","DT.ODA.ODAT.MP.ZS","DT.ODA.ODAT.PC.ZS","NY.ADJ.NNTY.KD.ZG","NY.ADJ.NNTY.PC.KD.ZG","NY.ADJ.NNTY.PC.KD","NY.ADJ.DNGY.GN.ZS","NY.ADJ.DMIN.GN.ZS","NY.ADJ.DRES.GN.ZS","NY.ADJ.DFOR.GN.ZS","NY.ADJ.DCO2.GN.ZS","NY.ADJ.DKAP.GN.ZS","NY.ADJ.AEDU.GN.ZS","NY.ADJ.ICTR.GN.ZS","NY.ADJ.NNAT.GN.ZS","NY.ADJ.SVNX.GN.ZS","NY.ADJ.SVNG.GN.ZS","NY.ADJ.DPEM.GN.ZS","GB.XPD.RSDV.GD.ZS","SP.POP.SCIE.RD.P6","SP.POP.TECH.RD.P6","EG.ELC.ACCS.ZS","EG.ELC.ACCS.UR.ZS","EG.ELC.ACCS.RU.ZS","EG.CFT.ACCS.ZS","EG.CFT.ACCS.RU.ZS","EG.CFT.ACCS.UR.ZS","IT.NET.BBND.P2","IT.MLT.MAIN.P2","IT.CEL.SETS.P2","IT.NET.USER.ZS","IT.NET.SECR.P6","IS.SHP.GCNW.XQ","IQ.CPA.PUBS.XQ","GC.REV.XGRT.GD.ZS","GC.XPN.TOTL.GD.ZS","NE.CON.GOVT.ZS","NE.CON.GOVT.KD.ZG","GC.DOD.TOTL.GD.ZS","IC.LGL.CRED.XQ","IC.BUS.EASE.XQ","AG.LND.ARBL.ZS","AG.LND.ARBL.HA.PC","AG.LND.AGRI.ZS","AG.LND.FRST.ZS","AG.LND.CROP.ZS","ER.PTD.TOTL.ZS","ER.LND.PTLD.ZS","ER.MRN.PTMR.ZS","ER.H2O.FWTL.ZS","ER.H2O.FWAG.ZS","ER.H2O.FWDM.ZS","ER.H2O.FWIN.ZS","ER.H2O.INTR.PC","ER.H2O.FWST.ZS","EN.ATM.PM25.MC.M3","EN.ATM.PM25.MC.ZS","EN.ATM.PM25.MC.T1.ZS","EN.ATM.PM25.MC.T2.ZS","EN.ATM.PM25.MC.T3.ZS","AG.LND.PRCP.MM","EN.CLC.MDAT.ZS"]
    years = (2000, 2020)

    core.download_data_batch(countries, indicators, years)

    dataset = core.create_dataset(countries, indicators, years)

    dataset.save(filepath="data/datasets/oecd_2000_raw", format="hdf5")

def preprocess_dataset(name: str, p_value: float = 0.8, folder_path: str = "data/datasets/", raw: str = "oecd_2000_raw") -> Dataset:
    raw_dataset = Dataset.load(f"{folder_path}/{raw}.h5")
    original_indicator_count = raw_dataset.n_indicators()
    print(f"Raw dataset contains {original_indicator_count} indicators.")
    print(f"Raw dataset contains {raw_dataset.n_countries()} countries and {raw_dataset.n_years()} years.")

    # forward declaring a deep copy such that individual steps could be omitted
    preprocessed_dataset: Dataset = raw_dataset.copy()
    prev_indicator_count = raw_dataset.n_indicators()

    # Remove indicators with missing values above a threshold
    preprocessed_dataset, prev_indicator_count = apply_preprocessing_step(
        dataset=preprocessed_dataset,
        func=Preprocessor.remove_missing_per_country,
        func_kwargs={"threshold": 0.5},
        description_template="Removed indicators for missing above a per country threshold of {threshold}.",
        prev_count=prev_indicator_count
    )

    # Remove indicators whose global variance is below a threshold
    preprocessed_dataset, prev_indicator_count = apply_preprocessing_step(
        dataset=preprocessed_dataset,
        func=Preprocessor.remove_variance,
        func_kwargs={"threshold": 0.05},
        description_template="Removed indicators for global variance below {threshold}.",
        prev_count=prev_indicator_count
    )

    # Remove indicators with too many constant countries
    preprocessed_dataset, prev_indicator_count = apply_preprocessing_step(
        dataset=preprocessed_dataset,
        func=Preprocessor.remove_constant,
        func_kwargs={"variance_threshold": 0.05, "country_threshold": 0.5},
        description_template="Removed indicators with too many constant countries (variance < {variance_threshold} and min active countries < {country_threshold}).",
        prev_count=prev_indicator_count
    )

    # Removal based on correlation
    # Making a copy of the dataset to avoid modifying the original to avoid leakage
    print("Removing indicators based on correlation...")
    pruned_dataset = apply_correlation_pruning(preprocessed_dataset, threshold=p_value)

    removed_indicators = set(preprocessed_dataset.indicators) - set(pruned_dataset.indicators)

    print(f"Removed {len(removed_indicators)} indicators based on correlation pruning. Remaining indicators: {pruned_dataset.n_indicators()}")

    preprocessed_dataset = exclude_indicators_from_dataset(preprocessed_dataset, list(removed_indicators))

    preprocessed_dataset.save(f"{folder_path}/{name}", format='hdf5')

    return preprocessed_dataset

def ensure_test_complete(dataset: Dataset, name: str, folder_path: str = "data/datasets/"):
    # iterate over all countries, and find the country with least amount of missing values, to use as test set
    country_missing_values = {} # country -> (total missing values ratio, number of indicators with missing values)
    for country in dataset.countries:
        country_data = dataset.extract_country(country)
        total_missing_values = np.sum(np.isnan(country_data))
        indicators_with_missing_values = np.sum(np.isnan(country_data).any(axis=1))
        country_missing_values[country] = (total_missing_values / country_data.size, indicators_with_missing_values)

    print(country_missing_values)

    n_countries_to_take = dataset.n_countries() * 0.2

    sorted_countries = sorted(country_missing_values.items(), key=lambda x: x[1][0])
    least_missing_countries = [country for country, _ in sorted_countries[:int(n_countries_to_take)]]
    # print(f"Countries with least missing values: {least_missing_countries}")
    # indicators with missing values in these countries
    indicators_with_missing_values = set()
    for country in least_missing_countries:
        country_data = dataset.extract_country(country)
        missing_indicators = np.isnan(country_data).any(axis=1)
        # print(missing_indicators.shape)
        missing_indicators = np.where(missing_indicators)[0]
        # print(missing_indicators)
        missing_indicators = dataset.indicators[missing_indicators]
        # print(missing_indicators)
        indicators_with_missing_values.update(missing_indicators)
        # print(np.sum(missing_indicators))
        # print(missing_indicators)
    # print(f"Indicators with missing values in these countries: {len(indicators_with_missing_values)}")
    # print(indicators_with_missing_values)
    preprocessed_dataset = exclude_indicators_from_dataset(dataset, list(indicators_with_missing_values))
    # print(f"Remaining indicators after excluding those with missing values in the least missing countries: {preprocessed_dataset.n_indicators()}")
    preprocessed_dataset.save(f"{folder_path}/{name}", format='hdf5')

def preprocess_data():
    dataset_70 = preprocess_dataset("oecd_2000_70", p_value=0.7)
    dataset_80 = preprocess_dataset("oecd_2000_80", p_value=0.8)
    dataset_90 = preprocess_dataset("oecd_2000_90", p_value=0.9)

    ensure_test_complete(dataset_80, "oecd_2000_80_t")



if __name__ == "__main__":
    download_raw_data()
    preprocess_data()