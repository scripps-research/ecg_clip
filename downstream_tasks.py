# Copyright (c) 2026 The Scripps Research Institute
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import argparse
import numpy as np
import pandas as pd
import random

from pathlib import Path


# ECG report keywords
AMI_KWDS = [
    "acute anterior infarct",
    "acute anterolateral infarct",
    "acute anteroseptal infarct",
    "acute extensive infarct",
    "acute inferior and lateral infarct",
    "acute inferior infarct",
    "acute infarct",
    "acute lateral infarct",
    "acute septal infarct",
    "acute st elevation mi",
    "infarct, acute",
]
AFLUT_KWDS = [
    "a-flutter",
    "afib/flut",
    "atrial flutter",
]
AF_KWDS = [
    "a-flutter/fibrillation",
    "afib",
    "atrial fibrillation",
]
NSR_KWDS = ["sinus rhythm"]

# ICD code classifications
HCM_ICD = ["I421", "I422", "4251", "42511", "42518"]
AMYLOID_ICD = ["2773", "E85"]
CARDIAC_INVOLVEMENT_ICD = [
    "42731", "I48", "428", "I110",
    "I130", "I132", "I50", "425",
    "I42", "I43", "I255",
]

# Lab item IDs
HBA1C_ITEMID = 50852
CR_ITEMID = 50920


def preprocess_death(patients, admissions):
    """Merge admission and patient death times, preferring deathtime over dod."""
    admit_deaths = (
        admissions.loc[admissions["deathtime"].notna(), ["subject_id", "deathtime"]]
        .drop_duplicates()
    )
    df = patients[["subject_id", "dod"]].merge(admit_deaths, how="left")
    df["death"] = df["deathtime"].fillna(df["dod"])
    return df[["subject_id", "death"]]


def preprocess_diagnosis(diagnosis):
    """Flag ICD-based diagnoses and aggregate to hospital admission level."""
    diagnosis["hcm_icd"] = diagnosis["icd_code"].str.match("|".join(HCM_ICD))
    diagnosis["amyloid_icd"] = diagnosis["icd_code"].str.contains("|".join(AMYLOID_ICD))
    diagnosis["cardiac_involvement_icd"] = diagnosis["icd_code"].str.contains("|".join(CARDIAC_INVOLVEMENT_ICD))
    diagnosis = diagnosis.drop(["seq_num", "icd_code", "icd_version"], axis=1)
    diagnosis = diagnosis.groupby(["subject_id", "hadm_id"]).any().reset_index()
    diagnosis["ca_icd"] = diagnosis["amyloid_icd"] & diagnosis["cardiac_involvement_icd"]
    return diagnosis


def preprocess_procedures(procedures, hcup_icd9_pth, hcup_icd10_pth):
    """Map ICD procedure codes to HCUP classes; return days with major therapeutic procedures."""
    hcup_icd9 = pd.read_csv(hcup_icd9_pth, skiprows=2, encoding="unicode_escape")
    hcup_icd9.columns = ["icd_code", "icd_description", "hcup"]
    hcup_icd10 = pd.read_csv(hcup_icd10_pth, skiprows=1)
    hcup_icd10.columns = ["icd_code", "icd_description", "hcup", "hcup_name"]
    # ICD codes in both HCUP files are stored as quoted strings; [1:-1] strips the surrounding quotes
    hcup_icd9_map = dict(zip(hcup_icd9["icd_code"].str[1:-1].str.strip(), hcup_icd9["hcup"]))
    hcup_icd10_map = dict(zip(hcup_icd10["icd_code"].str[1:-1].str.strip(), hcup_icd10["hcup"]))
    hcup_icd_map = hcup_icd9_map | hcup_icd10_map

    # HCUP class 4 = major therapeutic procedures (surgery)
    procedures["hcup"] = procedures["icd_code"].map(hcup_icd_map)
    procedures["surgery"] = procedures["hcup"] == 4
    procedures = (
        procedures[["subject_id", "hadm_id", "chartdate", "surgery"]]
        .groupby(["subject_id", "hadm_id", "chartdate"])
        .any()
        .reset_index()
    )
    return procedures[procedures["surgery"]].reset_index(drop=True)


def preprocess_reports(machine_measurements):
    """Extract and keyword-match ECG reports into binary labels."""
    reports = machine_measurements[[f"report_{i}" for i in range(18)]]
    reports = reports.fillna("").agg(". ".join, axis=1).str.lower()
    return pd.DataFrame({
        "study_id": machine_measurements["study_id"],
        "nsr": reports.str.contains("|".join(NSR_KWDS), na=False),
        "afib": reports.str.contains("|".join(AF_KWDS), na=False),
        "aflutter": reports.str.contains("|".join(AFLUT_KWDS), na=False),
        "acute_mi": reports.str.contains("|".join(AMI_KWDS), na=False),
        "report": reports
    })


def acute_mi(records, ecg_labels):
    """Return one ECG per patient with acute MI labels derived from ECG report keywords."""
    df = records.merge(ecg_labels[["study_id", "acute_mi"]])
    df = (
        df
        .sort_values("ecg_time")
        .groupby("subject_id")
        .first()
        .reset_index()
    )
    return df[["subject_id", "study_id", "path", "acute_mi"]]


def ca(records, admissions, diagnosis):
    """Return one ECG per patient with a cardiac amyloidosis (CA) label.

    Positive: earliest ECG within 180 days before the first CA diagnosis admission.
    Negative: earliest ECG from patients with no CA diagnosis.
    """
    ca_diagnosis = diagnosis[diagnosis["ca_icd"]].merge(
        admissions[["hadm_id", "admittime"]], how="left"
    )
    index_time = (
        ca_diagnosis
        .sort_values("admittime")
        .groupby("subject_id")
        .first()
        .reset_index()
    )
    index_time = index_time.merge(records[["subject_id", "study_id", "ecg_time"]], how="left")
    # keep ECGs in the 180-day window before the first CA-diagnosis admission
    index_time = index_time[index_time["ecg_time"] < index_time["admittime"]]
    index_time = index_time[index_time["admittime"] < index_time["ecg_time"] + pd.Timedelta(days=180)]

    pos = records[records["study_id"].isin(index_time["study_id"])].copy()
    pos = pos[["subject_id", "study_id", "ecg_time", "path"]]
    pos["ca"] = True

    ca_subject_ids = ca_diagnosis["subject_id"].unique()
    neg = records[~records["subject_id"].isin(ca_subject_ids)].copy()
    neg["ca"] = False

    df = pd.concat([pos, neg], ignore_index=True)
    df = (
        df
        .sort_values("ecg_time")
        .groupby("subject_id")
        .first()
        .reset_index()
    )
    return df[["subject_id", "study_id", "path", "ca"]]


def hcm(records, diagnosis):
    """Return one ECG per patient with a binary HCM ICD label."""
    hcm_flags = diagnosis.groupby("subject_id")["hcm_icd"].any().reset_index()
    df = records.merge(hcm_flags, how="inner")
    df = (
        df
        .sort_values("ecg_time")
        .groupby("subject_id")
        .first()
        .reset_index()
        .rename(columns={"hcm_icd": "hcm"})
    )
    return df[["subject_id", "study_id", "path", "hcm"]]


def afib(records, ecg_labels):
    """Return one index NSR ECG per patient for AF prediction.

    Positive (AF/AFL patients): earliest NSR ECG within 31 days before first AF/AFL ECG.
    Negative (never-AF patients): earliest NSR ECG.
    """
    df = records.merge(ecg_labels[["study_id", "nsr", "afib", "aflutter"]])

    pos_index = (
        df[df["afib"] | df["aflutter"]]
        .groupby("subject_id")["ecg_time"].min()
        .sub(pd.Timedelta(days=31))
        .rename("index_time")
        .reset_index()
    )

    nsr = df[df["nsr"]]
    afib_subject_ids = df[df["afib"] | df["aflutter"]]["subject_id"].unique()
    pos_nsr = nsr[nsr["subject_id"].isin(afib_subject_ids)]
    neg_nsr = nsr[~nsr["subject_id"].isin(afib_subject_ids)]

    neg = neg_nsr[["subject_id", "study_id", "path", "ecg_time"]].copy()
    neg["af_prediction"] = False

    pos = pos_nsr.merge(pos_index)
    pos = pos[pos["ecg_time"] >= pos["index_time"]]
    pos = pos[["subject_id", "study_id", "path", "ecg_time"]].copy()
    pos["af_prediction"] = True

    df = pd.concat([pos, neg], ignore_index=True)
    df = (
        df
        .sort_values("ecg_time")
        .groupby("subject_id")
        .first()
        .reset_index()
    )
    return df[["subject_id", "study_id", "path", "af_prediction"]]


def ed_mortality(ed):
    """Return one ECG per patient with 30-day ED mortality label.

    Uses the first ECG per ED stay (ecg_no_within_stay == 0), then selects
    the earliest qualifying ECG across all stays per patient.
    """
    ed = ed.copy()
    ed.columns = ed.columns.str.removeprefix("general_")
    ed = ed.rename(columns={"file_name": "path"})
    ed["path"] = ed["path"].str.split("/", n=1).str[1]  # strip leading path component
    ed = ed.replace(-999, pd.NA)
    df = ed[ed["deterioration_mortality_28d"].notna()]
    df = df[df["ecg_no_within_stay"] == 0].copy()
    df["ed_mortality_30d"] = df["mortality_days"] <= 30
    df = (
        df
        .sort_values("ecg_time")
        .groupby("subject_id")
        .first()
        .reset_index()
    )
    return df[["subject_id", "study_id", "path", "ed_mortality_30d"]]


def surgery_mortality(records, procedures, death):
    """Return one ECG per patient with 30-day post-surgery mortality label.

    Selects the earliest ECG recorded 0–14 days before a surgical procedure.
    """
    df = procedures.merge(death, how="left")
    df["surgery_mortality_30d"] = (
        (df["death"].dt.date >= df["chartdate"])
        & (df["death"].dt.date <= df["chartdate"] + pd.Timedelta(days=30))
    )
    df = df.merge(records, how="left")
    df = df[df["chartdate"] > df["ecg_time"]]
    df = df[df["chartdate"] <= df["ecg_time"].dt.date + pd.Timedelta(days=14)]
    df = (
        df
        .sort_values(["ecg_time", "chartdate"])
        .groupby("subject_id")
        .first()
        .reset_index()
    )
    return df[["subject_id", "study_id", "path", "surgery_mortality_30d"]]


def ckd(records, labs, patients, admissions):
    """Return one ECG per patient with a 3-year chronic kidney disease (CKD) risk label.

    Positive: earliest ECG followed by eGFR <= 45 within 3 years.
    Negative: earliest ECG from patients with creatinine measurements but eGFR always > 45.
    eGFR is computed via the CKD-EPI formula.
    """
    lab_value = labs[labs["itemid"] == CR_ITEMID].copy()
    lab_value["cr"] = lab_value["comments"].str.extract(
        r"serum creatinine value of (\d+\.\d+)"
    ).astype(float)
    lab_value = lab_value[lab_value["cr"].notna()].reset_index(drop=True)

    pts = patients[["subject_id", "gender", "anchor_age", "anchor_year"]]
    lab_value = lab_value.merge(pts, how="left")
    lab_value["gender_F"] = (lab_value["gender"] == "F").astype(float)
    lab_value["age"] = lab_value["anchor_age"] + (lab_value["charttime"] - lab_value["anchor_year"]).dt.days / 365.25
    lab_value["age"] = lab_value["age"].round()

    ethnicity = (
        admissions[["subject_id", "race"]]
        .assign(black=lambda x: x["race"].str.contains("BLACK"))
        .groupby("subject_id")["black"]
        .any()
        .reset_index()
    )
    lab_value = lab_value.merge(ethnicity, how="left")
    lab_value["black"] = lab_value["black"].fillna(False).astype(float)

    lab_value = lab_value[lab_value["age"].notna() & lab_value["gender"].notna()]
    lab_value = lab_value[["subject_id", "charttime", "cr", "age", "gender_F", "black"]]
    lab_value = lab_value.reset_index(drop=True)

    # CKD-EPI eGFR formula
    lab_value["eGFR"] = (
        175 * lab_value["cr"] ** (-1.154)
        * lab_value["age"] ** (-0.203)
        * 0.7642 ** lab_value["gender_F"]
        * 1.212 ** lab_value["black"]
    )
    lab_value["ckd"] = lab_value["eGFR"] <= 45
    lab_value = lab_value[["subject_id", "charttime", "ckd"]]

    pos = (
        lab_value[lab_value["ckd"]]
        .sort_values("charttime")
        .groupby("subject_id")
        .first()
        .reset_index()
    )
    pos = pos.merge(records, how="left")
    pos = pos[pos["charttime"] > pos["ecg_time"]]
    pos = pos[pos["charttime"] <= pos["ecg_time"] + pd.Timedelta(days=365 * 3)]
    pos = pos[["subject_id", "study_id", "path", "ecg_time"]]
    pos["ckd"] = True

    neg = records[records["subject_id"].isin(lab_value["subject_id"])]
    neg = neg[~neg["subject_id"].isin(lab_value[lab_value["ckd"]]["subject_id"])]
    neg = neg[["subject_id", "study_id", "path", "ecg_time"]]
    neg["ckd"] = False

    df = pd.concat([pos, neg], ignore_index=True)
    df = (
        df
        .sort_values("ecg_time")
        .groupby("subject_id")
        .first()
        .reset_index()
    )
    df["study_id"] = df["study_id"].astype(int)
    return df[["subject_id", "study_id", "path", "ckd"]]


def dm(records, labs):
    """Return one ECG per patient with a 3-year diabetes mellitus (DM) risk label.

    Positive: earliest ECG followed by HbA1c >= 6.5 within 3 years.
    Negative: earliest ECG from patients with HbA1c measurements but never >= 6.5.
    """
    lab_value = labs[labs["itemid"] == HBA1C_ITEMID].copy()
    lab_value["dm"] = lab_value["valuenum"] >= 6.5

    pos = (
        lab_value[lab_value["dm"]]
        .sort_values("charttime")
        .groupby("subject_id")
        .first()
        .reset_index()
    )
    pos = pos.merge(records, how="left")
    pos = pos[pos["charttime"] > pos["ecg_time"]]
    pos = pos[pos["charttime"] <= pos["ecg_time"] + pd.Timedelta(days=365 * 3)]
    pos = pos[["subject_id", "study_id", "path", "ecg_time"]]
    pos["dm"] = True

    neg = records[records["subject_id"].isin(lab_value["subject_id"])]
    neg = neg[~neg["subject_id"].isin(lab_value[lab_value["dm"]]["subject_id"])]
    neg = neg[["subject_id", "study_id", "path", "ecg_time"]]
    neg["dm"] = False

    df = pd.concat([pos, neg], ignore_index=True)
    df = (
        df
        .sort_values("ecg_time")
        .groupby("subject_id")
        .first()
        .reset_index()
    )
    df["study_id"] = df["study_id"].astype(int)
    return df[["subject_id", "study_id", "path", "dm"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create downstream task datasets from MIMIC-IV ECG data."
    )
    parser.add_argument("--mimic_dir", type=Path, required=True,
                        help="Path to MIMIC-IV root directory")
    parser.add_argument("--mimic_ecg_dir", type=Path, required=True,
                        help="Path to MIMIC-IV-ECG root directory")
    parser.add_argument("--mimic_ext_mds_ed_dir", type=Path, required=True,
                        help="Path to MIMIC-IV-ED MDS root directory")
    parser.add_argument("--hcup_icd9_pth", type=Path, required=True,
                        help="Path to HCUP ICD-9 procedure class CSV")
    parser.add_argument("--hcup_icd10_pth", type=Path, required=True,
                        help="Path to HCUP ICD-10 procedure class CSV")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory where downstream task CSVs will be saved")
    parser.add_argument("--random_state", type=int, default=2026)
    args, _ = parser.parse_known_args()

    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # read tables
    records = pd.read_csv(args.mimic_ecg_dir / "record_list.csv")
    machine_measurements = pd.read_csv(
        args.mimic_ecg_dir / "machine_measurements.csv",
        low_memory=False,
    )
    patients = pd.read_csv(args.mimic_dir / "hosp/patients.csv")
    diagnosis = pd.read_csv(args.mimic_dir / "hosp/diagnoses_icd.csv")
    procedures = pd.read_csv(args.mimic_dir / "hosp/procedures_icd.csv")
    admissions = pd.read_csv(args.mimic_dir / "hosp/admissions.csv")
    ed = pd.read_csv(
        args.mimic_ext_mds_ed_dir / "mds_ed.csv",
        usecols=[
            "general_file_name",
            "general_subject_id",
            "general_study_id",
            "general_ecg_time",
            "general_ecg_no_within_stay",
            "general_mortality_days",
            "deterioration_mortality_28d",
        ],
        low_memory=False,
    )

    labs = pd.read_csv(
        args.mimic_dir / "hosp/labevents.csv",
        usecols=["subject_id", "itemid", "charttime", "valuenum", "comments"],
    )
    labs = labs[labs["itemid"].isin([HBA1C_ITEMID, CR_ITEMID])].reset_index(drop=True)

    # restrict all tables to subjects with at least one ECG
    subject_ids = records["subject_id"].unique()
    patients = patients[patients["subject_id"].isin(subject_ids)].reset_index(drop=True)
    diagnosis = diagnosis[diagnosis["subject_id"].isin(subject_ids)].reset_index(drop=True)
    procedures = procedures[procedures["subject_id"].isin(subject_ids)].reset_index(drop=True)
    admissions = admissions[admissions["subject_id"].isin(subject_ids)].reset_index(drop=True)

    # parse datetimes
    records["ecg_time"] = pd.to_datetime(records["ecg_time"])
    patients["anchor_year"] = pd.to_datetime(patients["anchor_year"], format="%Y")
    patients["dod"] = pd.to_datetime(patients["dod"])
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"])
    procedures["chartdate"] = pd.to_datetime(procedures["chartdate"])
    ed["general_ecg_time"] = pd.to_datetime(ed["general_ecg_time"])
    labs["charttime"] = pd.to_datetime(labs["charttime"])

    # preprocess shared data structures
    ecg_labels = preprocess_reports(machine_measurements)
    diagnosis = preprocess_diagnosis(diagnosis)
    death = preprocess_death(patients, admissions)
    procedures = preprocess_procedures(procedures, args.hcup_icd9_pth, args.hcup_icd10_pth)

    # create and save task datasets
    acute_mi(records, ecg_labels).to_csv(args.output_dir / "acute_mi.csv", index=False)
    ca(records, admissions, diagnosis).to_csv(args.output_dir / "ca.csv", index=False)
    hcm(records, diagnosis).to_csv(args.output_dir / "hcm.csv", index=False)
    afib(records, ecg_labels).to_csv(args.output_dir / "af_prediction.csv", index=False)
    ed_mortality(ed).to_csv(args.output_dir / "ed_mortality_30d.csv", index=False)
    surgery_mortality(records, procedures, death).to_csv(args.output_dir / "surgery_mortality_30d.csv", index=False)
    ckd(records, labs, patients, admissions).to_csv(args.output_dir / "ckd.csv", index=False)
    dm(records, labs).to_csv(args.output_dir / "dm.csv", index=False)

